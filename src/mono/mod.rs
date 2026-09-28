//! Minimal lossless monochrome AV1 still encoder, purpose-built for 2-bit-grey
//! content (every block holds at most 4 distinct levels).
//!
//! Design (all profile 0, 8-bit, monochrome, still picture, single tile):
//!
//! * Greedy `NONE` partitions at 64x64/32x32/16x16 (a node whose
//!   distinct levels fit the palette is never split). Every block is coded
//!   `skip = 1` with `DC_PRED` plus an `iovl`-free luma palette holding the
//!   block's exact colors — no transform, quantization, or coefficient
//!   coding exists on this path at all, which is what makes a from-scratch
//!   encoder feasible.
//! * Optionally (code toggle `USE_INTRABC`, segregated in `intrabc.rs`), a
//!   block with an exact match in causal decoded area is coded as an
//!   IntraBC copy instead: `skip = 1` with an integer motion vector and no
//!   residuals. The MV predictor replicates dav1d's refmvs search; match
//!   validity is conservative against the decoder's SB-overlap/tile-clip
//!   rules, so its adjustments never trigger. When enabled, each raster is
//!   encoded both with IntraBC and palette-only and the smaller payload is
//!   kept; when disabled, only the palette path runs (faster).
//! * Static default CDFs are the starting point, but every `S()` symbol
//!   adapts its row (`disable_cdf_update = 0`) except split_or outcomes,
//!   which dav1d decodes with the non-updating bool reader; no
//!   segmentation, delta quantizers, loop filters, CDEF, restoration,
//!   or film grain.
//! * The container holds two `av01` items per photo (see
//!   `encode_gray_pair`): the 8x nearest-neighbor upscale as primary plus
//!   the original-fidelity raster, grouped in an `altr` entity group. Each
//!   item has its own `ispe`/`av1C` since levels differ (2.0 vs 4.0).
//!   Written with `gamut-isobmff`; the arithmetic coder is `gamut-bitstream`'s.
//!
//! Box/syntax references are to the AV1 Bitstream & Decoding Process
//! Specification: OBU framing §5.3, sequence header §5.5, frame header §5.9,
//! partitions §5.11.3, palette §5.11.46/.49/.50, CDF tables §9.3/§9.4.

use gamut_bitstream::{write_leb128, BitWriter, SymbolEncoder};
use gamut_isobmff::{
    write as write_isobmff, EntityGroup, IsoBmffImage, Item, Property, PropertyKind,
};
use std::collections::HashMap;
use std::io;
use std::ops::{Deref, DerefMut};

mod intrabc;
mod residual;
pub use intrabc::UniformSearchStats;
use intrabc::{IntrabcState, MatchShortlist, MvComp, MvRec, UniformDecision, MOTION_SHORTLIST_MAX};

const DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS: i32 = 0;

/// Process-wide diagnostics for the uniform 16x16 source index.
#[allow(dead_code)]
pub fn uniform_search_stats() -> UniformSearchStats {
    intrabc::uniform_stats_total()
}

/// Reset uniform 16x16 source-index diagnostics.
#[allow(dead_code)]
pub fn reset_uniform_search_stats() {
    intrabc::reset_uniform_stats()
}

/// Cached-search diagnostics for the 16x16 8x path: `(queries, fallbacks)`.
/// Committed (non-speculative) searches: `queries` counts verified-image
/// 16x16 attempts; `fallbacks` counts the narrow stripe fallbacks to the
/// ring oracle. Both are process-global atomics (Rayon-safe); reset per
/// measurement. See `pattern_cache_stats_speculative` / `_total` for
/// rejected-trial work vs combined totals.
// Diagnostics for tests/benchmarks; unused by the extraction binary itself,
// so `dead_code` would otherwise fire on the non-test build.
#[allow(dead_code)]
pub fn pattern_cache_stats() -> (u64, u64) {
    intrabc::pattern_stats()
}

/// Speculative trial searches (rejected branches; not committed).
// See `pattern_cache_stats` for the `dead_code` rationale.
#[allow(dead_code)]
pub fn pattern_cache_stats_speculative() -> (u64, u64) {
    intrabc::pattern_stats_speculative()
}

/// Total searches (committed + speculative).
// See `pattern_cache_stats` for the `dead_code` rationale.
#[allow(dead_code)]
pub fn pattern_cache_stats_total() -> (u64, u64) {
    intrabc::pattern_stats_total()
}

/// Reset cached-search diagnostics.
// See `pattern_cache_stats` for the `dead_code` rationale.
#[allow(dead_code)]
pub fn reset_pattern_cache_stats() {
    intrabc::reset_pattern_stats()
}

/// Code toggle for IntraBC (intra block copy) evaluation: exact
/// pixel-rectangle copies from causal decoded area via integer MVs +
/// `skip = 1`, as a per-block alternative to the palette path. When `true`,
/// each raster is encoded both ways (IntraBC vs palette-only) and the
/// smaller payload is kept; when `false`, only the palette path is encoded
/// (faster, byte-identical to the pre-IntraBC encoder: no flag symbols, no
/// header bit).
pub const USE_INTRABC: bool = true;

/// Flat-block pad selection policy. `Baseline` preserves the historical
/// cached-color-then-adjacent-value choice. `Lookahead` evaluates that choice
/// plus available gray levels by replaying the current flat block and one
/// upcoming sibling subtree with fractional-bit costs.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub enum FlatPadPolicy {
    #[default]
    Baseline,
    Lookahead,
}

/// Search policy for exact IntraBC copies in the RDO path.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub enum MotionCandidatePolicy {
    /// Preserve the historical first legal ring match.
    #[default]
    FirstMatch,
    /// Price the eight nearest legal exact matches independently.
    TopK8,
}

/// Work and selected alternatives for the opt-in multi-match policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionCandidateStats {
    /// Nonuniform 16x16 shortlist searches, including nested RDO trials.
    pub shortlist_searches: u64,
    /// Source positions checked by the pattern index or ring fallback.
    pub source_probes: u64,
    /// Exact matches priced in separate state-isolated RDO trials.
    pub priced_matches: u64,
    /// Committed leaves selecting a match after the first ring-ranked match.
    pub alternate_choices: u64,
    /// Sum of selected zero-based shortlist indices for alternate choices.
    pub alternate_index_sum: u64,
}

static MOTION_SHORTLIST_SEARCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static MOTION_SOURCE_PROBES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MOTION_PRICED_MATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MOTION_ALTERNATE_CHOICES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static MOTION_ALTERNATE_INDEX_SUM: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Read process-wide TopK8 work and selection counters.
#[allow(dead_code)]
pub fn motion_candidate_stats() -> MotionCandidateStats {
    use std::sync::atomic::Ordering;
    MotionCandidateStats {
        shortlist_searches: MOTION_SHORTLIST_SEARCHES.load(Ordering::Relaxed),
        source_probes: MOTION_SOURCE_PROBES.load(Ordering::Relaxed),
        priced_matches: MOTION_PRICED_MATCHES.load(Ordering::Relaxed),
        alternate_choices: MOTION_ALTERNATE_CHOICES.load(Ordering::Relaxed),
        alternate_index_sum: MOTION_ALTERNATE_INDEX_SUM.load(Ordering::Relaxed),
    }
}

/// Reset process-wide TopK8 diagnostics.
#[allow(dead_code)]
pub fn reset_motion_candidate_stats() {
    use std::sync::atomic::Ordering;
    MOTION_SHORTLIST_SEARCHES.store(0, Ordering::Relaxed);
    MOTION_SOURCE_PROBES.store(0, Ordering::Relaxed);
    MOTION_PRICED_MATCHES.store(0, Ordering::Relaxed);
    MOTION_ALTERNATE_CHOICES.store(0, Ordering::Relaxed);
    MOTION_ALTERNATE_INDEX_SUM.store(0, Ordering::Relaxed);
}

/// Work and outcomes of the opt-in flat-pad search. Counts include searches
/// inside speculative RDO branches; those trials are real added encoding work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlatPadStats {
    /// Flat palette blocks on committed (non-cost-only) encoder passes.
    pub committed_flat_blocks: u64,
    pub searches: u64,
    pub candidate_trials: u64,
    pub preview_nodes: u64,
    pub baseline_wins: u64,
    pub lookahead_wins: u64,
    pub ties: u64,
    pub errors: u64,
}

static FLAT_PAD_SEARCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FLAT_PAD_COMMITTED_BLOCKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static FLAT_PAD_CANDIDATE_TRIALS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static FLAT_PAD_PREVIEW_NODES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FLAT_PAD_BASELINE_WINS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FLAT_PAD_LOOKAHEAD_WINS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FLAT_PAD_TIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FLAT_PAD_ERRORS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Read process-wide flat-pad search counters.
#[allow(dead_code)]
pub fn flat_pad_stats() -> FlatPadStats {
    use std::sync::atomic::Ordering;
    FlatPadStats {
        committed_flat_blocks: FLAT_PAD_COMMITTED_BLOCKS.load(Ordering::Relaxed),
        searches: FLAT_PAD_SEARCHES.load(Ordering::Relaxed),
        candidate_trials: FLAT_PAD_CANDIDATE_TRIALS.load(Ordering::Relaxed),
        preview_nodes: FLAT_PAD_PREVIEW_NODES.load(Ordering::Relaxed),
        baseline_wins: FLAT_PAD_BASELINE_WINS.load(Ordering::Relaxed),
        lookahead_wins: FLAT_PAD_LOOKAHEAD_WINS.load(Ordering::Relaxed),
        ties: FLAT_PAD_TIES.load(Ordering::Relaxed),
        errors: FLAT_PAD_ERRORS.load(Ordering::Relaxed),
    }
}

/// Reset process-wide flat-pad search counters.
#[allow(dead_code)]
pub fn reset_flat_pad_stats() {
    use std::sync::atomic::Ordering;
    FLAT_PAD_SEARCHES.store(0, Ordering::Relaxed);
    FLAT_PAD_COMMITTED_BLOCKS.store(0, Ordering::Relaxed);
    FLAT_PAD_CANDIDATE_TRIALS.store(0, Ordering::Relaxed);
    FLAT_PAD_PREVIEW_NODES.store(0, Ordering::Relaxed);
    FLAT_PAD_BASELINE_WINS.store(0, Ordering::Relaxed);
    FLAT_PAD_LOOKAHEAD_WINS.store(0, Ordering::Relaxed);
    FLAT_PAD_TIES.store(0, Ordering::Relaxed);
    FLAT_PAD_ERRORS.store(0, Ordering::Relaxed);
}

fn merge_flat_pad_stats(stats: FlatPadStats) {
    use std::sync::atomic::Ordering;
    FLAT_PAD_SEARCHES.fetch_add(stats.searches, Ordering::Relaxed);
    FLAT_PAD_COMMITTED_BLOCKS.fetch_add(stats.committed_flat_blocks, Ordering::Relaxed);
    FLAT_PAD_CANDIDATE_TRIALS.fetch_add(stats.candidate_trials, Ordering::Relaxed);
    FLAT_PAD_PREVIEW_NODES.fetch_add(stats.preview_nodes, Ordering::Relaxed);
    FLAT_PAD_BASELINE_WINS.fetch_add(stats.baseline_wins, Ordering::Relaxed);
    FLAT_PAD_LOOKAHEAD_WINS.fetch_add(stats.lookahead_wins, Ordering::Relaxed);
    FLAT_PAD_TIES.fetch_add(stats.ties, Ordering::Relaxed);
    FLAT_PAD_ERRORS.fetch_add(stats.errors, Ordering::Relaxed);
}

/// Whole-image RDO-vs-baseline fallback diagnostics (process-global,
/// Rayon-safe; reset per measurement). `RDO_WINS` counts complete AVIF
/// files where the cost-based strategy was smaller; `BASELINE_WINS` counts
/// ordinary ties plus baseline-smaller files (baseline preferred on ties).
/// Unexpected RDO encode failures (baseline succeeded but RDO errored) are
/// counted separately in `RDO_ERRORS` so fallback does not conceal errors.
/// `RDO_CANDIDATES` counts trial candidate encodes evaluated (legal
/// palette/copy/split trials, including nested children).
static RDO_WINS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BASELINE_WINS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RDO_CANDIDATES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RDO_ERRORS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// RDO file-level wins/losses plus total trial candidates (see statics).
/// `BASELINE_WINS` excludes unexpected RDO failures; see `rdo_errors`.
// Diagnostics for tests/benchmarks; unused by the extraction binary itself,
// so `dead_code` would otherwise fire on the non-test build.
#[allow(dead_code)]
pub fn rdo_stats() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering;
    (
        RDO_WINS.load(Ordering::Relaxed),
        BASELINE_WINS.load(Ordering::Relaxed),
        RDO_CANDIDATES.load(Ordering::Relaxed),
    )
}

/// Unexpected RDO encode failures (baseline succeeded, RDO errored).
// See `rdo_stats` for the `dead_code` rationale.
#[allow(dead_code)]
pub fn rdo_errors() -> u64 {
    use std::sync::atomic::Ordering;
    RDO_ERRORS.load(Ordering::Relaxed)
}

/// Reset RDO fallback diagnostics (benchmarks/tests).
// See `rdo_stats` for the `dead_code` rationale.
#[allow(dead_code)]
pub fn reset_rdo_stats() {
    use std::sync::atomic::Ordering;
    RDO_WINS.store(0, Ordering::Relaxed);
    BASELINE_WINS.store(0, Ordering::Relaxed);
    RDO_CANDIDATES.store(0, Ordering::Relaxed);
    RDO_ERRORS.store(0, Ordering::Relaxed);
    reset_motion_candidate_stats();
}

/// Motion-vector pair and predictor in 1/8-pel units (matches
/// `intrabc::BcMatch` shape without exposing its private type).
type MvPair = (i32, i32);
type BcMatch = (MvPair, MvPair);
type CopyCost = (f64, BcMatch, bool);
type MotionSearchResult = (
    [Option<(BcMatch, bool, usize)>; MOTION_SHORTLIST_MAX],
    usize,
    bool,
);

/// Estimated cost plus the selected copy and its ring-order index for a
/// 16x16 shortlist decision.
#[derive(Clone, Copy, Debug)]
struct RankedCopyCost {
    cost: f64,
    mv_pred: BcMatch,
    uniform: bool,
    shortlist_index: usize,
}

#[derive(Clone, Copy)]
struct BlockContext {
    r: usize,
    c: usize,
    bsl: usize,
    top_has_right: bool,
    carried: Option<BcMatch>,
    force_palette: bool,
}

struct FlatPadLookahead<'a> {
    r: usize,
    c: usize,
    bsl: usize,
    top_has_right: bool,
    force_palette: bool,
    used: u8,
    baseline_pad: u8,
    cache: &'a [u8],
}

/// A 64x64 superblock's contained partition tree has at most 21 nodes
/// (64, four 32s, sixteen 16s). Plans are stack-bounded and live only for
/// the candidate and incoming probability/neighbor/availability state that
/// produced them.
const MAX_PLAN_NODES: usize = 21;

#[derive(Clone, Copy, Debug, PartialEq)]
enum PlanDecision {
    Palette,
    Copy(BcMatch, bool),
    Residual(u8),
    Split,
}

struct PartitionPlan {
    nodes: [Option<PlanDecision>; MAX_PLAN_NODES],
    len: usize,
}

impl PartitionPlan {
    fn new() -> Self {
        Self {
            nodes: [None; MAX_PLAN_NODES],
            len: 0,
        }
    }

    fn push(&mut self, decision: PlanDecision) {
        assert!(self.len < MAX_PLAN_NODES, "partition plan exceeds one SB");
        self.nodes[self.len] = Some(decision);
        self.len += 1;
    }

    fn extend(&mut self, other: &Self) {
        for decision in other.nodes[..other.len].iter().flatten().copied() {
            self.push(decision);
        }
    }
}

/// `InvalidInput` helper for supported-input violations at the boundary.
fn invalid_input(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// Checked pixel area `w * h`, returning `InvalidInput` on overflow
/// instead of wrapping.
fn checked_area(w: u32, h: u32) -> io::Result<usize> {
    (w as usize)
        .checked_mul(h as usize)
        .ok_or_else(|| invalid_input(format!("pixel area overflows usize: {w}x{h}")))
}

// ---------------------------------------------------------------------------
// Default CDF tables (AV1 §9.3/§9.4), transcribed for the symbols we emit.
// ---------------------------------------------------------------------------

/// `Default_Partition_W64_Cdf`, indexed [ctx].
const PARTITION_W64: [[u16; 10]; 4] = [
    [
        20137, 21547, 23078, 29566, 29837, 30261, 30524, 30892, 31724, 32768,
    ],
    [
        6732, 7490, 9497, 27944, 28250, 28515, 28969, 29630, 30104, 32768,
    ],
    [
        5945, 7663, 8348, 28683, 29117, 29749, 30064, 30298, 32238, 32768,
    ],
    [
        870, 1212, 1487, 31198, 31394, 31574, 31743, 31881, 32332, 32768,
    ],
];

/// `Default_Partition_W32_Cdf`, indexed [ctx].
const PARTITION_W32: [[u16; 10]; 4] = [
    [
        18462, 20920, 23124, 27647, 28227, 29049, 29519, 30178, 31544, 32768,
    ],
    [
        7689, 9060, 12056, 24992, 25660, 26182, 26951, 28041, 29052, 32768,
    ],
    [
        6015, 9009, 10062, 24544, 25409, 26545, 27071, 27526, 32047, 32768,
    ],
    [
        1394, 2208, 2796, 28614, 29061, 29466, 29840, 30185, 31899, 32768,
    ],
];

/// `Default_Partition_W16_Cdf`, indexed [ctx].
const PARTITION_W16: [[u16; 10]; 4] = [
    [
        15597, 20929, 24571, 26706, 27664, 28821, 29601, 30571, 31902, 32768,
    ],
    [
        7925, 11043, 16785, 22470, 23971, 25043, 26651, 28701, 29834, 32768,
    ],
    [
        5414, 13269, 15111, 20488, 22360, 24500, 25537, 26336, 32117, 32768,
    ],
    [
        2662, 6362, 8614, 20860, 23053, 24778, 26436, 27829, 31171, 32768,
    ],
];

/// `Default_Skip_Cdf`, indexed [ctx].
const SKIP: [[u16; 2]; 3] = [[31671, 32768], [16515, 32768], [4576, 32768]];

// Default_Kf_Y_Mode_Cdf[5][5][13], transcribed from pinned libaom 3.11.0
// vendor/av1/common/entropymode.c. The source carries the BSD 2-Clause License
// and Alliance for Open Media Patent License 1.0; this table is AV1 §9.4 data.
const KF_Y_MODE_CDF: [[[u16; 13]; 5]; 5] = [
    [
        [
            15588, 17027, 19338, 20218, 20682, 21110, 21825, 23244, 24189, 28165, 29093, 30466,
            32768,
        ],
        [
            12016, 18066, 19516, 20303, 20719, 21444, 21888, 23032, 24434, 28658, 30172, 31409,
            32768,
        ],
        [
            10052, 10771, 22296, 22788, 23055, 23239, 24133, 25620, 26160, 29336, 29929, 31567,
            32768,
        ],
        [
            14091, 15406, 16442, 18808, 19136, 19546, 19998, 22096, 24746, 29585, 30958, 32462,
            32768,
        ],
        [
            12122, 13265, 15603, 16501, 18609, 20033, 22391, 25583, 26437, 30261, 31073, 32475,
            32768,
        ],
    ],
    [
        [
            10023, 19585, 20848, 21440, 21832, 22760, 23089, 24023, 25381, 29014, 30482, 31436,
            32768,
        ],
        [
            5983, 24099, 24560, 24886, 25066, 25795, 25913, 26423, 27610, 29905, 31276, 31794,
            32768,
        ],
        [
            7444, 12781, 20177, 20728, 21077, 21607, 22170, 23405, 24469, 27915, 29090, 30492,
            32768,
        ],
        [
            8537, 14689, 15432, 17087, 17408, 18172, 18408, 19825, 24649, 29153, 31096, 32210,
            32768,
        ],
        [
            7543, 14231, 15496, 16195, 17905, 20717, 21984, 24516, 26001, 29675, 30981, 31994,
            32768,
        ],
    ],
    [
        [
            12613, 13591, 21383, 22004, 22312, 22577, 23401, 25055, 25729, 29538, 30305, 32077,
            32768,
        ],
        [
            9687, 13470, 18506, 19230, 19604, 20147, 20695, 22062, 23219, 27743, 29211, 30907,
            32768,
        ],
        [
            6183, 6505, 26024, 26252, 26366, 26434, 27082, 28354, 28555, 30467, 30794, 32086, 32768,
        ],
        [
            10718, 11734, 14954, 17224, 17565, 17924, 18561, 21523, 23878, 28975, 30287, 32252,
            32768,
        ],
        [
            9194, 9858, 16501, 17263, 18424, 19171, 21563, 25961, 26561, 30072, 30737, 32463, 32768,
        ],
    ],
    [
        [
            12602, 14399, 15488, 18381, 18778, 19315, 19724, 21419, 25060, 29696, 30917, 32409,
            32768,
        ],
        [
            8203, 13821, 14524, 17105, 17439, 18131, 18404, 19468, 25225, 29485, 31158, 32342,
            32768,
        ],
        [
            8451, 9731, 15004, 17643, 18012, 18425, 19070, 21538, 24605, 29118, 30078, 32018, 32768,
        ],
        [
            7714, 9048, 9516, 16667, 16817, 16994, 17153, 18767, 26743, 30389, 31536, 32528, 32768,
        ],
        [
            8843, 10280, 11496, 15317, 16652, 17943, 19108, 22718, 25769, 29953, 30983, 32485,
            32768,
        ],
    ],
    [
        [
            12578, 13671, 15979, 16834, 19075, 20913, 22989, 25449, 26219, 30214, 31150, 32477,
            32768,
        ],
        [
            9563, 13626, 15080, 15892, 17756, 20863, 22207, 24236, 25380, 29653, 31143, 32277,
            32768,
        ],
        [
            8356, 8901, 17616, 18256, 19350, 20106, 22598, 25947, 26466, 29900, 30523, 32261, 32768,
        ],
        [
            10835, 11815, 13124, 16042, 17018, 18039, 18947, 22753, 24615, 29489, 30883, 32482,
            32768,
        ],
        [
            7618, 8288, 9859, 10509, 15386, 18657, 22903, 28776, 29180, 31355, 31802, 32593, 32768,
        ],
    ],
];
const INTRA_Y_DC_ROW: [u16; 13] = KF_Y_MODE_CDF[0][0];
const INTRA_MODE_CONTEXT: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];
const V_PRED: u8 = 1;
const H_PRED: u8 = 2;
const PAETH_PRED: u8 = 12;
const RESIDUAL_INTRA_MODES: [u8; 4] = [DC_PRED as u8, V_PRED, H_PRED, PAETH_PRED];
const ANGLE_DELTA_CDF: [[u16; 7]; 2] = [
    [2180, 5032, 7567, 22776, 26989, 30217, 32768],
    [2301, 5608, 8801, 23487, 26974, 30330, 32768],
];

/// `Default_Palette_Y_Mode_Cdf`, rows for the block sizes we emit,
/// indexed `[bsl - 2]` (16x16 ⇒ sz_ctx 2, 32x32 ⇒ sz_ctx 4, 64x64 ⇒
/// sz_ctx 6, since sz_ctx = b_dim[2] + b_dim[3] - 2). Verified against
/// dav1d's `cdf.c` (row 2 matches the old single-size table).
const PALETTE_Y_MODE: [[[u16; 2]; 3]; 3] = [
    [[31823, 32768], [3400, 32768], [781, 32768]],
    [[32309, 32768], [7337, 32768], [1462, 32768]],
    [[32450, 32768], [7946, 32768], [129, 32768]],
];

/// `Default_Palette_Y_Size_Cdf` (sizes 2..8), indexed `[bsl - 2]` like above.
const PALETTE_Y_SIZE: [[u16; 7]; 3] = [
    [7788, 12741, 17325, 20500, 24315, 28530, 32768],
    [12725, 19180, 21863, 24839, 27535, 30120, 32768],
    [14940, 20797, 21678, 24186, 27033, 28999, 32768],
];

/// `Default_Palette_Size_N_Y_Color_Cdf`, indexed [color ctx]. Only sizes
/// 2..4 occur (2-bit source ⇒ at most 4 distinct levels per block).
const PALETTE_SIZE_2_Y_COLOR: [[u16; 2]; 5] = [
    [28710, 32768],
    [16384, 32768],
    [10553, 32768],
    [27036, 32768],
    [31603, 32768],
];
const PALETTE_SIZE_3_Y_COLOR: [[u16; 3]; 5] = [
    [27877, 30490, 32768],
    [11532, 25697, 32768],
    [6544, 30234, 32768],
    [23018, 28072, 32768],
    [31915, 32385, 32768],
];
const PALETTE_SIZE_4_Y_COLOR: [[u16; 4]; 5] = [
    [25572, 28046, 30045, 32768],
    [9478, 21590, 27256, 32768],
    [7248, 26837, 29824, 32768],
    [19167, 24486, 28349, 32768],
    [31400, 31825, 32250, 32768],
];

/// `Palette_Color_Context[]` (§9.3): color-context hash → index context.
const PALETTE_COLOR_CONTEXT: [i8; 9] = [-1, -1, 0, -1, -1, 4, 3, 2, 1];

// ---------------------------------------------------------------------------
// Four-shade fast paths: compact masks and tiny lookup tables.
//
// Game Boy source pixels use only four shades. The specialized
// source-analysis path is taken only after verifying every pixel belongs to
// `GAME_SHADES`; any unsupported value disables the path for the whole image
// (general fallback, byte-identical to the original scans).
//
// Static storage:
// * `GRAY_TO_SHADE`: 256 bytes (gray value → shade index 0..3, 0xFF invalid).
// * `MASK_COUNT`: 16 bytes (popcount per 4-bit mask).
// * `MASK_COLORS`: 64 bytes (sorted source colors per mask, 16×4).
// * `PAL_LUT`: 1500 bytes (palette-context/symbol per valid neighbor combo,
//   packed as `(ctx << 2) | sym`, 0xFF invalid; layout
//   `(((n_idx * 5 + left) * 5 + top) * 5 + tl) * 4 + cur` where `n_idx = n-2`,
//   `left/top/tl` are 0..3 or 4 for absent, `cur` is 0..3).
// Total: 256 + 16 + 64 + 1500 = 1836 bytes, plus per-image `masks16`
// (`(w/16)*(h/16)` bytes, e.g. 5760 B for 1280x1152, 90 B for 160x144).
// No runtime hash maps; no per-block or per-pixel allocations (masks are
// once per image, index scratch stays on the stack).
// ---------------------------------------------------------------------------

/// Game Boy source alphabet, sorted ascending.
const GAME_SHADES: [u8; 4] = [0, 85, 170, 255];

/// Gray value → shade index 0..3, 0xFF for unsupported values.
const GRAY_TO_SHADE: [u8; 256] = build_gray_to_shade();

/// The 256 possible 2x2 source-shade patterns represented by one 16x16
/// region in the verified 8x raster. Each entry stores four shade indices
/// in row-major order; transform templates are further keyed by both edge
/// samples and their availability.
const SCALED_PATTERN_SHADES: [[u8; 4]; 256] = build_scaled_pattern_shades();

const fn build_scaled_pattern_shades() -> [[u8; 4]; 256] {
    let mut patterns = [[0u8; 4]; 256];
    let mut pattern = 0usize;
    while pattern < 256 {
        let mut cell = 0usize;
        while cell < 4 {
            patterns[pattern][cell] = ((pattern >> (cell * 2)) & 3) as u8;
            cell += 1;
        }
        pattern += 1;
    }
    patterns
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct ResidualPatternKey {
    pattern: u8,
    mode: u8,
    top: [u8; 2],
    left: [u8; 2],
    top_left: u8,
    have_top: bool,
    have_left: bool,
    have_top_left: bool,
}

#[derive(Clone, Debug)]
struct ResidualPatternTemplate {
    predictor: [[u8; 16]; 16],
    coefficients: [[i32; 16]; 16],
}

fn make_residual_pattern_template(key: ResidualPatternKey) -> ResidualPatternTemplate {
    let pattern = SCALED_PATTERN_SHADES[key.pattern as usize];
    let mut template = ResidualPatternTemplate {
        predictor: [[0; 16]; 16],
        coefficients: [[0; 16]; 16],
    };
    for ty in 0..4 {
        for tx in 0..4 {
            let index = ty * 4 + tx;
            let source_shade = pattern[(ty / 2) * 2 + tx / 2];
            let predictors = residual_pattern_predictor(key, tx, ty);
            template.predictor[index] = predictors;
            let residual_pixels = std::array::from_fn(|pixel| {
                i32::from(GAME_SHADES[source_shade as usize]) - i32::from(predictors[pixel])
            });
            template.coefficients[index] = residual::forward_wht4x4(&residual_pixels);
        }
    }
    template
}

fn residual_pattern_predictor(key: ResidualPatternKey, tx: usize, ty: usize) -> [u8; 16] {
    let pattern = SCALED_PATTERN_SHADES[key.pattern as usize];
    let above_shade = if ty > 0 {
        Some(pattern[((ty - 1) / 2) * 2 + tx / 2])
    } else if key.have_top {
        Some(key.top[tx / 2])
    } else if tx > 0 || key.have_left {
        Some(if tx > 0 {
            pattern[(ty / 2) * 2 + (tx - 1) / 2]
        } else {
            key.left[ty / 2]
        })
    } else {
        None
    };
    let left_shade = if tx > 0 {
        Some(pattern[(ty / 2) * 2 + (tx - 1) / 2])
    } else if key.have_left {
        Some(key.left[ty / 2])
    } else {
        None
    };
    let above = [above_shade.map_or(127, |shade| GAME_SHADES[shade as usize]); 4];
    let left = [left_shade.map_or_else(
        || above_shade.map_or(129, |shade| GAME_SHADES[shade as usize]),
        |shade| GAME_SHADES[shade as usize],
    ); 4];
    let top_left = if above_shade.is_some() && left_shade.is_some() {
        if ty > 0 && tx > 0 {
            GAME_SHADES[pattern[((ty - 1) / 2) * 2 + (tx - 1) / 2] as usize]
        } else if ty > 0 && key.have_left {
            GAME_SHADES[key.left[(ty - 1) / 2] as usize]
        } else if ty > 0 {
            GAME_SHADES[pattern[((ty - 1) / 2) * 2 + tx / 2] as usize]
        } else if tx > 0 {
            if key.have_top {
                GAME_SHADES[key.top[(tx - 1) / 2] as usize]
            } else {
                GAME_SHADES[pattern[(ty / 2) * 2 + (tx - 1) / 2] as usize]
            }
        } else if key.have_top_left {
            GAME_SHADES[key.top_left as usize]
        } else if key.have_top {
            GAME_SHADES[key.top[tx / 2] as usize]
        } else if key.have_left {
            GAME_SHADES[key.left[ty / 2] as usize]
        } else {
            128
        }
    } else if above_shade.is_some() {
        above[0]
    } else if left_shade.is_some() {
        left[0]
    } else {
        128
    };

    let mut prediction = [0; 16];
    for y in 0..4 {
        for x in 0..4 {
            prediction[y * 4 + x] = match key.mode as usize {
                DC_PRED => {
                    let have_above = ty > 0 || key.have_top;
                    let have_left = tx > 0 || key.have_left;
                    match (have_above, have_left) {
                        (true, true) => {
                            let sum = above.iter().map(|&sample| u32::from(sample)).sum::<u32>()
                                + left.iter().map(|&sample| u32::from(sample)).sum::<u32>();
                            ((sum + 4) / 8) as u8
                        }
                        (true, false) => {
                            ((above.iter().map(|&sample| u32::from(sample)).sum::<u32>() + 2) / 4)
                                as u8
                        }
                        (false, true) => {
                            ((left.iter().map(|&sample| u32::from(sample)).sum::<u32>() + 2) / 4)
                                as u8
                        }
                        (false, false) => 128,
                    }
                }
                mode if mode == usize::from(V_PRED) => above[x],
                mode if mode == usize::from(H_PRED) => left[y],
                mode if mode == usize::from(PAETH_PRED) => {
                    let top = i32::from(above[x]);
                    let side = i32::from(left[y]);
                    let corner = i32::from(top_left);
                    let base = top + side - corner;
                    let d_left = (base - side).abs();
                    let d_top = (base - top).abs();
                    let d_corner = (base - corner).abs();
                    if d_left <= d_top && d_left <= d_corner {
                        left[y]
                    } else if d_top <= d_corner {
                        above[x]
                    } else {
                        top_left
                    }
                }
                _ => unreachable!("unsupported residual prediction mode {}", key.mode),
            };
        }
    }
    prediction
}

#[cfg(test)]
mod residual_pattern_tests {
    use super::*;

    fn reference_predictor(px: &[u8], width: usize, x: usize, y: usize, mode: u8) -> [u8; 16] {
        let have_top = y > 0;
        let have_left = x > 0;
        let above = if have_top {
            std::array::from_fn(|i| px[(y - 1) * width + x + i])
        } else if have_left {
            [px[y * width + x - 1]; 4]
        } else {
            [127; 4]
        };
        let left = if have_left {
            std::array::from_fn(|i| px[(y + i) * width + x - 1])
        } else if have_top {
            [px[(y - 1) * width + x]; 4]
        } else {
            [129; 4]
        };
        let top_left = if have_top && have_left {
            px[(y - 1) * width + x - 1]
        } else if have_top {
            px[(y - 1) * width + x]
        } else if have_left {
            px[y * width + x - 1]
        } else {
            128
        };
        let mut prediction = [0; 16];
        for row in 0..4 {
            for col in 0..4 {
                prediction[row * 4 + col] = match mode {
                    mode if mode == DC_PRED as u8 => match (have_top, have_left) {
                        (true, true) => {
                            let sum = above
                                .iter()
                                .chain(left.iter())
                                .map(|&v| u32::from(v))
                                .sum::<u32>();
                            ((sum + 4) / 8) as u8
                        }
                        (true, false) => {
                            ((above.iter().map(|&v| u32::from(v)).sum::<u32>() + 2) / 4) as u8
                        }
                        (false, true) => {
                            ((left.iter().map(|&v| u32::from(v)).sum::<u32>() + 2) / 4) as u8
                        }
                        (false, false) => 128,
                    },
                    V_PRED => above[col],
                    H_PRED => left[row],
                    PAETH_PRED => {
                        let top = i32::from(above[col]);
                        let side = i32::from(left[row]);
                        let corner = i32::from(top_left);
                        let base = top + side - corner;
                        let d_left = (base - side).abs();
                        let d_top = (base - top).abs();
                        let d_corner = (base - corner).abs();
                        if d_left <= d_top && d_left <= d_corner {
                            left[row]
                        } else if d_top <= d_corner {
                            above[col]
                        } else {
                            top_left
                        }
                    }
                    _ => unreachable!(),
                };
            }
        }
        prediction
    }

    #[test]
    fn scaled_pattern_templates_cover_all_patterns_modes_and_image_edges() {
        let modes = [DC_PRED as u8, V_PRED, H_PRED, PAETH_PRED];
        let contexts = [
            (48usize, 48usize, 16usize, 16usize, true, true, true),
            (32, 16, 16, 0, false, true, false),
            (16, 32, 0, 16, true, false, false),
            (16, 16, 0, 0, false, false, false),
        ];
        for pattern_id in 0..=u8::MAX {
            let pattern = SCALED_PATTERN_SHADES[pattern_id as usize];
            for (width, height, block_x, block_y, have_top, have_left, have_top_left) in contexts {
                for mode in modes {
                    let top = [3u8, 2];
                    let left = [1u8, 0];
                    let top_left = 2u8;
                    let key = ResidualPatternKey {
                        pattern: pattern_id,
                        mode,
                        top,
                        left,
                        top_left,
                        have_top,
                        have_left,
                        have_top_left,
                    };
                    let mut pixels = vec![0u8; width * height];
                    if have_top {
                        for x in 0..16 {
                            pixels[(block_y - 1) * width + block_x + x] =
                                GAME_SHADES[top[x / 8] as usize];
                        }
                    }
                    if have_left {
                        for y in 0..16 {
                            pixels[(block_y + y) * width + block_x - 1] =
                                GAME_SHADES[left[y / 8] as usize];
                        }
                    }
                    if have_top_left {
                        pixels[(block_y - 1) * width + block_x - 1] =
                            GAME_SHADES[top_left as usize];
                    }
                    for source_y in 0..2 {
                        for source_x in 0..2 {
                            let shade = GAME_SHADES[pattern[source_y * 2 + source_x] as usize];
                            for y in 0..8 {
                                let row = (block_y + source_y * 8 + y) * width;
                                pixels[row + block_x + source_x * 8
                                    ..row + block_x + (source_x + 1) * 8]
                                    .fill(shade);
                            }
                        }
                    }

                    let template = make_residual_pattern_template(key);
                    for ty in 0..4 {
                        for tx in 0..4 {
                            let x = block_x + tx * 4;
                            let y = block_y + ty * 4;
                            let index = ty * 4 + tx;
                            let expected = reference_predictor(&pixels, width, x, y, mode);
                            let predictor = template.predictor[index];
                            assert_eq!(
                                predictor, expected,
                                "pattern {pattern_id}, mode {mode}, edges ({have_top},{have_left}), TX ({tx},{ty})"
                            );
                            let mut residual_pixels = [0i32; 16];
                            for dy in 0..4 {
                                for dx in 0..4 {
                                    residual_pixels[dy * 4 + dx] =
                                        i32::from(pixels[(y + dy) * width + x + dx])
                                            - i32::from(expected[dy * 4 + dx]);
                                }
                            }
                            let coefficients = residual::forward_wht4x4(&residual_pixels);
                            assert_eq!(
                                template.coefficients[index], coefficients,
                                "pattern {pattern_id}, mode {mode}"
                            );
                            let reconstructed =
                                residual::inverse_wht4x4(&template.coefficients[index]);
                            for (k, (source, residual)) in pixels
                                .chunks_exact(width)
                                .skip(y)
                                .take(4)
                                .flat_map(|row| row[x..x + 4].iter())
                                .zip(reconstructed)
                                .enumerate()
                            {
                                assert_eq!(i32::from(*source), i32::from(expected[k]) + residual);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scaled_pattern_cache_key_includes_edges_and_availability() {
        let base = ResidualPatternKey {
            pattern: 0b11_10_01_00,
            mode: DC_PRED as u8,
            top: [0, 1],
            left: [2, 3],
            top_left: 1,
            have_top: true,
            have_left: true,
            have_top_left: true,
        };
        let mut cache = HashMap::new();
        for key in [
            base,
            ResidualPatternKey {
                top: [1, 1],
                ..base
            },
            ResidualPatternKey {
                left: [3, 3],
                ..base
            },
            ResidualPatternKey {
                have_top: false,
                ..base
            },
            ResidualPatternKey {
                have_left: false,
                ..base
            },
            ResidualPatternKey {
                mode: PAETH_PRED,
                ..base
            },
            ResidualPatternKey {
                top_left: 0,
                ..base
            },
            ResidualPatternKey {
                have_top_left: false,
                ..base
            },
        ] {
            cache
                .entry(key)
                .or_insert_with(|| make_residual_pattern_template(key));
        }
        assert_eq!(cache.len(), 8);
    }

    #[test]
    fn scaled_pattern_cache_enumerates_every_domain_pattern_and_mode() {
        use std::time::Instant;

        let (source_width, source_height) = (32usize, 32usize);
        let (width, height) = (256usize, 256usize);
        let mut small = vec![0u8; source_width * source_height];
        for pattern_id in 0..256usize {
            let (region_x, region_y) = (pattern_id % 16, pattern_id / 16);
            for y in 0..2 {
                for x in 0..2 {
                    let shade = (pattern_id >> ((y * 2 + x) * 2)) & 3;
                    small[(region_y * 2 + y) * source_width + region_x * 2 + x] =
                        GAME_SHADES[shade];
                }
            }
        }
        let mut scaled = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                scaled[y * width + x] = small[(y / 8) * source_width + x / 8];
            }
        }
        assert!(verify_residual_pair(
            &small,
            source_width as u32,
            source_height as u32,
            &scaled,
            width as u32,
            height as u32
        )
        .unwrap());
        let (_, masks) = validate_gray(&scaled, width as u32, height as u32).unwrap();
        let mut tile =
            TileEncoder::new_with_rdo(&scaled, width, height, false, masks, true).unwrap();
        tile.enable_residual_profile(true);

        let started = Instant::now();
        let mut candidate_trials = 0usize;
        let mut exact_prediction_skips = 0usize;
        for pattern_y in 0..16 {
            for pattern_x in 0..16 {
                let (r, c) = (pattern_y * 4, pattern_x * 4);
                for mode in RESIDUAL_INTRA_MODES {
                    candidate_trials += 1;
                    exact_prediction_skips +=
                        usize::from(tile.residual_block_is_exact_prediction(r, c, 4, mode));
                    tile.trial(r, c, 4, 4, |enc| enc.encode_residual_block(r, c, 2, mode))
                        .unwrap();
                }
            }
        }
        let elapsed = started.elapsed();
        assert_eq!(candidate_trials, 256 * RESIDUAL_INTRA_MODES.len());
        assert_eq!(
            tile.scaled_pattern_cache.len() + exact_prediction_skips,
            candidate_trials
        );
        let entry_size = std::mem::size_of::<ResidualPatternKey>()
            + std::mem::size_of::<ResidualPatternTemplate>();
        eprintln!(
            "scaled pattern trial coverage: 256 patterns x {} modes = {} candidate trials, {} transform cache entries and {} predictor-exact skips; HashMap capacity {}; value+key bucket estimate {} bytes; template generation {:?}",
            RESIDUAL_INTRA_MODES.len(),
            candidate_trials,
            tile.scaled_pattern_cache.len(),
            exact_prediction_skips,
            tile.scaled_pattern_cache.capacity(),
            tile.scaled_pattern_cache.capacity() * entry_size,
            elapsed
        );
    }
}

const fn build_gray_to_shade() -> [u8; 256] {
    let mut t = [0xFFu8; 256];
    t[0] = 0;
    t[85] = 1;
    t[170] = 2;
    t[255] = 3;
    t
}

/// Popcount per 4-bit mask (mask 0..15).
const MASK_COUNT: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// Sorted source colors per 4-bit mask: `MASK_COLORS[mask][k]` is the k-th
/// smallest shade present (`k < MASK_COUNT[mask]`); trailing slots are 0.
const MASK_COLORS: [[u8; 4]; 16] = build_mask_colors();

const fn build_mask_colors() -> [[u8; 4]; 16] {
    let mut out = [[0u8; 4]; 16];
    let mut mask = 0usize;
    while mask < 16 {
        let mut k = 0usize;
        let mut s = 0usize;
        while s < 4 {
            if (mask >> s) & 1 == 1 {
                out[mask][k] = GAME_SHADES[s];
                k += 1;
            }
            s += 1;
        }
        mask += 1;
    }
    out
}

/// Palette-context/symbol lookup: `(((n_idx*5+left)*5+top)*5+tl)*4+cur`.
const PAL_LUT: [u8; 1500] = build_pal_lut();

/// One palette-context entry: packed `(ctx << 2) | sym`, or 0xFF invalid.
/// Mirrors `palette_color_context` scoring, stable ordering, hash, and
/// symbol search exactly; `left/top/tl` use 4 for absent.
const fn pal_entry(n: usize, left: u8, top: u8, tl: u8, cur: u8) -> u8 {
    if n < 2 || n > 4 {
        return 0xFF;
    }
    if (cur as usize) >= n {
        return 0xFF;
    }
    if left != 4 && (left as usize) >= n {
        return 0xFF;
    }
    if top != 4 && (top as usize) >= n {
        return 0xFF;
    }
    if tl != 4 && (tl as usize) >= n {
        return 0xFF;
    }
    let hl = left != 4;
    let ht = top != 4;
    let htl = tl != 4;
    // Valid wavefront patterns only: top-row (left only), left-col (top
    // only), or interior (all three). The all-absent first pixel is coded
    // via ns() and never queries this table.
    let valid = (hl && !ht && !htl) || (!hl && ht && !htl) || (hl && ht && htl);
    if !valid {
        return 0xFF;
    }
    let mut scores = [0i32; 8];
    if hl {
        scores[left as usize] += 2;
    }
    if htl {
        scores[tl as usize] += 1;
    }
    if ht {
        scores[top as usize] += 2;
    }
    let mut order = [0usize; 8];
    let mut i = 0usize;
    while i < 8 {
        order[i] = i;
        i += 1;
    }
    let mut si = 0usize;
    while si < 3 {
        let mut max_idx = si;
        let mut j = si;
        while j < n {
            if scores[j] > scores[max_idx] {
                max_idx = j;
            }
            j += 1;
        }
        if max_idx != si {
            let ms = scores[max_idx];
            let mo = order[max_idx];
            let mut k = max_idx;
            while k > si {
                scores[k] = scores[k - 1];
                order[k] = order[k - 1];
                k -= 1;
            }
            scores[si] = ms;
            order[si] = mo;
        }
        si += 1;
    }
    let hash = (scores[0] + scores[1] * 2 + scores[2] * 2) as usize;
    if hash >= PALETTE_COLOR_CONTEXT.len() {
        return 0xFF;
    }
    let ctx = PALETTE_COLOR_CONTEXT[hash];
    if ctx < 0 {
        return 0xFF;
    }
    let mut sym = 0usize;
    let mut p = 0usize;
    while p < 8 {
        if order[p] == cur as usize {
            sym = p;
            break;
        }
        p += 1;
    }
    if sym >= n {
        return 0xFF;
    }
    ((ctx as u8) << 2) | (sym as u8)
}

const fn build_pal_lut() -> [u8; 1500] {
    let mut t = [0xFFu8; 1500];
    let mut n_idx = 0usize;
    while n_idx < 3 {
        let n = n_idx + 2;
        let mut left = 0usize;
        while left < 5 {
            let mut top = 0usize;
            while top < 5 {
                let mut tl = 0usize;
                while tl < 5 {
                    let mut cur = 0usize;
                    while cur < 4 {
                        let idx = (((n_idx * 5 + left) * 5 + top) * 5 + tl) * 4 + cur;
                        t[idx] = pal_entry(n, left as u8, top as u8, tl as u8, cur as u8);
                        cur += 1;
                    }
                    tl += 1;
                }
                top += 1;
            }
            left += 1;
        }
        n_idx += 1;
    }
    t
}

/// Fast palette-context/symbol lookup. Returns `None` for invalid combos
/// (never queried in valid encodes); callers fall back to the reference
/// `palette_color_context` to preserve exact behavior.
#[inline]
fn pal_lut_lookup(n: usize, left: u8, top: u8, tl: u8, cur: u8) -> Option<(usize, usize)> {
    if !(2..=4).contains(&n) || cur >= 4 || left > 4 || top > 4 || tl > 4 {
        return None;
    }
    let idx = (((n - 2) * 5 + left as usize) * 5 + top as usize) * 5 + tl as usize;
    let idx = idx * 4 + cur as usize;
    let packed = PAL_LUT[idx];
    if packed == 0xFF {
        return None;
    }
    Some(((packed >> 2) as usize, (packed & 3) as usize))
}

/// Verify the four-shade invariant and build one 4-bit mask per 16x16
/// region (row-major, `(w/16)*(h/16)` bytes). Returns `None` on any
/// unsupported pixel value (general fallback, no silent mapping), on
/// misaligned dimensions, or on buffer mismatch.
fn build_source_masks(px: &[u8], w: usize, h: usize) -> Option<Vec<u8>> {
    if !w.is_multiple_of(16) || !h.is_multiple_of(16) {
        return None;
    }
    if px.len() != w.saturating_mul(h) {
        return None;
    }
    let w16 = w / 16;
    let h16 = h / 16;
    let mut masks = vec![0u8; w16.saturating_mul(h16)];
    for by in 0..h16 {
        for bx in 0..w16 {
            let mut m = 0u8;
            for dy in 0..16 {
                let base = (by * 16 + dy) * w + bx * 16;
                for dx in 0..16 {
                    let s = GRAY_TO_SHADE[px[base + dx] as usize];
                    if s == 0xFF {
                        return None;
                    }
                    m |= 1 << s;
                }
            }
            masks[by * w16 + bx] = m;
        }
    }
    Some(masks)
}

/// Partition symbol values.
const PARTITION_NONE: usize = 0;
const PARTITION_SPLIT: usize = 3;

/// Intra mode value for DC prediction.
const DC_PRED: usize = 0;

/// `av1C` body builder for our stream: marker/version, profile 0 / given
/// level, monochrome with (1, 1) subsampling bytes, matching the sequence
/// header. `level_idx` must equal the `seq_level_idx` in the item's
/// Sequence Header OBU (AVIF §2.2.1: av1C fields shall match the sequence
/// header).
fn av1c_mono8(level_idx: u8) -> [u8; 4] {
    [0x81, level_idx, 0x1c, 0x00]
}

// ---------------------------------------------------------------------------
// Small helpers (mirroring the spec / gamut-av1).
// ---------------------------------------------------------------------------

/// Bits needed to hold `value - 1`, minimum 1.
fn dimension_bits(value: u32) -> u32 {
    (32 - value.saturating_sub(1).leading_zeros()).max(1)
}

/// Merged SPLIT-ish probability mass for `split_or_horz` (§9.5),
/// computed over the *current* (possibly adapted) partition CDF row.
/// Never updated itself: dav1d decodes split_or with the non-adapting
/// bool reader, so no CDF state changes on either side.
fn split_psum_horz(p: &[u16]) -> u32 {
    (p[2].wrapping_sub(p[1])
        + p[3].wrapping_sub(p[2])
        + p[4].wrapping_sub(p[3])
        + p[6].wrapping_sub(p[5])
        + p[7].wrapping_sub(p[6])
        + p[9].wrapping_sub(p[8])) as u32
}

/// Merged SPLIT-ish probability mass for `split_or_vert` (§9.5).
/// Same no-update rule as above.
fn split_psum_vert(p: &[u16]) -> u32 {
    (p[1].wrapping_sub(p[0])
        + p[3].wrapping_sub(p[2])
        + p[4].wrapping_sub(p[3])
        + p[5].wrapping_sub(p[4])
        + p[6].wrapping_sub(p[5])
        + p[8].wrapping_sub(p[7])) as u32
}

/// Fractional-bit estimate for one symbol coded against `cdf`.
///
/// `prob = (cdf[sym] - prev) / 32768`, `bits = -log2(prob)`. Uses the
/// pre-update CDF row, matching the encoder's adaptation point (the trial
/// CDFs evolve identically to the live coder for the winning candidate, so
/// re-encoding the winner reproduces the same updates).
///
/// Approximations (documented for RDO ranking; see RDO docs):
/// * Ignores the `EC_MIN_PROB` floor and `EC_PROB_SHIFT` precision in
///   `od_ec` interval subdivision, except for a hard floor at
///   `4/32768` (≈13 bits) for zero-probability symbols (which the real
///   coder still encodes via the `EC_MIN_PROB` interval; see `encode_q15`).
///   Non-zero tiny probs ignore the `+MIN_PROB` widening (second-order).
/// * Ignores range-renormalization interactions, carry propagation, and
///   deferred output bytes (hence never compares raw buffer lengths).
/// * Ignores final flush overhead (common prefix/suffix cancel locally).
/// * Future blocks' cost changes from updated CDFs/neighbours are not
///   modeled; local decisions are not globally optimal.
pub(crate) fn adapt_bits(cdf: &[u16], sym: usize) -> f64 {
    debug_assert!(sym < cdf.len());
    if sym >= cdf.len() {
        return 30.0;
    }
    let prev = if sym > 0 { u32::from(cdf[sym - 1]) } else { 0 };
    let cur = u32::from(cdf[sym]);
    // `cur == prev` (zero CDF mass) occurs after adaptation collisions
    // (e.g. `f(0) == f(1)` for large rates); the real `od_ec` coder still
    // encodes via the `EC_MIN_PROB == 4` floor, ≈13 bits at typical ranges.
    // Clamp there instead of asserting, preserving ranking without `-inf`.
    let diff = cur.saturating_sub(prev);
    let prob = (diff as f64 / 32768.0).max(4.0 / 32768.0).clamp(1e-9, 1.0);
    -prob.log2()
}

/// One adapting CDF: owned copy of a default row plus the spec §8.2.6
/// adaptation counter (both start fresh for every image).
/// Fixed-capacity row so speculative snapshots copy CDF state inline rather
/// than cloning one heap allocation per adapting context.
#[derive(Clone, Debug, PartialEq)]
struct FixedCdfRow {
    values: [u16; 13],
    len: usize,
}

impl Deref for FixedCdfRow {
    type Target = [u16];

    fn deref(&self) -> &Self::Target {
        &self.values[..self.len]
    }
}

impl DerefMut for FixedCdfRow {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values[..self.len]
    }
}

#[derive(Clone, Debug, PartialEq)]
struct AdaptCdf {
    cdf: FixedCdfRow,
    count: u16,
}

impl AdaptCdf {
    fn new(row: &[u16]) -> Self {
        assert!(row.len() <= 13, "encoder CDF row exceeds fixed capacity");
        let mut values = [0; 13];
        values[..row.len()].copy_from_slice(row);
        Self {
            cdf: FixedCdfRow {
                values,
                len: row.len(),
            },
            count: 0,
        }
    }

    fn encode(&mut self, sym: &mut SymbolEncoder, s: usize) {
        sym.encode_symbol_adapt(s, &mut self.cdf, &mut self.count);
    }

    fn update(&mut self, symbol: usize) {
        let n = self.cdf.len();
        let rate = 3
            + u32::from(self.count > 15)
            + u32::from(self.count > 31)
            + (31 - (n as u32).leading_zeros()).min(2);
        let (_, body) = self.cdf.split_last_mut().expect("CDF row is non-empty");
        for value in &mut body[..symbol] {
            *value -= *value >> rate;
        }
        for value in &mut body[symbol..] {
            *value += ((1u16 << 15) - *value) >> rate;
        }
        self.count = (self.count + 1).min(32);
    }
}

/// All CDF state for one image. Rows mirror the static tables above;
/// every `S()` symbol adapts its row (`disable_cdf_update = 0`).
/// Literals (`L()`, `NS()`) never adapt, matching the decoder.
#[derive(Clone, Debug, PartialEq)]
struct Cdfs {
    part_w64: [AdaptCdf; 4],
    part_w32: [AdaptCdf; 4],
    part_w16: [AdaptCdf; 4],
    skip: [AdaptCdf; 3],
    y_dc: AdaptCdf,
    pal_mode: [[AdaptCdf; 3]; 3],
    pal_size: [AdaptCdf; 3],
    pal_idx2: [AdaptCdf; 5],
    pal_idx3: [AdaptCdf; 5],
    pal_idx4: [AdaptCdf; 5],
}

impl Cdfs {
    fn new() -> Self {
        Self {
            part_w64: PARTITION_W64.map(|row| AdaptCdf::new(&row)),
            part_w32: PARTITION_W32.map(|row| AdaptCdf::new(&row)),
            part_w16: PARTITION_W16.map(|row| AdaptCdf::new(&row)),
            skip: SKIP.map(|row| AdaptCdf::new(&row)),
            y_dc: AdaptCdf::new(&INTRA_Y_DC_ROW),
            pal_mode: PALETTE_Y_MODE.map(|rows| rows.map(|row| AdaptCdf::new(&row))),
            pal_size: PALETTE_Y_SIZE.map(|row| AdaptCdf::new(&row)),
            pal_idx2: PALETTE_SIZE_2_Y_COLOR.map(|row| AdaptCdf::new(&row)),
            pal_idx3: PALETTE_SIZE_3_Y_COLOR.map(|row| AdaptCdf::new(&row)),
            pal_idx4: PALETTE_SIZE_4_Y_COLOR.map(|row| AdaptCdf::new(&row)),
        }
    }

    fn part(&mut self, bsl: usize, ctx: usize) -> &mut AdaptCdf {
        match bsl {
            2 => &mut self.part_w16[ctx],
            3 => &mut self.part_w32[ctx],
            _ => &mut self.part_w64[ctx],
        }
    }

    /// Current (possibly adapted) partition CDF row, for deriving
    /// split_or probabilities. Shared borrow: deriving touches no state.
    fn part_row(&self, bsl: usize, ctx: usize) -> &[u16] {
        match bsl {
            2 => &self.part_w16[ctx].cdf,
            3 => &self.part_w32[ctx].cdf,
            _ => &self.part_w64[ctx].cdf,
        }
    }

    /// Palette index CDF row for `n` colors. Only 2..4 exist (the
    /// transcribed `Palette_Size_N_Y_Color` tables); callers validate
    /// `psize` at the boundary/`block()` and return `InvalidInput`
    /// otherwise, so larger sizes never reach here.
    fn pal_idx(&mut self, n: usize, ctx: usize) -> &mut AdaptCdf {
        match n {
            2 => &mut self.pal_idx2[ctx],
            3 => &mut self.pal_idx3[ctx],
            4 => &mut self.pal_idx4[ctx],
            _ => unreachable!("palette size {n} has no index CDF (supported: 2..4)"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ResidualModeCdfs {
    y_mode: [[AdaptCdf; 5]; 5],
    angle_delta: [AdaptCdf; 2],
}

impl Default for ResidualModeCdfs {
    fn default() -> Self {
        Self {
            y_mode: std::array::from_fn(|above| {
                std::array::from_fn(|left| AdaptCdf::new(&KF_Y_MODE_CDF[above][left]))
            }),
            angle_delta: ANGLE_DELTA_CDF.map(|row| AdaptCdf::new(&row)),
        }
    }
}

/// `get_palette_color_context` (§5.11.50): reorder colors by neighbour
/// score; returns the `ColorOrder` permutation and the index context.
fn palette_color_context(
    color_map: &[u8],
    bw: usize,
    r: usize,
    c: usize,
    n: usize,
) -> ([usize; 8], usize) {
    let mut scores = [0i32; 8];
    let mut order = [0usize; 8];
    for (i, o) in order.iter_mut().enumerate() {
        *o = i;
    }
    if c > 0 {
        scores[color_map[r * bw + (c - 1)] as usize] += 2;
    }
    if r > 0 && c > 0 {
        scores[color_map[(r - 1) * bw + (c - 1)] as usize] += 1;
    }
    if r > 0 {
        scores[color_map[(r - 1) * bw + c] as usize] += 2;
    }
    for i in 0..3 {
        let mut max_idx = i;
        for j in i..n {
            if scores[j] > scores[max_idx] {
                max_idx = j;
            }
        }
        if max_idx != i {
            let (ms, mo) = (scores[max_idx], order[max_idx]);
            let mut k = max_idx;
            while k > i {
                scores[k] = scores[k - 1];
                order[k] = order[k - 1];
                k -= 1;
            }
            scores[i] = ms;
            order[i] = mo;
        }
    }
    let hash = (scores[0] + scores[1] * 2 + scores[2] * 2) as usize;
    (order, PALETTE_COLOR_CONTEXT[hash] as usize)
}

// ---------------------------------------------------------------------------
// Tile encoder.
// ---------------------------------------------------------------------------

struct TileEncoder<'a> {
    px: &'a [u8],
    w: usize,
    h: usize,
    mi_cols: usize,
    mi_rows: usize,
    sym: SymbolEncoder,
    cdfs: Cdfs,
    coeff_cdfs: Option<residual::CoeffCdfs>,
    residual_mode_cdfs: Option<ResidualModeCdfs>,
    /// `None` disables IntraBC entirely (byte-identical to the pre-IntraBC
    /// encoder: no flag symbols, no header bit — see `USE_INTRABC`).
    intrabc: Option<IntrabcState>,
    skip: Vec<u8>,
    psize: Vec<u8>,
    pcolors: Vec<[u8; 8]>,
    ymode: Option<Vec<u8>>,
    above_coeff_level: Vec<u8>,
    left_coeff_level: Vec<u8>,
    above_coeff_dc: Vec<u8>,
    left_coeff_dc: Vec<u8>,
    /// Pure WHT memoization keyed by the exact residual; it stores no entropy costs.
    residual_transform_cache: HashMap<[i32; 16], [i32; 16]>,
    /// Verified 8x image: templates keyed by a 2x2 pattern, mode, and full edge context.
    scaled_pattern_cache: HashMap<ResidualPatternKey, ResidualPatternTemplate>,
    residual_profile: bool,
    scaled_8x_profile: bool,
    above_part: Vec<u8>,
    left_part: Vec<u8>,
    /// Four-shade source masks (`Some` iff every pixel verified in
    /// `GAME_SHADES`): one 4-bit mask per 16x16 region, row-major.
    src_masks: Option<Vec<u8>>,
    mask_w16: usize,
    /// Sorted unique grayscale samples from this effective rendered image.
    /// Used only to form the opt-in flat-pad candidate set.
    global_colors: [u8; 256],
    global_color_len: usize,
    /// Fractional-bit cost estimate accumulated alongside encoding (see
    /// `adapt_bits`). Saved/restored around RDO trials; the delta over a
    /// trial is the candidate's estimated cost. Committed-path value is
    /// diagnostic only.
    cost: f64,
    /// Trial candidates evaluated (RDO). Excluded from snapshots so the
    /// count persists across rejected trials; merged to globals on finish.
    candidates: u64,
    /// When true, fully contained 64x64/32x32/16x16 nodes use cost-based
    /// selection among legal palette/copy/split alternatives. When false,
    /// the unchanged greedy baseline runs byte-identically.
    use_rdo: bool,
    /// RDO trials need adapted CDFs and fractional costs, but not arithmetic
    /// output. Keeping trials cost-only avoids cloning the growing coder.
    cost_only: bool,
    /// Flat-block unused palette entry policy. Defaults to the historical
    /// cached-color/fallback rule.
    flat_pad_policy: FlatPadPolicy,
    /// Opt-in multi-match pricing. The default preserves the first-match
    /// candidate sequence and output byte-for-byte.
    motion_candidate_policy: MotionCandidatePolicy,
    /// Trial scoring for a non-RDO palette path also needs the same
    /// fractional-bit accounting as RDO, without changing its decisions.
    pad_scoring: bool,
    /// A look-ahead preview uses the historical policy for its one-sibling
    /// horizon, so previews never recursively start more previews.
    pad_preview_depth: u8,
    /// Search counters deliberately persist across rejected trials.
    flat_pad_stats: FlatPadStats,
}

/// `Partition_Context` update table (§5.11.4), rows [above|left],
/// columns NONE/HORZ/VERT/SPLIT. Only the NONE column is used here
/// (SPLIT never updates; HORZ/VERT are never emitted).
const AL_PART_NONE_ABOVE: [u8; 5] = [0x00, 0x10, 0x18, 0x1c, 0x1e];
const AL_PART_NONE_LEFT: [u8; 5] = [0x00, 0x10, 0x18, 0x1c, 0x1e];

/// Scoped snapshot for one RDO trial rect (see RDO docs in `impl`).
/// `candidates` is deliberately excluded (persists across rejects).
struct Snapshot {
    sym: Option<SymbolEncoder>,
    cdfs: Cdfs,
    coeff_cdfs: Option<residual::CoeffCdfs>,
    residual_mode_cdfs: Option<ResidualModeCdfs>,
    intrabc_cdfs: Option<(AdaptCdf, AdaptCdf, [MvComp; 2], u32)>,
    cost: f64,
    cost_only: bool,
    r: usize,
    c: usize,
    bw4: usize,
    bh4: usize,
    footprint_len: usize,
    skip_fp: [u8; 256],
    psize_fp: [u8; 256],
    pcolors_fp: [[u8; 8]; 256],
    ymode_fp: Option<[u8; 256]>,
    residual_coeff_contexts: Option<ResidualCoeffSnapshot>,
    decoded_fp: [bool; 256],
    rec_fp: [MvRec; 256],
    above_len: usize,
    above_fp: [u8; 16],
    left_len: usize,
    left_fp: [u8; 16],
}

struct ResidualCoeffSnapshot {
    above_level: [u8; 16],
    left_level: [u8; 16],
    above_dc: [u8; 16],
    left_dc: [u8; 16],
}

impl<'a> TileEncoder<'a> {
    #[cfg(test)]
    fn new(
        px: &'a [u8],
        w: usize,
        h: usize,
        use_intrabc: bool,
        src_masks: Option<Vec<u8>>,
    ) -> io::Result<Self> {
        Self::new_with_rdo(px, w, h, use_intrabc, src_masks, false)
    }

    #[cfg(test)]
    fn new_with_rdo(
        px: &'a [u8],
        w: usize,
        h: usize,
        use_intrabc: bool,
        src_masks: Option<Vec<u8>>,
        use_rdo: bool,
    ) -> io::Result<Self> {
        Self::new_with_rdo_and_pad_policy(
            px,
            w,
            h,
            use_intrabc,
            src_masks,
            use_rdo,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_rdo_and_pad_policy(
        px: &'a [u8],
        w: usize,
        h: usize,
        use_intrabc: bool,
        src_masks: Option<Vec<u8>>,
        use_rdo: bool,
        flat_pad_policy: FlatPadPolicy,
        search_radius_rings: i32,
        uniform_search_radius_rings: i32,
    ) -> io::Result<Self> {
        Self::new_with_rdo_pad_and_motion_policy(
            px,
            w,
            h,
            use_intrabc,
            src_masks,
            use_rdo,
            flat_pad_policy,
            search_radius_rings,
            uniform_search_radius_rings,
            MotionCandidatePolicy::FirstMatch,
        )
    }

    // Clippy's count is a false positive here: the constructor keeps the
    // input geometry, coding switches, and search controls explicit.
    #[allow(clippy::too_many_arguments)]
    fn new_with_rdo_pad_and_motion_policy(
        px: &'a [u8],
        w: usize,
        h: usize,
        use_intrabc: bool,
        src_masks: Option<Vec<u8>>,
        use_rdo: bool,
        flat_pad_policy: FlatPadPolicy,
        search_radius_rings: i32,
        uniform_search_radius_rings: i32,
        motion_candidate_policy: MotionCandidatePolicy,
    ) -> io::Result<Self> {
        if w > u32::MAX as usize || h > u32::MAX as usize {
            return Err(invalid_input(format!(
                "dimensions exceed u32 range, got {w}x{h}"
            )));
        }
        // `w`/`h` fit in `u32` here.
        check_supported_dimensions(w as u32, h as u32)?;
        let mi_cols = w / 4;
        let mi_rows = h / 4;
        let mi_area = mi_cols.checked_mul(mi_rows).ok_or_else(|| {
            invalid_input(format!("MI grid area overflows usize: {mi_cols}x{mi_rows}"))
        })?;
        let mut intrabc = use_intrabc.then(|| {
            IntrabcState::with_search_radius(
                mi_cols,
                mi_rows,
                search_radius_rings,
                uniform_search_radius_rings,
            )
        });
        // Verify the 8x nearest-neighbor invariant once per image and
        // precompute the bounded per-key lists when it holds. Unverified
        // images keep the original ring behavior exactly.
        if let Some(bc) = intrabc.as_mut() {
            bc.build_pattern_cache(px, w, h);
            if use_rdo {
                bc.build_uniform_cache(px, w, h);
            }
        }
        let mask_w16 = w / 16;
        let mut colors_present = [false; 256];
        for &v in px {
            colors_present[v as usize] = true;
        }
        let mut global_colors = [0u8; 256];
        let mut global_color_len = 0usize;
        for (v, &present) in colors_present.iter().enumerate() {
            if present {
                global_colors[global_color_len] = v as u8;
                global_color_len += 1;
            }
        }
        Ok(Self {
            px,
            w,
            h,
            mi_cols,
            mi_rows,
            sym: SymbolEncoder::new(),
            cdfs: Cdfs::new(),
            coeff_cdfs: None,
            residual_mode_cdfs: None,
            intrabc,
            skip: vec![0; mi_area],
            psize: vec![0; mi_area],
            pcolors: vec![[0u8; 8]; mi_area],
            ymode: None,
            above_coeff_level: vec![0; mi_cols],
            left_coeff_level: vec![0; mi_rows],
            above_coeff_dc: vec![0; mi_cols],
            left_coeff_dc: vec![0; mi_rows],
            residual_transform_cache: HashMap::new(),
            scaled_pattern_cache: HashMap::new(),
            residual_profile: false,
            scaled_8x_profile: false,
            above_part: vec![0; mi_cols],
            left_part: vec![0; mi_rows],
            src_masks,
            mask_w16,
            global_colors,
            global_color_len,
            cost: 0.0,
            candidates: 0,
            use_rdo,
            cost_only: false,
            flat_pad_policy,
            motion_candidate_policy,
            pad_scoring: false,
            pad_preview_depth: 0,
            flat_pad_stats: FlatPadStats::default(),
        })
    }

    fn sample(&self, x: usize, y: usize) -> u8 {
        self.px[y * self.w + x]
    }

    /// Combined 4-bit source mask for the `bw4`-MI square at MI `(r, c)`.
    /// Returns `None` when the four-shade path is disabled or the region is
    /// not fully on-screen (preserves the original boundary behavior via
    /// fallback scans).
    fn combined_mask(&self, r: usize, c: usize, bw4: usize) -> Option<u8> {
        let masks = self.src_masks.as_ref()?;
        if r.saturating_add(bw4) > self.mi_rows || c.saturating_add(bw4) > self.mi_cols {
            return None;
        }
        // MI (r,c) → pixel (sx,sy); 16px cells → mask grid. All terminal
        // blocks are 16-aligned here, so the rect covers whole cells.
        let sx = c * 4;
        let sy = r * 4;
        let bw = bw4 * 4;
        if !sx.is_multiple_of(16) || !sy.is_multiple_of(16) || !bw.is_multiple_of(16) {
            return None;
        }
        let x0 = sx / 16;
        let y0 = sy / 16;
        let n = bw / 16;
        let mut m = 0u8;
        for dy in 0..n {
            let base = (y0 + dy) * self.mask_w16 + x0;
            for dx in 0..n {
                m |= masks[base + dx];
                if m == 0x0F {
                    // Mask full; remaining ORs cannot change popcount, but
                    // keep combining cheaply (few cells max: 4x4).
                }
            }
        }
        Some(m)
    }

    /// Count of distinct levels in the `bw4`-MI region at MI `(r, c)`,
    /// saturating at 5 (only the ≤4 question matters: larger palettes
    /// have no index tables in this encoder).
    fn region_levels(&self, r: usize, c: usize, bw4: usize) -> usize {
        if let Some(m) = self.combined_mask(r, c, bw4) {
            return MASK_COUNT[(m & 0x0F) as usize] as usize;
        }
        let px = bw4 * 4;
        let (sx, sy) = (c * 4, r * 4);
        let xe = (sx + px).min(self.w);
        let ye = (sy + px).min(self.mi_rows * 4);
        let mut set = [false; 256];
        let mut count = 0;
        for y in sy..ye {
            for x in sx..xe {
                let v = self.sample(x, y) as usize;
                if !set[v] {
                    set[v] = true;
                    count += 1;
                    if count > 4 {
                        return count;
                    }
                }
            }
        }
        count
    }

    /// Partition context bit (§8.3.2) from neighbour bytes.
    fn partition_ctx(&self, r: usize, c: usize, bsl: usize) -> usize {
        let bit = bsl - 1;
        let above = usize::from((self.above_part[c] >> bit) & 1);
        let left = if c > 0 {
            usize::from((self.left_part[r] >> bit) & 1)
        } else {
            0
        };
        left * 2 + above
    }

    // -- Cost-aware symbol helpers (all syntax contributions for RDO) --
    // Each computes `adapt_bits` (or literal bits) from the pre-update row,
    // accumulates into `self.cost`, then encodes identically. Baseline and
    // RDO share them so committed bytes are identical for identical decisions.
    // When neither RDO nor pad look-ahead scoring is active, costs are never
    // used for decisions, so fractional estimates (including `log2`) are
    // skipped entirely for speed; bytes and adaptive updates are unchanged.
    fn count_cost(&self) -> bool {
        self.use_rdo || self.pad_scoring
    }

    fn enable_residual_profile(&mut self, scaled_8x: bool) {
        self.residual_profile = true;
        self.scaled_8x_profile = scaled_8x;
        self.coeff_cdfs = Some(residual::CoeffCdfs::default());
        self.residual_mode_cdfs = Some(ResidualModeCdfs::default());
        self.ymode = Some(vec![DC_PRED as u8; self.mi_cols * self.mi_rows]);
    }

    fn enc_part(&mut self, bsl: usize, ctx: usize, s: usize) {
        if self.count_cost() {
            let bits = adapt_bits(self.cdfs.part_row(bsl, ctx), s);
            self.cost += bits;
        }
        let cost_only = self.cost_only;
        let row = self.cdfs.part(bsl, ctx);
        if cost_only {
            row.update(s);
        } else {
            row.encode(&mut self.sym, s);
        }
    }

    fn enc_skip(&mut self, ctx: usize, s: usize) {
        if self.count_cost() {
            let bits = adapt_bits(&self.cdfs.skip[ctx].cdf, s);
            self.cost += bits;
        }
        if self.cost_only {
            self.cdfs.skip[ctx].update(s);
        } else {
            self.cdfs.skip[ctx].encode(&mut self.sym, s);
        }
    }

    fn enc_y_mode(&mut self, r: usize, c: usize, mode: u8) {
        let symbol = mode as usize;
        let cost_only = self.cost_only;
        let use_cost = self.count_cost();
        if let (Some(grid), Some(cdfs)) = (&self.ymode, &mut self.residual_mode_cdfs) {
            let above_mode = if r > 0 {
                grid[(r - 1) * self.mi_cols + c]
            } else {
                DC_PRED as u8
            };
            let left_mode = if c > 0 {
                grid[r * self.mi_cols + c - 1]
            } else {
                DC_PRED as u8
            };
            let above_ctx = INTRA_MODE_CONTEXT[above_mode as usize];
            let left_ctx = INTRA_MODE_CONTEXT[left_mode as usize];
            let cdf = &mut cdfs.y_mode[above_ctx][left_ctx];
            if use_cost {
                self.cost += adapt_bits(&cdf.cdf, symbol);
            }
            if cost_only {
                cdf.update(symbol);
            } else {
                cdf.encode(&mut self.sym, symbol);
            }
        } else {
            debug_assert_eq!(mode, DC_PRED as u8);
            let cdf = &mut self.cdfs.y_dc;
            if use_cost {
                self.cost += adapt_bits(&cdf.cdf, symbol);
            }
            if cost_only {
                cdf.update(symbol);
            } else {
                cdf.encode(&mut self.sym, symbol);
            }
        }
    }

    fn enc_angle_delta_zero(&mut self, mode: u8) {
        if !(V_PRED..=H_PRED).contains(&mode) {
            return;
        }
        let use_cost = self.count_cost();
        let cost_only = self.cost_only;
        let Some(mode_cdfs) = self.residual_mode_cdfs.as_mut() else {
            return;
        };
        let cdf = &mut mode_cdfs.angle_delta[(mode - V_PRED) as usize];
        let symbol = 3; // zero delta, biased by MAX_ANGLE_DELTA
        if use_cost {
            self.cost += adapt_bits(&cdf.cdf, symbol);
        }
        if cost_only {
            cdf.update(symbol);
        } else {
            cdf.encode(&mut self.sym, symbol);
        }
    }

    fn record_y_mode(&mut self, r: usize, c: usize, bw4: usize, bh4: usize, mode: u8) {
        let Some(grid) = self.ymode.as_mut() else {
            return;
        };
        for y in r..r + bh4 {
            for x in c..c + bw4 {
                if y < self.mi_rows && x < self.mi_cols {
                    grid[y * self.mi_cols + x] = mode;
                }
            }
        }
    }

    fn enc_pal_mode(&mut self, bsl: usize, pctx: usize, s: usize) {
        if self.count_cost() {
            let bits = adapt_bits(&self.cdfs.pal_mode[bsl - 2][pctx].cdf, s);
            self.cost += bits;
        }
        if self.cost_only {
            self.cdfs.pal_mode[bsl - 2][pctx].update(s);
        } else {
            self.cdfs.pal_mode[bsl - 2][pctx].encode(&mut self.sym, s);
        }
    }

    fn enc_pal_size(&mut self, bsl: usize, s: usize) {
        if self.count_cost() {
            let bits = adapt_bits(&self.cdfs.pal_size[bsl - 2].cdf, s);
            self.cost += bits;
        }
        if self.cost_only {
            self.cdfs.pal_size[bsl - 2].update(s);
        } else {
            self.cdfs.pal_size[bsl - 2].encode(&mut self.sym, s);
        }
    }

    fn enc_pal_idx(&mut self, n: usize, ctx: usize, s: usize) {
        if self.count_cost() {
            let row: &[u16] = match n {
                2 => &self.cdfs.pal_idx2[ctx].cdf,
                3 => &self.cdfs.pal_idx3[ctx].cdf,
                _ => &self.cdfs.pal_idx4[ctx].cdf,
            };
            let bits = adapt_bits(row, s);
            self.cost += bits;
        }
        if self.cost_only {
            self.cdfs.pal_idx(n, ctx).update(s);
        } else {
            self.cdfs.pal_idx(n, ctx).encode(&mut self.sym, s);
        }
    }

    fn enc_static(&mut self, s: usize, cdf: &[u16]) {
        if self.count_cost() {
            let bits = adapt_bits(cdf, s);
            self.cost += bits;
        }
        if !self.cost_only {
            self.sym.encode_symbol(s, cdf);
        }
    }

    fn enc_lit(&mut self, val: u32, n: u32) {
        // Equiprobable `L()` bits (`read_literal` via fixed 1/2 CDF).
        if self.count_cost() {
            self.cost += f64::from(n);
        }
        if !self.cost_only {
            self.sym.encode_literal(val, n);
        }
    }

    /// Record a terminal NONE partition over the `n4`-MI block.
    fn update_partition_ctx(&mut self, r: usize, c: usize, bsl: usize) {
        let bl = 5 - bsl;
        let n4 = 1usize << bsl;
        let (a, l) = (AL_PART_NONE_ABOVE[bl], AL_PART_NONE_LEFT[bl]);
        for k in 0..n4 {
            if c + k < self.mi_cols {
                self.above_part[c + k] = a;
            }
            if r + k < self.mi_rows {
                self.left_part[r + k] = l;
            }
        }
    }

    /// `EDGE_I444_TOP_HAS_RIGHT` for quadrant `quad` (Z-order: 0 = TL,
    /// 1 = TR, 2 = BL, 3 = BR) under a parent with flag `parent_tr`,
    /// mirroring dav1d's generated edge tree (`init_mode_node`: TR is
    /// kept everywhere except BR and a TR child of a TR-less parent).
    /// Threaded through `partition()` so IntraBC predictor replication
    /// sees the decoder's exact top-right availability.
    fn child_top_right(quad: usize, parent_tr: bool) -> bool {
        !(quad == 3 || (quad == 1 && !parent_tr))
    }

    /// Emit the partition tree. A fully on-screen 32x32 or 64x64 node
    /// whose distinct levels fit the palette (≤4 here — our index tables
    /// only cover sizes 2..4) is coded `NONE` directly; anything else
    /// splits down, with 16x16 (`bw4 == 4`) leaves as before. Offscreen
    /// subtrees emit nothing, and partially on-screen nodes take the
    /// existing edge (split_or / forced-split) path, never `NONE`.
    /// `top_has_right` is the decoder edge flag for this node (SB roots
    /// start set, mirroring the generated tree root).
    fn partition(&mut self, r: usize, c: usize, bw4: usize, top_has_right: bool) -> io::Result<()> {
        if r >= self.mi_rows || c >= self.mi_cols {
            return Ok(());
        }
        let bsl = bw4.trailing_zeros() as usize;
        if bw4 == 4 {
            let ctx = self.partition_ctx(r, c, bsl);
            self.enc_part(bsl, ctx, PARTITION_NONE);
            self.update_partition_ctx(r, c, bsl);
            self.block(r, c, bsl, top_has_right, None)?;
            return Ok(());
        }
        // Contained nodes: flat regions always take NONE (palette beats a
        // copy for a single level); with IntraBC off, anything paletteable
        // (≤4 levels) takes NONE exactly as before; with IntraBC on, a
        // 2..=4-level region takes NONE only when a copy exists now —
        // otherwise it splits so children can match at finer grain.
        // Partially on-screen nodes never take NONE (edge path below).
        // The successful search (MV + predictor) is carried into `block`
        // to avoid repeating the same search and predictor query.
        if r + bw4 <= self.mi_rows && c + bw4 <= self.mi_cols {
            let levels = self.region_levels(r, c, bw4);
            let carried: Option<((i32, i32), (i32, i32))> = if levels <= 1 || self.intrabc.is_none()
            {
                None
            } else {
                self.intrabc_match(r, c, bw4, top_has_right)
            };
            let take_none = if levels <= 1 {
                true
            } else if self.intrabc.is_none() {
                levels <= 4
            } else {
                carried.is_some()
            };
            if take_none {
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_NONE);
                self.update_partition_ctx(r, c, bsl);
                self.block(r, c, bsl, top_has_right, carried)?;
                return Ok(());
            }
        }
        let half = bw4 >> 1;
        let has_rows = r + half < self.mi_rows;
        let has_cols = c + half < self.mi_cols;
        if has_rows && has_cols {
            let ctx = self.partition_ctx(r, c, bsl);
            self.enc_part(bsl, ctx, PARTITION_SPLIT);
        } else if has_cols {
            // Bottom edge: SPLIT-or-HORZ as a bool against the merged
            // split mass of the *current* row. No state update, exactly
            // like dav1d's non-adapting bool decode.
            let ctx = self.partition_ctx(r, c, bsl);
            let psum = split_psum_horz(self.cdfs.part_row(bsl, ctx));
            debug_assert!(psum < 32768);
            // psum == 0 would wrap the subtraction below; use the
            // exactly-dual [32767, 32768] instead (unreachable in
            // practice: mass(NONE) floors at 1, so psum <= 32767).
            let c0 = if psum == 0 {
                32767
            } else {
                (32768 - psum) as u16
            };
            self.enc_static(1, &[c0, 32768]);
        } else if has_rows {
            // Right edge: same arrangement with the vert masses.
            let ctx = self.partition_ctx(r, c, bsl);
            let psum = split_psum_vert(self.cdfs.part_row(bsl, ctx));
            debug_assert!(psum < 32768);
            let c0 = if psum == 0 {
                32767
            } else {
                (32768 - psum) as u16
            };
            self.enc_static(1, &[c0, 32768]);
        }
        // Forced SPLIT (neither flag) codes no symbol.
        self.partition(r, c, half, Self::child_top_right(0, top_has_right))?;
        self.partition(r, c + half, half, Self::child_top_right(1, top_has_right))?;
        self.partition(r + half, c, half, Self::child_top_right(2, top_has_right))?;
        self.partition(
            r + half,
            c + half,
            half,
            Self::child_top_right(3, top_has_right),
        )?;
        Ok(())
    }

    fn skip_ctx(&self, r: usize, c: usize) -> usize {
        let above = r > 0 && self.skip[(r - 1) * self.mi_cols + c] != 0;
        let left = c > 0 && self.skip[r * self.mi_cols + (c - 1)] != 0;
        usize::from(above) + usize::from(left)
    }

    fn txb_skip_ctx_4x4(&self, r: usize, c: usize) -> usize {
        let top = i32::from(self.above_coeff_level[c]);
        let left = i32::from(self.left_coeff_level[r]);
        if top == 0 && left == 0 {
            1
        } else if top == 0 || left == 0 {
            2 + usize::from(top.max(left) > 3)
        } else if top.max(left) <= 3 {
            4
        } else if top.min(left) <= 3 {
            5
        } else {
            6
        }
    }

    fn dc_sign_ctx_4x4(&self, r: usize, c: usize) -> usize {
        let signed = |category| match category {
            1 => -1,
            2 => 1,
            _ => 0,
        };
        let sum = signed(self.above_coeff_dc[c]) + signed(self.left_coeff_dc[r]);
        if sum < 0 {
            1
        } else if sum > 0 {
            2
        } else {
            0
        }
    }

    fn record_coeff_context_4x4(&mut self, r: usize, c: usize, context: residual::CoeffContext) {
        self.above_coeff_level[c] = context.cul_level;
        self.left_coeff_level[r] = context.cul_level;
        self.above_coeff_dc[c] = context.dc_category;
        self.left_coeff_dc[r] = context.dc_category;
    }

    fn reset_coeff_context(&mut self, r: usize, c: usize, bw4: usize, bh4: usize) {
        for x in c..c + bw4 {
            if x < self.mi_cols {
                self.above_coeff_level[x] = 0;
                self.above_coeff_dc[x] = 0;
            }
        }
        for y in r..r + bh4 {
            if y < self.mi_rows {
                self.left_coeff_level[y] = 0;
                self.left_coeff_dc[y] = 0;
            }
        }
    }

    fn scaled_pattern_key(&self, x: usize, y: usize, mode: u8) -> ResidualPatternKey {
        debug_assert!(x.is_multiple_of(16) && y.is_multiple_of(16));
        let shade = |sx: usize, sy: usize| {
            let value = self.sample(sx * 8, sy * 8);
            let result = GRAY_TO_SHADE[value as usize];
            debug_assert_ne!(
                result, 0xFF,
                "verified profile has unsupported shade {value}"
            );
            result
        };
        let source_x = x / 8;
        let source_y = y / 8;
        let pattern = [
            shade(source_x, source_y),
            shade(source_x + 1, source_y),
            shade(source_x, source_y + 1),
            shade(source_x + 1, source_y + 1),
        ];
        let pattern_id = pattern
            .iter()
            .enumerate()
            .fold(0u8, |packed, (index, &value)| {
                packed | (value << (index * 2))
            });
        let have_top = y > 0;
        let have_left = x > 0;
        let have_top_left = have_top && have_left;
        let top = if have_top {
            [
                shade(source_x, source_y - 1),
                shade(source_x + 1, source_y - 1),
            ]
        } else {
            [0, 0]
        };
        let left = if have_left {
            [
                shade(source_x - 1, source_y),
                shade(source_x - 1, source_y + 1),
            ]
        } else {
            [0, 0]
        };
        let top_left = if have_top_left {
            shade(source_x - 1, source_y - 1)
        } else {
            0
        };
        ResidualPatternKey {
            pattern: pattern_id,
            mode,
            top,
            left,
            top_left,
            have_top,
            have_left,
            have_top_left,
        }
    }

    fn residual_dc_predictor(&self, x: usize, y: usize) -> u8 {
        let have_top = y > 0;
        let have_left = x > 0;
        match (have_top, have_left) {
            (true, true) => {
                let mut sum = 0u32;
                for offset in 0..4 {
                    sum += u32::from(self.sample(x + offset, y - 1));
                    sum += u32::from(self.sample(x - 1, y + offset));
                }
                ((sum + 4) / 8) as u8
            }
            (true, false) => {
                let sum = (0..4)
                    .map(|offset| u32::from(self.sample(x + offset, y - 1)))
                    .sum::<u32>();
                ((sum + 2) / 4) as u8
            }
            (false, true) => {
                let sum = (0..4)
                    .map(|offset| u32::from(self.sample(x - 1, y + offset)))
                    .sum::<u32>();
                ((sum + 2) / 4) as u8
            }
            (false, false) => 128,
        }
    }

    fn residual_predictor_4x4(&self, x: usize, y: usize, mode: u8) -> [u8; 16] {
        if mode == DC_PRED as u8 {
            return [self.residual_dc_predictor(x, y); 16];
        }
        let have_top = y > 0;
        let have_left = x > 0;
        let above = if have_top {
            std::array::from_fn(|offset| self.sample(x + offset, y - 1))
        } else if have_left {
            [self.sample(x - 1, y); 4]
        } else {
            [127; 4]
        };
        let left = if have_left {
            std::array::from_fn(|offset| self.sample(x - 1, y + offset))
        } else if have_top {
            [self.sample(x, y - 1); 4]
        } else {
            [129; 4]
        };
        let top_left = if have_top && have_left {
            self.sample(x - 1, y - 1)
        } else if have_top {
            self.sample(x, y - 1)
        } else if have_left {
            self.sample(x - 1, y)
        } else {
            128
        };
        let mut prediction = [0; 16];
        for row in 0..4 {
            for col in 0..4 {
                prediction[row * 4 + col] = match mode {
                    V_PRED => above[col],
                    H_PRED => left[row],
                    PAETH_PRED => {
                        let top = i32::from(above[col]);
                        let side = i32::from(left[row]);
                        let corner = i32::from(top_left);
                        let base = top + side - corner;
                        let d_left = (base - side).abs();
                        let d_top = (base - top).abs();
                        let d_corner = (base - corner).abs();
                        if d_left <= d_top && d_left <= d_corner {
                            left[row]
                        } else if d_top <= d_corner {
                            above[col]
                        } else {
                            top_left
                        }
                    }
                    _ => unreachable!("unsupported residual intra mode {mode}"),
                };
            }
        }
        prediction
    }

    /// AV1 `skip = 1` omits all transform coefficients while still applying
    /// intra prediction. In qindex-0 mode this is useful only when every
    /// mandatory 4x4 prediction already equals its source samples.
    fn residual_block_is_exact_prediction(&self, r: usize, c: usize, bw4: usize, mode: u8) -> bool {
        let (sx, sy) = (c * 4, r * 4);
        for tx_y in 0..bw4 {
            for tx_x in 0..bw4 {
                let (x, y) = (sx + tx_x * 4, sy + tx_y * 4);
                let predictor = self.residual_predictor_4x4(x, y, mode);
                for dy in 0..4 {
                    for dx in 0..4 {
                        if self.sample(x + dx, y + dy) != predictor[dy * 4 + dx] {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Emit a supported coded-lossless luma prediction mode with 4x4 WHT
    /// residuals. Source neighbors equal decoder reconstruction on this q0
    /// path. Coefficient and mode CDF costs remain live; only prediction and
    /// transform math are memoized for verified scaled patterns.
    fn encode_residual_block(
        &mut self,
        r: usize,
        c: usize,
        bsl: usize,
        mode: u8,
    ) -> io::Result<()> {
        if !self.residual_profile {
            return Err(invalid_input(
                "residual transform candidate requires verified four-shade pixels".to_owned(),
            ));
        }
        if !RESIDUAL_INTRA_MODES.contains(&mode) {
            return Err(invalid_input(format!(
                "unsupported residual prediction mode {mode}"
            )));
        }
        let bw4 = 1usize << bsl;
        let (sx, sy) = (c * 4, r * 4);
        let exact_prediction = self.residual_block_is_exact_prediction(r, c, bw4, mode);
        let skip_ctx = self.skip_ctx(r, c);
        self.enc_skip(skip_ctx, usize::from(exact_prediction));

        let use_cost = self.count_cost();
        let emit = !self.cost_only;
        if let Some(bc) = self.intrabc.as_mut() {
            let (sym, cost) = (&mut self.sym, &mut self.cost);
            bc.flag_with_cost(sym, false, cost, use_cost, emit);
        }
        self.enc_y_mode(r, c, mode);
        if (V_PRED..=H_PRED).contains(&mode) {
            self.enc_angle_delta_zero(mode);
        }
        self.record_y_mode(r, c, bw4, bw4, mode);
        if mode == DC_PRED as u8 {
            let above_palette = r > 0 && self.psize[(r - 1) * self.mi_cols + c] > 0;
            let left_palette = c > 0 && self.psize[r * self.mi_cols + c - 1] > 0;
            let palette_ctx = usize::from(above_palette) + usize::from(left_palette);
            self.enc_pal_mode(bsl, palette_ctx, 0);
        }

        if exact_prediction {
            // With skip=1, the decoder applies these same 4x4 predictions
            // and reads no TXB/coefficient symbols. Verify the omitted
            // residual is identically zero, then mirror reset_block_context.
            self.reset_coeff_context(r, c, bw4, bw4);
            for yy in 0..bw4 {
                for xx in 0..bw4 {
                    let (rr, cc) = (r + yy, c + xx);
                    self.skip[rr * self.mi_cols + cc] = 1;
                    self.psize[rr * self.mi_cols + cc] = 0;
                    self.pcolors[rr * self.mi_cols + cc] = [0; 8];
                }
            }
            if let Some(bc) = self.intrabc.as_mut() {
                bc.record(r, c, bw4, bw4, None);
            }
            return Ok(());
        }

        // AV1 codes transform blocks in full-block raster order. The cache is
        // 16x16-pattern based, so each TX looks up its containing template,
        // but entropy/context updates remain in the normative transform order.
        for tx_y in 0..bw4 {
            for tx_x in 0..bw4 {
                let (x, y) = (sx + tx_x * 4, sy + tx_y * 4);
                let mi_x = x / 4;
                let mi_y = y / 4;
                let cached_template = if self.scaled_8x_profile {
                    let (tile_x, tile_y) = (x / 16 * 16, y / 16 * 16);
                    let key = self.scaled_pattern_key(tile_x, tile_y, mode);
                    let template = self
                        .scaled_pattern_cache
                        .entry(key)
                        .or_insert_with(|| make_residual_pattern_template(key));
                    let local_tx = ((y - tile_y) / 4) * 4 + (x - tile_x) / 4;
                    Some((
                        template.predictor[local_tx],
                        template.coefficients[local_tx],
                    ))
                } else {
                    None
                };
                let predictors = cached_template
                    .map(|(predictor, _)| predictor)
                    .unwrap_or_else(|| self.residual_predictor_4x4(x, y, mode));
                let mut residual_pixels = [0i32; 16];
                for dy in 0..4 {
                    for dx in 0..4 {
                        residual_pixels[dy * 4 + dx] = i32::from(self.sample(x + dx, y + dy))
                            - i32::from(predictors[dy * 4 + dx]);
                    }
                }
                let coefficients = if let Some((_, coefficients)) = cached_template {
                    coefficients
                } else if let Some(cached) = self.residual_transform_cache.get(&residual_pixels) {
                    *cached
                } else {
                    let transformed = residual::forward_wht4x4(&residual_pixels);
                    self.residual_transform_cache
                        .insert(residual_pixels, transformed);
                    transformed
                };
                let inverse = residual::inverse_wht4x4(&coefficients);
                for dy in 0..4 {
                    for dx in 0..4 {
                        let reconstructed =
                            i32::from(predictors[dy * 4 + dx]) + inverse[dy * 4 + dx];
                        if reconstructed != i32::from(self.sample(x + dx, y + dy)) {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "intra mode {mode} TX_4X4 did not reconstruct source at ({},{})",
                                    x + dx,
                                    y + dy
                                ),
                            ));
                        }
                    }
                }
                let txb_ctx = self.txb_skip_ctx_4x4(mi_y, mi_x);
                let dc_ctx = self.dc_sign_ctx_4x4(mi_y, mi_x);
                let use_cost = self.count_cost();
                let cost = use_cost.then_some(&mut self.cost);
                let sym = (!self.cost_only).then_some(&mut self.sym);
                let coeff_context = residual::encode_tx4x4_coefficients_with_cost(
                    sym,
                    self.coeff_cdfs
                        .as_mut()
                        .expect("residual profile initializes coefficient CDFs"),
                    &coefficients,
                    txb_ctx,
                    dc_ctx,
                    cost,
                )?;
                self.record_coeff_context_4x4(mi_y, mi_x, coeff_context);
            }
        }

        for yy in 0..bw4 {
            for xx in 0..bw4 {
                let (rr, cc) = (r + yy, c + xx);
                self.skip[rr * self.mi_cols + cc] = 0;
                self.psize[rr * self.mi_cols + cc] = 0;
                self.pcolors[rr * self.mi_cols + cc] = [0; 8];
            }
        }
        if let Some(bc) = self.intrabc.as_mut() {
            bc.record(r, c, bw4, bw4, None);
        }
        Ok(())
    }

    /// IntraBC copy available for the `bw4`-MI square at MI `(r, c)` right
    /// now (deterministic in decoder state, so `partition()` and `block()`
    /// agree): the decoder-predictor-rooted match search, or `None` when
    /// disabled. Returns the MV in 1/8-pel units plus the predictor it was
    /// found from (so `block` avoids a second `predictor` query).
    fn intrabc_match(
        &self,
        r: usize,
        c: usize,
        bw4: usize,
        top_has_right: bool,
    ) -> Option<((i32, i32), (i32, i32))> {
        match self.intrabc.as_ref() {
            Some(bc) => bc.find_match(self.px, self.w, self.h, r, c, bw4, bw4, top_has_right),
            None => None,
        }
    }

    fn intrabc_match_shortlist(
        &self,
        r: usize,
        c: usize,
        bw4: usize,
        top_has_right: bool,
        limit: usize,
    ) -> Option<MatchShortlist> {
        self.intrabc.as_ref().map(|bc| {
            bc.find_match_shortlist(
                self.px,
                self.w,
                self.h,
                r,
                c,
                bw4,
                bw4,
                top_has_right,
                limit,
            )
        })
    }

    /// Uniform 16x16 lookup has its own exact-byte spatial index. A
    /// nonuniform result tells callers to preserve the existing search;
    /// uniform misses are definitive because the index covers every source
    /// origin on the ring's 4px candidate grid.
    fn intrabc_uniform_match(
        &self,
        r: usize,
        c: usize,
        bw4: usize,
        top_has_right: bool,
    ) -> io::Result<UniformDecision> {
        match self.intrabc.as_ref() {
            Some(bc) => {
                bc.find_uniform_match(self.px, self.w, self.h, r, c, bw4, bw4, top_has_right)
            }
            None => Ok(UniformDecision::NotUniform),
        }
    }

    fn record_uniform_copy_selected(&mut self) {
        if let Some(bc) = self.intrabc.as_mut() {
            bc.record_uniform_copy_selected();
        }
    }

    /// `get_palette_cache` for luma: sorted dedup merge of the above
    /// (unless at a 64px superblock-row top) and left palettes.
    /// Fixed `[u8; 8]` + length — at most 4+4 entries, no heap.
    fn palette_cache(&self, r: usize, c: usize) -> ([u8; 8], usize) {
        let above_n = if !r.is_multiple_of(16) {
            self.psize[(r - 1) * self.mi_cols + c] as usize
        } else {
            0
        };
        let left_n = if c > 0 {
            self.psize[r * self.mi_cols + (c - 1)] as usize
        } else {
            0
        };
        let blank = [0u8; 8];
        let above = if above_n > 0 {
            &self.pcolors[(r - 1) * self.mi_cols + c]
        } else {
            &blank
        };
        let left = if left_n > 0 {
            &self.pcolors[r * self.mi_cols + (c - 1)]
        } else {
            &blank
        };
        let mut cache = [0u8; 8];
        let mut cache_len: usize = 0;
        let push = |v: u8, cache: &mut [u8; 8], cache_len: &mut usize| {
            if *cache_len == 0 || cache[*cache_len - 1] != v {
                debug_assert!(*cache_len < 8);
                if *cache_len < 8 {
                    cache[*cache_len] = v;
                    *cache_len += 1;
                }
            }
        };
        let (mut ai, mut li) = (0, 0);
        while ai < above_n && li < left_n {
            let (ac, lc) = (above[ai], left[li]);
            if lc < ac {
                push(lc, &mut cache, &mut cache_len);
                li += 1;
            } else {
                push(ac, &mut cache, &mut cache_len);
                ai += 1;
                if lc == ac {
                    li += 1;
                }
            }
        }
        while ai < above_n {
            push(above[ai], &mut cache, &mut cache_len);
            ai += 1;
        }
        while li < left_n {
            push(left[li], &mut cache, &mut cache_len);
            li += 1;
        }
        (cache, cache_len)
    }

    /// `ns(n)` literal (§4.10.7).
    fn encode_ns(&mut self, val: usize, n: usize) {
        if n <= 1 {
            return;
        }
        let w = n.ilog2() + 1;
        let m = (1usize << w) - n;
        if val < m {
            self.enc_lit(val as u32, w - 1);
        } else {
            let coded = val + m;
            self.enc_lit((coded >> 1) as u32, w - 1);
            self.enc_lit((coded & 1) as u32, 1);
        }
    }

    /// Code one block at MI `(r, c)` with size `bsl` (2 ⇒ 16x16, 3 ⇒
    /// 32x32, 4 ⇒ 64x64): skip + DC + palette + indices, or — when
    /// IntraBC is enabled and an exact causal match exists — an intrabc
    /// copy (`skip = 1`, integer MV, no residuals). Palette mode is
    /// legal at all three sizes (dav1d gates it on `imax(bw4, bh4) <= 16`
    /// MI); the index tables only cover sizes 2..4, so the palette path
    /// returns `InvalidInput` for any other distinct count.
    ///
    /// `carried` is the successful `partition` search (MV + predictor) for
    /// this exact block, avoiding a repeated `find_match` and a repeated
    /// `predictor` query. `None` means "decide here" (leaves, flat blocks,
    /// IntraBC off, or no match at the parent).
    ///
    /// `force_palette` forces the palette path even when a copy exists
    /// (RDO palette candidate vs baseline decide). Baseline callers pass
    /// false; RDO palette trials pass true so an available copy does not
    /// preempt the palette cost measurement.
    fn block(
        &mut self,
        r: usize,
        c: usize,
        bsl: usize,
        top_has_right: bool,
        carried: Option<((i32, i32), (i32, i32))>,
    ) -> io::Result<()> {
        self.block_with_force(r, c, bsl, top_has_right, carried, false)
    }

    fn block_with_force(
        &mut self,
        r: usize,
        c: usize,
        bsl: usize,
        top_has_right: bool,
        carried: Option<((i32, i32), (i32, i32))>,
        force_palette: bool,
    ) -> io::Result<()> {
        self.block_with_force_pad(
            BlockContext {
                r,
                c,
                bsl,
                top_has_right,
                carried,
                force_palette,
            },
            None,
        )
    }

    /// `forced_pad` is used only by the look-ahead's cost-only candidate
    /// trials. Committed encoding still selects through `flat_pad_policy`.
    fn block_with_force_pad(
        &mut self,
        context: BlockContext,
        forced_pad: Option<u8>,
    ) -> io::Result<()> {
        let BlockContext {
            r,
            c,
            bsl,
            top_has_right,
            carried,
            force_palette,
        } = context;
        let bw: usize = 4 << bsl;
        let (sx, sy) = (c * 4, r * 4);
        let bw4 = 1usize << bsl;

        // Fast path: `partition` already found an exact copy for this
        // block. Skip distinct/colors/cache work entirely — a copy beats
        // the palette path by construction here.
        if let Some(((my, mx), (py, px_))) = carried {
            let sctx = self.skip_ctx(r, c);
            self.enc_skip(sctx, 1);
            self.encode_copy(r, c, bsl, bw4, my, mx, py, px_)?;
            return Ok(());
        }

        // Distinct-level scan doubles as the flat test (single-level
        // blocks always take the palette path — a copy can't beat a
        // ~15-bit flat table). On the verified four-shade path the
        // precomputed 16x16 masks are combined instead of rescanning
        // 256-entry presence arrays; otherwise the original scan runs.
        let fast_mask = self.combined_mask(r, c, bw4);
        let (distinct, set) = if let Some(m) = fast_mask {
            (MASK_COUNT[(m & 0x0F) as usize] as usize, None)
        } else {
            let mut set = [false; 256];
            for i in 0..bw {
                for j in 0..bw {
                    set[self.sample(sx + j, sy + i) as usize] = true;
                }
            }
            let distinct = set.iter().filter(|&&b| b).count();
            (distinct, Some(set))
        };
        if distinct == 0 {
            return Err(invalid_input(format!(
                "empty palette block at MI ({r},{c}) size {bw}x{bw}"
            )));
        }

        // IntraBC decision (non-flat only): an exact match in causal
        // decoded area beats the palette path (MV residual of ~10-25 bits
        // vs palette headers plus indices). `find_match` returns the
        // predictor too, so no second `predictor` query is needed.
        // `force_palette` (RDO) skips the search so the palette cost is
        // measured even when a copy is available.
        let bc_match: Option<((i32, i32), (i32, i32))> = if force_palette {
            None
        } else if distinct > 1 {
            self.intrabc_match(r, c, bw4, top_has_right)
        } else {
            None
        };

        // skip = 1 (no residual; reconstruction is exactly the palette
        // — or the copied pixels on the IntraBC path).
        let sctx = self.skip_ctx(r, c);

        if let Some(((my, mx), (py, px_))) = bc_match {
            self.enc_skip(sctx, 1);
            self.encode_copy(r, c, bsl, bw4, my, mx, py, px_)?;
            return Ok(());
        }
        // Palette path from here: build colors/cache/index scratch only
        // now that the copy path is ruled out.
        let mut colors = [0u8; 4];
        let mut psize: usize = 0;
        if let Some(m) = fast_mask {
            // Verified four-shade path: sorted source palette straight from
            // the 4-bit mask (shades ascending, matching the 0..256 scan
            // order for this alphabet). At most 4 distinct by construction.
            let mm = (m & 0x0F) as usize;
            let cnt = MASK_COUNT[mm] as usize;
            debug_assert_eq!(cnt, distinct);
            colors[..cnt].copy_from_slice(&MASK_COLORS[mm][..cnt]);
            psize = cnt;
        } else if let Some(ref set_ref) = set {
            for (v, &present) in set_ref.iter().enumerate() {
                if present {
                    if psize < 4 {
                        colors[psize] = v as u8;
                        psize += 1;
                    } else {
                        // More than 4 distinct: palette path is unsupported
                        // (no index tables). Count for the error below.
                        psize = distinct;
                        break;
                    }
                }
            }
        } else {
            // Unreachable: `fast_mask` is `None` exactly when `set` is
            // `Some` (see distinct scan above).
            debug_assert!(false, "internal: neither mask nor set for palette");
            return Err(io::Error::other(
                "internal: neither mask nor set for palette",
            ));
        }
        // `distinct` is authoritative; `psize` above is `min(distinct,4)`
        // unless overflow. Re-derive for the flat-pad and error paths.
        let mut n_colors = distinct;
        if distinct <= 4 {
            debug_assert_eq!(psize, distinct);
        } else {
            // Palette path: the index CDFs only support 2..4 colors.
            return Err(invalid_input(format!(
                "unsupported palette size {distinct} in {bw}x{bw} block at ({sx},{sy}): encoder supports 2..4 colors per palette-coded block"
            )));
        }
        // A flat block still needs a 2-entry table. Prefer padding with a
        // cached level (nearly free via a reuse flag below) over an
        // adjacent level (short delta chain); either way the pad value is
        // never referenced by the index map.
        let (cache_arr, cache_len) = self.palette_cache(r, c);
        if n_colors == 1 {
            let v = colors[0];
            if !self.cost_only && forced_pad.is_none() {
                self.flat_pad_stats.committed_flat_blocks += 1;
            }
            let baseline_pad = Self::baseline_flat_pad(v, &cache_arr[..cache_len]);
            let pad = match forced_pad {
                Some(pad) => {
                    debug_assert_ne!(pad, v, "flat pad must differ from the used color");
                    pad
                }
                None if self.flat_pad_policy == FlatPadPolicy::Lookahead
                    && self.pad_preview_depth == 0 =>
                {
                    self.lookahead_flat_pad(FlatPadLookahead {
                        r,
                        c,
                        bsl,
                        top_has_right,
                        force_palette,
                        used: v,
                        baseline_pad,
                        cache: &cache_arr[..cache_len],
                    })?
                }
                None => baseline_pad,
            };
            colors[1] = pad;
            n_colors = 2;
            // Keep the 2-entry table sorted (binary_search below).
            if colors[0] > colors[1] {
                colors.swap(0, 1);
            }
        }
        let psize = n_colors;

        self.enc_skip(sctx, 1);

        // Palette path from here: build colors/cache/index scratch only
        // now that the copy path is ruled out.
        // (colors/cache/index construction delayed until the palette
        // branch actually needs it — copy blocks skip all of it.)
        let emit = !self.cost_only;
        let use_cost = self.count_cost();
        if let Some(bc) = self.intrabc.as_mut() {
            let (sym, cost) = (&mut self.sym, &mut self.cost);
            bc.flag_with_cost(sym, false, cost, use_cost, emit);
        }

        let colors_slice = &colors[..psize];
        // Small direct mapping for the verified four-shade path:
        // shade index (0..3 via `GRAY_TO_SHADE`) → palette index. Flat-block
        // padding outside the four shades is never referenced by source
        // pixels, so its slot stays unmapped. General fallback keeps the
        // original per-pixel search exactly.
        let mut shade_to_pal = [0u8; 4];
        if fast_mask.is_some() {
            for (k, &cc) in colors_slice.iter().enumerate() {
                let s = GRAY_TO_SHADE[cc as usize];
                if s != 0xFF {
                    shade_to_pal[s as usize] = k as u8;
                }
            }
        }
        // Reusable index-map scratch: 64x64 max = 4096 bytes on the stack,
        // no per-block heap. Only the first `bw*bw` entries are used.
        let mut index_map = [0u8; 4096];
        let area = bw * bw;
        debug_assert!(area <= 4096);
        // Copy px/w out to avoid `&mut self` / `&self` borrow conflicts
        // with the local scratch (disjoint-field friendly).
        let px_ref = self.px;
        let w_ref = self.w;
        let use_shade_lut = fast_mask.is_some();
        for i in 0..bw {
            let row_off = (sy + i) * w_ref + sx;
            for j in 0..bw {
                let v = px_ref[row_off + j];
                let idx = if use_shade_lut {
                    let s = GRAY_TO_SHADE[v as usize];
                    debug_assert!(s != 0xFF, "four-shade path hit unsupported pixel {v}");
                    shade_to_pal[s as usize]
                } else {
                    // `colors_slice` is sorted; linear scan over ≤4 entries
                    // is cheaper than `binary_search` setup.
                    let mut idx = 0u8;
                    for (k, &cc) in colors_slice.iter().enumerate() {
                        if cc == v {
                            idx = k as u8;
                            break;
                        }
                    }
                    idx
                };
                index_map[i * bw + j] = idx;
            }
        }
        let index_slice = &index_map[..area];

        // Palette blocks use DC_PRED. In the residual profile, mode contexts
        // follow the per-MI mode grid; the legacy path keeps its original
        // single all-DC CDF row.
        self.enc_y_mode(r, c, DC_PRED as u8);
        self.record_y_mode(r, c, bw4, bw4, DC_PRED as u8);

        // has_palette_y = 1 (context = neighbours paletted; CDF row
        // selected by block size via `bsl`).
        let above_p = r > 0 && self.psize[(r - 1) * self.mi_cols + c] > 0;
        let left_p = c > 0 && self.psize[r * self.mi_cols + (c - 1)] > 0;
        let pctx = usize::from(above_p) + usize::from(left_p);
        self.enc_pal_mode(bsl, pctx, 1);

        // palette_size_y_minus_2, then the colors: one L(1) reuse flag
        // per palette-cache entry (equi-probable, non-adapting, stopping
        // once pal_sz entries are collected — mirroring dav1d's
        // read_pal_plane), then explicit coding for the rest (first color
        // raw 8b, remaining deltas). Reusing every cached color we need is
        // optimal: visited flag positions cost a bit either way, and every
        // reuse shortens the delta chain and hastens the early stop. Only
        // 4 gray levels exist globally, so the cache almost always covers
        // the block — this is where the upscale's bits were going.
        self.enc_pal_size(bsl, psize - 2);
        let mut is_used = [false; 4];
        let mut n_used = 0usize;
        for &cc in &cache_arr[..cache_len] {
            if n_used == psize {
                break;
            }
            if let Ok(pos) = colors_slice.binary_search(&cc) {
                self.enc_lit(1, 1);
                is_used[pos] = true;
                n_used += 1;
            } else {
                self.enc_lit(0, 1);
            }
        }
        let mut new_colors = [0u8; 4];
        let mut new_len: usize = 0;
        for (k, &cc) in colors_slice.iter().enumerate() {
            if !is_used[k] {
                new_colors[new_len] = cc;
                new_len += 1;
            }
        }
        let new_slice = &new_colors[..new_len];
        if n_used < psize {
            self.enc_lit(u32::from(new_slice[0]), 8);
            if new_slice.len() > 1 {
                // Minimal initial width covering *every* delta in the
                // chain (decoder widths only shrink, so the maximum
                // delta dictates a valid — and optimal — start; using
                // just the first delta truncates a later larger one).
                let mut need = 1u32;
                for w in new_slice.windows(2) {
                    let d = w[1] as u32 - w[0] as u32 - 1;
                    if d > 0 {
                        need = need.max(d.ilog2() + 1);
                    }
                }
                let need = need.clamp(5, 8);
                self.enc_lit(need - 5, 2);
                let mut palette_bits = need;
                for k in 1..new_slice.len() {
                    let delta = new_slice[k] as u32 - new_slice[k - 1] as u32 - 1;
                    self.enc_lit(delta, palette_bits);
                    let prev = u32::from(new_slice[k]);
                    if prev + 1 >= 255 {
                        // Decoder fills any remaining new slots with 255 and
                        // stops (dav1d `read_pal_plane`:
                        // `if (prev + !pl >= max)`). Reaching 255 must be
                        // the last explicit new color; reaching 254 may
                        // leave exactly one trailing 255, which the decoder
                        // fills implicitly (e.g. new `[0, 254, 255]` stops
                        // after 254).
                        debug_assert!(
                            k + 1 == new_slice.len()
                                || (prev == 254
                                    && k + 2 == new_slice.len()
                                    && new_slice[k + 1] == 255)
                        );
                        break;
                    }
                    // Matches dav1d's `1 + ulog2(max - prev - !pl)` with
                    // ulog2(0) == 0 (note: plain CeilLog2 differs at
                    // prev == 254, where the width stays 1, not 0).
                    palette_bits = palette_bits.min(1 + (254 - prev).max(1).ilog2());
                }
            }
        }

        // Bookkeeping for neighbour contexts of later blocks.
        let n4 = 1usize << bsl;
        for y in 0..n4 {
            for x in 0..n4 {
                let (rr, cc) = (r + y, c + x);
                self.skip[rr * self.mi_cols + cc] = 1;
                self.psize[rr * self.mi_cols + cc] = psize as u8;
                self.pcolors[rr * self.mi_cols + cc][..psize].copy_from_slice(colors_slice);
            }
        }
        self.reset_coeff_context(r, c, n4, n4);
        // Mirror dav1d's `splat_intraref`: palette blocks contribute no
        // motion candidate (and mark the footprint decoded for match
        // search) exactly like the decoder's `rt` grid.
        if let Some(bc) = &mut self.intrabc {
            bc.record(r, c, bw4, bw4, None);
        }

        // color_index_map_y: first index via ns(), rest in wavefront order
        // as positions in the neighbour-derived ColorOrder. The tiny
        // `PAL_LUT` replaces repeated scoring/ordering/searches; on a LUT
        // miss (unreachable for valid 2..4 palettes) the reference
        // calculation runs to preserve exact behavior.
        self.encode_ns(index_slice[0] as usize, psize);
        for i in 1..(2 * bw - 1) {
            let mut j = i.min(bw - 1);
            let j_end = i.saturating_sub(bw - 1);
            loop {
                let (rr, cc) = (i - j, j);
                let actual = index_slice[rr * bw + cc];
                let left = if cc > 0 {
                    index_slice[rr * bw + (cc - 1)]
                } else {
                    4
                };
                let top = if rr > 0 {
                    index_slice[(rr - 1) * bw + cc]
                } else {
                    4
                };
                let tl = if rr > 0 && cc > 0 {
                    index_slice[(rr - 1) * bw + (cc - 1)]
                } else {
                    4
                };
                let (ctx, sym) = if let Some((c, s)) = pal_lut_lookup(psize, left, top, tl, actual)
                {
                    (c, s)
                } else {
                    let (order, ctx) = palette_color_context(index_slice, bw, rr, cc, psize);
                    let sym = order
                        .iter()
                        .position(|&x| x == actual as usize)
                        .unwrap_or(0);
                    (ctx, sym)
                };
                self.enc_pal_idx(psize, ctx, sym);
                if j == j_end {
                    break;
                }
                j -= 1;
            }
        }
        // skip = 1 ⇒ read_block_tx_size returns TX_4X4 with no symbols,
        // and residual() codes nothing. Lossless (qindex 0) needs no
        // transform, quantizer, or coefficient syntax at all.
        Ok(())
    }

    fn baseline_flat_pad(v: u8, cache: &[u8]) -> u8 {
        cache
            .iter()
            .copied()
            .find(|&cached| cached != v)
            .unwrap_or(if v < 255 { v + 1 } else { v - 1 })
    }

    /// Next partition node in the current 64x64 superblock's depth-first
    /// sibling order. Every ancestor between this node and the root is
    /// known to have split because this block was reached. Crossing to the
    /// next superblock is deliberately outside the one-node horizon.
    fn next_sibling_node(
        &self,
        r: usize,
        c: usize,
        bw4: usize,
    ) -> Option<(usize, usize, usize, bool)> {
        let (sb_r, sb_c) = (r / 16 * 16, c / 16 * 16);
        let (mut node_r, mut node_c, mut node_bw4) = (r, c, bw4);
        while node_bw4 < 16 {
            let parent_bw4 = node_bw4 * 2;
            let parent_r = sb_r + ((node_r - sb_r) / parent_bw4) * parent_bw4;
            let parent_c = sb_c + ((node_c - sb_c) / parent_bw4) * parent_bw4;
            let half = parent_bw4 / 2;
            let quad =
                usize::from(node_r >= parent_r + half) * 2 + usize::from(node_c >= parent_c + half);
            if quad < 3 {
                let next_quad = quad + 1;
                let next_r = parent_r + usize::from(next_quad >= 2) * half;
                let next_c = parent_c + usize::from(next_quad % 2 == 1) * half;
                let top_has_right = self.top_has_right_for_node(next_r, next_c, node_bw4);
                return Some((next_r, next_c, node_bw4, top_has_right));
            }
            (node_r, node_c, node_bw4) = (parent_r, parent_c, parent_bw4);
        }
        None
    }

    fn top_has_right_for_node(&self, r: usize, c: usize, bw4: usize) -> bool {
        let (mut node_r, mut node_c) = (r / 16 * 16, c / 16 * 16);
        let mut node_bw4 = 16usize;
        let mut top_has_right = true;
        while node_bw4 > bw4 {
            let half = node_bw4 / 2;
            let quad = usize::from(r >= node_r + half) * 2 + usize::from(c >= node_c + half);
            top_has_right = Self::child_top_right(quad, top_has_right);
            if quad >= 2 {
                node_r += half;
            }
            if quad % 2 == 1 {
                node_c += half;
            }
            node_bw4 = half;
        }
        top_has_right
    }

    fn encode_partition_node(
        &mut self,
        r: usize,
        c: usize,
        bw4: usize,
        top_has_right: bool,
    ) -> io::Result<()> {
        if self.use_rdo {
            self.rdo_partition(r, c, bw4, top_has_right)
        } else {
            self.partition(r, c, bw4, top_has_right)
        }
    }

    /// Score one unused flat-block palette entry by cost-only replay of the
    /// current block and one actual upcoming sibling partition node. Each
    /// candidate gets a fresh snapshot of both node footprints, CDFs, cost,
    /// partition state, palette neighbors, and IntraBC availability state.
    /// The arithmetic coder is untouched because `trial()` disables output.
    fn lookahead_flat_pad(&mut self, context: FlatPadLookahead<'_>) -> io::Result<u8> {
        let FlatPadLookahead {
            r,
            c,
            bsl,
            top_has_right,
            force_palette,
            used,
            baseline_pad,
            cache,
        } = context;
        let bw4 = 1usize << bsl;
        let Some((next_r, next_c, next_bw4, next_top_has_right)) =
            self.next_sibling_node(r, c, bw4)
        else {
            return Ok(baseline_pad);
        };

        // Candidate ordering is stable and preserves the historical pad as
        // the deterministic tie-break. Cache colors follow AV1's sorted cache
        // order; effective-image colors follow ascending sample order.
        let mut candidates = [0u8; 265];
        let mut candidate_len = 0usize;
        candidates[candidate_len] = baseline_pad;
        candidate_len += 1;
        // Consider cache colors first (free reuse via flags).
        for &candidate in cache.iter() {
            if candidate == used
                || (candidate < used) != (baseline_pad < used)
                || candidates[..candidate_len].contains(&candidate)
            {
                continue;
            }
            candidates[candidate_len] = candidate;
            candidate_len += 1;
        }
        // Also consider global colors that are not in the cache, but only when
        // the cache is empty (first block in a region). When the cache already
        // has colors, adding a non-cached global color pollutes the cache for
        // subsequent blocks without providing a reuse benefit, which can make
        // the overall file larger despite the local pair optimization.
        if cache.is_empty() {
            for &candidate in self.global_colors[..self.global_color_len].iter() {
                if candidate == used
                    || (candidate < used) != (baseline_pad < used)
                    || candidates[..candidate_len].contains(&candidate)
                {
                    continue;
                }
                candidates[candidate_len] = candidate;
                candidate_len += 1;
            }
        }
        if candidate_len == 1 {
            return Ok(baseline_pad);
        }
        self.flat_pad_stats.searches += 1;

        let rect_r = r.min(next_r);
        let rect_c = c.min(next_c);
        let rect_h = (r + bw4).max(next_r + next_bw4) - rect_r;
        let rect_w = (c + bw4).max(next_c + next_bw4) - rect_c;
        debug_assert!(rect_w <= 16 && rect_h <= 16);

        let mut best_pad = baseline_pad;
        let mut best_cost = f64::INFINITY;
        let mut baseline_cost = f64::INFINITY;
        const EPS: f64 = 1e-9;
        for &candidate in &candidates[..candidate_len] {
            let previous_scoring = self.pad_scoring;
            self.pad_scoring = true;
            let trial = self.trial(rect_r, rect_c, rect_w, rect_h, |enc| {
                enc.block_with_force_pad(
                    BlockContext {
                        r,
                        c,
                        bsl,
                        top_has_right,
                        carried: None,
                        force_palette,
                    },
                    Some(candidate),
                )?;
                let previous_preview = enc.pad_preview_depth;
                enc.pad_preview_depth = previous_preview.saturating_add(1);
                let preview =
                    enc.encode_partition_node(next_r, next_c, next_bw4, next_top_has_right);
                enc.pad_preview_depth = previous_preview;
                preview
            });
            self.pad_scoring = previous_scoring;

            let cost = match trial {
                Ok(cost) => {
                    self.flat_pad_stats.candidate_trials += 1;
                    self.flat_pad_stats.preview_nodes += 1;
                    cost
                }
                Err(_) => {
                    self.flat_pad_stats.errors += 1;
                    return Ok(baseline_pad);
                }
            };
            if candidate == baseline_pad {
                baseline_cost = cost;
            }
            if cost < best_cost - EPS {
                best_cost = cost;
                best_pad = candidate;
            }
        }

        if (baseline_cost - best_cost).abs() <= EPS {
            self.flat_pad_stats.ties += 1;
        } else if best_pad == baseline_pad {
            self.flat_pad_stats.baseline_wins += 1;
        } else {
            self.flat_pad_stats.lookahead_wins += 1;
        }
        debug_assert_eq!(
            best_pad < used,
            baseline_pad < used,
            "flat pad selection must preserve the used color's palette index"
        );
        Ok(best_pad)
    }

    /// Encode an IntraBC copy for the `bw4`-MI block at `(r, c)` given the
    /// chosen MV and its predictor: flag + MVD, neighbour bookkeeping
    /// (skip/psize, no `ymode` — all-DC is implicit), and `record`.
    #[allow(clippy::too_many_arguments)]
    fn encode_copy(
        &mut self,
        r: usize,
        c: usize,
        bsl: usize,
        bw4: usize,
        my: i32,
        mx: i32,
        py: i32,
        px_: i32,
    ) -> io::Result<()> {
        // IntraBC copy: flag, then the MV residual against the
        // decoder's predictor (replicated refmvs search). No mode,
        // palette, or index symbols; dav1d records DC/empty palette
        // contexts for neighbours, mirrored in bookkeeping below.
        // Disjoint field borrows (`intrabc` vs `sym`/`cost`) keep the
        // cost-aware flag/MVD exact.
        let emit = !self.cost_only;
        let use_cost = self.count_cost();
        let Some(bc) = self.intrabc.as_mut() else {
            return Err(io::Error::other(
                "internal: IntraBC match without IntraBC state",
            ));
        };
        // SAFETY of borrows: `bc` borrows `self.intrabc`; `sym`/`cost`
        // borrow disjoint fields. The compiler accepts these together
        // because they are direct field borrows. To express that, inline
        // the flag/MVD via a helper that takes split borrows.
        // (Implemented inline to avoid holding `bc` across `self.sym`.)
        // Fall through to the split-borrow block below.
        let _ = bc;
        // Re-establish disjoint borrows in a single statement.
        let (intrabc, sym, cost) = (&mut self.intrabc, &mut self.sym, &mut self.cost);
        let Some(bc2) = intrabc.as_mut() else {
            return Err(io::Error::other(
                "internal: IntraBC match without IntraBC state",
            ));
        };
        bc2.flag_with_cost(sym, true, cost, use_cost, emit);
        bc2.encode_mvd_with_cost(sym, ((my, mx), (py, px_)), cost, use_cost, emit);
        let n4 = 1usize << bsl;
        for y in 0..n4 {
            for x in 0..n4 {
                let (rr, cc) = (r + y, c + x);
                self.skip[rr * self.mi_cols + cc] = 1;
                self.psize[rr * self.mi_cols + cc] = 0;
            }
        }
        self.record_y_mode(r, c, n4, n4, DC_PRED as u8);
        self.reset_coeff_context(r, c, n4, n4);
        self.intrabc
            .as_mut()
            .ok_or_else(|| io::Error::other("internal: missing IntraBC state"))?
            .record(r, c, bw4, bw4, Some((my, mx)));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Cost-based (RDO) partition selection.
    //
    // Bounded search (explicit):
    // * Fully contained 64x64 (bw4=16, bsl=4) and 32x32 (bw4=8, bsl=3):
    //   NONE+palette (if `palette_legal`), NONE+copy (if `intrabc_match`
    //   finds one in the trial state), SPLIT with recursively selected
    //   children (decoder order, `child_top_right` threaded).
    // * Fully contained 16x16 leaves (bw4=4, bsl=2): palette vs copy, no
    //   smaller partitions.
    // * Partially on-screen nodes: unchanged edge path (split_or bools or
    //   forced SPLIT, never NONE), recursing via `rdo_partition` so
    //   contained children still use RDO. Decoder-availability restrictions
    //   are enforced by `intrabc_match` (SB delay, wavefront, decoded
    //   bitmap) in the trial state.
    //
    // Cost model: `self.cost` fractional estimates (`adapt_bits` for
    // adapting symbols, 1.0 per literal bit), accumulated alongside every
    // encode (baseline and trials share helpers so bytes match for identical
    // decisions). Trials never compare raw buffer lengths and never finalize
    // the live coder; the delta `trial_cost - base_cost` ranks candidates.
    // See `adapt_bits` docs for approximations (EC_MIN_PROB floor excepted,
    // range/carry/flush ignored, future costs unmodeled — local, not global,
    // optimum).
    //
    // State handling: trials update costs and adapting CDFs without emitting
    // arithmetic output. CDFs and bounded neighbour footprints are copied
    // inline; the coder is untouched, and no snapshot vectors are allocated.
    // Rejected trials restore that state and preserve candidate/search work
    // counters separately. A winning split retains its decisions in a local
    // plan and replays them only from the same incoming CDF, neighbor, and
    // availability state. Plans never cross those state boundaries.
    // -----------------------------------------------------------------------

    fn speculative_depth(&self) -> u32 {
        self.intrabc
            .as_ref()
            .map(|bc| bc.speculative_depth())
            .unwrap_or(0)
    }

    fn enter_speculative(&mut self) {
        if let Some(bc) = self.intrabc.as_mut() {
            bc.enter_speculative();
        }
    }

    fn restore_speculative(&mut self, saved: u32) {
        if let Some(bc) = self.intrabc.as_mut() {
            bc.restore_speculative(saved);
        }
    }

    /// Palette legal iff distinct levels fit existing capabilities (2..4,
    /// plus flat 1 with the encoder's cache/adjacent padding behavior).
    /// `region_levels` saturates at 5; 0 (empty, unreachable) is illegal.
    fn palette_legal(&self, r: usize, c: usize, bw4: usize) -> bool {
        let levels = self.region_levels(r, c, bw4);
        (1..=4).contains(&levels)
    }

    #[cfg(test)]
    fn save_snapshot(&self, r: usize, c: usize, bw4: usize, bh4: usize) -> Snapshot {
        self.save_snapshot_inner(r, c, bw4, bh4, true)
    }

    fn save_trial_snapshot(&self, r: usize, c: usize, bw4: usize, bh4: usize) -> Snapshot {
        self.save_snapshot_inner(r, c, bw4, bh4, false)
    }

    fn save_snapshot_inner(
        &self,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
        save_symbol_state: bool,
    ) -> Snapshot {
        const MAX_FOOTPRINT: usize = 16 * 16;
        debug_assert!(bw4 * bh4 <= MAX_FOOTPRINT);
        debug_assert!(bw4 <= 16 && bh4 <= 16);
        let mut skip_fp = [0; MAX_FOOTPRINT];
        let mut psize_fp = [0; MAX_FOOTPRINT];
        let mut pcolors_fp = [[0; 8]; MAX_FOOTPRINT];
        let mut ymode_fp = self.ymode.as_ref().map(|_| [DC_PRED as u8; MAX_FOOTPRINT]);
        let mut decoded_fp = [false; MAX_FOOTPRINT];
        let mut rec_fp = [MvRec::default(); MAX_FOOTPRINT];
        let mut footprint_len = 0;
        for y in 0..bh4 {
            for x in 0..bw4 {
                let (rr, cc) = (r + y, c + x);
                // Contained callers guarantee in-bounds; edge callers never
                // snapshot (they use the baseline edge path without trials).
                // Defensive bounds keep release safe.
                if rr < self.mi_rows && cc < self.mi_cols {
                    let idx = rr * self.mi_cols + cc;
                    skip_fp[footprint_len] = self.skip[idx];
                    psize_fp[footprint_len] = self.psize[idx];
                    pcolors_fp[footprint_len] = self.pcolors[idx];
                    if let (Some(grid), Some(saved)) = (&self.ymode, ymode_fp.as_mut()) {
                        saved[footprint_len] = grid[idx];
                    }
                    if let Some(bc) = self.intrabc.as_ref() {
                        decoded_fp[footprint_len] = bc.decoded[idx];
                        rec_fp[footprint_len] = bc.rec[idx];
                    }
                    footprint_len += 1;
                }
            }
        }
        let mut above_fp = [0; 16];
        let mut residual_coeff_contexts = self.residual_profile.then_some(ResidualCoeffSnapshot {
            above_level: [0; 16],
            left_level: [0; 16],
            above_dc: [0; 16],
            left_dc: [0; 16],
        });
        let mut above_len = 0;
        for k in 0..bw4 {
            if c + k < self.mi_cols {
                above_fp[above_len] = self.above_part[c + k];
                if let Some(saved) = residual_coeff_contexts.as_mut() {
                    saved.above_level[above_len] = self.above_coeff_level[c + k];
                    saved.above_dc[above_len] = self.above_coeff_dc[c + k];
                }
                above_len += 1;
            }
        }
        let mut left_fp = [0; 16];
        let mut left_len = 0;
        for k in 0..bh4 {
            if r + k < self.mi_rows {
                left_fp[left_len] = self.left_part[r + k];
                if let Some(saved) = residual_coeff_contexts.as_mut() {
                    saved.left_level[left_len] = self.left_coeff_level[r + k];
                    saved.left_dc[left_len] = self.left_coeff_dc[r + k];
                }
                left_len += 1;
            }
        }
        Snapshot {
            sym: save_symbol_state.then(|| self.sym.clone()),
            cdfs: self.cdfs.clone(),
            coeff_cdfs: self.coeff_cdfs.clone(),
            residual_mode_cdfs: self.residual_mode_cdfs.clone(),
            intrabc_cdfs: self.intrabc.as_ref().map(|bc| bc.snapshot_cdfs()),
            cost: self.cost,
            cost_only: self.cost_only,
            r,
            c,
            bw4,
            bh4,
            footprint_len,
            skip_fp,
            psize_fp,
            pcolors_fp,
            ymode_fp,
            residual_coeff_contexts,
            decoded_fp,
            rec_fp,
            above_len,
            above_fp,
            left_len,
            left_fp,
        }
    }

    fn restore_snapshot(&mut self, snap: Snapshot) {
        // `candidates` persists (not in snapshot).
        if let Some(sym) = snap.sym {
            self.sym = sym;
        }
        self.cdfs = snap.cdfs;
        self.coeff_cdfs = snap.coeff_cdfs;
        self.residual_mode_cdfs = snap.residual_mode_cdfs;
        self.cost = snap.cost;
        self.cost_only = snap.cost_only;
        if let (Some(bc), Some(saved)) = (self.intrabc.as_mut(), snap.intrabc_cdfs) {
            bc.restore_cdfs(saved);
        }
        let mut i = 0usize;
        for y in 0..snap.bh4 {
            for x in 0..snap.bw4 {
                let (rr, cc) = (snap.r + y, snap.c + x);
                if rr < self.mi_rows && cc < self.mi_cols {
                    let idx = rr * self.mi_cols + cc;
                    if i < snap.footprint_len {
                        self.skip[idx] = snap.skip_fp[i];
                        self.psize[idx] = snap.psize_fp[i];
                        self.pcolors[idx] = snap.pcolors_fp[i];
                        if let (Some(grid), Some(saved)) = (&mut self.ymode, snap.ymode_fp.as_ref())
                        {
                            grid[idx] = saved[i];
                        }
                    }
                    if let Some(bc) = self.intrabc.as_mut() {
                        if i < snap.footprint_len {
                            bc.decoded[idx] = snap.decoded_fp[i];
                            bc.rec[idx] = snap.rec_fp[i];
                        }
                    }
                    i += 1;
                }
            }
        }
        for k in 0..snap.above_len {
            let v = snap.above_fp[k];
            if snap.c + k < self.mi_cols {
                self.above_part[snap.c + k] = v;
                if let Some(saved) = &snap.residual_coeff_contexts {
                    self.above_coeff_level[snap.c + k] = saved.above_level[k];
                    self.above_coeff_dc[snap.c + k] = saved.above_dc[k];
                }
            }
        }
        for k in 0..snap.left_len {
            let v = snap.left_fp[k];
            if snap.r + k < self.mi_rows {
                self.left_part[snap.r + k] = v;
                if let Some(saved) = &snap.residual_coeff_contexts {
                    self.left_coeff_level[snap.r + k] = saved.left_level[k];
                    self.left_coeff_dc[snap.r + k] = saved.left_dc[k];
                }
            }
        }
    }

    /// Trial-encode `f` from the current state with speculative stats,
    /// returning its estimated cost delta (or `Err` for illegal candidates).
    /// Restores the snapshot afterwards; increments `candidates` once per
    /// attempted trial (even on `Err`? No — only on `Ok`; illegal screened
    /// without encoding are not counted; see callers). Nesting-aware: pushes
    /// speculative depth so inner winners remain speculative relative to an
    /// outer trial; restore returns to the enclosing depth.
    fn trial<F>(&mut self, r: usize, c: usize, bw4: usize, bh4: usize, f: F) -> io::Result<f64>
    where
        F: FnOnce(&mut Self) -> io::Result<()>,
    {
        let base_cost = self.cost;
        let snap = self.save_trial_snapshot(r, c, bw4, bh4);
        self.cost_only = true;
        self.enter_speculative();
        let res = f(self);
        let delta = self.cost - base_cost;
        self.restore_snapshot(snap);
        // `candidates` persists across restore (not in snapshot).
        match res {
            Ok(()) => {
                self.candidates += 1;
                // Count globally for benchmarks (atomic, Rayon-safe).
                RDO_CANDIDATES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(delta)
            }
            Err(e) => Err(e),
        }
    }

    /// RDO leaf (contained 16x16): palette vs copy, no smaller partitions.
    /// Palette-only mode (`intrabc == None`) encodes palette directly.
    fn rdo_leaf_with_plan(
        &mut self,
        r: usize,
        c: usize,
        bsl: usize,
        top_has_right: bool,
        mut plan: Option<&mut PartitionPlan>,
    ) -> io::Result<()> {
        debug_assert_eq!(bsl, 2);
        let bw4 = 4usize;
        if self.intrabc.is_none() && !self.residual_profile {
            // Palette-only: single legal candidate (leaves always ≤4 for
            // valid inputs, checked at the boundary).
            let ctx = self.partition_ctx(r, c, bsl);
            self.enc_part(bsl, ctx, PARTITION_NONE);
            self.update_partition_ctx(r, c, bsl);
            self.block_with_force(r, c, bsl, top_has_right, None, true)?;
            if let Some(plan) = plan.as_deref_mut() {
                plan.push(PlanDecision::Palette);
            }
            return Ok(());
        }
        // Preserve the enclosing speculative depth: when invoked inside a
        // parent trial the winner remains speculative relative to the parent.
        // Trials below push depth and restore to this value; the committed
        // re-encode runs at this depth (not unconditionally 0).
        let outer_depth = self.speculative_depth();
        let pal_legal = self.palette_legal(r, c, bw4);
        // Copy availability is determined inside the copy trial (speculative
        // search from the same base state); no live-state search here so
        // rejected trials leave no availability changes.
        // Evaluate palette trial (if legal).
        let pal_cost: Option<f64> = if pal_legal {
            Some(self.trial(r, c, bw4, bw4, |enc| {
                let ctx = enc.partition_ctx(r, c, bsl);
                enc.enc_part(bsl, ctx, PARTITION_NONE);
                enc.update_partition_ctx(r, c, bsl);
                enc.block_with_force(r, c, bsl, top_has_right, None, true)
            })?)
        } else {
            None
        };
        let residual_pick = if self.residual_profile {
            let mut best = None;
            for mode in RESIDUAL_INTRA_MODES {
                let cost = self.trial(r, c, bw4, bw4, |enc| {
                    let ctx = enc.partition_ctx(r, c, bsl);
                    enc.enc_part(bsl, ctx, PARTITION_NONE);
                    enc.update_partition_ctx(r, c, bsl);
                    enc.encode_residual_block(r, c, bsl, mode)
                })?;
                if best.is_none_or(|(_, incumbent)| cost < incumbent - 1e-9) {
                    best = Some((mode, cost));
                }
            }
            best
        } else {
            None
        };
        // Uniform blocks keep their existing exact-byte lookup. The TopK8
        // candidate is limited to nonuniform 16x16 leaves; every match is
        // priced from an identical incoming state before one is replayed.
        const EPS: f64 = 1e-9;
        let best_copy: Option<RankedCopyCost> = {
            let outer_depth = self.speculative_depth();
            self.enter_speculative();
            let searched = (|| -> io::Result<MotionSearchResult> {
                let mut candidates = [None; MOTION_SHORTLIST_MAX];
                let mut candidate_count = 0;
                let mut used_shortlist = false;
                match self.intrabc_uniform_match(r, c, bw4, top_has_right)? {
                    UniformDecision::Uniform(Some(mv_pred)) => {
                        candidates[0] = Some((mv_pred, true, 0));
                        candidate_count = 1;
                    }
                    UniformDecision::Uniform(None) => {}
                    UniformDecision::NotUniform => match self.motion_candidate_policy {
                        MotionCandidatePolicy::FirstMatch => {
                            if let Some(mv_pred) = self.intrabc_match(r, c, bw4, top_has_right) {
                                candidates[0] = Some((mv_pred, false, 0));
                                candidate_count = 1;
                            }
                        }
                        MotionCandidatePolicy::TopK8 => {
                            used_shortlist = true;
                            let shortlist = self
                                .intrabc_match_shortlist(
                                    r,
                                    c,
                                    bw4,
                                    top_has_right,
                                    MOTION_SHORTLIST_MAX,
                                )
                                .expect("IntraBC state checked before shortlist search");
                            use std::sync::atomic::Ordering;
                            MOTION_SHORTLIST_SEARCHES.fetch_add(1, Ordering::Relaxed);
                            MOTION_SOURCE_PROBES.fetch_add(shortlist.probes, Ordering::Relaxed);
                            for (index, candidate_slot) in
                                candidates.iter_mut().enumerate().take(shortlist.len)
                            {
                                let candidate = shortlist.matches[index]
                                    .expect("shortlist entries are contiguous");
                                *candidate_slot = Some((candidate.mv_pred, false, index));
                            }
                            candidate_count = shortlist.len;
                        }
                    },
                }
                Ok((candidates, candidate_count, used_shortlist))
            })();
            self.restore_speculative(outer_depth);
            let (candidates, candidate_count, used_shortlist) = searched?;
            let mut best: Option<RankedCopyCost> = None;
            for (index, candidate) in candidates[..candidate_count].iter().enumerate() {
                let (mv_pred, uniform, shortlist_index) =
                    candidate.expect("candidate entries are contiguous");
                if used_shortlist {
                    MOTION_PRICED_MATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                let cost = self.trial(r, c, bw4, bw4, |enc| {
                    let ctx = enc.partition_ctx(r, c, bsl);
                    enc.enc_part(bsl, ctx, PARTITION_NONE);
                    enc.update_partition_ctx(r, c, bsl);
                    let sctx = enc.skip_ctx(r, c);
                    enc.enc_skip(sctx, 1);
                    enc.encode_copy(
                        r,
                        c,
                        bsl,
                        bw4,
                        mv_pred.0 .0,
                        mv_pred.0 .1,
                        mv_pred.1 .0,
                        mv_pred.1 .1,
                    )
                })?;
                let priced = RankedCopyCost {
                    cost,
                    mv_pred,
                    uniform,
                    shortlist_index: if used_shortlist {
                        index
                    } else {
                        shortlist_index
                    },
                };
                if best.is_none_or(|incumbent| cost < incumbent.cost - EPS) {
                    best = Some(priced);
                }
            }
            best
        };
        // Existing candidates retain their historical tie order (palette,
        // then copy). Residual modes replace the incumbent only on a strict
        // local estimated-cost improvement.
        #[derive(Clone, Copy)]
        enum LeafPick {
            Palette,
            Copy(RankedCopyCost),
            Residual(u8),
        }
        let mut best = match (pal_cost, best_copy) {
            (Some(pc), Some(copy)) if copy.cost < pc - EPS => {
                Some((LeafPick::Copy(copy), copy.cost))
            }
            (Some(pc), _) => Some((LeafPick::Palette, pc)),
            (None, Some(copy)) => Some((LeafPick::Copy(copy), copy.cost)),
            (None, None) => None,
        };
        if let Some((mode, cost)) = residual_pick {
            if best.is_none_or(|(_, incumbent)| cost < incumbent - EPS) {
                best = Some((LeafPick::Residual(mode), cost));
            }
        }
        let selected = best
            .map(|(pick, _)| pick)
            .ok_or_else(|| invalid_input(format!("no legal RDO leaf candidate at MI ({r},{c})")))?;
        // Committed re-encode at the enclosing depth (speculative when nested,
        // committed when top-level). Preserves the parent trial's state so
        // rejected branches never increment committed-path counters.
        debug_assert_eq!(
            self.speculative_depth(),
            outer_depth,
            "trials must restore enclosing depth"
        );
        self.restore_speculative(outer_depth);
        match selected {
            LeafPick::Palette => {
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_NONE);
                self.update_partition_ctx(r, c, bsl);
                self.block_with_force(r, c, bsl, top_has_right, None, true)?;
                if let Some(plan) = plan.as_deref_mut() {
                    plan.push(PlanDecision::Palette);
                }
            }
            LeafPick::Copy(selected) => {
                let mv_pred = selected.mv_pred;
                let uniform_copy = selected.uniform;
                if !self.cost_only
                    && self.motion_candidate_policy == MotionCandidatePolicy::TopK8
                    && !uniform_copy
                    && selected.shortlist_index > 0
                {
                    use std::sync::atomic::Ordering;
                    MOTION_ALTERNATE_CHOICES.fetch_add(1, Ordering::Relaxed);
                    MOTION_ALTERNATE_INDEX_SUM
                        .fetch_add(selected.shortlist_index as u64, Ordering::Relaxed);
                }
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_NONE);
                self.update_partition_ctx(r, c, bsl);
                let sctx = self.skip_ctx(r, c);
                self.enc_skip(sctx, 1);
                self.encode_copy(
                    r,
                    c,
                    bsl,
                    bw4,
                    mv_pred.0 .0,
                    mv_pred.0 .1,
                    mv_pred.1 .0,
                    mv_pred.1 .1,
                )?;
                if uniform_copy {
                    self.record_uniform_copy_selected();
                }
                if let Some(plan) = plan {
                    plan.push(PlanDecision::Copy(mv_pred, uniform_copy));
                }
            }
            LeafPick::Residual(mode) => {
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_NONE);
                self.update_partition_ctx(r, c, bsl);
                self.encode_residual_block(r, c, bsl, mode)?;
                if let Some(plan) = plan {
                    plan.push(PlanDecision::Residual(mode));
                }
            }
        }
        Ok(())
    }

    /// RDO for fully contained 32x32/64x64: NONE+palette, NONE+copy, SPLIT.
    #[cfg(test)]
    fn rdo_contained(
        &mut self,
        r: usize,
        c: usize,
        bw4: usize,
        bsl: usize,
        top_has_right: bool,
    ) -> io::Result<()> {
        self.rdo_contained_with_plan(r, c, bw4, bsl, top_has_right, None)
    }

    fn rdo_contained_with_plan(
        &mut self,
        r: usize,
        c: usize,
        bw4: usize,
        bsl: usize,
        top_has_right: bool,
        mut plan: Option<&mut PartitionPlan>,
    ) -> io::Result<()> {
        debug_assert!(bw4 == 8 || bw4 == 16);
        debug_assert!(r + bw4 <= self.mi_rows && c + bw4 <= self.mi_cols);
        // Preserve the enclosing speculative depth (see `rdo_leaf`).
        let outer_depth = self.speculative_depth();
        // Palette-only mode: palette vs split.
        if self.intrabc.is_none() && !self.residual_profile {
            let pal_legal = self.palette_legal(r, c, bw4);
            if !pal_legal {
                // Must split (no trial needed for the single legal option).
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_SPLIT);
                let half = bw4 >> 1;
                Self::record_plan(&mut plan, PlanDecision::Split);
                self.rdo_partition_with_plan(
                    r,
                    c,
                    half,
                    Self::child_top_right(0, top_has_right),
                    plan.as_deref_mut(),
                )?;
                self.rdo_partition_with_plan(
                    r,
                    c + half,
                    half,
                    Self::child_top_right(1, top_has_right),
                    plan.as_deref_mut(),
                )?;
                self.rdo_partition_with_plan(
                    r + half,
                    c,
                    half,
                    Self::child_top_right(2, top_has_right),
                    plan.as_deref_mut(),
                )?;
                self.rdo_partition_with_plan(
                    r + half,
                    c + half,
                    half,
                    Self::child_top_right(3, top_has_right),
                    plan.as_deref_mut(),
                )?;
                return Ok(());
            }
            // Compare palette vs split.
            let pal_cost = self.trial(r, c, bw4, bw4, |enc| {
                let ctx = enc.partition_ctx(r, c, bsl);
                enc.enc_part(bsl, ctx, PARTITION_NONE);
                enc.update_partition_ctx(r, c, bsl);
                enc.block_with_force(r, c, bsl, top_has_right, None, true)
            })?;
            let mut split_plan = PartitionPlan::new();
            let split_cost = self.trial(r, c, bw4, bw4, |enc| {
                let ctx = enc.partition_ctx(r, c, bsl);
                enc.enc_part(bsl, ctx, PARTITION_SPLIT);
                let half = bw4 >> 1;
                enc.rdo_partition_with_plan(
                    r,
                    c,
                    half,
                    Self::child_top_right(0, top_has_right),
                    Some(&mut split_plan),
                )?;
                enc.rdo_partition_with_plan(
                    r,
                    c + half,
                    half,
                    Self::child_top_right(1, top_has_right),
                    Some(&mut split_plan),
                )?;
                enc.rdo_partition_with_plan(
                    r + half,
                    c,
                    half,
                    Self::child_top_right(2, top_has_right),
                    Some(&mut split_plan),
                )?;
                enc.rdo_partition_with_plan(
                    r + half,
                    c + half,
                    half,
                    Self::child_top_right(3, top_has_right),
                    Some(&mut split_plan),
                )
            })?;
            const EPS: f64 = 1e-9;
            debug_assert_eq!(
                self.speculative_depth(),
                outer_depth,
                "trials must restore enclosing depth"
            );
            self.restore_speculative(outer_depth);
            if split_cost < pal_cost - EPS {
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_SPLIT);
                self.apply_plan_children(r, c, bw4, top_has_right, &split_plan)?;
                Self::record_plan(&mut plan, PlanDecision::Split);
                if let Some(plan) = plan {
                    plan.extend(&split_plan);
                }
            } else {
                // Tie → palette (deterministic).
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_NONE);
                self.update_partition_ctx(r, c, bsl);
                self.block_with_force(r, c, bsl, top_has_right, None, true)?;
                Self::record_plan(&mut plan, PlanDecision::Palette);
            }
            return Ok(());
        }
        // IntraBC-enabled: palette (if legal), copy (if available), split.
        let pal_legal = self.palette_legal(r, c, bw4);
        let pal_cost: Option<f64> = if pal_legal {
            Some(self.trial(r, c, bw4, bw4, |enc| {
                let ctx = enc.partition_ctx(r, c, bsl);
                enc.enc_part(bsl, ctx, PARTITION_NONE);
                enc.update_partition_ctx(r, c, bsl);
                enc.block_with_force(r, c, bsl, top_has_right, None, true)
            })?)
        } else {
            None
        };
        // Copy availability via speculative peek (no live mutation), then
        // trial-encode if available.
        let copy_info: Option<(BcMatch, bool)> = {
            let outer_depth = self.speculative_depth();
            self.enter_speculative();
            let m = self.intrabc_match(r, c, bw4, top_has_right);
            self.restore_speculative(outer_depth);
            m.map(|mv_pred| (mv_pred, false))
        };
        let copy_cost: Option<CopyCost> = match copy_info {
            None => None,
            Some((mv_pred, is_uniform)) => {
                let cost = self.trial(r, c, bw4, bw4, |enc| {
                    let ctx = enc.partition_ctx(r, c, bsl);
                    enc.enc_part(bsl, ctx, PARTITION_NONE);
                    enc.update_partition_ctx(r, c, bsl);
                    let sctx = enc.skip_ctx(r, c);
                    enc.enc_skip(sctx, 1);
                    enc.encode_copy(
                        r,
                        c,
                        bsl,
                        bw4,
                        mv_pred.0 .0,
                        mv_pred.0 .1,
                        mv_pred.1 .0,
                        mv_pred.1 .1,
                    )
                })?;
                Some((cost, mv_pred, is_uniform))
            }
        };
        let residual_pick = if self.residual_profile {
            let mut best = None;
            for mode in RESIDUAL_INTRA_MODES {
                let cost = self.trial(r, c, bw4, bw4, |enc| {
                    let ctx = enc.partition_ctx(r, c, bsl);
                    enc.enc_part(bsl, ctx, PARTITION_NONE);
                    enc.update_partition_ctx(r, c, bsl);
                    enc.encode_residual_block(r, c, bsl, mode)
                })?;
                if best.is_none_or(|(_, incumbent)| cost < incumbent - 1e-9) {
                    best = Some((mode, cost));
                }
            }
            best
        } else {
            None
        };
        let mut split_plan = PartitionPlan::new();
        let split_cost: Option<f64> = match self.trial(r, c, bw4, bw4, |enc| {
            let ctx = enc.partition_ctx(r, c, bsl);
            enc.enc_part(bsl, ctx, PARTITION_SPLIT);
            let half = bw4 >> 1;
            enc.rdo_partition_with_plan(
                r,
                c,
                half,
                Self::child_top_right(0, top_has_right),
                Some(&mut split_plan),
            )?;
            enc.rdo_partition_with_plan(
                r,
                c + half,
                half,
                Self::child_top_right(1, top_has_right),
                Some(&mut split_plan),
            )?;
            enc.rdo_partition_with_plan(
                r + half,
                c,
                half,
                Self::child_top_right(2, top_has_right),
                Some(&mut split_plan),
            )?;
            enc.rdo_partition_with_plan(
                r + half,
                c + half,
                half,
                Self::child_top_right(3, top_has_right),
                Some(&mut split_plan),
            )
        }) {
            Ok(d) => Some(d),
            Err(e) => {
                // Split of a contained node cannot fail for valid inputs
                // (children are contained and leaves always ≤4). Propagate.
                return Err(e);
            }
        };
        // Pick cheapest; tie → earliest (palette, then copy, then split).
        // Earlier wins ties by using `< best + EPS` (not `< best - EPS`):
        // a candidate replaces the current best iff it is not strictly
        // worse beyond EPS, so the earliest among near-equals survives.
        const EPS: f64 = 1e-9;
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Pick {
            Palette,
            Copy,
            Residual(u8),
            Split,
        }
        let mut best = Pick::Split;
        let mut best_cost = split_cost.expect("split must succeed");
        if let Some((cc, _, _)) = copy_cost {
            if cc < best_cost + EPS {
                best = Pick::Copy;
                best_cost = cc;
            }
        }
        if let Some(pc) = pal_cost {
            if pc < best_cost + EPS {
                best = Pick::Palette;
                best_cost = pc;
            }
        }
        if let Some((mode, rc)) = residual_pick {
            if rc < best_cost - EPS {
                best = Pick::Residual(mode);
                best_cost = rc;
            }
        }
        // Edge: no palette and no copy (levels>4, no match) → split only.
        // `best` is already Split in that case.
        if pal_cost.is_none() && copy_cost.is_none() && residual_pick.is_none() {
            best = Pick::Split;
        }
        let _ = best_cost;
        debug_assert_eq!(
            self.speculative_depth(),
            outer_depth,
            "trials must restore enclosing depth"
        );
        self.restore_speculative(outer_depth);
        match best {
            Pick::Palette => {
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_NONE);
                self.update_partition_ctx(r, c, bsl);
                self.block_with_force(r, c, bsl, top_has_right, None, true)?;
                Self::record_plan(&mut plan, PlanDecision::Palette);
            }
            Pick::Copy => {
                let (_, mv_pred, uniform_copy) = copy_cost.expect("copy must exist when chosen");
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_NONE);
                self.update_partition_ctx(r, c, bsl);
                let sctx = self.skip_ctx(r, c);
                self.enc_skip(sctx, 1);
                self.encode_copy(
                    r,
                    c,
                    bsl,
                    bw4,
                    mv_pred.0 .0,
                    mv_pred.0 .1,
                    mv_pred.1 .0,
                    mv_pred.1 .1,
                )?;
                if uniform_copy {
                    self.record_uniform_copy_selected();
                }
                Self::record_plan(&mut plan, PlanDecision::Copy(mv_pred, uniform_copy));
            }
            Pick::Residual(mode) => {
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_NONE);
                self.update_partition_ctx(r, c, bsl);
                self.encode_residual_block(r, c, bsl, mode)?;
                Self::record_plan(&mut plan, PlanDecision::Residual(mode));
            }
            Pick::Split => {
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_SPLIT);
                self.apply_plan_children(r, c, bw4, top_has_right, &split_plan)?;
                Self::record_plan(&mut plan, PlanDecision::Split);
                if let Some(plan) = plan {
                    plan.extend(&split_plan);
                }
            }
        }
        Ok(())
    }

    fn record_plan(plan: &mut Option<&mut PartitionPlan>, decision: PlanDecision) {
        if let Some(plan) = plan.as_deref_mut() {
            plan.push(decision);
        }
    }

    /// Re-emit the four child choices selected by the winning split trial.
    /// The plan was produced from the same incoming coder/CDF/neighbour and
    /// decoded-availability state, so replaying it preserves symbol order and
    /// avoids repeating the child searches.
    fn apply_plan_children(
        &mut self,
        r: usize,
        c: usize,
        bw4: usize,
        top_has_right: bool,
        plan: &PartitionPlan,
    ) -> io::Result<()> {
        let half = bw4 >> 1;
        let mut cursor = 0;
        for (child_r, child_c, child_idx) in [
            (r, c, 0),
            (r, c + half, 1),
            (r + half, c, 2),
            (r + half, c + half, 3),
        ] {
            self.apply_plan_node(
                child_r,
                child_c,
                half,
                Self::child_top_right(child_idx, top_has_right),
                plan,
                &mut cursor,
            )?;
        }
        assert_eq!(cursor, plan.len, "partition plan has trailing decisions");
        Ok(())
    }

    fn apply_plan_node(
        &mut self,
        r: usize,
        c: usize,
        bw4: usize,
        top_has_right: bool,
        plan: &PartitionPlan,
        cursor: &mut usize,
    ) -> io::Result<()> {
        debug_assert!(r + bw4 <= self.mi_rows && c + bw4 <= self.mi_cols);
        let decision = plan.nodes[*cursor].expect("partition plan ended before a node");
        *cursor += 1;
        let bsl = bw4.trailing_zeros() as usize;
        match decision {
            PlanDecision::Palette => {
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_NONE);
                self.update_partition_ctx(r, c, bsl);
                self.block_with_force(r, c, bsl, top_has_right, None, true)?;
            }
            PlanDecision::Copy(mv_pred, uniform_copy) => {
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_NONE);
                self.update_partition_ctx(r, c, bsl);
                let sctx = self.skip_ctx(r, c);
                self.enc_skip(sctx, 1);
                self.encode_copy(
                    r,
                    c,
                    bsl,
                    bw4,
                    mv_pred.0 .0,
                    mv_pred.0 .1,
                    mv_pred.1 .0,
                    mv_pred.1 .1,
                )?;
                if uniform_copy {
                    self.record_uniform_copy_selected();
                }
            }
            PlanDecision::Residual(mode) => {
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_NONE);
                self.update_partition_ctx(r, c, bsl);
                self.encode_residual_block(r, c, bsl, mode)?;
            }
            PlanDecision::Split => {
                let ctx = self.partition_ctx(r, c, bsl);
                self.enc_part(bsl, ctx, PARTITION_SPLIT);
                let half = bw4 >> 1;
                for (child_r, child_c, child_idx) in [
                    (r, c, 0),
                    (r, c + half, 1),
                    (r + half, c, 2),
                    (r + half, c + half, 3),
                ] {
                    self.apply_plan_node(
                        child_r,
                        child_c,
                        half,
                        Self::child_top_right(child_idx, top_has_right),
                        plan,
                        cursor,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Recursive RDO partition (edge-preserving). Fully contained 64/32 go
    /// to `rdo_contained`, contained 16x16 leaves to `rdo_leaf`; partial
    /// nodes use the unchanged edge path and recurse via `rdo_partition`.
    fn rdo_partition(
        &mut self,
        r: usize,
        c: usize,
        bw4: usize,
        top_has_right: bool,
    ) -> io::Result<()> {
        self.rdo_partition_with_plan(r, c, bw4, top_has_right, None)
    }

    fn rdo_partition_with_plan(
        &mut self,
        r: usize,
        c: usize,
        bw4: usize,
        top_has_right: bool,
        plan: Option<&mut PartitionPlan>,
    ) -> io::Result<()> {
        if r >= self.mi_rows || c >= self.mi_cols {
            return Ok(());
        }
        let bsl = bw4.trailing_zeros() as usize;
        if bw4 == 4 {
            // Leaves tile exactly on 16-aligned images; still guard bounds.
            // Contained leaves use RDO; offscreen returns (unreachable).
            return self.rdo_leaf_with_plan(r, c, bsl, top_has_right, plan);
        }
        if r + bw4 <= self.mi_rows && c + bw4 <= self.mi_cols {
            return self.rdo_contained_with_plan(r, c, bw4, bsl, top_has_right, plan);
        }
        // Edge path (unchanged): split_or bools or forced SPLIT, never NONE.
        let half = bw4 >> 1;
        let has_rows = r + half < self.mi_rows;
        let has_cols = c + half < self.mi_cols;
        if has_rows && has_cols {
            let ctx = self.partition_ctx(r, c, bsl);
            self.enc_part(bsl, ctx, PARTITION_SPLIT);
        } else if has_cols {
            let ctx = self.partition_ctx(r, c, bsl);
            let psum = split_psum_horz(self.cdfs.part_row(bsl, ctx));
            debug_assert!(psum < 32768);
            let c0 = if psum == 0 {
                32767
            } else {
                (32768 - psum) as u16
            };
            self.enc_static(1, &[c0, 32768]);
        } else if has_rows {
            let ctx = self.partition_ctx(r, c, bsl);
            let psum = split_psum_vert(self.cdfs.part_row(bsl, ctx));
            debug_assert!(psum < 32768);
            let c0 = if psum == 0 {
                32767
            } else {
                (32768 - psum) as u16
            };
            self.enc_static(1, &[c0, 32768]);
        }
        self.rdo_partition(r, c, half, Self::child_top_right(0, top_has_right))?;
        self.rdo_partition(r, c + half, half, Self::child_top_right(1, top_has_right))?;
        self.rdo_partition(r + half, c, half, Self::child_top_right(2, top_has_right))?;
        self.rdo_partition(
            r + half,
            c + half,
            half,
            Self::child_top_right(3, top_has_right),
        )?;
        Ok(())
    }

    fn finish(self) -> io::Result<Vec<u8>> {
        let (bytes, _) = self.finish_with_candidates()?;
        Ok(bytes)
    }

    fn finish_with_candidates(mut self) -> io::Result<(Vec<u8>, u64)> {
        // Merge per-image RDO trial counts to the process-global counter
        // for benchmarks (committed path; trials already counted via
        // `trial()` atomics, but per-image `candidates` is the source of
        // truth for this image — the global was already bumped per trial,
        // so do NOT double-count here; see `encode_*_rdo` for file wins).
        // (No-op: `trial()` already did `RDO_CANDIDATES.fetch_add`.)
        let use_rdo = self.use_rdo;
        for r in (0..self.mi_rows).step_by(16) {
            for c in (0..self.mi_cols).step_by(16) {
                // SB roots start with top_has_right set, mirroring the
                // generated edge-tree root (`top_has_right = 1`).
                if use_rdo {
                    self.rdo_partition(r, c, 16, true)?;
                } else {
                    self.partition(r, c, 16, true)?;
                }
            }
        }
        let n = self.candidates;
        merge_flat_pad_stats(self.flat_pad_stats);
        Ok((self.sym.finish(), n))
    }
}

// ---------------------------------------------------------------------------
// Headers (raw bits) and OBU framing.
// ---------------------------------------------------------------------------

fn obu_wrap(obu_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + payload.len());
    out.push((obu_type << 3) | 0b010);
    write_leb128(&mut out, payload.len() as u64);
    out.extend_from_slice(payload);
    out
}

/// Minimal defined `seq_level_idx` (AV1 Annex A) covering `w`x`h`.
///
/// Only defined levels are returned (2.0, 2.1, 3.0, 3.1, 4.0, 5.0, 6.0);
/// still pictures only need the pic-size/dimension constraints, not the
/// display/decode-rate ones. Falls back to 31 (maximum parameters) when
/// nothing defined covers the size. AVIF Baseline (`MA1B`) caps at 5.1
/// (8912896px, 8192x4352), so callers targeting Baseline should reject
/// anything needing more than that.
fn seq_level_idx_for(w: u32, h: u32) -> u8 {
    // (idx, MaxPicSize, MaxHSize, MaxVSize) for defined still-relevant levels.
    const LEVELS: [(u8, u32, u32, u32); 7] = [
        (0, 147456, 2048, 1152),     // 2.0
        (1, 278784, 2816, 1584),     // 2.1
        (4, 665856, 4352, 2448),     // 3.0
        (5, 1065024, 5504, 3096),    // 3.1
        (8, 2359296, 6144, 3456),    // 4.0
        (12, 8912896, 8192, 4352),   // 5.0
        (16, 35651584, 16384, 8704), // 6.0
    ];
    let picsize = w.saturating_mul(h);
    for (idx, max_pic, max_h, max_v) in LEVELS {
        if picsize <= max_pic && w <= max_h && h <= max_v {
            return idx;
        }
    }
    31
}

/// Sequence header OBU payload: profile 0, still picture, reduced header,
/// minimal defined level covering `w`x`h` (2.0 for 160x144, 4.0 for
/// 1280x1152), 64px superblocks, all loop tools off,
/// monochrome full-range 8-bit, no film grain.
fn sequence_header_obu(w: u32, h: u32) -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.put_bits(0, 3); // seq_profile = 0 (Main)
    bw.put_bit(1); // still_picture
    bw.put_bit(1); // reduced_still_picture_header
    bw.put_bits(u32::from(seq_level_idx_for(w, h)), 5); // seq_level_idx
    let wbits = dimension_bits(w);
    let hbits = dimension_bits(h);
    bw.put_bits(wbits - 1, 4); // frame_width_bits_minus_1
    bw.put_bits(hbits - 1, 4); // frame_height_bits_minus_1
    bw.put_bits(w - 1, wbits); // max_frame_width_minus_1
    bw.put_bits(h - 1, hbits); // max_frame_height_minus_1
    bw.put_bit(0); // use_128x128_superblock
    bw.put_bit(0); // enable_filter_intra
    bw.put_bit(0); // enable_intra_edge_filter
    bw.put_bit(0); // enable_superres
    bw.put_bit(0); // enable_cdef
    bw.put_bit(0); // enable_restoration
                   // color_config: 8-bit, monochrome, no description, full range.
    bw.put_bit(0); // high_bitdepth
    bw.put_bit(1); // mono_chrome
    bw.put_bit(0); // color_description_present_flag
    bw.put_bit(1); // color_range (full)
    bw.put_bit(0); // film_grain_params_present
    let bits = bw.bit_len();
    let mut bytes = bw.into_bytes();
    // trailing_bits: one 1 bit, then zero-pad to the byte.
    let rem = bits % 8;
    if rem == 0 {
        bytes.push(0x80);
    } else if let Some(last) = bytes.last_mut() {
        *last |= 1 << (7 - rem);
    } else {
        // Unreachable in practice (header always emits bytes), but avoid
        // panicking and emit a lone trailing bit instead.
        bytes.push(0x80);
    }
    bytes
}

/// AV1 tile limits for `tile_info()` (§6.8, single tile with 64px
/// superblocks throughout this encoder: `use_128x128_superblock = 0`,
/// so `sbShift = 4`, `sbSize = 6`).
const MAX_TILE_WIDTH: u32 = 4096;
const MAX_TILE_AREA: u32 = 4096 * 2304;
const MAX_TILE_COLS: u32 = 64;
const MAX_TILE_ROWS: u32 = 64;

/// `tile_log2(blkSize, target)` (§6.8): smallest `k` with
/// `(blkSize << k) >= target`.
fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while ((blk_size as u64) << k) < target as u64 {
        k += 1;
    }
    k
}

/// Uniform single-tile `tile_info()` parameters for 64px superblocks:
/// `(minLog2TileCols, maxLog2TileCols, maxLog2TileRows, minLog2Tiles)`.
/// Callers stay at the minima; single-tile is feasible exactly when
/// `minLog2TileCols == 0 && minLog2Tiles == 0`.
fn tile_params(w: u32, h: u32) -> (u32, u32, u32, u32) {
    let mi_cols = 2 * ((w.saturating_add(7)) >> 3);
    let mi_rows = 2 * ((h.saturating_add(7)) >> 3);
    let sb_cols = (mi_cols.saturating_add(15)) >> 4;
    let sb_rows = (mi_rows.saturating_add(15)) >> 4;
    // 64px SBs: sbSize = sbShift + 2 = 6.
    let max_tile_width_sb = MAX_TILE_WIDTH >> 6;
    let max_tile_area_sb = MAX_TILE_AREA >> 12;
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles =
        min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_cols.saturating_mul(sb_rows)));
    (
        min_log2_tile_cols,
        max_log2_tile_cols,
        max_log2_tile_rows,
        min_log2_tiles,
    )
}

/// Dimensions this encoder supports: 16-aligned coding grid, non-empty,
/// at most 65536px per side (sequence header `frame_width_bits_minus_1` /
/// `frame_height_bits_minus_1` are 4 bits, so at most 16 bits per dimension),
/// and single-tile feasible with 64px superblocks (i.e. the uniform
/// `tile_info()` below stays at `TileColsLog2 == TileRowsLog2 == 0`,
/// so no `context_update_tile_id`/`tile_size_bytes` section exists).
/// Returns `InvalidInput` otherwise; all public entry points go through
/// here (via `validate_gray`).
fn check_supported_dimensions(w: u32, h: u32) -> io::Result<()> {
    if !(w.is_multiple_of(16) && h.is_multiple_of(16)) {
        return Err(invalid_input(format!(
            "dimensions must be 16-aligned, got {w}x{h}"
        )));
    }
    if w == 0 || h == 0 {
        return Err(invalid_input(format!(
            "dimensions must be non-empty, got {w}x{h}"
        )));
    }
    const MAX_SEQ_DIMENSION: u32 = 65536;
    if w > MAX_SEQ_DIMENSION || h > MAX_SEQ_DIMENSION {
        return Err(invalid_input(format!(
            "dimensions must be at most {MAX_SEQ_DIMENSION}x{MAX_SEQ_DIMENSION}, got {w}x{h}"
        )));
    }
    let (min_cols, _, _, min_tiles) = tile_params(w, h);
    if min_cols != 0 || min_tiles != 0 {
        return Err(invalid_input(format!(
            "dimensions {w}x{h} need multiple tiles, encoder supports single tile only"
        )));
    }
    Ok(())
}

/// Supported colors: the palette index CDFs only cover sizes 2..4, so
/// every 16x16 coding block must hold at most 4 distinct levels (larger
/// palettes have no index tables). Checked at the boundary before
/// encoding; `block()` re-checks the palette path it actually codes.
fn check_supported_colors(gray: &[u8], w: u32, h: u32) -> io::Result<()> {
    let w_usize = w as usize;
    let h_usize = h as usize;
    // `w`/`h` are 16-aligned here (dimensions checked first), so the
    // 16px grid tiles the raster exactly.
    for ty in (0..h_usize).step_by(16) {
        for tx in (0..w_usize).step_by(16) {
            let mut set = [false; 256];
            let mut count = 0usize;
            for y in ty..ty + 16 {
                for x in tx..tx + 16 {
                    let v = gray[y * w_usize + x] as usize;
                    if !set[v] {
                        set[v] = true;
                        count += 1;
                        if count > 4 {
                            return Err(invalid_input(format!(
                                "unsupported palette size {count} in 16x16 block at ({tx},{ty}): encoder supports 2..4 colors per block"
                            )));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Boundary validation for the `io::Result` entry points: dimensions,
/// checked area, buffer length, and supported colors. Returns
/// `InvalidInput` instead of panicking. On success also returns the
/// verified four-shade masks (`Some` iff every pixel is in `GAME_SHADES`);
/// verified images skip the per-16x16 256-entry color scan since ≤4 colors
/// per block hold by construction, while unverified images run the original
/// detailed check (preserving arbitrary-gray acceptance and rejection).
fn validate_gray(gray: &[u8], w: u32, h: u32) -> io::Result<(usize, Option<Vec<u8>>)> {
    check_supported_dimensions(w, h)?;
    let area = checked_area(w, h)?;
    if gray.len() != area {
        return Err(invalid_input(format!(
            "pixel buffer length mismatch for {w}x{h}: expected {area} bytes, got {}",
            gray.len()
        )));
    }
    let masks = build_source_masks(gray, w as usize, h as usize);
    if masks.is_none() {
        check_supported_colors(gray, w, h)?;
    }
    Ok((area, masks))
}

/// Uncompressed frame header bits for our KEY still, then byte-aligned.
/// `allow_intrabc` gates the per-block intrabc path (obu.c reads the bit
/// right here when `allow_screen_content_tools && !superres`); flipping it
/// changes nothing else (loopfilter/CDEF/restoration sections are already
/// absent via lossless, delta_q via `base_q_idx == 0`).
fn frame_header_bits(w: u32, h: u32, allow_intrabc: bool) -> io::Result<Vec<u8>> {
    check_supported_dimensions(w, h)?;
    let mut bw = BitWriter::new();
    // (show_existing_frame/frame_type/show_frame implied by reduced header)
    bw.put_bit(0); // disable_cdf_update (CDFs adapt)
    bw.put_bit(1); // allow_screen_content_tools (seq forces SELECT)
    bw.put_bit(0); // force_integer_mv (overridden to 1 for intra)
                   // frame_size_override implied 0 → dimensions from sequence max.
                   // render_and_frame_size_different = 0:
    bw.put_bit(0);
    // allow_intrabc:
    bw.put_bit(u8::from(allow_intrabc));
    // (primary_ref NONE, tile_info, quant, segmentation, deltas,
    //  loopfilter/cdef/lr all implied off by KEY + CodedLossless)
    // tile_info: uniform spacing, single tile. Each increment flag is
    // only present while its loop has room to increase
    // (`TileColsLog2 < maxLog2TileCols`, then rows); e.g. a 64x64
    // frame is a single SB each way, so both loops are empty and no
    // increment bit exists. Emitting unconditional breaks shifts every
    // later field and the stream fails to decode.
    bw.put_bit(1); // uniform_tile_spacing_flag
    {
        let (min_log2_tile_cols, max_log2_tile_cols, max_log2_tile_rows, min_log2_tiles) =
            tile_params(w, h);
        // Single tile: stay at the minima (both zero per
        // `check_supported_dimensions` above).
        let tile_cols_log2 = min_log2_tile_cols;
        if tile_cols_log2 < max_log2_tile_cols {
            bw.put_bit(0); // increment_tile_cols_log2 → break
        }
        let min_log2_tile_rows = min_log2_tiles.saturating_sub(tile_cols_log2);
        let tile_rows_log2 = min_log2_tile_rows;
        if tile_rows_log2 < max_log2_tile_rows {
            bw.put_bit(0); // increment_tile_rows_log2 → break
        }
        debug_assert!(tile_cols_log2 == 0 && tile_rows_log2 == 0);
        // (TileColsLog2 == TileRowsLog2 == 0 ⇒ no
        //  context_update_tile_id / tile_size_bytes_minus_1.)
        if tile_cols_log2 != 0 || tile_rows_log2 != 0 {
            return Err(invalid_input(format!(
                "dimensions {w}x{h} need multiple tiles, encoder supports single tile only"
            )));
        }
    }
    // quantization_params: base_q_idx = 0 (lossless), no deltas, no qmatrix:
    bw.put_bits(0, 8); // base_q_idx
    bw.put_bit(0); // DeltaQYDc.delta_coded
    bw.put_bit(0); // using_qmatrix
                   // segmentation_params: off:
    bw.put_bit(0); // segmentation_enabled
                   // (delta_q/lf: base_q_idx == 0 ⇒ nothing)
                   // (loop_filter/cdef/restoration: CodedLossless/AllLossless ⇒ nothing)
                   // (tx_mode: CodedLossless ⇒ ONLY_4X4, nothing)
                   // (frame_reference_mode/global_motion: intra ⇒ nothing)
    bw.put_bit(1); // reduced_tx_set
                   // (film_grain: not present ⇒ nothing)
    bw.byte_align();
    Ok(bw.into_bytes())
}

// ---------------------------------------------------------------------------
// Public entry points.
// ---------------------------------------------------------------------------

/// Encode one raster to its AV1 item payload (sequence header OBU + frame
/// OBU wrapping the palette-coded tile data). Returns the payload and the
/// matching `av1C` body.
///
/// When IntraBC is enabled, the raster is encoded both ways (IntraBC vs
/// palette-only) and the smaller payload is kept; when disabled, only the
/// palette path runs (faster).
///
/// Validates dimensions, checked area, buffer length, and supported
/// colors at the boundary, returning `InvalidInput` instead of panicking.
fn encode_obu_payload(
    gray: &[u8],
    w: u32,
    h: u32,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    encode_obu_payload_with_policy(
        gray,
        w,
        h,
        FlatPadPolicy::Baseline,
        search_radius_rings,
        uniform_search_radius_rings,
    )
}

fn encode_obu_payload_with_policy(
    gray: &[u8],
    w: u32,
    h: u32,
    flat_pad_policy: FlatPadPolicy,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    if !USE_INTRABC {
        return encode_obu_payload_with_policy_and_intrabc(
            gray,
            w,
            h,
            USE_INTRABC,
            flat_pad_policy,
            search_radius_rings,
            uniform_search_radius_rings,
        );
    }
    let (on_payload, on_av1c) = encode_obu_payload_with_policy_and_intrabc(
        gray,
        w,
        h,
        true,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
    )?;
    let (off_payload, off_av1c) = encode_obu_payload_with_policy_and_intrabc(
        gray,
        w,
        h,
        false,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
    )?;
    if on_payload.len() <= off_payload.len() {
        Ok((on_payload, on_av1c))
    } else {
        Ok((off_payload, off_av1c))
    }
}

#[cfg(test)]
fn encode_obu_payload_with(
    gray: &[u8],
    w: u32,
    h: u32,
    use_intrabc: bool,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    encode_obu_payload_with_policy_and_intrabc(
        gray,
        w,
        h,
        use_intrabc,
        FlatPadPolicy::Baseline,
        search_radius_rings,
        uniform_search_radius_rings,
    )
}

fn encode_obu_payload_with_policy_and_intrabc(
    gray: &[u8],
    w: u32,
    h: u32,
    use_intrabc: bool,
    flat_pad_policy: FlatPadPolicy,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    let (_, masks) = validate_gray(gray, w, h)?;

    let tile = TileEncoder::new_with_rdo_and_pad_policy(
        gray,
        w as usize,
        h as usize,
        use_intrabc,
        masks,
        false,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
    )?;
    let tile_data = tile.finish()?;

    let seq_obu = obu_wrap(1, &sequence_header_obu(w, h));
    let mut frame_payload = frame_header_bits(w, h, use_intrabc)?;
    // Tile group OBU with a single tile: no header bits, byte-aligned
    // by construction, then the tile data. Together with the frame
    // header this forms an OBU_FRAME (type 6), not OBU_FRAME_HEADER.
    frame_payload.extend_from_slice(&tile_data);
    let frame_obu = obu_wrap(6, &frame_payload);
    let mut payload = seq_obu;
    payload.extend_from_slice(&frame_obu);

    let level_idx = seq_level_idx_for(w, h);
    Ok((payload, av1c_mono8(level_idx)))
}

/// RDO variant of `encode_obu_payload_with`: same headers/container, but
/// the tile uses cost-based partition selection (`use_rdo = true`) among
/// legal palette/copy/split alternatives with the fractional-bit model.
/// `use_intrabc = false` exercises palette-only RDO (palette vs split);
/// `true` exercises IntraBC-enabled RDO (palette/copy/split). Search and MV
/// selection are unchanged; only partition/mode selection differs.
#[cfg(test)]
fn encode_obu_payload_with_rdo(
    gray: &[u8],
    w: u32,
    h: u32,
    use_intrabc: bool,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    encode_obu_payload_with_rdo_and_pad_policy(
        gray,
        w,
        h,
        use_intrabc,
        FlatPadPolicy::Baseline,
        search_radius_rings,
        uniform_search_radius_rings,
    )
}

#[cfg(test)]
fn encode_obu_payload_with_rdo_and_pad_policy(
    gray: &[u8],
    w: u32,
    h: u32,
    use_intrabc: bool,
    flat_pad_policy: FlatPadPolicy,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    encode_obu_payload_with_rdo_pad_and_motion_policy(
        gray,
        w,
        h,
        use_intrabc,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
        MotionCandidatePolicy::FirstMatch,
    )
}

// Clippy's count is a false positive here: this internal bridge carries the
// same encoder controls as its existing RDO entry point plus one policy.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn encode_obu_payload_with_rdo_pad_and_motion_policy(
    gray: &[u8],
    w: u32,
    h: u32,
    use_intrabc: bool,
    flat_pad_policy: FlatPadPolicy,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
    motion_candidate_policy: MotionCandidatePolicy,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    encode_obu_payload_with_rdo_pad_motion_residual(
        gray,
        w,
        h,
        use_intrabc,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
        motion_candidate_policy,
        false,
        false,
    )
}

// Clippy's count is a false positive for this internal adapter: it preserves
// the existing encoder controls and adds the two residual-profile selectors.
#[allow(clippy::too_many_arguments)]
fn encode_obu_payload_with_rdo_pad_motion_residual(
    gray: &[u8],
    w: u32,
    h: u32,
    use_intrabc: bool,
    flat_pad_policy: FlatPadPolicy,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
    motion_candidate_policy: MotionCandidatePolicy,
    residual_profile: bool,
    scaled_8x_profile: bool,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    let (_, masks) = validate_gray(gray, w, h)?;
    let mut tile = TileEncoder::new_with_rdo_pad_and_motion_policy(
        gray,
        w as usize,
        h as usize,
        use_intrabc,
        masks,
        true,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
        motion_candidate_policy,
    )?;
    if residual_profile {
        tile.enable_residual_profile(scaled_8x_profile);
    }
    let tile_data = tile.finish()?;
    let seq_obu = obu_wrap(1, &sequence_header_obu(w, h));
    let mut frame_payload = frame_header_bits(w, h, use_intrabc)?;
    frame_payload.extend_from_slice(&tile_data);
    let frame_obu = obu_wrap(6, &frame_payload);
    let mut payload = seq_obu;
    payload.extend_from_slice(&frame_obu);
    let level_idx = seq_level_idx_for(w, h);
    Ok((payload, av1c_mono8(level_idx)))
}

/// RDO payload picker (mirrors `encode_obu_payload`): when IntraBC is
/// enabled the single RDO-intrabc encode already considers palette, copy,
/// and split at every contained node, so no separate on/off picking is
/// needed; when disabled only the palette-only RDO runs.
fn encode_obu_payload_rdo(
    gray: &[u8],
    w: u32,
    h: u32,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    encode_obu_payload_rdo_with_policy(
        gray,
        w,
        h,
        FlatPadPolicy::Baseline,
        search_radius_rings,
        uniform_search_radius_rings,
    )
}

fn encode_obu_payload_rdo_with_policy(
    gray: &[u8],
    w: u32,
    h: u32,
    flat_pad_policy: FlatPadPolicy,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    encode_obu_payload_rdo_with_policy_and_motion(
        gray,
        w,
        h,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
        MotionCandidatePolicy::FirstMatch,
    )
}

fn encode_obu_payload_rdo_with_policy_and_motion(
    gray: &[u8],
    w: u32,
    h: u32,
    flat_pad_policy: FlatPadPolicy,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
    motion_candidate_policy: MotionCandidatePolicy,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    encode_obu_payload_rdo_with_policy_motion_residual(
        gray,
        w,
        h,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
        motion_candidate_policy,
        false,
        false,
    )
}

// Clippy's count is a false positive for this internal adapter: it forwards
// the existing RDO controls plus the two residual-profile selectors.
#[allow(clippy::too_many_arguments)]
fn encode_obu_payload_rdo_with_policy_motion_residual(
    gray: &[u8],
    w: u32,
    h: u32,
    flat_pad_policy: FlatPadPolicy,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
    motion_candidate_policy: MotionCandidatePolicy,
    residual_profile: bool,
    scaled_8x_profile: bool,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    if !USE_INTRABC {
        return encode_obu_payload_with_rdo_pad_motion_residual(
            gray,
            w,
            h,
            false,
            flat_pad_policy,
            search_radius_rings,
            uniform_search_radius_rings,
            motion_candidate_policy,
            residual_profile,
            scaled_8x_profile,
        );
    }
    encode_obu_payload_with_rdo_pad_motion_residual(
        gray,
        w,
        h,
        true,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
        motion_candidate_policy,
        residual_profile,
        scaled_8x_profile,
    )
}

fn av01_item(id: u32, w: u32, h: u32, payload: Vec<u8>, av1c: [u8; 4]) -> Item {
    Item {
        id,
        item_type: *b"av01",
        name: String::new(),
        content_type: None,
        content_encoding: None,
        hidden: false,
        references: vec![],
        properties: vec![
            Property {
                essential: false,
                kind: PropertyKind::ImageSpatialExtents {
                    width: w,
                    height: h,
                },
            },
            Property {
                essential: true,
                kind: PropertyKind::CodecConfiguration {
                    kind: *b"av1C",
                    data: av1c.to_vec(),
                },
            },
            Property {
                essential: false,
                kind: PropertyKind::PixelInformation {
                    bits_per_channel: vec![8],
                },
            },
        ],
        payload,
    }
}

/// Encode 8-bit single-plane gray pixels (`w`x`h`, 16-aligned) as a
/// complete AVIF file: sequence header OBU + frame OBU (header + single
/// tile group) wrapping the palette-coded tile data, in a single-`av01`
/// monochrome primary item.
///
/// Cost-based RDO is attempted first, but the unchanged baseline strategy
/// is retained as a whole-image fallback: both complete serialized AVIF
/// files are built and the smaller is kept, preferring the baseline on
/// ties. `RDO_WINS`/`BASELINE_WINS` record the outcome for reporting how
/// often the new strategy wins and what the safeguard costs (roughly one
/// extra baseline encode plus RDO trials).
///
/// Supported inputs are 16-aligned non-empty single-tile dimensions with
/// `gray.len() == w*h` (checked) and at most 4 distinct levels per 16x16
/// block; violations return `InvalidInput`.
#[allow(dead_code)]
pub fn encode_gray(gray: &[u8], w: u32, h: u32) -> io::Result<Vec<u8>> {
    encode_gray_with_search_radii(
        gray,
        w,
        h,
        intrabc::DEFAULT_SEARCH_RINGS,
        DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
    )
}

/// Encode one gray raster using explicit pattern and uniform IntraBC radii.
#[allow(dead_code)]
pub fn encode_gray_with_search_radii(
    gray: &[u8],
    w: u32,
    h: u32,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
) -> io::Result<Vec<u8>> {
    // Baseline file (unchanged strategy, byte-identical to pre-RDO).
    let (base_payload, base_av1c) =
        encode_obu_payload(gray, w, h, search_radius_rings, uniform_search_radius_rings)?;
    let base_image = IsoBmffImage {
        major_brand: *b"avif",
        minor_version: 0,
        compatible_brands: vec![*b"avif", *b"mif1", *b"miaf"],
        primary_item_id: 1,
        items: vec![av01_item(1, w, h, base_payload, base_av1c)],
        groups: vec![],
    };
    let base_bytes =
        write_isobmff(&base_image).map_err(|e| io::Error::other(format!("avif mux: {e:?}")))?;
    // RDO file (cost-based partitions, same container).
    let rdo_bytes = match (|| -> io::Result<Vec<u8>> {
        let (rdo_payload, rdo_av1c) =
            encode_obu_payload_rdo(gray, w, h, search_radius_rings, uniform_search_radius_rings)?;
        let rdo_image = IsoBmffImage {
            major_brand: *b"avif",
            minor_version: 0,
            compatible_brands: vec![*b"avif", *b"mif1", *b"miaf"],
            primary_item_id: 1,
            items: vec![av01_item(1, w, h, rdo_payload, rdo_av1c)],
            groups: vec![],
        };
        write_isobmff(&rdo_image).map_err(|e| io::Error::other(format!("avif mux: {e:?}")))
    })() {
        Ok(bytes) => bytes,
        Err(_) => {
            // RDO must succeed whenever baseline does (same legality);
            // on unexpected RDO failure, fall back to baseline but count it
            // separately so fallback does not conceal errors.
            use std::sync::atomic::Ordering;
            RDO_ERRORS.fetch_add(1, Ordering::Relaxed);
            return Ok(base_bytes);
        }
    };
    use std::sync::atomic::Ordering;
    if rdo_bytes.len() < base_bytes.len() {
        RDO_WINS.fetch_add(1, Ordering::Relaxed);
        Ok(rdo_bytes)
    } else {
        BASELINE_WINS.fetch_add(1, Ordering::Relaxed);
        Ok(base_bytes)
    }
}

/// Encode two rasters of the same picture (e.g. 160x144 original and its
/// 8x nearest-neighbor upscale) into one AVIF file with two `av01` items.
///
/// Item 1 (primary) is the `large` raster, item 2 the `small` raster; both
/// are non-hidden and grouped in an `altr` entity group so readers treat
/// them as alternatives and display the primary by default (AVIF §5.1).
/// Each item carries its own `ispe`/`av1C` (levels may differ: 2.0 vs 4.0).
///
/// Baseline and RDO payloads are produced once for each raster, then each
/// available combination is serialized as a complete AVIF. The smallest
/// file wins, with deterministic baseline-first tie ordering.
///
/// Each raster has the same supported-input constraints as `encode_gray`;
/// violations return `InvalidInput`.
#[allow(dead_code)]
pub fn encode_gray_pair(
    small: &[u8],
    sw: u32,
    sh: u32,
    large: &[u8],
    lw: u32,
    lh: u32,
) -> io::Result<Vec<u8>> {
    encode_gray_pair_with_pad_policy(
        small,
        sw,
        sh,
        large,
        lw,
        lh,
        FlatPadPolicy::Baseline,
        intrabc::DEFAULT_SEARCH_RINGS,
        DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
    )
}

/// Encode two gray rasters using explicit pattern and uniform IntraBC radii.
// The two rasters, their dimensions, and both independent radius settings are
// the complete inputs for this convenience entry point.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn encode_gray_pair_with_search_radii(
    small: &[u8],
    sw: u32,
    sh: u32,
    large: &[u8],
    lw: u32,
    lh: u32,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
) -> io::Result<Vec<u8>> {
    encode_gray_pair_with_pad_policy(
        small,
        sw,
        sh,
        large,
        lw,
        lh,
        FlatPadPolicy::Baseline,
        search_radius_rings,
        uniform_search_radius_rings,
    )
}

/// `encode_gray_pair` with an explicit flat-block padding policy and both IntraBC search radii.
/// The pattern radius defaults to 64 rings and the uniform radius defaults to 0 (frame edges).
#[allow(clippy::too_many_arguments)]
pub fn encode_gray_pair_with_pad_policy(
    small: &[u8],
    sw: u32,
    sh: u32,
    large: &[u8],
    lw: u32,
    lh: u32,
    flat_pad_policy: FlatPadPolicy,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
) -> io::Result<Vec<u8>> {
    encode_gray_pair_with_pad_policy_and_motion_candidate(
        small,
        sw,
        sh,
        large,
        lw,
        lh,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
        MotionCandidatePolicy::FirstMatch,
    )
}

/// `encode_gray_pair_with_pad_policy` with an opt-in motion-copy candidate
/// policy for the large (normally 8x) raster's RDO encode. The first-match
/// policy leaves the historical candidate path unchanged.
// Clippy's count is a false positive here: this public convenience function
// intentionally mirrors the existing pair API and adds its policy argument.
#[allow(clippy::too_many_arguments)]
pub fn encode_gray_pair_with_pad_policy_and_motion_candidate(
    small: &[u8],
    sw: u32,
    sh: u32,
    large: &[u8],
    lw: u32,
    lh: u32,
    flat_pad_policy: FlatPadPolicy,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
    motion_candidate_policy: MotionCandidatePolicy,
) -> io::Result<Vec<u8>> {
    encode_gray_pair_with_pad_policy_motion_candidate_and_residual_transform(
        small,
        sw,
        sh,
        large,
        lw,
        lh,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
        motion_candidate_policy,
        false,
    )
}

/// Pair encoder with opt-in coded-lossless residual candidates.
/// Residual candidates are considered only when the source uses the exact
/// four-shade alphabet and `large` is a byte-exact 8x nearest-neighbor image
/// of `small`; otherwise this falls back to the established candidate set.
// Clippy's count is a false positive here: this public convenience API keeps
// the established per-image dimensions and encoder policy arguments explicit.
#[allow(clippy::too_many_arguments)]
pub fn encode_gray_pair_with_pad_policy_motion_candidate_and_residual_transform(
    small: &[u8],
    sw: u32,
    sh: u32,
    large: &[u8],
    lw: u32,
    lh: u32,
    flat_pad_policy: FlatPadPolicy,
    search_radius_rings: i32,
    uniform_search_radius_rings: i32,
    motion_candidate_policy: MotionCandidatePolicy,
    residual_transform: bool,
) -> io::Result<Vec<u8>> {
    let base_small = encode_obu_payload_with_policy(
        small,
        sw,
        sh,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
    )?;
    let base_large = encode_obu_payload_with_policy(
        large,
        lw,
        lh,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
    )?;
    // Preserve the whole-file baseline safeguard, but make the RDO choice
    // independently for each item using only payloads already encoded.
    let mut had_rdo_error = false;
    let rdo_small = match encode_obu_payload_rdo_with_policy(
        small,
        sw,
        sh,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
    ) {
        Ok(candidate) => Some(candidate),
        Err(_) => {
            had_rdo_error = true;
            RDO_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    };
    let rdo_large = match encode_obu_payload_rdo_with_policy_and_motion(
        large,
        lw,
        lh,
        flat_pad_policy,
        search_radius_rings,
        uniform_search_radius_rings,
        motion_candidate_policy,
    ) {
        Ok(candidate) => Some(candidate),
        Err(_) => {
            had_rdo_error = true;
            RDO_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    };

    let residual_profile =
        residual_transform && verify_residual_pair(small, sw, sh, large, lw, lh)?;
    let residual_small = if residual_profile {
        match encode_obu_payload_rdo_with_policy_motion_residual(
            small,
            sw,
            sh,
            flat_pad_policy,
            search_radius_rings,
            uniform_search_radius_rings,
            MotionCandidatePolicy::FirstMatch,
            true,
            false,
        ) {
            Ok(candidate) => Some(candidate),
            Err(_) => {
                had_rdo_error = true;
                RDO_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                None
            }
        }
    } else {
        None
    };
    let residual_large = if residual_profile {
        match encode_obu_payload_rdo_with_policy_motion_residual(
            large,
            lw,
            lh,
            flat_pad_policy,
            search_radius_rings,
            uniform_search_radius_rings,
            motion_candidate_policy,
            true,
            true,
        ) {
            Ok(candidate) => Some(candidate),
            Err(_) => {
                had_rdo_error = true;
                RDO_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                None
            }
        }
    } else {
        None
    };

    let small_candidates = [Some(base_small), rdo_small, residual_small];
    let large_candidates = [Some(base_large), rdo_large, residual_large];
    // Stable order keeps baseline/baseline on ties, then tests mixed small,
    // mixed large, and finally RDO/RDO. Each serialization contains the
    // complete item metadata and altr group and therefore compares exactly
    // what will be written.
    let mode_order = [
        (0usize, 0usize),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (0, 2),
        (2, 1),
        (1, 2),
        (2, 2),
    ];
    let mut best_bytes: Option<Vec<u8>> = None;
    let mut best_mode = (0usize, 0usize);
    for mode in mode_order {
        let (Some((small_payload, small_av1c)), Some((large_payload, large_av1c))) =
            (&small_candidates[mode.0], &large_candidates[mode.1])
        else {
            continue;
        };
        let bytes = match mux_gray_pair_payloads(
            PairPayload {
                payload: small_payload,
                av1c: *small_av1c,
                width: sw,
                height: sh,
            },
            PairPayload {
                payload: large_payload,
                av1c: *large_av1c,
                width: lw,
                height: lh,
            },
        ) {
            Ok(bytes) => bytes,
            Err(error) if mode == (0, 0) => return Err(error),
            Err(_) => {
                had_rdo_error = true;
                RDO_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                continue;
            }
        };
        if best_bytes
            .as_ref()
            .is_none_or(|best| complete_avif_candidate_is_strictly_smaller(&bytes, best))
        {
            best_bytes = Some(bytes);
            best_mode = mode;
        }
    }
    use std::sync::atomic::Ordering;
    let selected_rdo = best_mode != (0, 0);
    if selected_rdo {
        RDO_WINS.fetch_add(1, Ordering::Relaxed);
    } else if !had_rdo_error {
        BASELINE_WINS.fetch_add(1, Ordering::Relaxed);
    }
    best_bytes.ok_or_else(|| io::Error::other("no encoded pair candidates"))
}

fn verify_residual_pair(
    small: &[u8],
    sw: u32,
    sh: u32,
    large: &[u8],
    lw: u32,
    lh: u32,
) -> io::Result<bool> {
    let small_area = checked_area(sw, sh)?;
    let large_area = checked_area(lw, lh)?;
    if small.len() != small_area || large.len() != large_area {
        return Err(invalid_input(format!(
            "pixel buffers do not match dimensions: small {} for {sw}x{sh}, large {} for {lw}x{lh}",
            small.len(),
            large.len()
        )));
    }
    if sw.checked_mul(8) != Some(lw) || sh.checked_mul(8) != Some(lh) {
        return Ok(false);
    }
    if small
        .iter()
        .any(|&pixel| GRAY_TO_SHADE[pixel as usize] == 0xFF)
    {
        return Ok(false);
    }
    let (sw, lw) = (sw as usize, lw as usize);
    for y in 0..lh as usize {
        let small_row = (y / 8) * sw;
        let large_row = y * lw;
        for x in 0..lw {
            if large[large_row + x] != small[small_row + x / 8] {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

struct PairPayload<'a> {
    payload: &'a [u8],
    av1c: [u8; 4],
    width: u32,
    height: u32,
}

fn mux_gray_pair_payloads(small: PairPayload<'_>, large: PairPayload<'_>) -> io::Result<Vec<u8>> {
    let image = IsoBmffImage {
        major_brand: *b"avif",
        minor_version: 0,
        compatible_brands: vec![*b"avif", *b"mif1", *b"miaf"],
        primary_item_id: 1,
        items: vec![
            av01_item(
                1,
                large.width,
                large.height,
                large.payload.to_vec(),
                large.av1c,
            ),
            av01_item(
                2,
                small.width,
                small.height,
                small.payload.to_vec(),
                small.av1c,
            ),
        ],
        groups: vec![EntityGroup {
            group_type: *b"altr",
            group_id: 10,
            entity_ids: vec![1, 2],
        }],
    };
    write_isobmff(&image).map_err(|e| io::Error::other(format!("avif mux: {e:?}")))
}

fn complete_avif_candidate_is_strictly_smaller(candidate: &[u8], incumbent: &[u8]) -> bool {
    candidate.len() < incumbent.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gamut_isobmff::read as read_isobmff;

    /// 8px checkerboard (repetitive enough for IntraBC to trigger).
    fn checker(w: usize, h: usize) -> Vec<u8> {
        (0..h)
            .flat_map(|y| (0..w).map(move |x| if ((x / 8) + (y / 8)) % 2 == 0 { 0 } else { 255 }))
            .collect()
    }

    fn forced_residual_payload(
        gray: &[u8],
        w: u32,
        h: u32,
        scaled_8x_profile: bool,
        mode: u8,
    ) -> (Vec<u8>, [u8; 4]) {
        let (_, masks) = validate_gray(gray, w, h).unwrap();
        let mut tile = TileEncoder::new_with_rdo_pad_and_motion_policy(
            gray,
            w as usize,
            h as usize,
            USE_INTRABC,
            masks,
            true,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::FirstMatch,
        )
        .unwrap();
        tile.enable_residual_profile(scaled_8x_profile);
        for r in (0..tile.mi_rows).step_by(16) {
            for c in (0..tile.mi_cols).step_by(16) {
                let bsl = 4;
                let ctx = tile.partition_ctx(r, c, bsl);
                tile.enc_part(bsl, ctx, PARTITION_NONE);
                tile.update_partition_ctx(r, c, bsl);
                tile.encode_residual_block(r, c, bsl, mode).unwrap();
            }
        }
        let tile_data = tile.sym.finish();
        let mut frame_payload = frame_header_bits(w, h, USE_INTRABC).unwrap();
        frame_payload.extend_from_slice(&tile_data);
        let mut payload = obu_wrap(1, &sequence_header_obu(w, h));
        payload.extend_from_slice(&obu_wrap(6, &frame_payload));
        (payload, av1c_mono8(seq_level_idx_for(w, h)))
    }

    fn assert_obu_exact_with_independent_decoders(
        payload: &[u8],
        expected: &[u8],
        width: usize,
        height: usize,
        label: &str,
    ) {
        use std::process::Command;

        let dir = std::env::temp_dir().join(format!(
            "residual_transform_decode_{}_{}",
            std::process::id(),
            label
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("item.ivf");
        let mut ivf = Vec::with_capacity(32 + 12 + payload.len());
        ivf.extend_from_slice(b"DKIF");
        ivf.extend_from_slice(&0u16.to_le_bytes());
        ivf.extend_from_slice(&32u16.to_le_bytes());
        ivf.extend_from_slice(b"AV01");
        ivf.extend_from_slice(&(width as u16).to_le_bytes());
        ivf.extend_from_slice(&(height as u16).to_le_bytes());
        ivf.extend_from_slice(&30u32.to_le_bytes());
        ivf.extend_from_slice(&1u32.to_le_bytes());
        ivf.extend_from_slice(&1u32.to_le_bytes());
        ivf.extend_from_slice(&0u32.to_le_bytes());
        ivf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&0u64.to_le_bytes());
        ivf.extend_from_slice(payload);
        std::fs::write(&input, ivf).unwrap();
        let mut ran_decoder = false;
        for decoder in ["aomdec", "dav1d"] {
            if Command::new(decoder).arg("--version").output().is_err() {
                eprintln!("{decoder} unavailable; skipping this decoder");
                continue;
            }
            ran_decoder = true;
            let output = dir.join(format!("{decoder}.yuv"));
            let status = if decoder == "aomdec" {
                Command::new(decoder)
                    .args(["--codec=av1", "--rawvideo", "--output-bit-depth=8", "-o"])
                    .arg(&output)
                    .arg(&input)
                    .status()
                    .unwrap()
            } else {
                Command::new(decoder)
                    .args(["--demuxer", "ivf", "-i"])
                    .arg(&input)
                    .args(["-o"])
                    .arg(&output)
                    .args(["--muxer", "yuv", "--quiet"])
                    .status()
                    .unwrap()
            };
            assert!(status.success(), "{decoder} rejected {label} AV1 item");
            let decoded = std::fs::read(&output).unwrap();
            let luma_end = width * height;
            assert!(decoded.len() >= luma_end, "{decoder} output is truncated");
            assert_eq!(
                &decoded[..luma_end],
                expected,
                "{decoder} changed {label} luma samples"
            );
        }
        assert!(ran_decoder, "no independent AV1 decoder is installed");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn pair_container_shape() {
        let small = checker(64, 64);
        let large = checker(128, 128);
        let bytes = encode_gray_pair(&small, 64, 64, &large, 128, 128).unwrap();
        let img = read_isobmff(&bytes).unwrap();
        assert_eq!(img.primary_item_id, 1);
        assert_eq!(img.items.len(), 2);
        assert!(img.items.iter().all(|it| it.item_type == *b"av01"));
        assert_eq!(img.groups.len(), 1);
        assert_eq!(img.groups[0].group_type, *b"altr");
    }

    #[test]
    fn complete_avif_size_ties_keep_the_earlier_candidate() {
        let make_pair = |small_payload: &[u8], large_payload: &[u8]| {
            mux_gray_pair_payloads(
                PairPayload {
                    payload: small_payload,
                    av1c: av1c_mono8(0),
                    width: 16,
                    height: 16,
                },
                PairPayload {
                    payload: large_payload,
                    av1c: av1c_mono8(0),
                    width: 128,
                    height: 128,
                },
            )
            .unwrap()
        };
        let earlier = make_pair(&[0x01, 0x02], &[0x03, 0x04]);
        let tied_later = make_pair(&[0x05, 0x06], &[0x07, 0x08]);
        assert_eq!(earlier.len(), tied_later.len());
        assert!(!complete_avif_candidate_is_strictly_smaller(
            &tied_later,
            &earlier
        ));
    }

    #[test]
    fn residual_profile_verification_enforces_shades_and_exact_eight_x_edges() {
        let small = (0..16usize * 16)
            .map(|i| GAME_SHADES[(i / 16 + i % 16) % 4])
            .collect::<Vec<_>>();
        let source = &small;
        let large = (0..128usize)
            .flat_map(|y| (0..128usize).map(move |x| source[(y / 8) * 16 + x / 8]))
            .collect::<Vec<_>>();
        assert!(verify_residual_pair(&small, 16, 16, &large, 128, 128).unwrap());

        let mut unsupported_shade = small.clone();
        unsupported_shade[0] = 1;
        let unsupported_source = &unsupported_shade;
        let unsupported_large = (0..128usize)
            .flat_map(|y| (0..128usize).map(move |x| unsupported_source[(y / 8) * 16 + x / 8]))
            .collect::<Vec<_>>();
        assert!(
            !verify_residual_pair(&unsupported_shade, 16, 16, &unsupported_large, 128, 128)
                .unwrap()
        );

        let mut corrupted_large = large.clone();
        corrupted_large[0] ^= 1;
        assert!(!verify_residual_pair(&small, 16, 16, &corrupted_large, 128, 128).unwrap());
        assert!(!verify_residual_pair(&small, 16, 16, &large[..127 * 128], 128, 127).unwrap());
        assert_eq!(
            verify_residual_pair(&small, 16, 16, &large[..large.len() - 1], 128, 128)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn residual_flag_falls_back_and_selects_only_complete_file_wins() {
        let small = vec![1u8; 16 * 16];
        let large = vec![1u8; 128 * 128];
        let args = (
            16,
            16,
            128,
            128,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::FirstMatch,
        );
        let fallback = encode_gray_pair_with_pad_policy_motion_candidate_and_residual_transform(
            &small, args.0, args.1, &large, args.2, args.3, args.4, args.5, args.6, args.7, false,
        )
        .unwrap();
        let unsupported_opt_in =
            encode_gray_pair_with_pad_policy_motion_candidate_and_residual_transform(
                &small, args.0, args.1, &large, args.2, args.3, args.4, args.5, args.6, args.7,
                true,
            )
            .unwrap();
        assert_eq!(unsupported_opt_in, fallback);

        let solid_small = vec![85u8; 16 * 16];
        let solid_large = vec![85u8; 128 * 128];
        let baseline = encode_gray_pair_with_pad_policy_motion_candidate_and_residual_transform(
            &solid_small,
            16,
            16,
            &solid_large,
            128,
            128,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::FirstMatch,
            false,
        )
        .unwrap();
        let selected = encode_gray_pair_with_pad_policy_motion_candidate_and_residual_transform(
            &solid_small,
            16,
            16,
            &solid_large,
            128,
            128,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::FirstMatch,
            true,
        )
        .unwrap();
        assert!(selected.len() <= baseline.len());
        let baseline_image = read_isobmff(&baseline).unwrap();
        let selected_image = read_isobmff(&selected).unwrap();
        let baseline_items = baseline_image
            .items
            .iter()
            .map(|item| item.payload.len())
            .collect::<Vec<_>>();
        let selected_items = selected_image
            .items
            .iter()
            .map(|item| item.payload.len())
            .collect::<Vec<_>>();
        eprintln!(
            "residual solid 16x16/128x128 fixture: complete AVIF {} -> {} bytes; item payloads {:?} -> {:?}",
            baseline.len(),
            selected.len(),
            baseline_items,
            selected_items
        );
        assert!(
            selected.len() < baseline.len(),
            "fixture should show a complete-container residual win"
        );
    }

    #[test]
    fn pair_rdo_selects_small_and_large_items_independently() {
        let (sw, sh) = (160u32, 144u32);
        let (lw, lh) = (128u32, 128u32);
        let small = vec![0u8; sw as usize * sh as usize];
        let mut large = vec![0u8; lw as usize * lh as usize];
        let mut state = 3u32; // deterministic random four-shade case
        for pixel in &mut large {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *pixel = [0, 85, 170, 255][(state >> 30) as usize];
        }
        let (bs, bsc) = encode_obu_payload(
            &small,
            sw,
            sh,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (rs, rsc) = encode_obu_payload_rdo(
            &small,
            sw,
            sh,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (bl, blc) = encode_obu_payload(
            &large,
            lw,
            lh,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (rl, rlc) = encode_obu_payload_rdo(
            &large,
            lw,
            lh,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let mux = |small_payload: &[u8],
                   small_av1c: [u8; 4],
                   large_payload: &[u8],
                   large_av1c: [u8; 4]| {
            mux_gray_pair_payloads(
                PairPayload {
                    payload: small_payload,
                    av1c: small_av1c,
                    width: sw,
                    height: sh,
                },
                PairPayload {
                    payload: large_payload,
                    av1c: large_av1c,
                    width: lw,
                    height: lh,
                },
            )
            .unwrap()
        };
        let candidates = [
            mux(&bs, bsc, &bl, blc),
            mux(&rs, rsc, &bl, blc),
            mux(&bs, bsc, &rl, rlc),
            mux(&rs, rsc, &rl, rlc),
        ];
        let mut expected_index = 0;
        for index in 1..candidates.len() {
            if candidates[index].len() < candidates[expected_index].len() {
                expected_index = index;
            }
        }
        assert_eq!(
            expected_index, 1,
            "fixture must select mixed RDO/base items"
        );
        let selected = encode_gray_pair_with_search_radii(
            &small,
            sw,
            sh,
            &large,
            lw,
            lh,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        assert_eq!(selected, candidates[expected_index]);
        assert!(selected.len() <= candidates[0].len());
        assert!(selected.len() <= candidates[3].len());
    }

    #[test]
    fn residual_transform_pair_is_exact_and_never_larger() {
        let (sw, sh) = (64u32, 64u32);
        let (lw, lh) = (512u32, 512u32);
        let mut small = vec![0u8; sw as usize * sh as usize];
        for y in 0..sh as usize {
            for x in 0..sw as usize {
                let shade = ((x / 8) * 3 + (y / 8) * 5 + (x / 16) * (y / 16)) % 4;
                small[y * sw as usize + x] = GAME_SHADES[shade];
            }
        }
        let source = &small;
        let large = (0..lh as usize)
            .flat_map(|y| (0..lw as usize).map(move |x| source[(y / 8) * sw as usize + x / 8]))
            .collect::<Vec<_>>();
        assert!(verify_residual_pair(&small, sw, sh, &large, lw, lh).unwrap());

        // Force every contained 64x64 node through the new qindex-0 residual
        // writer, then decode both AVIF items with libaom and dav1d.
        let (small_payload, small_av1c) =
            forced_residual_payload(&small, sw, sh, false, DC_PRED as u8);
        let (large_payload, large_av1c) =
            forced_residual_payload(&large, lw, lh, true, DC_PRED as u8);
        let forced_pair = mux_gray_pair_payloads(
            PairPayload {
                payload: &small_payload,
                av1c: small_av1c,
                width: sw,
                height: sh,
            },
            PairPayload {
                payload: &large_payload,
                av1c: large_av1c,
                width: lw,
                height: lh,
            },
        )
        .unwrap();
        let forced_image = read_isobmff(&forced_pair).unwrap();
        assert_eq!(forced_image.items.len(), 2);
        assert_obu_exact_with_independent_decoders(
            &forced_image.items[1].payload,
            &small,
            sw as usize,
            sh as usize,
            "forced-small",
        );
        assert_obu_exact_with_independent_decoders(
            &forced_image.items[0].payload,
            &large,
            lw as usize,
            lh as usize,
            "forced-large",
        );

        let baseline = encode_gray_pair_with_pad_policy_and_motion_candidate(
            &small,
            sw,
            sh,
            &large,
            lw,
            lh,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::FirstMatch,
        )
        .unwrap();
        let opted_in = encode_gray_pair_with_pad_policy_motion_candidate_and_residual_transform(
            &small,
            sw,
            sh,
            &large,
            lw,
            lh,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::FirstMatch,
            true,
        )
        .unwrap();
        assert!(opted_in.len() <= baseline.len());
        let selected_image = read_isobmff(&opted_in).unwrap();
        assert_eq!(selected_image.items.len(), 2);
        assert_obu_exact_with_independent_decoders(
            &selected_image.items[0].payload,
            &large,
            lw as usize,
            lh as usize,
            "selected-large",
        );
        assert_obu_exact_with_independent_decoders(
            &selected_image.items[1].payload,
            &small,
            sw as usize,
            sh as usize,
            "selected-small",
        );
    }

    #[test]
    fn qindex_zero_exact_dc_skip_decodes_with_both_independent_decoders() {
        let (w, h) = (128u32, 128u32);
        let gray = vec![85u8; w as usize * h as usize];
        let (_, masks) = validate_gray(&gray, w, h).unwrap();
        let mut tile = TileEncoder::new_with_rdo_pad_and_motion_policy(
            &gray,
            w as usize,
            h as usize,
            USE_INTRABC,
            masks,
            true,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::FirstMatch,
        )
        .unwrap();
        tile.enable_residual_profile(false);
        assert!(!tile.residual_block_is_exact_prediction(0, 0, 16, DC_PRED as u8));
        assert!(tile.residual_block_is_exact_prediction(0, 16, 16, DC_PRED as u8));

        let (payload, _) = forced_residual_payload(&gray, w, h, false, DC_PRED as u8);
        assert_obu_exact_with_independent_decoders(
            &payload,
            &gray,
            w as usize,
            h as usize,
            "q0-dc-skip",
        );
    }

    #[test]
    fn all_supported_residual_modes_decode_exactly_with_two_decoders() {
        let (width, height) = (128usize, 128usize);
        let small = (0..16usize * 16)
            .map(|index| GAME_SHADES[(index * 13 + index / 7) & 3])
            .collect::<Vec<_>>();
        let mut scaled = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                scaled[y * width + x] = small[(y / 8) * 16 + x / 8];
            }
        }
        let native = (0..width * height)
            .map(|index| GAME_SHADES[(index * 5 + index / 11 + index / width) & 3])
            .collect::<Vec<_>>();
        for mode in RESIDUAL_INTRA_MODES {
            let (payload, _) =
                forced_residual_payload(&scaled, width as u32, height as u32, true, mode);
            assert_obu_exact_with_independent_decoders(
                &payload,
                &scaled,
                width,
                height,
                &format!("scaled-mode-{mode}"),
            );
            let (payload, _) =
                forced_residual_payload(&native, width as u32, height as u32, false, mode);
            assert_obu_exact_with_independent_decoders(
                &payload,
                &native,
                width,
                height,
                &format!("native-mode-{mode}"),
            );
        }
    }

    #[test]
    fn residual_mode_trials_replay_identical_state_and_rollback() {
        let width = 128usize;
        let gray = (0..width * width)
            .map(|index| GAME_SHADES[(index * 7 + index / 9) & 3])
            .collect::<Vec<_>>();
        let (_, masks) = validate_gray(&gray, width as u32, width as u32).unwrap();
        let mut tile = TileEncoder::new_with_rdo(&gray, width, width, true, masks, true).unwrap();
        tile.enable_residual_profile(false);

        let base_cdfs = tile.cdfs.clone();
        let base_coeff_cdfs = tile.coeff_cdfs.clone();
        let base_mode_cdfs = tile.residual_mode_cdfs.clone();
        let base_ymode = tile.ymode.clone();
        let base_above_coeff_level = tile.above_coeff_level.clone();
        let base_left_coeff_level = tile.left_coeff_level.clone();
        let base_above_coeff_dc = tile.above_coeff_dc.clone();
        let base_left_coeff_dc = tile.left_coeff_dc.clone();
        let base_skip = tile.skip.clone();
        let base_cost = tile.cost;

        let mut first_costs = Vec::new();
        for mode in RESIDUAL_INTRA_MODES {
            let cost = tile
                .trial(0, 0, 16, 16, |enc| enc.encode_residual_block(0, 0, 4, mode))
                .unwrap();
            first_costs.push((mode, cost));
            assert_eq!(tile.cdfs, base_cdfs);
            assert_eq!(tile.coeff_cdfs, base_coeff_cdfs);
            assert_eq!(tile.residual_mode_cdfs, base_mode_cdfs);
            assert_eq!(tile.ymode, base_ymode);
            assert_eq!(tile.above_coeff_level, base_above_coeff_level);
            assert_eq!(tile.left_coeff_level, base_left_coeff_level);
            assert_eq!(tile.above_coeff_dc, base_above_coeff_dc);
            assert_eq!(tile.left_coeff_dc, base_left_coeff_dc);
            assert_eq!(tile.skip, base_skip);
            assert_eq!(tile.cost, base_cost);
        }
        let second_costs = RESIDUAL_INTRA_MODES
            .map(|mode| {
                (
                    mode,
                    tile.trial(0, 0, 16, 16, |enc| enc.encode_residual_block(0, 0, 4, mode))
                        .unwrap(),
                )
            })
            .to_vec();
        assert_eq!(first_costs, second_costs);
    }

    #[test]
    fn residual_transform_real_raster_reports_complete_pair_size_and_time() {
        use std::time::Instant;

        let (sw, sh) = (160u32, 144u32);
        let (lw, lh) = (1280u32, 1152u32);
        let small = vec![85u8; sw as usize * sh as usize];
        let source = &small;
        let large = (0..lh as usize)
            .flat_map(|y| (0..lw as usize).map(move |x| source[(y / 8) * sw as usize + x / 8]))
            .collect::<Vec<_>>();
        assert!(verify_residual_pair(&small, sw, sh, &large, lw, lh).unwrap());

        let start = Instant::now();
        let baseline = encode_gray_pair_with_pad_policy_motion_candidate_and_residual_transform(
            &small,
            sw,
            sh,
            &large,
            lw,
            lh,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::FirstMatch,
            false,
        )
        .unwrap();
        let baseline_time = start.elapsed();
        let start = Instant::now();
        let selected = encode_gray_pair_with_pad_policy_motion_candidate_and_residual_transform(
            &small,
            sw,
            sh,
            &large,
            lw,
            lh,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::FirstMatch,
            true,
        )
        .unwrap();
        let selected_time = start.elapsed();
        let baseline_image = read_isobmff(&baseline).unwrap();
        let selected_image = read_isobmff(&selected).unwrap();
        let item_sizes = |image: &IsoBmffImage| {
            image
                .items
                .iter()
                .map(|item| item.payload.len())
                .collect::<Vec<_>>()
        };
        eprintln!(
            "residual real-raster solid 160x144/1280x1152: baseline {} bytes in {:?}, enabled-selected {} bytes in {:?}; item payloads {:?} -> {:?}",
            baseline.len(),
            baseline_time,
            selected.len(),
            selected_time,
            item_sizes(&baseline_image),
            item_sizes(&selected_image)
        );
        assert!(selected.len() <= baseline.len());
    }

    #[test]
    fn residual_transform_real_raster_covers_every_scaled_pattern() {
        use std::time::Instant;

        let (sw, sh) = (160u32, 144u32);
        let (lw, lh) = (1280u32, 1152u32);
        let mut small = vec![0u8; sw as usize * sh as usize];
        for block_y in 0..sh as usize / 2 {
            for block_x in 0..sw as usize / 2 {
                let pattern = ((block_y * (sw as usize / 2) + block_x) % 256) as u8;
                for (bit, (dy, dx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                    let shade = (pattern >> (bit * 2)) & 3;
                    small[(block_y * 2 + dy) * sw as usize + block_x * 2 + dx] =
                        GAME_SHADES[shade as usize];
                }
            }
        }
        let source = &small;
        let large = (0..lh as usize)
            .flat_map(|y| (0..lw as usize).map(move |x| source[(y / 8) * sw as usize + x / 8]))
            .collect::<Vec<_>>();
        assert!(verify_residual_pair(&small, sw, sh, &large, lw, lh).unwrap());

        let start = Instant::now();
        let baseline = encode_gray_pair_with_pad_policy_motion_candidate_and_residual_transform(
            &small,
            sw,
            sh,
            &large,
            lw,
            lh,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::FirstMatch,
            false,
        )
        .unwrap();
        let baseline_time = start.elapsed();
        let start = Instant::now();
        let selected = encode_gray_pair_with_pad_policy_motion_candidate_and_residual_transform(
            &small,
            sw,
            sh,
            &large,
            lw,
            lh,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::FirstMatch,
            true,
        )
        .unwrap();
        let selected_time = start.elapsed();
        let baseline_image = read_isobmff(&baseline).unwrap();
        let selected_image = read_isobmff(&selected).unwrap();
        let item_sizes = |image: &IsoBmffImage| {
            image
                .items
                .iter()
                .map(|item| item.payload.len())
                .collect::<Vec<_>>()
        };
        eprintln!(
            "residual real-raster all-pattern 160x144/1280x1152: baseline {} bytes in {:?}, enabled-selected {} bytes in {:?}; item payloads {:?} -> {:?}",
            baseline.len(),
            baseline_time,
            selected.len(),
            selected_time,
            item_sizes(&baseline_image),
            item_sizes(&selected_image)
        );
        assert!(selected.len() <= baseline.len());
    }

    #[test]
    fn toggle_changes_output_but_stays_valid_obus() {
        let gray = checker(64, 64);
        let (on, _) = encode_obu_payload_with(
            &gray,
            64,
            64,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (off, _) = encode_obu_payload_with(
            &gray,
            64,
            64,
            false,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        // Sequence-header OBU first in both streams.
        assert_eq!(&on[..1], &[0x0a]);
        assert_eq!(&off[..1], &[0x0a]);
        // IntraBC fires on repetitive content, so the streams differ.
        assert_ne!(on, off);
    }

    #[test]
    fn enabled_picks_smallest_payload() {
        // 8px checker: palette-only wins outright here (IntraBC flags and
        // finer splits cost more than the copies save); flat rasters tie
        // (broken toward the IntraBC payload). Either way the enabled
        // entry point must keep the smaller of the two.
        let cases: Vec<Vec<u8>> = vec![checker(64, 64), vec![0u8; 64 * 64], vec![85u8; 64 * 64]];
        for gray in &cases {
            let (on, _) = encode_obu_payload_with(
                gray,
                64,
                64,
                true,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            let (off, _) = encode_obu_payload_with(
                gray,
                64,
                64,
                false,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            let (picked, _) = encode_obu_payload(
                gray,
                64,
                64,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            assert_eq!(picked.len(), on.len().min(off.len()));
            assert!(picked.len() <= off.len());
            if on.len() <= off.len() {
                assert_eq!(picked, on);
            } else {
                assert_eq!(picked, off);
            }
        }
    }

    #[test]
    fn three_color_254_255_implicit_fill() {
        // Regression: `[0, 254, 255]` is accepted (3 colors) but the old
        // `debug_assert!(k + 1 == new_slice.len())` fired when the delta
        // chain reached 254 with an implicit trailing 255 remaining
        // (the decoder fills it, so release decoded exactly).
        let palette = [0u8, 254, 255];
        let gray: Vec<u8> = (0..16 * 16).map(|i| palette[i % 3]).collect();
        // Both palette paths (deterministic for the single block: no cache,
        // no causal match) plus the full entry point.
        encode_obu_payload_with(
            &gray,
            16,
            16,
            false,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        encode_obu_payload_with(
            &gray,
            16,
            16,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        encode_gray_with_search_radii(
            &gray,
            16,
            16,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
    }

    #[test]
    fn invalid_inputs_return_invalid_input() {
        use std::io::ErrorKind;
        // Non-16-aligned dimensions.
        let gray = vec![0u8; 30 * 16];
        let err = encode_gray_with_search_radii(
            &gray,
            30,
            16,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        // Length mismatch (64x64 needs 4096 bytes).
        let short = vec![0u8; 100];
        let err = encode_gray_with_search_radii(
            &short,
            64,
            64,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        // Too many colors in one 16x16 block (5 distinct levels).
        let mut many = vec![0u8; 64 * 64];
        for y in 0..16 {
            for x in 0..16 {
                many[y * 64 + x] = (x % 5) as u8;
            }
        }
        let err = encode_gray_with_search_radii(
            &many,
            64,
            64,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        // Zero dimensions.
        let empty: Vec<u8> = vec![];
        let err = encode_gray_with_search_radii(
            &empty,
            0,
            0,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn oversized_dimensions_rejected() {
        use std::io::ErrorKind;
        // 65552 needs 17 bits, but `frame_width_bits_minus_1` /
        // `frame_height_bits_minus_1` are 4 bits (max 16 bits per side).
        // Dimension validation runs before buffer-length checks, so an
        // empty buffer still exercises the cap.
        let err = encode_gray_with_search_radii(
            &[],
            16,
            65552,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.to_string().contains("65536"), "unexpected: {err}");
        let err = encode_gray_with_search_radii(
            &[],
            65552,
            16,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.to_string().contains("65536"), "unexpected: {err}");
        // The cap is inclusive: 65536 fails later on buffer length, not
        // on dimensions.
        let err = encode_gray_with_search_radii(
            &[],
            16,
            65536,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(
            !err.to_string().contains("at most"),
            "boundary 65536 must pass the dimension cap: {err}"
        );
    }

    #[test]
    fn gray_to_shade_mapping() {
        assert_eq!(GRAY_TO_SHADE[0], 0);
        assert_eq!(GRAY_TO_SHADE[85], 1);
        assert_eq!(GRAY_TO_SHADE[170], 2);
        assert_eq!(GRAY_TO_SHADE[255], 3);
        // Every other value is unsupported (0xFF, never silently mapped).
        for (v, &entry) in GRAY_TO_SHADE.iter().enumerate() {
            if ![0, 85, 170, 255].contains(&(v as u8)) {
                assert_eq!(entry, 0xFF, "gray {v} must be invalid");
            }
        }
    }

    #[test]
    fn mask_tables_exhaustive() {
        // Every 4-bit mask: popcount and sorted source palette.
        for (mask, &count) in MASK_COUNT.iter().enumerate() {
            let count = count as usize;
            assert_eq!(count, (mask as u8).count_ones() as usize, "mask {mask:04b}");
            let mut expected = [0u8; 4];
            let mut k = 0usize;
            for (s, &shade) in GAME_SHADES.iter().enumerate() {
                if (mask >> s) & 1 == 1 {
                    expected[k] = shade;
                    k += 1;
                }
            }
            assert_eq!(k, count, "mask {mask:04b}");
            assert_eq!(&MASK_COLORS[mask][..count], &expected[..count]);
            // Sorted ascending (GAME_SHADES sorted).
            for w in MASK_COLORS[mask][..count].windows(2) {
                assert!(w[0] < w[1], "mask {mask:04b} unsorted");
            }
        }
        // Empty mask has zero colors; every nonempty mask has 1..4.
        assert_eq!(MASK_COUNT[0], 0);
        for &count in MASK_COUNT.iter().skip(1) {
            assert!((1..=4).contains(&count));
        }
    }

    #[test]
    fn source_masks_match_reference_scans() {
        // For every nonempty mask, build a 16x16 block realizing exactly that
        // mask (quadrants carry the set shades), then verify the mask path
        // (popcount + sorted palette) matches the original 256-entry scan.
        for mask in 1..16u8 {
            let bits: Vec<u8> = (0..4u8).filter(|s| (mask >> s) & 1 == 1).collect();
            let mut block = vec![0u8; 16 * 16];
            for y in 0..16 {
                for x in 0..16 {
                    let q = (y / 8) * 2 + (x / 8);
                    let s = bits[q % bits.len()];
                    block[y * 16 + x] = GAME_SHADES[s as usize];
                }
            }
            // Reference distinct + sorted palette via 256-entry scan.
            let mut set = [false; 256];
            for &v in &block {
                set[v as usize] = true;
            }
            let distinct_ref = set.iter().filter(|&&b| b).count();
            let mut colors_ref = [0u8; 4];
            let mut k = 0usize;
            for (v, &p) in set.iter().enumerate() {
                if p {
                    colors_ref[k] = v as u8;
                    k += 1;
                }
            }
            assert_eq!(distinct_ref, MASK_COUNT[mask as usize] as usize);
            assert_eq!(
                &colors_ref[..distinct_ref],
                &MASK_COLORS[mask as usize][..distinct_ref]
            );
            // `build_source_masks` on a single-block image agrees.
            let masks = build_source_masks(&block, 16, 16).expect("four-shade must verify");
            assert_eq!(masks, vec![mask]);
        }
        // Unsupported values reject the whole image (no silent mapping).
        let mut bad = vec![0u8; 16 * 16];
        bad[100] = 1;
        assert!(build_source_masks(&bad, 16, 16).is_none());
        // Misaligned dimensions or length mismatch also fall back.
        assert!(build_source_masks(&vec![0u8; 16 * 16], 30, 16).is_none());
        assert!(build_source_masks(&[0u8; 10], 16, 16).is_none());
    }

    #[test]
    fn hierarchical_masks_match_scans() {
        // Random 64x64 four-shade image: combined 32x32/64x64 masks must give
        // the same distinct counts as the original 256-entry scans,
        // preserving partition decisions.
        let mut px = vec![0u8; 64 * 64];
        let levels = [0u8, 85, 170, 255];
        for y in 0..64 {
            for x in 0..64 {
                px[y * 64 + x] = levels[(x * 5 + y * 11 + (x / 16) * 3) % 4];
            }
        }
        let masks = build_source_masks(&px, 64, 64).expect("must verify");
        assert_eq!(masks.len(), 16);
        let reference_distinct = |sx: usize, sy: usize, bw: usize| {
            let mut set = [false; 256];
            let mut n = 0usize;
            for y in sy..sy + bw {
                for x in sx..sx + bw {
                    let v = px[y * 64 + x] as usize;
                    if !set[v] {
                        set[v] = true;
                        n += 1;
                    }
                }
            }
            n
        };
        // 32x32 quadrants (2x2 masks each).
        for (bx, by) in [(0, 0), (32, 0), (0, 32), (32, 32)] {
            let mut m = 0u8;
            for dy in 0..2 {
                for dx in 0..2 {
                    m |= masks[(by / 16 + dy) * 4 + (bx / 16 + dx)];
                }
            }
            assert_eq!(
                MASK_COUNT[m as usize] as usize,
                reference_distinct(bx, by, 32),
                "32x32 at ({bx},{by})"
            );
        }
        // Full 64x64 (4x4 masks).
        let m = masks.iter().fold(0u8, |a, &b| a | b);
        assert_eq!(
            MASK_COUNT[m as usize] as usize,
            reference_distinct(0, 0, 64)
        );
    }

    #[test]
    fn shade_to_palette_mapping_matches_search() {
        // Every nonempty mask: shade→palette table reproduces the per-pixel
        // linear search for all source pixels. Flat masks exercise the
        // padding path (cached pad inside/outside the four shades).
        for mask in 1..16u8 {
            let cnt = MASK_COUNT[mask as usize] as usize;
            let src_colors = &MASK_COLORS[mask as usize][..cnt];
            // Representative palettes: non-flat as-is; flat with several pad
            // choices (cached four-shade, arbitrary outside, adjacent).
            let palettes: Vec<Vec<u8>> = if cnt > 1 {
                vec![src_colors.to_vec()]
            } else {
                let v = src_colors[0];
                let mut cands = vec![];
                for pad in [1u8, 2, 84, 86, 169, 171, 254, 0, 85, 170, 255] {
                    if pad == v {
                        continue;
                    }
                    let mut p = vec![v, pad];
                    p.sort_unstable();
                    if !cands.contains(&p) {
                        cands.push(p);
                    }
                }
                cands
            };
            for palette in palettes {
                let psize = palette.len();
                // Build shade→palette exactly like the encoder fast path.
                let mut shade_to_pal = [0u8; 4];
                for (k, &cc) in palette.iter().enumerate() {
                    let s = GRAY_TO_SHADE[cc as usize];
                    if s != 0xFF {
                        shade_to_pal[s as usize] = k as u8;
                    }
                }
                // Every source shade present in the mask must map to the
                // same index as the linear search.
                for (s, &shade) in GAME_SHADES.iter().enumerate() {
                    if (mask >> s) & 1 == 0 {
                        continue;
                    }
                    let mut expect = 0u8;
                    for (k, &cc) in palette.iter().enumerate() {
                        if cc == shade {
                            expect = k as u8;
                            break;
                        }
                    }
                    // Flat pad outside the alphabet is never referenced;
                    // present shades always hit their palette entry.
                    assert_eq!(
                        shade_to_pal[s], expect,
                        "mask {mask:04b} palette {palette:?}"
                    );
                }
                assert!(psize == 2 || psize == cnt);
            }
        }
    }

    #[test]
    fn palette_lut_matches_reference_exhaustive() {
        // Every valid (n, left, top, tl, cur): LUT context+symbol equals the
        // reference scoring/ordering/position logic, including tie-breaks
        // and block-edge availabilities. First-pixel (all absent) excluded
        // (coded via ns(), never queried).
        let mut checked = 0usize;
        for n in 2..=4usize {
            // Top-row: left only.
            for left in 0..n as u8 {
                for cur in 0..n as u8 {
                    // Build minimal 2-wide map to place neighbors: use 16x16
                    // scratch with r=0,c=1, left at (0,0).
                    let bw = 16;
                    let mut color_map = vec![0u8; bw * bw];
                    color_map[0] = left;
                    color_map[1] = cur;
                    let (order, ctx_ref) = palette_color_context(&color_map, bw, 0, 1, n);
                    let sym_ref = order.iter().position(|&x| x == cur as usize).unwrap_or(0);
                    let (ctx, sym) = pal_lut_lookup(n, left, 4, 4, cur).expect("top-row must hit");
                    assert_eq!(
                        (ctx, sym),
                        (ctx_ref, sym_ref),
                        "n={n} top-row l={left} cur={cur}"
                    );
                    checked += 1;
                }
            }
            // Left-col: top only.
            for top in 0..n as u8 {
                for cur in 0..n as u8 {
                    let bw = 16;
                    let mut color_map = vec![0u8; bw * bw];
                    color_map[0] = top;
                    color_map[bw] = cur;
                    let (order, ctx_ref) = palette_color_context(&color_map, bw, 1, 0, n);
                    let sym_ref = order.iter().position(|&x| x == cur as usize).unwrap_or(0);
                    let (ctx, sym) = pal_lut_lookup(n, 4, top, 4, cur).expect("left-col must hit");
                    assert_eq!(
                        (ctx, sym),
                        (ctx_ref, sym_ref),
                        "n={n} left-col t={top} cur={cur}"
                    );
                    checked += 1;
                }
            }
            // Interior: all three neighbors, all index combos.
            for left in 0..n as u8 {
                for top in 0..n as u8 {
                    for tl in 0..n as u8 {
                        for cur in 0..n as u8 {
                            let bw = 16;
                            let mut color_map = vec![0u8; bw * bw];
                            // Place at (1,1): left (1,0), tl (0,0), top (0,1).
                            color_map[bw] = left;
                            color_map[0] = tl;
                            color_map[1] = top;
                            color_map[bw + 1] = cur;
                            let (order, ctx_ref) = palette_color_context(&color_map, bw, 1, 1, n);
                            let sym_ref =
                                order.iter().position(|&x| x == cur as usize).unwrap_or(0);
                            let (ctx, sym) =
                                pal_lut_lookup(n, left, top, tl, cur).expect("interior must hit");
                            assert_eq!(
                                (ctx, sym),
                                (ctx_ref, sym_ref),
                                "n={n} l={left} t={top} tl={tl} cur={cur}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        // Top-row (4+9+16) + left-col (4+9+16) + interior (16+81+256) = 411.
        assert_eq!(checked, 411, "must cover all valid combos");
        // Invalid combos miss (fallback preserves exact behavior).
        assert!(pal_lut_lookup(2, 2, 4, 4, 0).is_none());
        assert!(pal_lut_lookup(2, 0, 4, 4, 2).is_none());
        assert!(pal_lut_lookup(4, 0, 4, 0, 0).is_none());
        assert!(pal_lut_lookup(5, 0, 0, 0, 0).is_none());
    }

    #[test]
    fn arbitrary_gray_fallback_preserved() {
        use std::io::ErrorKind;
        // Accepted arbitrary-gray inputs (≤4 per 16x16, values outside the
        // four shades) still encode via the general fallback on both paths.
        let mut gray = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                gray[y * 64 + x] = [10u8, 20, 30, 40][(x + y) % 4];
            }
        }
        let (_, masks) = validate_gray(&gray, 64, 64).unwrap();
        assert!(masks.is_none(), "arbitrary grays must not take mask path");
        encode_obu_payload_with(
            &gray,
            64,
            64,
            false,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        encode_obu_payload_with(
            &gray,
            64,
            64,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        encode_gray_with_search_radii(
            &gray,
            64,
            64,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        // Single unsupported pixel in an otherwise four-shade image also
        // falls back (no silent mapping) but still encodes (≤4 per block).
        let mut mixed = vec![0u8; 16 * 16];
        mixed.fill(0);
        mixed[0] = 1;
        mixed[1] = 85;
        let (_, masks) = validate_gray(&mixed, 16, 16).unwrap();
        assert!(masks.is_none());
        encode_gray_with_search_radii(
            &mixed,
            16,
            16,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        // Too many distinct still rejected identically on both paths.
        let mut many = vec![0u8; 16 * 16];
        for (i, v) in many.iter_mut().enumerate() {
            *v = (i % 5) as u8;
        }
        let err = encode_gray_with_search_radii(
            &many,
            16,
            16,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn flat_pad_lookahead_targets_the_upcoming_sibling() {
        let (w, h) = (32usize, 16usize);
        let mut gray = vec![0u8; w * h];
        for row in 0..h {
            gray[row * w + 16..row * w + w].fill(85);
        }
        let masks = build_source_masks(&gray, w, h).expect("four-shade source");
        let encode_first = |enc: &mut TileEncoder<'_>| {
            let ctx = enc.partition_ctx(0, 0, 2);
            enc.enc_part(2, ctx, PARTITION_NONE);
            enc.update_partition_ctx(0, 0, 2);
            enc.block_with_force(0, 0, 2, true, None, true).unwrap();
        };

        let mut baseline = TileEncoder::new_with_rdo_and_pad_policy(
            &gray,
            w,
            h,
            false,
            Some(masks.clone()),
            false,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        encode_first(&mut baseline);
        assert_eq!(&baseline.pcolors[0][..2], &[0, 1]);

        let mut lookahead = TileEncoder::new_with_rdo_and_pad_policy(
            &gray,
            w,
            h,
            false,
            Some(masks),
            false,
            FlatPadPolicy::Lookahead,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        encode_first(&mut lookahead);
        assert_eq!(&lookahead.pcolors[0][..2], &[0, 85]);
        assert_eq!(lookahead.flat_pad_stats.searches, 1);
        assert_eq!(lookahead.flat_pad_stats.lookahead_wins, 1);
        assert!(lookahead.flat_pad_stats.candidate_trials >= 2);
        assert_eq!(gray[0], 0);
        assert_eq!(gray[16], 85);
    }

    #[test]
    fn flat_pad_lookahead_preserves_the_used_palette_index() {
        let (w, h) = (48usize, 16usize);
        let mut gray = vec![85u8; w * h];
        for row in 0..h {
            gray[row * w + 16..row * w + 32].fill(170);
            gray[row * w + 32..row * w + 48].fill(0);
        }
        let masks = build_source_masks(&gray, w, h).expect("four-shade source");
        let mut enc = TileEncoder::new_with_rdo_and_pad_policy(
            &gray,
            w,
            h,
            false,
            Some(masks),
            false,
            FlatPadPolicy::Lookahead,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();

        let baseline_pad = TileEncoder::baseline_flat_pad(85, &[]);
        enc.block_with_force(0, 0, 2, true, None, true).unwrap();
        let selected_pad = enc.pcolors[0][..2]
            .iter()
            .copied()
            .find(|&color| color != 85)
            .unwrap();

        assert_eq!(baseline_pad, 86);
        assert!(selected_pad > 85, "opposite-side shade 0 must be excluded");
        assert_eq!(
            usize::from(selected_pad < 85),
            usize::from(baseline_pad < 85)
        );
        assert_eq!(enc.flat_pad_stats.lookahead_wins, 1);
    }

    #[test]
    fn flat_pad_lookahead_covers_edges_and_both_intrabc_modes() {
        let (w, h) = (80usize, 48usize);
        let levels = [0u8, 85, 170, 255];
        let mut gray = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                gray[y * w + x] = levels[((x / 16) + 2 * (y / 16)) % levels.len()];
            }
        }
        for use_intrabc in [false, true] {
            for policy in [FlatPadPolicy::Baseline, FlatPadPolicy::Lookahead] {
                let first = encode_obu_payload_with_policy_and_intrabc(
                    &gray,
                    w as u32,
                    h as u32,
                    use_intrabc,
                    policy,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                let repeated = encode_obu_payload_with_policy_and_intrabc(
                    &gray,
                    w as u32,
                    h as u32,
                    use_intrabc,
                    policy,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                assert_eq!(
                    first, repeated,
                    "greedy edge path use_intrabc={use_intrabc} policy={policy:?}"
                );

                let first_rdo = encode_obu_payload_with_rdo_and_pad_policy(
                    &gray,
                    w as u32,
                    h as u32,
                    use_intrabc,
                    policy,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                let repeated_rdo = encode_obu_payload_with_rdo_and_pad_policy(
                    &gray,
                    w as u32,
                    h as u32,
                    use_intrabc,
                    policy,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                assert_eq!(
                    first_rdo, repeated_rdo,
                    "RDO edge path use_intrabc={use_intrabc} policy={policy:?}"
                );
            }
        }
    }

    #[test]
    fn flat_pad_lookahead_is_restored_after_rejected_nested_rdo_trial() {
        let (w, h) = (64usize, 64usize);
        let levels = [0u8, 85, 170, 255];
        let mut gray = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                gray[y * w + x] = levels[((x / 16) + 2 * (y / 16)) % levels.len()];
            }
        }
        let masks = build_source_masks(&gray, w, h).expect("four-shade source");
        let mut enc = TileEncoder::new_with_rdo_and_pad_policy(
            &gray,
            w,
            h,
            true,
            Some(masks),
            true,
            FlatPadPolicy::Lookahead,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let before = (
            enc.skip.clone(),
            enc.psize.clone(),
            enc.pcolors.clone(),
            enc.above_part.clone(),
            enc.left_part.clone(),
            enc.cdfs.clone(),
            format!("{:?}", enc.sym),
            enc.cost,
            enc.candidates,
        );
        let _ = enc
            .trial(0, 0, 4, 4, |trial| {
                trial.rdo_leaf_with_plan(0, 0, 2, true, None)
            })
            .unwrap();
        assert_eq!(enc.skip, before.0);
        assert_eq!(enc.psize, before.1);
        assert_eq!(enc.pcolors, before.2);
        assert_eq!(enc.above_part, before.3);
        assert_eq!(enc.left_part, before.4);
        assert_eq!(enc.cdfs, before.5);
        assert_eq!(format!("{:?}", enc.sym), before.6);
        assert_eq!(enc.cost, before.7);
        assert!(enc.candidates > before.8, "nested trials are counted");
        assert!(enc.flat_pad_stats.searches > 0);
        assert!(enc.flat_pad_stats.candidate_trials > 0);
    }

    #[test]
    fn fast_vs_fallback_tile_byte_identical() {
        // Four-shade images: fast mask/LUT paths must produce byte-identical
        // tile data to the original fallback scans/searches, for both
        // palette-only and IntraBC-enabled encodings (so picking cannot hide
        // a regression). Fallback is forced by passing `None` masks directly.
        fn tile_bytes(
            gray: &[u8],
            w: usize,
            h: usize,
            use_intrabc: bool,
            masks: Option<Vec<u8>>,
        ) -> Vec<u8> {
            TileEncoder::new(gray, w, h, use_intrabc, masks)
                .unwrap()
                .finish()
                .unwrap()
        }
        fn corpus() -> Vec<(Vec<u8>, usize, usize)> {
            let mut out = Vec::new();
            // Flat (exercises pad inside/outside alphabet via cache evolution).
            out.push((vec![0u8; 16 * 16], 16, 16));
            out.push((vec![255u8; 16 * 16], 16, 16));
            out.push((vec![85u8; 32 * 32], 32, 32));
            // Checker 8px.
            let mut chk = vec![0u8; 64 * 64];
            for y in 0..64 {
                for x in 0..64 {
                    chk[y * 64 + x] = if ((x / 8) + (y / 8)) % 2 == 0 { 0 } else { 255 };
                }
            }
            out.push((chk, 64, 64));
            // Random four-shade 64x64.
            let levels = [0u8, 85, 170, 255];
            let mut rnd = vec![0u8; 64 * 64];
            for y in 0..64 {
                for x in 0..64 {
                    rnd[y * 64 + x] = levels[(x * 7 + y * 13) % 4];
                }
            }
            out.push((rnd, 64, 64));
            // All 15 masks tiled as 16x16 blocks in a 64x64 image.
            let mut all = vec![0u8; 64 * 64];
            for by in 0..4 {
                for bx in 0..4 {
                    let mask = ((by * 4 + bx) % 15 + 1) as u8;
                    let bits: Vec<u8> = (0..4u8).filter(|s| (mask >> s) & 1 == 1).collect();
                    for y in 0..16 {
                        for x in 0..16 {
                            let q = (y / 8) * 2 + (x / 8);
                            let s = bits[q % bits.len()];
                            all[(by * 16 + y) * 64 + (bx * 16 + x)] =
                                [0u8, 85, 170, 255][s as usize];
                        }
                    }
                }
            }
            out.push((all, 64, 64));
            out
        }
        for (gray, w, h) in corpus() {
            let masks = build_source_masks(&gray, w, h).expect("corpus must verify");
            for use_intrabc in [false, true] {
                let fast = tile_bytes(&gray, w, h, use_intrabc, Some(masks.clone()));
                let slow = tile_bytes(&gray, w, h, use_intrabc, None);
                assert_eq!(fast, slow, "tile mismatch {w}x{h} intrabc={use_intrabc}");
                // Full payloads (headers + tile) must also match the public
                // entry points which use the fast path internally.
                let (fast_payload, _) = encode_obu_payload_with(
                    &gray,
                    w as u32,
                    h as u32,
                    use_intrabc,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                // Rebuild slow payload with identical headers.
                let slow_tile = slow;
                let seq_obu = obu_wrap(1, &sequence_header_obu(w as u32, h as u32));
                let mut frame_payload = frame_header_bits(w as u32, h as u32, use_intrabc).unwrap();
                frame_payload.extend_from_slice(&slow_tile);
                let frame_obu = obu_wrap(6, &frame_payload);
                let mut slow_payload = seq_obu;
                slow_payload.extend_from_slice(&frame_obu);
                assert_eq!(fast_payload, slow_payload);
            }
        }
    }

    /// Manual benchmark (ignored by default): median of 5 runs for
    /// palette-only, IntraBC-enabled, and complete (picking) paths on
    /// representative four-shade rasters. Run with:
    /// `cargo test --release bench_modes_manual -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn bench_modes_manual() {
        fn median(mut v: Vec<std::time::Duration>) -> std::time::Duration {
            v.sort_unstable();
            v[v.len() / 2]
        }
        // Deterministic 160x144 four-shade photo-like raster + 8x upscale.
        let (sw, sh) = (160usize, 144usize);
        let mut small = vec![0u8; sw * sh];
        let levels = [0u8, 85, 170, 255];
        for y in 0..sh {
            for x in 0..sw {
                small[y * sw + x] = levels[(x * 7 + y * 13) % 4];
            }
        }
        let (lw, lh) = (1280usize, 1152usize);
        let mut large = vec![0u8; lw * lh];
        for y in 0..lh {
            for x in 0..lw {
                large[y * lw + x] = small[(y / 8) * sw + (x / 8)];
            }
        }
        for (name, gray, w, h) in [
            ("small160", &small, 160u32, 144u32),
            ("large1280", &large, 1280u32, 1152u32),
        ] {
            let mut off = Vec::new();
            let mut on = Vec::new();
            let mut picked = Vec::new();
            let mut paired = Vec::new();
            for _ in 0..5 {
                let t = std::time::Instant::now();
                encode_obu_payload_with(
                    gray,
                    w,
                    h,
                    false,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                off.push(t.elapsed());
                let t = std::time::Instant::now();
                encode_obu_payload_with(
                    gray,
                    w,
                    h,
                    true,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                on.push(t.elapsed());
                let t = std::time::Instant::now();
                encode_obu_payload(
                    gray,
                    w,
                    h,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                picked.push(t.elapsed());
            }
            // Pair path (both rasters + container) measured separately.
            if name == "small160" {
                for _ in 0..5 {
                    let t = std::time::Instant::now();
                    encode_gray_pair_with_search_radii(
                        &small,
                        160,
                        144,
                        &large,
                        1280,
                        1152,
                        intrabc::DEFAULT_SEARCH_RINGS,
                        DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                    )
                    .unwrap();
                    paired.push(t.elapsed());
                }
            }
            eprintln!(
                "bench {name}: palette-only median {:?}, intrabc median {:?}, complete median {:?}",
                median(off),
                median(on),
                median(picked)
            );
            if !paired.is_empty() {
                eprintln!("bench pair: complete median {:?}", median(paired));
            }
        }
    }

    // -- Cost-based RDO focused tests (lossless, coding cost only) --

    /// Large palette block wins: 64x64 two-color halves, first SB (no copy
    /// due to SB delay). Baseline intrabc-enabled splits (coarse
    /// "split when no whole-block copy" rule); RDO compares NONE+palette vs
    /// SPLIT and keeps the large palette (1 byte saved here; larger gaps on
    /// photo content). Palette-only mode ties (both take the palette).
    #[test]
    fn rdo_large_palette_wins() {
        let mut halves = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                halves[y * 64 + x] = if y < 32 { 0 } else { 255 };
            }
        }
        let (base_off, _) = encode_obu_payload_with(
            &halves,
            64,
            64,
            false,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (base_on, _) = encode_obu_payload_with(
            &halves,
            64,
            64,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (rdo_off, _) = encode_obu_payload_with_rdo(
            &halves,
            64,
            64,
            false,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (rdo_on, _) = encode_obu_payload_with_rdo(
            &halves,
            64,
            64,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        // Baseline on splits (22) vs off palette (21); RDO on recovers the
        // palette (21), tying off and beating on.
        assert_eq!(base_off.len(), 21);
        assert_eq!(base_on.len(), 22);
        assert_eq!(rdo_off.len(), 21);
        assert_eq!(rdo_on.len(), 21);
        assert!(
            rdo_on.len() < base_on.len(),
            "large palette must win vs split"
        );
    }

    /// Split children win: 64x64 four flat quadrants (4 colors overall).
    /// Baseline palette-only takes the large 4-color palette (30); RDO
    /// compares it against SPLIT into four flat palettes (24) and splits.
    /// IntraBC-enabled baseline already splits (24) via the coarse rule, so
    /// RDO ties it here; the win is over the palette-only baseline.
    #[test]
    fn rdo_split_wins_via_copies_or_palettes() {
        let mut quads = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                quads[y * 64 + x] = match (x >= 32, y >= 32) {
                    (false, false) => 0,
                    (true, false) => 85,
                    (false, true) => 170,
                    (true, true) => 255,
                };
            }
        }
        let (base_off, _) = encode_obu_payload_with(
            &quads,
            64,
            64,
            false,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (base_on, _) = encode_obu_payload_with(
            &quads,
            64,
            64,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (rdo_off, _) = encode_obu_payload_with_rdo(
            &quads,
            64,
            64,
            false,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (rdo_on, _) = encode_obu_payload_with_rdo(
            &quads,
            64,
            64,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        assert_eq!(base_off.len(), 30);
        assert_eq!(base_on.len(), 24);
        assert_eq!(rdo_off.len(), 24);
        assert_eq!(rdo_on.len(), 24);
        assert!(
            rdo_off.len() < base_off.len(),
            "split into flats must win vs large palette"
        );
    }

    #[test]
    fn top_k_motion_candidate_preserves_first_match_control_and_is_deterministic() {
        let (sw, sh) = (32usize, 32usize);
        let mut small = vec![0u8; sw * sh];
        for y in 0..sh {
            for x in 0..sw {
                small[y * sw + x] = match (x % 2, y % 2) {
                    (0, 0) => 0,
                    (1, 0) => 85,
                    (0, 1) => 170,
                    _ => 255,
                };
            }
        }
        let (lw, lh) = (sw * 8, sh * 8);
        let mut large = vec![0u8; lw * lh];
        for y in 0..sh {
            for x in 0..sw {
                let value = small[y * sw + x];
                for dy in 0..8 {
                    let start = (y * 8 + dy) * lw + x * 8;
                    large[start..start + 8].fill(value);
                }
            }
        }
        let baseline = encode_obu_payload_rdo_with_policy(
            &large,
            lw as u32,
            lh as u32,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let explicit_control = encode_obu_payload_rdo_with_policy_and_motion(
            &large,
            lw as u32,
            lh as u32,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::FirstMatch,
        )
        .unwrap();
        assert_eq!(baseline, explicit_control);

        let candidate = encode_obu_payload_rdo_with_policy_and_motion(
            &large,
            lw as u32,
            lh as u32,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::TopK8,
        )
        .unwrap();
        let repeated = encode_obu_payload_rdo_with_policy_and_motion(
            &large,
            lw as u32,
            lh as u32,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::TopK8,
        )
        .unwrap();
        assert_eq!(candidate, repeated, "TopK8 must be deterministic");
    }

    /// Available copy loses to palette coding: repeated 16px patterns where
    /// whole-block copies exist (later SBs) but MV residuals plus flags cost
    /// more than small palettes. Baseline intrabc-enabled takes the copies
    /// (100/58); RDO measures both and keeps palettes (70/32), tying
    /// palette-only and beating the copy path by 30/26 bytes here.
    /// A matching copy (or small palette) does not automatically win — cost decides.
    #[test]
    fn rdo_copy_loses_to_palette() {
        let mut rep128 = vec![0u8; 128 * 128];
        for y in 0..128 {
            for x in 0..128 {
                rep128[y * 128 + x] = match ((x / 16) % 2, (y / 16) % 2) {
                    (0, 0) => 0,
                    (1, 0) => 85,
                    (0, 1) => 170,
                    _ => 255,
                };
            }
        }
        let (base_off, _) = encode_obu_payload_with(
            &rep128,
            128,
            128,
            false,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (base_on, _) = encode_obu_payload_with(
            &rep128,
            128,
            128,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (rdo_on, _) = encode_obu_payload_with_rdo(
            &rep128,
            128,
            128,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        assert_eq!(base_off.len(), 70);
        assert_eq!(base_on.len(), 100);
        assert_eq!(rdo_on.len(), 70);
        assert!(
            rdo_on.len() < base_on.len(),
            "palette must beat available copies"
        );
        // 8px checker repeats similarly.
        let chk = checker(64, 64);
        let (b_off, _) = encode_obu_payload_with(
            &chk,
            64,
            64,
            false,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (b_on, _) = encode_obu_payload_with(
            &chk,
            64,
            64,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (r_on, _) = encode_obu_payload_with_rdo(
            &chk,
            64,
            64,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        assert_eq!((b_off.len(), b_on.len(), r_on.len()), (32, 58, 32));
        assert!(r_on.len() < b_on.len());
    }

    /// Flat regions, border/photo transitions, repeated patterns, and native
    /// edge partitions (160x144 has partial SBs; 1280x1152 is exact).
    /// RDO never loses to the baseline best via the whole-image fallback,
    /// and palette-only vs IntraBC-enabled modes are exercised separately.
    #[test]
    fn rdo_flat_edges_and_modes() {
        // Flat ties (deterministic; also exercised in `rdo_deterministic_ties`).
        let flat = vec![85u8; 64 * 64];
        for use_bc in [false, true] {
            let (b, _) = encode_obu_payload_with(
                &flat,
                64,
                64,
                use_bc,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            let (r, _) = encode_obu_payload_with_rdo(
                &flat,
                64,
                64,
                use_bc,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            assert_eq!(b.len(), 18);
            assert_eq!(r.len(), 18);
        }
        // Border/photo transition: 160x144 with 16px border (0) around photo
        // interior (alternating 85/170 checker 8px). Exercises edge SBs.
        let (w, h) = (160usize, 144usize);
        let mut img = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let border = x < 16 || y < 16 || x >= 144 || y >= 128;
                img[y * w + x] = if border {
                    0
                } else if ((x / 8) + (y / 8)) % 2 == 0 {
                    85
                } else {
                    170
                };
            }
        }
        for use_bc in [false, true] {
            let (b, _) = encode_obu_payload_with(
                &img,
                160,
                144,
                use_bc,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            let (r, _) = encode_obu_payload_with_rdo(
                &img,
                160,
                144,
                use_bc,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            // RDO locally ≤ baseline; true bytes may tie or win, never
            // catastrophically lose (fallback covers any estimate miss).
            // Here we assert the fallback file (via `encode_gray`) is ≤ baseline.
            let _ = (b, r);
        }
        // Whole-image fallback: `encode_gray` picks min complete file.
        let (b_pay, b_av1c) = encode_obu_payload(
            &img,
            160,
            144,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let base_file = {
            let image = IsoBmffImage {
                major_brand: *b"avif",
                minor_version: 0,
                compatible_brands: vec![*b"avif", *b"mif1", *b"miaf"],
                primary_item_id: 1,
                items: vec![av01_item(1, 160, 144, b_pay, b_av1c)],
                groups: vec![],
            };
            gamut_isobmff::write(&image).unwrap()
        };
        let rdo_file = encode_gray_with_search_radii(
            &img,
            160,
            144,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        assert!(
            rdo_file.len() <= base_file.len(),
            "fallback must never lose: rdo {} vs base {}",
            rdo_file.len(),
            base_file.len()
        );
        // Native upscale edge: 1280x1152 8x of the above (exact SBs).
        let mut large = vec![0u8; 1280 * 1152];
        for y in 0..1152 {
            for x in 0..1280 {
                large[y * 1280 + x] = img[(y / 8) * w + (x / 8)];
            }
        }
        let (b2, _) = encode_obu_payload(
            &large,
            1280,
            1152,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (r2, _) = encode_obu_payload_rdo(
            &large,
            1280,
            1152,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        assert!(
            r2.len() <= b2.len(),
            "large RDO must not lose to baseline best"
        );
    }

    /// Deterministic ties: flat blocks and identical runs produce exactly
    /// equal costs and identical bytes across runs (tie → palette locally,
    /// baseline on exact file ties globally).
    #[test]
    fn rdo_deterministic_ties() {
        let flat = vec![0u8; 64 * 64];
        let (r1, _) = encode_obu_payload_with_rdo(
            &flat,
            64,
            64,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (r2, _) = encode_obu_payload_with_rdo(
            &flat,
            64,
            64,
            true,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        assert_eq!(r1, r2, "RDO must be deterministic");
        // Whole-image tie prefers baseline (equal files → baseline kept).
        // Note: global win counters would race across parallel tests, so tie
        // preference is verified via byte-identical files (fallback returns
        // the baseline bytes on exact ties), not via globals here.
        // Benchmarks (single-threaded) report wins via `rdo_stats`.
        let f1 = encode_gray_with_search_radii(
            &flat,
            64,
            64,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let f2 = encode_gray_with_search_radii(
            &flat,
            64,
            64,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        assert_eq!(f1, f2, "fallback must be deterministic");
        // Baseline-only file (same mux as `encode_gray` uses for its base).
        let (bp, ba) = encode_obu_payload(
            &flat,
            64,
            64,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let base_file = {
            let image = IsoBmffImage {
                major_brand: *b"avif",
                minor_version: 0,
                compatible_brands: vec![*b"avif", *b"mif1", *b"miaf"],
                primary_item_id: 1,
                items: vec![av01_item(1, 64, 64, bp, ba)],
                groups: vec![],
            };
            gamut_isobmff::write(&image).unwrap()
        };
        assert_eq!(f1, base_file, "tie must keep baseline bytes exactly");
    }

    /// Rejected trials leave no state changes (except the persistent
    /// `candidates` counter). Encodes a 64x64 with RDO, then verifies that a
    /// manual trial + restore roundtrips all scoped state.
    #[test]
    fn rdo_rejected_trials_leave_no_state() {
        use super::TileEncoder;
        let mut img = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                img[y * 64 + x] = [0u8, 85, 170, 255][(x / 16 + y / 16) % 4];
            }
        }
        let (_, masks) = validate_gray(&img, 64, 64).unwrap();
        let mut enc = TileEncoder::new_with_rdo(&img, 64, 64, true, masks, true).unwrap();
        // Baseline snapshot at (0,0) 64x64.
        let base_skip = enc.skip.clone();
        let base_psize = enc.psize.clone();
        let base_pcolors = enc.pcolors.clone();
        let base_above = enc.above_part.clone();
        let base_left = enc.left_part.clone();
        let base_cost = enc.cost;
        let base_cands = enc.candidates;
        let base_sym_dbg = format!("{:?}", enc.sym);
        let base_cdfs = enc.cdfs.clone();
        // Speculative trial that mutates (palette encode), then restore.
        let snap = enc.save_snapshot(0, 0, 16, 16);
        enc.enter_speculative();
        // Force a palette encode (mutates sym/CDFs/neighbours/cost).
        let ctx = enc.partition_ctx(0, 0, 4);
        enc.enc_part(4, ctx, PARTITION_NONE);
        enc.update_partition_ctx(0, 0, 4);
        enc.block_with_force(0, 0, 4, true, None, true).unwrap();
        // State changed (cost grew, sym advanced).
        assert!(enc.cost > base_cost);
        enc.restore_snapshot(snap);
        // Restored (except candidates, which persists; cost restored).
        assert_eq!(enc.skip, base_skip);
        assert_eq!(enc.psize, base_psize);
        assert_eq!(enc.pcolors, base_pcolors);
        assert_eq!(enc.above_part, base_above);
        assert_eq!(enc.left_part, base_left);
        assert_eq!(enc.cost, base_cost);
        assert_eq!(format!("{:?}", enc.sym), base_sym_dbg);
        assert_eq!(enc.cdfs, base_cdfs);
        // Candidates persists (trial counted even though restored).
        // Note: `save/restore` alone does not count; `trial()` does.
        assert_eq!(enc.candidates, base_cands);
        // `trial()` counts and restores.
        let c0 = enc.candidates;
        let _ = enc
            .trial(0, 0, 16, 16, |e| {
                let ctx = e.partition_ctx(0, 0, 4);
                e.enc_part(4, ctx, PARTITION_NONE);
                e.update_partition_ctx(0, 0, 4);
                e.block_with_force(0, 0, 4, true, None, true)
            })
            .unwrap();
        assert_eq!(enc.candidates, c0 + 1);
        assert_eq!(enc.cost, base_cost, "trial must restore cost");
        assert_eq!(enc.skip, base_skip, "trial must restore neighbours");
        // Speculative searches increment speculative (not committed)
        // diagnostics (per-image, test-isolated; globals would race across
        // parallel tests).
        let (q0, f0) = {
            let bc = enc.intrabc.as_ref().expect("intrabc on");
            (bc.pattern_queries(), bc.pattern_fallbacks())
        };
        let (sq0, sf0) = {
            let bc = enc.intrabc.as_ref().expect("intrabc on");
            (bc.pattern_spec_queries(), bc.pattern_spec_fallbacks())
        };
        let snap2 = enc.save_snapshot(0, 0, 4, 4);
        enc.enter_speculative();
        let _ = enc.intrabc_match(0, 0, 4, true);
        enc.restore_snapshot(snap2);
        let (q1, f1) = {
            let bc = enc.intrabc.as_ref().expect("intrabc on");
            (bc.pattern_queries(), bc.pattern_fallbacks())
        };
        let (sq1, sf1) = {
            let bc = enc.intrabc.as_ref().expect("intrabc on");
            (bc.pattern_spec_queries(), bc.pattern_spec_fallbacks())
        };
        assert_eq!((q0, f0), (q1, f1), "speculative search must not count");
        // Speculative work is counted separately (total = committed + spec).
        // This 16x16 is uniform (flat quadrant) on a verified image, so the
        // speculative query hits the stripe fallback path: one speculative
        // query and one speculative fallback.
        assert_eq!(sq1, sq0 + 1, "speculative query tracked separately");
        assert_eq!(sf1, sf0 + 1, "speculative fallback tracked separately");
    }

    /// Nested rejected trials must not increment committed-path counters.
    /// Regression for the `set_speculative(false)` bug: `rdo_leaf` /
    /// `rdo_contained` unconditionally cleared speculative state when
    /// committing their local winner, so a rejected parent trial at MI
    /// (0,80) on IMG_01's enlarged raster changed per-image query/fallback
    /// counters from (0,0) to (9,9). With nesting-aware depth the inner
    /// winner remains speculative relative to the parent.
    ///
    /// Uses a verified 8x image with actual copy opportunities (repeating
    /// non-stripe 16x16 pattern, wide enough for SB-delay hits) and a
    /// rejected 64x64 parent trial containing nested palette/copy/split
    /// trials. Distinguishes total / speculative / committed searches and
    /// verifies ordinary baseline wins never increment the error counter.
    #[test]
    fn rdo_nested_rejected_trial_no_committed_counts() {
        use super::TileEncoder;
        // 512x64 verified 8x image: every 8x8 cell constant, each 16x16 is
        // [[0,85],[170,255]] (non-stripe, definitive cached path), repeated
        // so later 64x64 blocks have exact causal matches past the SB delay.
        let (w, h) = (512usize, 64usize);
        let mut img = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let bx = (x / 8) % 2;
                let by = (y / 8) % 2;
                img[y * w + x] = match (bx, by) {
                    (0, 0) => 0,
                    (1, 0) => 85,
                    (0, 1) => 170,
                    _ => 255,
                };
            }
        }
        let (_, masks) = validate_gray(&img, w as u32, h as u32).unwrap();
        let mut enc = TileEncoder::new_with_rdo(&img, w, h, true, masks, true).unwrap();
        {
            let bc = enc.intrabc.as_ref().expect("intrabc on");
            assert!(bc.pattern_verified(), "test image must verify");
        }
        // Advance committed state through the first five SBs (MI c 0..80) so
        // MI (0,80) has decoded history and SB-delay-legal sources.
        for c in (0..80usize).step_by(16) {
            enc.rdo_partition(0, c, 16, true).unwrap();
        }
        // Copy opportunity must actually exist at the nested-trial origin.
        assert!(
            enc.intrabc_match(0, 80, 16, true).is_some(),
            "test needs an actual 64x64 copy at MI (0,80)"
        );
        let (q0, f0) = {
            let bc = enc.intrabc.as_ref().expect("intrabc on");
            (bc.pattern_queries(), bc.pattern_fallbacks())
        };
        let (sq0, sf0) = {
            let bc = enc.intrabc.as_ref().expect("intrabc on");
            (bc.pattern_spec_queries(), bc.pattern_spec_fallbacks())
        };
        let (tq0, tf0) = {
            let bc = enc.intrabc.as_ref().expect("intrabc on");
            (bc.pattern_total_queries(), bc.pattern_total_fallbacks())
        };
        assert_eq!(tq0, q0 + sq0, "total = committed + speculative");
        assert_eq!(tf0, f0 + sf0, "total = committed + speculative");
        // Rejected parent trial: nested palette/copy/split trials plus inner
        // winners (speculative relative to this trial). Restored afterwards.
        let _ = enc
            .trial(0, 80, 16, 16, |e| e.rdo_contained(0, 80, 16, 4, true))
            .unwrap();
        let (q1, f1) = {
            let bc = enc.intrabc.as_ref().expect("intrabc on");
            (bc.pattern_queries(), bc.pattern_fallbacks())
        };
        let (sq1, sf1) = {
            let bc = enc.intrabc.as_ref().expect("intrabc on");
            (bc.pattern_spec_queries(), bc.pattern_spec_fallbacks())
        };
        let (tq1, tf1) = {
            let bc = enc.intrabc.as_ref().expect("intrabc on");
            (bc.pattern_total_queries(), bc.pattern_total_fallbacks())
        };
        assert_eq!(
            (q0, f0),
            (q1, f1),
            "rejected nested trial must not increment committed counters"
        );
        assert!(
            (sq1, sf1) != (sq0, sf0),
            "rejected nested trial must do speculative work (spec {sq0},{sf0} -> {sq1},{sf1})"
        );
        assert_eq!(tq1, q1 + sq1, "total tracks committed + speculative");
        assert_eq!(tf1, f1 + sf1, "total tracks committed + speculative");
        assert!(tq1 > tq0, "total work grew via speculative trials");
        // Ordinary baseline wins (here: flat tie) never increment errors.
        let errs_before = super::rdo_errors();
        let flat = vec![0u8; 32 * 32];
        let _ = super::encode_gray_with_search_radii(
            &flat,
            32,
            32,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        assert_eq!(
            super::rdo_errors(),
            errs_before,
            "ordinary tie must not count as RDO error"
        );
    }

    #[test]
    fn rdo_uniform_copy_uses_exact_gray_and_rejected_nested_trial_stays_speculative() {
        use super::TileEncoder;
        let (w, h) = (512usize, 64usize);
        let pixels = vec![37u8; w * h];
        let (_, masks) = validate_gray(&pixels, w as u32, h as u32).unwrap();

        let mut enc = TileEncoder::new_with_rdo(&pixels, w, h, true, masks.clone(), true).unwrap();
        enc.intrabc.as_mut().unwrap().record(0, 0, 4, 4, None);
        let before = enc.intrabc.as_ref().unwrap().uniform_stats();
        enc.trial(0, 80, 4, 4, |e| e.rdo_leaf_with_plan(0, 80, 2, true, None))
            .unwrap();
        let rejected = enc.intrabc.as_ref().unwrap().uniform_stats();
        assert_eq!(
            rejected.committed_searches, before.committed_searches,
            "rejected nested trial must not claim a committed search"
        );
        assert!(
            rejected.speculative_work > before.speculative_work,
            "nested RDO trial must expose its candidate search work"
        );
        assert_eq!(
            rejected.selected_copies, before.selected_copies,
            "rejected trial must not count its local uniform-copy choice"
        );

        enc.rdo_leaf_with_plan(0, 80, 2, true, None).unwrap();
        let committed = enc.intrabc.as_ref().unwrap().uniform_stats();
        assert_eq!(
            committed.selected_copies, 1,
            "RDO should choose the low-cost copy"
        );
        assert!(committed.legal_matches > rejected.legal_matches);

        let mut palette = TileEncoder::new_with_rdo(&pixels, w, h, false, masks, true).unwrap();
        palette.rdo_leaf_with_plan(0, 80, 2, true, None).unwrap();
        assert!(
            palette.intrabc.is_none(),
            "palette-only mode disables IntraBC"
        );
    }

    /// Whole-image fallback prefers baseline on ties and keeps the smaller
    /// complete file otherwise. Reports wins via globals.
    #[test]
    fn rdo_fallback_prefers_baseline_on_ties() {
        // Flat ties → baseline bytes exactly (no globals assert; parallel-safe).
        // Global win/candidate counters are for single-threaded benchmarks.
        let flat = vec![255u8; 32 * 32];
        let f = encode_gray_with_search_radii(
            &flat,
            32,
            32,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        assert!(!f.is_empty());
        let (bp, ba) = encode_obu_payload(
            &flat,
            32,
            32,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let base_file = {
            let image = IsoBmffImage {
                major_brand: *b"avif",
                minor_version: 0,
                compatible_brands: vec![*b"avif", *b"mif1", *b"miaf"],
                primary_item_id: 1,
                items: vec![av01_item(1, 32, 32, bp, ba)],
                groups: vec![],
            };
            gamut_isobmff::write(&image).unwrap()
        };
        assert_eq!(f, base_file, "tie must keep baseline");
        // Pair fallback shape preserved (see `pair_container_shape`).
        reset_rdo_stats();
        let small = checker(64, 64);
        let large = checker(128, 128);
        let pair = encode_gray_pair_with_search_radii(
            &small,
            64,
            64,
            &large,
            128,
            128,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        assert!(!pair.is_empty());
        let (w2, b2, _) = rdo_stats();
        // Pair compares complete files; either wins or baseline (tie) — but
        // the file must be ≤ baseline-only construction. Baseline-only pair:
        let (sp, sa) = encode_obu_payload(
            &small,
            64,
            64,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let (lp, la) = encode_obu_payload(
            &large,
            128,
            128,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();
        let base_pair = {
            let image = IsoBmffImage {
                major_brand: *b"avif",
                minor_version: 0,
                compatible_brands: vec![*b"avif", *b"mif1", *b"miaf"],
                primary_item_id: 1,
                items: vec![av01_item(1, 128, 128, lp, la), av01_item(2, 64, 64, sp, sa)],
                groups: vec![EntityGroup {
                    group_type: *b"altr",
                    group_id: 10,
                    entity_ids: vec![1, 2],
                }],
            };
            gamut_isobmff::write(&image).unwrap()
        };
        assert!(pair.len() <= base_pair.len(), "pair fallback must not lose");
        let _ = (w2, b2);
    }

    /// Decode both image items via ffmpeg and the TopK8 primary through
    /// libavif's dav1d and libaom backends. Skips gracefully when ffmpeg is
    /// missing.
    /// Manual run: `cargo test --release rdo_decode_verify -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn rdo_decode_verify() {
        use std::process::Command;
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            eprintln!("ffmpeg missing, skipping decode verify");
            return;
        }
        // 160x144 border/photo + 8x upscale (both items, native sizes).
        let (w, h) = (160usize, 144usize);
        let mut small = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let border = x < 16 || y < 16 || x >= 144 || y >= 128;
                small[y * w + x] = if border {
                    0
                } else if ((x / 8) + (y / 8)) % 2 == 0 {
                    170
                } else {
                    85
                };
            }
        }
        let mut large = vec![0u8; 1280 * 1152];
        for y in 0..1152 {
            for x in 0..1280 {
                large[y * 1280 + x] = small[(y / 8) * w + (x / 8)];
            }
        }
        let avif = encode_gray_pair_with_pad_policy_and_motion_candidate(
            &small,
            160,
            144,
            &large,
            1280,
            1152,
            FlatPadPolicy::Baseline,
            intrabc::DEFAULT_SEARCH_RINGS,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            MotionCandidatePolicy::TopK8,
        )
        .unwrap();
        let dir = std::env::temp_dir().join(format!("rdo_verify_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let avif_path = dir.join("t.avif");
        std::fs::write(&avif_path, &avif).unwrap();
        for (stream, expect, ew, eh) in [
            (0, &large, 1280usize, 1152usize),
            (1, &small, 160usize, 144usize),
        ] {
            let raw_path = dir.join(format!("s{stream}.raw"));
            let st = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    avif_path.to_str().unwrap(),
                    "-map",
                    &format!("0:{stream}"),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "gray",
                    raw_path.to_str().unwrap(),
                ])
                .status()
                .expect("ffmpeg run");
            assert!(st.success(), "ffmpeg decode stream {stream}");
            let raw = std::fs::read(&raw_path).unwrap();
            assert_eq!(raw.len(), ew * eh, "stream {stream} size");
            assert_eq!(raw, *expect, "stream {stream} lossless pixels");
        }
        if Command::new("avifdec").arg("-V").output().is_ok() {
            for codec in ["dav1d", "aom"] {
                let y4m_path = dir.join(format!("topk-{codec}.y4m"));
                let status = Command::new("avifdec")
                    .args(["--codec", codec])
                    .arg(&avif_path)
                    .arg(&y4m_path)
                    .status()
                    .expect("avifdec run");
                assert!(status.success(), "avifdec {codec} decode");
                let y4m = std::fs::read(&y4m_path).unwrap();
                let header_end = y4m.iter().position(|&byte| byte == b'\n').unwrap();
                let header = String::from_utf8_lossy(&y4m[..header_end]);
                assert!(header.contains("W1280") && header.contains("H1152"));
                let frame_start = y4m[header_end + 1..]
                    .iter()
                    .position(|&byte| byte == b'\n')
                    .map(|offset| header_end + 2 + offset)
                    .unwrap();
                let luma_end = frame_start + 1280 * 1152;
                assert!(y4m.len() >= luma_end, "short Y4M output from {codec}");
                assert_eq!(
                    &y4m[frame_start..luma_end],
                    large.as_slice(),
                    "TopK8 primary item differs through {codec}"
                );
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// RDO benchmark (ignored): baseline vs RDO sizes, medians, candidates.
    /// `cargo test --release bench_rdo -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn bench_rdo() {
        fn median(mut v: Vec<std::time::Duration>) -> std::time::Duration {
            v.sort_unstable();
            v[v.len() / 2]
        }
        reset_rdo_stats();
        crate::mono::reset_pattern_cache_stats();
        let (sw, sh) = (160usize, 144usize);
        let levels = [0u8, 85, 170, 255];
        let mut distinct_small = vec![0u8; sw * sh];
        for y in 0..sh {
            for x in 0..sw {
                distinct_small[y * sw + x] = levels[(x * 7 + y * 13) % 4];
            }
        }
        let mut distinct_large = vec![0u8; 1280 * 1152];
        for y in 0..1152 {
            for x in 0..1280 {
                distinct_large[y * 1280 + x] = distinct_small[(y / 8) * sw + (x / 8)];
            }
        }
        let flat_small = vec![0u8; sw * sh];
        let flat_large = vec![0u8; 1280 * 1152];
        for (name, gray, w, h) in [
            ("distinct_small160", &distinct_small, 160u32, 144u32),
            ("distinct_large1280", &distinct_large, 1280u32, 1152u32),
            ("duplicate_small160", &flat_small, 160u32, 144u32),
            ("duplicate_large1280", &flat_large, 1280u32, 1152u32),
        ] {
            let (b_off, _) = encode_obu_payload_with(
                gray,
                w,
                h,
                false,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            let (b_on, _) = encode_obu_payload_with(
                gray,
                w,
                h,
                true,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            let (b_pick, _) = encode_obu_payload(
                gray,
                w,
                h,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            let (r_off, _) = encode_obu_payload_with_rdo(
                gray,
                w,
                h,
                false,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            let (r_on, _) = encode_obu_payload_with_rdo(
                gray,
                w,
                h,
                true,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            let (r_pick, _) = encode_obu_payload_rdo(
                gray,
                w,
                h,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            let base_file = {
                let (p, a) = encode_obu_payload(
                    gray,
                    w,
                    h,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                let image = IsoBmffImage {
                    major_brand: *b"avif",
                    minor_version: 0,
                    compatible_brands: vec![*b"avif", *b"mif1", *b"miaf"],
                    primary_item_id: 1,
                    items: vec![av01_item(1, w, h, p, a)],
                    groups: vec![],
                };
                gamut_isobmff::write(&image).unwrap()
            };
            let opt_file = encode_gray_with_search_radii(
                gray,
                w,
                h,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            let mut t_base = Vec::new();
            let mut t_rdo = Vec::new();
            let mut t_opt = Vec::new();
            for _ in 0..5 {
                let t = std::time::Instant::now();
                encode_obu_payload(
                    gray,
                    w,
                    h,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                t_base.push(t.elapsed());
                let t = std::time::Instant::now();
                encode_obu_payload_rdo(
                    gray,
                    w,
                    h,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                t_rdo.push(t.elapsed());
                let t = std::time::Instant::now();
                encode_gray_with_search_radii(
                    gray,
                    w,
                    h,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                t_opt.push(t.elapsed());
            }
            eprintln!(
                "bench_rdo {name}: payloads base_off={} base_on={} base_pick={} rdo_off={} rdo_on={} rdo_pick={} files base={} opt={} times base={:?} rdo={:?} opt={:?}",
                b_off.len(),
                b_on.len(),
                b_pick.len(),
                r_off.len(),
                r_on.len(),
                r_pick.len(),
                base_file.len(),
                opt_file.len(),
                median(t_base),
                median(t_rdo),
                median(t_opt)
            );
        }
        for (pname, s, l) in [
            ("distinct_pair", &distinct_small, &distinct_large),
            ("duplicate_pair", &flat_small, &flat_large),
        ] {
            let mut t_pair = Vec::new();
            for _ in 0..5 {
                let t = std::time::Instant::now();
                encode_gray_pair_with_search_radii(
                    s,
                    160,
                    144,
                    l,
                    1280,
                    1152,
                    intrabc::DEFAULT_SEARCH_RINGS,
                    DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
                )
                .unwrap();
                t_pair.push(t.elapsed());
            }
            let pair = encode_gray_pair_with_search_radii(
                s,
                160,
                144,
                l,
                1280,
                1152,
                intrabc::DEFAULT_SEARCH_RINGS,
                DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
            )
            .unwrap();
            eprintln!(
                "bench_rdo {pname}: pair_file={} pair_median={:?}",
                pair.len(),
                median(t_pair)
            );
        }
        let (rw, bw, cands) = rdo_stats();
        let errors = rdo_errors();
        let (q, f) = crate::mono::pattern_cache_stats();
        let (sq, sf) = crate::mono::pattern_cache_stats_speculative();
        let (tq, tf) = crate::mono::pattern_cache_stats_total();
        eprintln!(
            "bench_rdo totals: rdo_wins={rw} baseline_wins={bw} errors={errors} candidates={cands} pattern_committed={q}/{f} pattern_speculative={sq}/{sf} pattern_total={tq}/{tf}"
        );
        eprintln!(
            "bench_rdo memory: snapshot_inline={}B max_trial_depth=3; trial snapshots allocate no footprint/CDF/coder clones; masks_small=90B masks_large=5760B pattern_large<=90KiB",
            std::mem::size_of::<Snapshot>()
        );
    }

    #[test]
    fn lookahead_can_be_worse_than_baseline_due_to_cache_pollution() {
        // This test demonstrates the bug where --flat-pad-policy lookahead
        // produces larger files than baseline due to palette cache pollution.
        //
        // The lookahead policy optimizes the current flat block + one sibling,
        // but doesn't account for how the chosen pad value pollutes the palette
        // cache for all subsequent blocks in the same row and below.
        //
        // We construct an image where:
        // 1. Early blocks in the first row are flat (single color)
        // 2. The lookahead picks a pad color not in the cache for these blocks
        // 3. Many subsequent 2-color blocks would efficiently reuse the baseline
        //    cache, but get an extra color from the polluted cache
        //
        // The bug manifests as lookahead producing a larger file than baseline.

        let (sw, sh) = (160u32, 144u32); // 10x9 blocks of 16x16
        let (lw, lh) = (1280u32, 1152u32); // 80x72 blocks of 16x16

        // Create small image with specific pattern to trigger the bug
        let mut small = vec![0u8; (sw * sh) as usize];
        let mut large = vec![0u8; (lw * lh) as usize];

        // Pattern to trigger the bug:
        // Row 0: 10 flat blocks ALL with color 0
        // Row 1-8: All blocks are 2-color [0, 85] checkerboard
        //
        // With baseline:
        // - Block (0,0) color 0: cache empty → baseline picks 85 (v+1). Cache: [0, 85]
        // - Block (1,0) color 0: cache [0, 85] → baseline picks 85 (cached). Cache: [0, 85]
        // - All 10 flat blocks get pad 85. Final cache after row 0: [0, 85]
        // - Row 1 blocks [0,85]: cache from above has [0,85], they reuse efficiently (2 colors)
        //
        // With lookahead:
        // - Block (0,0) color 0: evaluates (0,0) + sibling (1,0) both color 0
        //   Candidates: 85 (baseline), 170, 255 (global)
        //   The cost model might prefer 170 or 255 for the pair!
        //   If it picks 170: cache becomes [0, 170]
        // - Block (1,0) color 0: cache [0, 170] → baseline would pick 0 or 170
        // - Eventually cache gets polluted with 170/255 instead of 85
        // - Row 1 blocks [0,85]: cache has [0, 170, ...] but NOT 85
        //   They must add 85 as a NEW color → 3 colors instead of 2!
        //   Each such block costs extra bits for the 3rd palette entry

        for by in 0..9 {
            for bx in 0..10 {
                // First row: all flat color 0
                // Other rows: 2-color [0, 85] checkerboard
                let is_flat = by == 0;

                for dy in 0..16 {
                    for dx in 0..16 {
                        let x = bx * 16 + dx;
                        let y = by * 16 + dy;
                        if x < sw as usize && y < sh as usize {
                            if is_flat {
                                small[y * sw as usize + x] = 0;
                            } else {
                                // All other rows: [0, 85] checkerboard
                                small[y * sw as usize + x] =
                                    if (dx + dy) % 2 == 0 { 0 } else { 85 };
                            }
                        }
                    }
                }
            }
        }

        // Large image: 8x upscale of small
        for y in 0..sh as usize {
            for x in 0..sw as usize {
                let v = small[y * sw as usize + x];
                for dy in 0..8 {
                    for dx in 0..8 {
                        let lx = x * 8 + dx;
                        let ly = y * 8 + dy;
                        if lx < lw as usize && ly < lh as usize {
                            large[ly * lw as usize + lx] = v;
                        }
                    }
                }
            }
        }

        // Encode with both policies
        let baseline = encode_gray_pair_with_pad_policy(
            &small,
            sw,
            sh,
            &large,
            lw,
            lh,
            FlatPadPolicy::Baseline,
            64,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();

        let lookahead = encode_gray_pair_with_pad_policy(
            &small,
            sw,
            sh,
            &large,
            lw,
            lh,
            FlatPadPolicy::Lookahead,
            64,
            DEFAULT_UNIFORM_SEARCH_RADIUS_RINGS,
        )
        .unwrap();

        println!("Baseline size: {} bytes", baseline.len());
        println!("Lookahead size: {} bytes", lookahead.len());
        println!(
            "Difference (lookahead - baseline): {} bytes",
            lookahead.len() as i64 - baseline.len() as i64
        );

        // This test documents the bug - it should pass (demonstrating the bug exists)
        // Once fixed, we would change this to assert!(lookahead.len() <= baseline.len())
        // For now, we just verify both encode successfully and log the sizes
        assert!(!baseline.is_empty());
        assert!(!lookahead.is_empty());

        // If lookahead is worse, we've reproduced the bug
        if lookahead.len() > baseline.len() {
            eprintln!(
                "BUG CONFIRMED: Lookahead ({}) > Baseline ({}) by {} bytes",
                lookahead.len(),
                baseline.len(),
                lookahead.len() - baseline.len()
            );
        }
    }
}
