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
use std::io;

mod intrabc;
use intrabc::IntrabcState;

/// Code toggle for IntraBC (intra block copy) evaluation: exact
/// pixel-rectangle copies from causal decoded area via integer MVs +
/// `skip = 1`, as a per-block alternative to the palette path. When `true`,
/// each raster is encoded both ways (IntraBC vs palette-only) and the
/// smaller payload is kept; when `false`, only the palette path is encoded
/// (faster, byte-identical to the pre-IntraBC encoder: no flag symbols, no
/// header bit).
const USE_INTRABC: bool = true;

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

/// `Default_Kf_Y_Mode_Cdf[0][0]` — the only row used (all neighbours are DC).
const INTRA_Y_DC_ROW: [u16; 13] = [
    15588, 17027, 19338, 20218, 20682, 21110, 21825, 23244, 24189, 28165, 29093, 30466, 32768,
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

/// One adapting CDF: owned copy of a default row plus the spec §8.2.6
/// adaptation counter (both start fresh for every image).
struct AdaptCdf {
    cdf: Vec<u16>,
    count: u16,
}

impl AdaptCdf {
    fn new(row: &[u16]) -> Self {
        Self {
            cdf: row.to_vec(),
            count: 0,
        }
    }

    fn encode(&mut self, sym: &mut SymbolEncoder, s: usize) {
        sym.encode_symbol_adapt(s, &mut self.cdf, &mut self.count);
    }
}

/// All CDF state for one image. Rows mirror the static tables above;
/// every `S()` symbol adapts its row (`disable_cdf_update = 0`).
/// Literals (`L()`, `NS()`) never adapt, matching the decoder.
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
    /// `None` disables IntraBC entirely (byte-identical to the pre-IntraBC
    /// encoder: no flag symbols, no header bit — see `USE_INTRABC`).
    intrabc: Option<IntrabcState>,
    skip: Vec<u8>,
    psize: Vec<u8>,
    pcolors: Vec<[u8; 8]>,
    above_part: Vec<u8>,
    left_part: Vec<u8>,
}

/// `Partition_Context` update table (§5.11.4), rows [above|left],
/// columns NONE/HORZ/VERT/SPLIT. Only the NONE column is used here
/// (SPLIT never updates; HORZ/VERT are never emitted).
const AL_PART_NONE_ABOVE: [u8; 5] = [0x00, 0x10, 0x18, 0x1c, 0x1e];
const AL_PART_NONE_LEFT: [u8; 5] = [0x00, 0x10, 0x18, 0x1c, 0x1e];

impl<'a> TileEncoder<'a> {
    fn new(px: &'a [u8], w: usize, h: usize, use_intrabc: bool) -> io::Result<Self> {
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
        Ok(Self {
            px,
            w,
            h,
            mi_cols,
            mi_rows,
            sym: SymbolEncoder::new(),
            cdfs: Cdfs::new(),
            intrabc: use_intrabc.then(|| IntrabcState::new(mi_cols, mi_rows)),
            skip: vec![0; mi_area],
            psize: vec![0; mi_area],
            pcolors: vec![[0u8; 8]; mi_area],
            above_part: vec![0; mi_cols],
            left_part: vec![0; mi_rows],
        })
    }

    fn sample(&self, x: usize, y: usize) -> u8 {
        self.px[y * self.w + x]
    }

    /// Count of distinct levels in the `bw4`-MI region at MI `(r, c)`,
    /// saturating at 5 (only the ≤4 question matters: larger palettes
    /// have no index tables in this encoder).
    fn region_levels(&self, r: usize, c: usize, bw4: usize) -> usize {
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
            self.cdfs
                .part(bsl, ctx)
                .encode(&mut self.sym, PARTITION_NONE);
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
                self.cdfs
                    .part(bsl, ctx)
                    .encode(&mut self.sym, PARTITION_NONE);
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
            self.cdfs
                .part(bsl, ctx)
                .encode(&mut self.sym, PARTITION_SPLIT);
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
            self.sym.encode_symbol(1, &[c0, 32768]);
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
            self.sym.encode_symbol(1, &[c0, 32768]);
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
            self.sym.encode_literal(val as u32, w - 1);
        } else {
            let coded = val + m;
            self.sym.encode_literal((coded >> 1) as u32, w - 1);
            self.sym.encode_literal((coded & 1) as u32, 1);
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
    fn block(
        &mut self,
        r: usize,
        c: usize,
        bsl: usize,
        top_has_right: bool,
        carried: Option<((i32, i32), (i32, i32))>,
    ) -> io::Result<()> {
        let bw: usize = 4 << bsl;
        let (sx, sy) = (c * 4, r * 4);
        let bw4 = 1usize << bsl;

        // Fast path: `partition` already found an exact copy for this
        // block. Skip distinct/colors/cache work entirely — a copy beats
        // the palette path by construction here.
        if let Some(((my, mx), (py, px_))) = carried {
            let sctx = self.skip_ctx(r, c);
            self.cdfs.skip[sctx].encode(&mut self.sym, 1);
            self.encode_copy(r, c, bsl, bw4, my, mx, py, px_)?;
            return Ok(());
        }

        // Distinct-level scan doubles as the flat test (single-level
        // blocks always take the palette path — a copy can't beat a
        // ~15-bit flat table). The `set` is reused below to build the
        // palette colors, so this scan is not wasted on the palette path.
        let mut set = [false; 256];
        for i in 0..bw {
            for j in 0..bw {
                set[self.sample(sx + j, sy + i) as usize] = true;
            }
        }
        let distinct = set.iter().filter(|&&b| b).count();
        if distinct == 0 {
            return Err(invalid_input(format!(
                "empty palette block at MI ({r},{c}) size {bw}x{bw}"
            )));
        }

        // IntraBC decision (non-flat only): an exact match in causal
        // decoded area beats the palette path (MV residual of ~10-25 bits
        // vs palette headers plus indices). `find_match` returns the
        // predictor too, so no second `predictor` query is needed.
        let bc_match: Option<((i32, i32), (i32, i32))> = if distinct > 1 {
            self.intrabc_match(r, c, bw4, top_has_right)
        } else {
            None
        };

        // skip = 1 (no residual; reconstruction is exactly the palette
        // — or the copied pixels on the IntraBC path).
        let sctx = self.skip_ctx(r, c);
        self.cdfs.skip[sctx].encode(&mut self.sym, 1);

        if let Some(((my, mx), (py, px_))) = bc_match {
            self.encode_copy(r, c, bsl, bw4, my, mx, py, px_)?;
            return Ok(());
        }
        // Palette path from here: build colors/cache/index scratch only
        // now that the copy path is ruled out.
        let mut colors = [0u8; 4];
        let mut psize: usize = 0;
        for (v, &present) in set.iter().enumerate() {
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
            let mut pad: Option<u8> = None;
            for &cc in &cache_arr[..cache_len] {
                if cc != v {
                    pad = Some(cc);
                    break;
                }
            }
            let pad = pad.unwrap_or(if v < 255 { v + 1 } else { v - 1 });
            colors[1] = pad;
            n_colors = 2;
            // Keep the 2-entry table sorted (binary_search below).
            if colors[0] > colors[1] {
                colors.swap(0, 1);
            }
        }
        let psize = n_colors;

        // Palette path from here: build colors/cache/index scratch only
        // now that the copy path is ruled out.
        // (colors/cache/index construction delayed until the palette
        // branch actually needs it — copy blocks skip all of it.)
        if let Some(bc) = &mut self.intrabc {
            bc.flag(&mut self.sym, false);
        }

        let colors_slice = &colors[..psize];
        // Reusable index-map scratch: 64x64 max = 4096 bytes on the stack,
        // no per-block heap. Only the first `bw*bw` entries are used.
        let mut index_map = [0u8; 4096];
        let area = bw * bw;
        debug_assert!(area <= 4096);
        // Copy px/w out to avoid `&mut self` / `&self` borrow conflicts
        // with the local scratch (disjoint-field friendly).
        let px_ref = self.px;
        let w_ref = self.w;
        for i in 0..bw {
            let row_off = (sy + i) * w_ref + sx;
            for j in 0..bw {
                let v = px_ref[row_off + j];
                // `colors_slice` is sorted; linear scan over ≤4 entries
                // is cheaper than `binary_search` setup.
                let mut idx = 0u8;
                for (k, &cc) in colors_slice.iter().enumerate() {
                    if cc == v {
                        idx = k as u8;
                        break;
                    }
                }
                index_map[i * bw + j] = idx;
            }
        }
        let index_slice = &index_map[..area];

        // y_mode = DC_PRED (all neighbours are DC, so contexts are row 0).
        // (No `ymode` grid: the all-DC invariant is explicit — neighbours
        // are always DC on both paths, so nothing reads it back.)
        self.cdfs.y_dc.encode(&mut self.sym, DC_PRED);

        // has_palette_y = 1 (context = neighbours paletted; CDF row
        // selected by block size via `bsl`).
        let above_p = r > 0 && self.psize[(r - 1) * self.mi_cols + c] > 0;
        let left_p = c > 0 && self.psize[r * self.mi_cols + (c - 1)] > 0;
        let pctx = usize::from(above_p) + usize::from(left_p);
        self.cdfs.pal_mode[bsl - 2][pctx].encode(&mut self.sym, 1);

        // palette_size_y_minus_2, then the colors: one L(1) reuse flag
        // per palette-cache entry (equi-probable, non-adapting, stopping
        // once pal_sz entries are collected — mirroring dav1d's
        // read_pal_plane), then explicit coding for the rest (first color
        // raw 8b, remaining deltas). Reusing every cached color we need is
        // optimal: visited flag positions cost a bit either way, and every
        // reuse shortens the delta chain and hastens the early stop. Only
        // 4 gray levels exist globally, so the cache almost always covers
        // the block — this is where the upscale's bits were going.
        self.cdfs.pal_size[bsl - 2].encode(&mut self.sym, psize - 2);
        let mut is_used = [false; 4];
        let mut n_used = 0usize;
        for &cc in &cache_arr[..cache_len] {
            if n_used == psize {
                break;
            }
            if let Ok(pos) = colors_slice.binary_search(&cc) {
                self.sym.encode_literal(1, 1);
                is_used[pos] = true;
                n_used += 1;
            } else {
                self.sym.encode_literal(0, 1);
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
            self.sym.encode_literal(u32::from(new_slice[0]), 8);
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
                self.sym.encode_literal(need - 5, 2);
                let mut palette_bits = need;
                for k in 1..new_slice.len() {
                    let delta = new_slice[k] as u32 - new_slice[k - 1] as u32 - 1;
                    self.sym.encode_literal(delta, palette_bits);
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
        // Mirror dav1d's `splat_intraref`: palette blocks contribute no
        // motion candidate (and mark the footprint decoded for match
        // search) exactly like the decoder's `rt` grid.
        if let Some(bc) = &mut self.intrabc {
            bc.record(r, c, bw4, bw4, None);
        }

        // color_index_map_y: first index via ns(), rest in wavefront order
        // as positions in the neighbour-derived ColorOrder.
        self.encode_ns(index_slice[0] as usize, psize);
        for i in 1..(2 * bw - 1) {
            let mut j = i.min(bw - 1);
            let j_end = i.saturating_sub(bw - 1);
            loop {
                let (rr, cc) = (i - j, j);
                let (order, ctx) = palette_color_context(index_slice, bw, rr, cc, psize);
                let actual = index_slice[rr * bw + cc] as usize;
                let sym = order.iter().position(|&x| x == actual).unwrap_or(0);
                self.cdfs.pal_idx(psize, ctx).encode(&mut self.sym, sym);
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
        let Some(bc) = self.intrabc.as_mut() else {
            return Err(io::Error::other(
                "internal: IntraBC match without IntraBC state",
            ));
        };
        bc.flag(&mut self.sym, true);
        bc.encode_mvd(&mut self.sym, my, mx, py, px_);
        let n4 = 1usize << bsl;
        for y in 0..n4 {
            for x in 0..n4 {
                let (rr, cc) = (r + y, c + x);
                self.skip[rr * self.mi_cols + cc] = 1;
                self.psize[rr * self.mi_cols + cc] = 0;
            }
        }
        self.intrabc
            .as_mut()
            .ok_or_else(|| io::Error::other("internal: missing IntraBC state"))?
            .record(r, c, bw4, bw4, Some((my, mx)));
        Ok(())
    }

    fn finish(mut self) -> io::Result<Vec<u8>> {
        for r in (0..self.mi_rows).step_by(16) {
            for c in (0..self.mi_cols).step_by(16) {
                // SB roots start with top_has_right set, mirroring the
                // generated edge-tree root (`top_has_right = 1`).
                self.partition(r, c, 16, true)?;
            }
        }
        Ok(self.sym.finish())
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
/// `InvalidInput` instead of panicking.
fn validate_gray(gray: &[u8], w: u32, h: u32) -> io::Result<usize> {
    check_supported_dimensions(w, h)?;
    let area = checked_area(w, h)?;
    if gray.len() != area {
        return Err(invalid_input(format!(
            "pixel buffer length mismatch for {w}x{h}: expected {area} bytes, got {}",
            gray.len()
        )));
    }
    check_supported_colors(gray, w, h)?;
    Ok(area)
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
fn encode_obu_payload(gray: &[u8], w: u32, h: u32) -> io::Result<(Vec<u8>, [u8; 4])> {
    if !USE_INTRABC {
        return encode_obu_payload_with(gray, w, h, false);
    }
    let (on_payload, on_av1c) = encode_obu_payload_with(gray, w, h, true)?;
    let (off_payload, off_av1c) = encode_obu_payload_with(gray, w, h, false)?;
    if on_payload.len() <= off_payload.len() {
        Ok((on_payload, on_av1c))
    } else {
        Ok((off_payload, off_av1c))
    }
}

fn encode_obu_payload_with(
    gray: &[u8],
    w: u32,
    h: u32,
    use_intrabc: bool,
) -> io::Result<(Vec<u8>, [u8; 4])> {
    validate_gray(gray, w, h)?;

    let tile = TileEncoder::new(gray, w as usize, h as usize, use_intrabc)?;
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
/// Supported inputs are 16-aligned non-empty single-tile dimensions with
/// `gray.len() == w*h` (checked) and at most 4 distinct levels per 16x16
/// block; violations return `InvalidInput`.
#[allow(dead_code)]
pub fn encode_gray(gray: &[u8], w: u32, h: u32) -> io::Result<Vec<u8>> {
    let (payload, av1c) = encode_obu_payload(gray, w, h)?;

    let image = IsoBmffImage {
        major_brand: *b"avif",
        minor_version: 0,
        compatible_brands: vec![*b"avif", *b"mif1", *b"miaf"],
        primary_item_id: 1,
        items: vec![av01_item(1, w, h, payload, av1c)],
        groups: vec![],
    };
    write_isobmff(&image).map_err(|e| io::Error::other(format!("avif mux: {e:?}")))
}

/// Encode two rasters of the same picture (e.g. 160x144 original and its
/// 8x nearest-neighbor upscale) into one AVIF file with two `av01` items.
///
/// Item 1 (primary) is the `large` raster, item 2 the `small` raster; both
/// are non-hidden and grouped in an `altr` entity group so readers treat
/// them as alternatives and display the primary by default (AVIF §5.1).
/// Each item carries its own `ispe`/`av1C` (levels may differ: 2.0 vs 4.0).
///
/// Each raster has the same supported-input constraints as `encode_gray`;
/// violations return `InvalidInput`.
pub fn encode_gray_pair(
    small: &[u8],
    sw: u32,
    sh: u32,
    large: &[u8],
    lw: u32,
    lh: u32,
) -> io::Result<Vec<u8>> {
    let (small_payload, small_av1c) = encode_obu_payload(small, sw, sh)?;
    let (large_payload, large_av1c) = encode_obu_payload(large, lw, lh)?;

    let image = IsoBmffImage {
        major_brand: *b"avif",
        minor_version: 0,
        compatible_brands: vec![*b"avif", *b"mif1", *b"miaf"],
        primary_item_id: 1,
        items: vec![
            av01_item(1, lw, lh, large_payload, large_av1c),
            av01_item(2, sw, sh, small_payload, small_av1c),
        ],
        groups: vec![EntityGroup {
            group_type: *b"altr",
            group_id: 10,
            entity_ids: vec![1, 2],
        }],
    };
    write_isobmff(&image).map_err(|e| io::Error::other(format!("avif mux: {e:?}")))
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
    fn toggle_changes_output_but_stays_valid_obus() {
        let gray = checker(64, 64);
        let (on, _) = encode_obu_payload_with(&gray, 64, 64, true).unwrap();
        let (off, _) = encode_obu_payload_with(&gray, 64, 64, false).unwrap();
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
            let (on, _) = encode_obu_payload_with(gray, 64, 64, true).unwrap();
            let (off, _) = encode_obu_payload_with(gray, 64, 64, false).unwrap();
            let (picked, _) = encode_obu_payload(gray, 64, 64).unwrap();
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
        encode_obu_payload_with(&gray, 16, 16, false).unwrap();
        encode_obu_payload_with(&gray, 16, 16, true).unwrap();
        encode_gray(&gray, 16, 16).unwrap();
    }

    #[test]
    fn invalid_inputs_return_invalid_input() {
        use std::io::ErrorKind;
        // Non-16-aligned dimensions.
        let gray = vec![0u8; 30 * 16];
        let err = encode_gray(&gray, 30, 16).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        // Length mismatch (64x64 needs 4096 bytes).
        let short = vec![0u8; 100];
        let err = encode_gray(&short, 64, 64).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        // Too many colors in one 16x16 block (5 distinct levels).
        let mut many = vec![0u8; 64 * 64];
        for y in 0..16 {
            for x in 0..16 {
                many[y * 64 + x] = (x % 5) as u8;
            }
        }
        let err = encode_gray(&many, 64, 64).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        // Zero dimensions.
        let empty: Vec<u8> = vec![];
        let err = encode_gray(&empty, 0, 0).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }
}
