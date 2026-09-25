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

use gamut_bitstream::{BitWriter, SymbolEncoder, write_leb128};
use gamut_isobmff::{
    EntityGroup, IsoBmffImage, Item, Property, PropertyKind, write as write_isobmff,
};
use std::io;

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
    15588, 17027, 19338, 20218, 20682, 21110, 21825, 23244, 24189, 28165, 29093, 30466,
    32768,
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
            pal_mode: PALETTE_Y_MODE
                .map(|rows| rows.map(|row| AdaptCdf::new(&row))),
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

    fn pal_idx(&mut self, n: usize, ctx: usize) -> &mut AdaptCdf {
        match n {
            2 => &mut self.pal_idx2[ctx],
            3 => &mut self.pal_idx3[ctx],
            _ => &mut self.pal_idx4[ctx],
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
    mi_cols: usize,
    mi_rows: usize,
    sym: SymbolEncoder,
    cdfs: Cdfs,
    ymode: Vec<u8>,
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
    fn new(px: &'a [u8], w: usize, h: usize) -> Self {
        assert!(w.is_multiple_of(16) && h.is_multiple_of(16), "dimensions must be 16-aligned");
        let mi_cols = w / 4;
        let mi_rows = h / 4;
        Self {
            px,
            w,
            mi_cols,
            mi_rows,
            sym: SymbolEncoder::new(),
            cdfs: Cdfs::new(),
            ymode: vec![0; mi_cols * mi_rows],
            skip: vec![0; mi_cols * mi_rows],
            psize: vec![0; mi_cols * mi_rows],
            pcolors: vec![[0u8; 8]; mi_cols * mi_rows],
            above_part: vec![0; mi_cols],
            left_part: vec![0; mi_rows],
        }
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

    /// Emit the partition tree. A fully on-screen 32x32 or 64x64 node
    /// whose distinct levels fit the palette (≤4 here — our index tables
    /// only cover sizes 2..4) is coded `NONE` directly; anything else
    /// splits down, with 16x16 (`bw4 == 4`) leaves as before. Offscreen
    /// subtrees emit nothing, and partially on-screen nodes take the
    /// existing edge (split_or / forced-split) path, never `NONE`.
    fn partition(&mut self, r: usize, c: usize, bw4: usize) {
        if r >= self.mi_rows || c >= self.mi_cols {
            return;
        }
        let bsl = bw4.trailing_zeros() as usize;
        if bw4 == 4 {
            let ctx = self.partition_ctx(r, c, bsl);
            self.cdfs.part(bsl, ctx).encode(&mut self.sym, PARTITION_NONE);
            self.update_partition_ctx(r, c, bsl);
            self.block(r, c, bsl);
            return;
        }
        if r + bw4 <= self.mi_rows && c + bw4 <= self.mi_cols && self.region_levels(r, c, bw4) <= 4
        {
            let ctx = self.partition_ctx(r, c, bsl);
            self.cdfs.part(bsl, ctx).encode(&mut self.sym, PARTITION_NONE);
            self.update_partition_ctx(r, c, bsl);
            self.block(r, c, bsl);
            return;
        }
        let half = bw4 >> 1;
        let has_rows = r + half < self.mi_rows;
        let has_cols = c + half < self.mi_cols;
        if has_rows && has_cols {
            let ctx = self.partition_ctx(r, c, bsl);
            self.cdfs.part(bsl, ctx).encode(&mut self.sym, PARTITION_SPLIT);
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
            let c0 = if psum == 0 { 32767 } else { (32768 - psum) as u16 };
            self.sym.encode_symbol(1, &[c0, 32768]);
        } else if has_rows {
            // Right edge: same arrangement with the vert masses.
            let ctx = self.partition_ctx(r, c, bsl);
            let psum = split_psum_vert(self.cdfs.part_row(bsl, ctx));
            debug_assert!(psum < 32768);
            let c0 = if psum == 0 { 32767 } else { (32768 - psum) as u16 };
            self.sym.encode_symbol(1, &[c0, 32768]);
        }
        // Forced SPLIT (neither flag) codes no symbol.
        self.partition(r, c, half);
        self.partition(r, c + half, half);
        self.partition(r + half, c, half);
        self.partition(r + half, c + half, half);
    }

    fn skip_ctx(&self, r: usize, c: usize) -> usize {
        let above = r > 0 && self.skip[(r - 1) * self.mi_cols + c] != 0;
        let left = c > 0 && self.skip[r * self.mi_cols + (c - 1)] != 0;
        usize::from(above) + usize::from(left)
    }

    /// `get_palette_cache` for luma: sorted dedup merge of the above
    /// (unless at a 64px superblock-row top) and left palettes.
    fn palette_cache(&self, r: usize, c: usize) -> Vec<u8> {
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
        let mut cache = Vec::new();
        let mut push = |v: u8| {
            if cache.last() != Some(&v) {
                cache.push(v);
            }
        };
        let (mut ai, mut li) = (0, 0);
        while ai < above_n && li < left_n {
            let (ac, lc) = (above[ai], left[li]);
            if lc < ac {
                push(lc);
                li += 1;
            } else {
                push(ac);
                ai += 1;
                if lc == ac {
                    li += 1;
                }
            }
        }
        while ai < above_n {
            push(above[ai]);
            ai += 1;
        }
        while li < left_n {
            push(left[li]);
            li += 1;
        }
        cache
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
    /// 32x32, 4 ⇒ 64x64): skip + DC + palette + indices. Palette mode is
    /// legal at all three sizes (dav1d gates it on `imax(bw4, bh4) <= 16`
    /// MI); the index tables only cover sizes 2..4, so callers must keep
    /// distinct levels ≤ 4.
    fn block(&mut self, r: usize, c: usize, bsl: usize) {
        let bw: usize = 4 << bsl;
        let (sx, sy) = (c * 4, r * 4);

        // Palette colors: sorted distinct levels (≤4 for 2-bit content).
        let mut set = [false; 256];
        for i in 0..bw {
            for j in 0..bw {
                set[self.sample(sx + j, sy + i) as usize] = true;
            }
        }
        let mut colors: Vec<u8> = (0..256).filter(|&v| set[v]).map(|v| v as u8).collect();
        assert!(!colors.is_empty() && colors.len() <= 8);
        // A flat block still needs a 2-entry table. Prefer padding with a
        // cached level (nearly free via a reuse flag below) over an
        // adjacent level (short delta chain); either way the pad value is
        // never referenced by the index map.
        let cache = self.palette_cache(r, c);
        if colors.len() == 1 {
            let v = colors[0];
            let pad = cache
                .iter()
                .find(|&&cc| cc != v)
                .copied()
                .unwrap_or(if v < 255 { v + 1 } else { v - 1 });
            colors.push(pad);
            colors.sort_unstable();
        }
        let psize = colors.len();

        let mut index_map = vec![0u8; bw * bw];
        for i in 0..bw {
            for j in 0..bw {
                let v = self.sample(sx + j, sy + i);
                index_map[i * bw + j] = colors.binary_search(&v).unwrap_or(0) as u8;
            }
        }

        // skip = 1 (no residual; reconstruction is exactly the palette).
        let sctx = self.skip_ctx(r, c);
        self.cdfs.skip[sctx].encode(&mut self.sym, 1);

        // y_mode = DC_PRED (all neighbours are DC, so contexts are row 0).
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
        let mut is_used = vec![false; psize];
        let mut n_used = 0usize;
        for &cc in &cache {
            if n_used == psize {
                break;
            }
            if let Ok(pos) = colors.binary_search(&cc) {
                self.sym.encode_literal(1, 1);
                is_used[pos] = true;
                n_used += 1;
            } else {
                self.sym.encode_literal(0, 1);
            }
        }
        let new_colors: Vec<u8> = colors
            .iter()
            .zip(is_used.iter())
            .filter(|(_, &u)| !u)
            .map(|(&c, _)| c)
            .collect();
        if n_used < psize {
            self.sym.encode_literal(u32::from(new_colors[0]), 8);
            if new_colors.len() > 1 {
                // Minimal initial width covering the first delta
                // (decoder computes bits = 5 + this field).
                let first_delta = new_colors[1] as u32 - new_colors[0] as u32 - 1;
                let need = if first_delta == 0 {
                    1
                } else {
                    first_delta.ilog2() + 1
                }
                .max(5)
                .min(8);
                self.sym.encode_literal(need - 5, 2);
                let mut palette_bits = need;
                for k in 1..new_colors.len() {
                    let delta = new_colors[k] as u32 - new_colors[k - 1] as u32 - 1;
                    self.sym.encode_literal(delta, palette_bits);
                    let prev = u32::from(new_colors[k]);
                    if prev + 1 >= 255 {
                        // Decoder fills any remaining slots with 255 and
                        // stops; only reachable for a trailing 255.
                        debug_assert!(k + 1 == new_colors.len());
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
                self.ymode[rr * self.mi_cols + cc] = DC_PRED as u8;
                self.skip[rr * self.mi_cols + cc] = 1;
                self.psize[rr * self.mi_cols + cc] = psize as u8;
                self.pcolors[rr * self.mi_cols + cc][..psize].copy_from_slice(&colors);
            }
        }

        // color_index_map_y: first index via ns(), rest in wavefront order
        // as positions in the neighbour-derived ColorOrder.
        self.encode_ns(index_map[0] as usize, psize);
        for i in 1..(2 * bw - 1) {
            let mut j = i.min(bw - 1);
            let j_end = i.saturating_sub(bw - 1);
            loop {
                let (rr, cc) = (i - j, j);
                let (order, ctx) = palette_color_context(&index_map, bw, rr, cc, psize);
                let actual = index_map[rr * bw + cc] as usize;
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
    }

    fn finish(mut self) -> Vec<u8> {
        for r in (0..self.mi_rows).step_by(16) {
            for c in (0..self.mi_cols).step_by(16) {
                self.partition(r, c, 16);
            }
        }
        self.sym.finish()
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
    } else {
        *bytes.last_mut().unwrap() |= 1 << (7 - rem);
    }
    bytes
}

/// Uncompressed frame header bits for our KEY still, then byte-aligned.
fn frame_header_bits() -> Vec<u8> {
    let mut bw = BitWriter::new();
    // (show_existing_frame/frame_type/show_frame implied by reduced header)
    bw.put_bit(0); // disable_cdf_update (CDFs adapt)
    bw.put_bit(1); // allow_screen_content_tools (seq forces SELECT)
    bw.put_bit(0); // force_integer_mv (overridden to 1 for intra)
    // frame_size_override implied 0 → dimensions from sequence max.
    // render_and_frame_size_different = 0:
    bw.put_bit(0);
    // allow_intrabc = 0:
    bw.put_bit(0);
    // (primary_ref NONE, tile_info, quant, segmentation, deltas,
    //  loopfilter/cdef/lr all implied off by KEY + CodedLossless)
    // tile_info: uniform spacing, single tile (both increments 0):
    bw.put_bit(1); // uniform_tile_spacing_flag
    bw.put_bit(0); // increment_tile_cols_log2 → break
    bw.put_bit(0); // increment_tile_rows_log2 → break
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
    bw.into_bytes()
}

// ---------------------------------------------------------------------------
// Public entry points.
// ---------------------------------------------------------------------------

/// Encode one raster to its AV1 item payload (sequence header OBU + frame
/// OBU wrapping the palette-coded tile data). Returns the payload and the
/// matching `av1C` body.
fn encode_obu_payload(gray: &[u8], w: u32, h: u32) -> (Vec<u8>, [u8; 4]) {
    assert_eq!(gray.len(), w as usize * h as usize);
    assert!(w.is_multiple_of(16) && h.is_multiple_of(16));

    let tile = TileEncoder::new(gray, w as usize, h as usize);
    let tile_data = tile.finish();

    let seq_obu = obu_wrap(1, &sequence_header_obu(w, h));
    let mut frame_payload = frame_header_bits();
    // Tile group OBU with a single tile: no header bits, byte-aligned
    // by construction, then the tile data. Together with the frame
    // header this forms an OBU_FRAME (type 6), not OBU_FRAME_HEADER.
    frame_payload.extend_from_slice(&tile_data);
    let frame_obu = obu_wrap(6, &frame_payload);
    let mut payload = seq_obu;
    payload.extend_from_slice(&frame_obu);

    let level_idx = seq_level_idx_for(w, h);
    (payload, av1c_mono8(level_idx))
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
#[allow(dead_code)]
pub fn encode_gray(gray: &[u8], w: u32, h: u32) -> io::Result<Vec<u8>> {
    let (payload, av1c) = encode_obu_payload(gray, w, h);

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
pub fn encode_gray_pair(
    small: &[u8],
    sw: u32,
    sh: u32,
    large: &[u8],
    lw: u32,
    lh: u32,
) -> io::Result<Vec<u8>> {
    let (small_payload, small_av1c) = encode_obu_payload(small, sw, sh);
    let (large_payload, large_av1c) = encode_obu_payload(large, lw, lh);

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
