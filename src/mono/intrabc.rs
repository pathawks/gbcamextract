//! IntraBC (intra block copy) for monochrome stills, segregated from the
//! palette path: exact pixel-rectangle copies from causal decoded area via
//! integer motion vectors plus `skip = 1` (no residuals ever).
//!
//! Mirrors dav1d's intrabc path: `allow_intrabc` gating (obu.c), the
//! `intra = !intrabc-flag` decision (decode.c), the `{0,-1}`-refpair
//! candidate search in full (refmvs.c primary top/left/top-right, corner,
//! secondary rows/columns, then the two weight bubble sorts), the
//! entry-0/default predictor selection, integer-only `read_mv_residual`
//! (mv_prec = -1, so no fractional symbols), and the SB-overlap/tile-clip
//! validity rules. Non-intrabc neighbours contribute nothing (dav1d splats
//! `INVALID_MV` for them via `splat_intraref`).
//!
//! MV units are 1/8 pel (`i32`); we only emit multiples of 32 (4px grid)
//! so every source rect stays MI-aligned and the decoded-bitmap check is
//! exact.

use super::AdaptCdf;
use gamut_bitstream::SymbolEncoder;

// ---------------------------------------------------------------------------
// Default CDF tables (AV1 §9.3/§9.4, verified against dav1d's `cdf.c`).
// ---------------------------------------------------------------------------

/// `Default_Intrabc_Cdf`: single adapting row, no context.
const INTRABC_FLAG: [u16; 2] = [30531, 32768];

/// `Default_Mv_Joint_Cdf`: ZERO, H, V, HV.
const MV_JOINT: [u16; 4] = [4096, 11264, 19328, 32768];

/// `Default_Mv_Comp.Classes_Cdf` (11 values: classes 0..10).
const MV_CLASSES: [u16; 11] = [
    28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757, 32762, 32767, 32768,
];

/// `Default_Mv_Comp.Class0_Cdf`.
const MV_CLASS0: [u16; 2] = [27648, 32768];

/// `Default_Mv_Comp.ClassN_Cdf[n]`, one row per bit position.
const MV_CLASSN: [[u16; 2]; 10] = [
    [17408, 32768],
    [17920, 32768],
    [18944, 32768],
    [20480, 32768],
    [22528, 32768],
    [24576, 32768],
    [28672, 32768],
    [29952, 32768],
    [29952, 32768],
    [30720, 32768],
];

/// `Default_Mv_Comp.Sign_Cdf` (shared by both components here).
const MV_SIGN: [u16; 2] = [16384, 32768];

/// `read_mv_residual` with mv_prec = -1 (force-integer path) never touches
/// the fractional tables, so they are not transcribed.

// ---------------------------------------------------------------------------
// Motion-vector component CDFs (one set per component: 0 = vertical).
// ---------------------------------------------------------------------------

struct MvComp {
    sign: AdaptCdf,
    classes: AdaptCdf,
    class0: AdaptCdf,
    classn: [AdaptCdf; 10],
}

impl MvComp {
    fn new() -> Self {
        Self {
            sign: AdaptCdf::new(&MV_SIGN),
            classes: AdaptCdf::new(&MV_CLASSES),
            class0: AdaptCdf::new(&MV_CLASS0),
            classn: MV_CLASSN.map(|row| AdaptCdf::new(&row)),
        }
    }

    /// Code one nonzero residual component, a multiple of 8 (integer pel
    /// in 1/8-pel units), mirroring `read_mv_component_diff` with
    /// mv_prec < 0 (`diff = ((up << 3) | 0b111) + 1`, signed).
    fn encode(&mut self, sym: &mut SymbolEncoder, d: i32) {
        debug_assert!(d != 0 && d % 8 == 0);
        let m = (d.abs() / 8 - 1) as u32;
        self.sign.encode(sym, usize::from(d < 0));
        let cl = if m == 0 { 0 } else { m.ilog2() };
        debug_assert!(cl <= 10);
        self.classes.encode(sym, cl as usize);
        if cl == 0 {
            self.class0.encode(sym, m as usize);
        } else {
            let rem = m - (1 << cl);
            for n in 0..cl {
                self.classn[n as usize].encode(sym, ((rem >> n) & 1) as usize);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-MI records for predictor replication (dav1d's `rt` grid, simplified:
// absolute frame coords, no 32-entry wrapping since we only read the
// immediate above/left rows).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct MvRec {
    intrabc: bool,
    my: i32,
    mx: i32,
    bw4: usize,
    bh4: usize,
}

/// Search radius cap, in 4px steps (±256px). Bounds MV class widths and
/// keeps the match search cheap; misses just fall back to palette.
const SEARCH_RINGS: i32 = 64;

/// Pixel-match step: 4px grid keeps every source rect MI-aligned.
const MATCH_STEP: i32 = 4;

pub struct IntrabcState {
    flag: AdaptCdf,
    joint: AdaptCdf,
    comp: [MvComp; 2],
    cols: usize,
    rows: usize,
    rec: Vec<MvRec>,
    decoded: Vec<bool>,
}

impl IntrabcState {
    pub fn new(cols: usize, rows: usize) -> Self {
        let blank = MvRec {
            intrabc: false,
            my: 0,
            mx: 0,
            bw4: 4,
            bh4: 4,
        };
        Self {
            flag: AdaptCdf::new(&INTRABC_FLAG),
            joint: AdaptCdf::new(&MV_JOINT),
            comp: [MvComp::new(), MvComp::new()],
            cols,
            rows,
            rec: vec![blank; cols * rows],
            decoded: vec![false; cols * rows],
        }
    }

    pub fn flag(&mut self, sym: &mut SymbolEncoder, use_bc: bool) {
        self.flag.encode(sym, usize::from(use_bc));
    }

    fn at(&self, r: usize, c: usize) -> MvRec {
        self.rec[r * self.cols + c]
    }

    /// dav1d `add_spatial_candidate` for our `{0,-1}` refpair with `mf = 0`:
    /// intrabc neighbours merge into the stack by value (dups bump the
    /// weight, which decides the final order); anything else contributes
    /// nothing. At most 8 entries (later distinct values are dropped).
    fn consider(
        &self,
        stack: &mut Vec<((i32, i32), u32)>,
        r: usize,
        c: usize,
        weight: u32,
    ) {
        let rec = self.at(r, c);
        if !rec.intrabc {
            return;
        }
        let mv = (rec.my, rec.mx);
        if let Some(entry) = stack.iter_mut().find(|(m, _)| *m == mv) {
            entry.1 += weight;
        } else if stack.len() < 8 {
            stack.push((mv, weight));
        }
    }

    /// Bounds-checked candidate probe: out-of-frame cells (reachable only
    /// via the secondary odd-col overshoot at the right tile edge)
    /// contribute nothing, matching the benign padded reads in practice.
    fn consider_opt(
        &self,
        stack: &mut Vec<((i32, i32), u32)>,
        r: usize,
        c: usize,
        weight: u32,
    ) {
        if r < self.rows && c < self.cols {
            self.consider(stack, r, c, weight);
        }
    }

    /// One `scan_row` replication: single fast-path add when the span fits
    /// the first neighbour, else a walk stepping by neighbour widths.
    /// Returns dav1d's `n_rows` contribution (`weight >> 1` with
    /// `weight = max(2, min(2*max_rows, first_h))`, else 1 after a walk).
    fn scan_row_at(
        &self,
        stack: &mut Vec<((i32, i32), u32)>,
        r: usize,
        c: usize,
        bw4: usize,
        w4: usize,
        max_rows: u32,
        step: usize,
    ) -> u32 {
        let first = self.at(r, c);
        if bw4 <= first.bw4 {
            let len = step.max(bw4.min(first.bw4)) as u32;
            let weight = 2.max((2 * max_rows).min(first.bh4 as u32));
            self.consider(stack, r, c, len * weight);
            return weight >> 1;
        }
        let mut len = step.max(bw4.min(first.bw4));
        let mut x = 0;
        loop {
            self.consider_opt(stack, r, c + x, (len * 2) as u32);
            x += len;
            if x >= w4 {
                break;
            }
            if c + x >= self.cols {
                break;
            }
            len = step.max(self.at(r, c + x).bw4);
        }
        1
    }

    /// Mirror of `scan_row_at` down the left column.
    fn scan_col_at(
        &self,
        stack: &mut Vec<((i32, i32), u32)>,
        r: usize,
        c: usize,
        bh4: usize,
        h4: usize,
        max_cols: u32,
        step: usize,
    ) -> u32 {
        let first = self.at(r, c);
        if bh4 <= first.bh4 {
            let len = step.max(bh4.min(first.bh4)) as u32;
            let weight = 2.max((2 * max_cols).min(first.bw4 as u32));
            self.consider(stack, r, c, len * weight);
            return weight >> 1;
        }
        let mut len = step.max(bh4.min(first.bh4));
        let mut y = 0;
        loop {
            self.consider_opt(stack, r + y, c, (len * 2) as u32);
            y += len;
            if y >= h4 {
                break;
            }
            if r + y >= self.rows {
                break;
            }
            len = step.max(self.at(r + y, c).bh4);
        }
        1
    }

    /// Bubble-sort `stack[begin..end]` by descending weight (stable:
    /// strict `<` only), mirroring dav1d's two sort regions.
    fn sort_region(stack: &mut [((i32, i32), u32)], begin: usize, end: usize) {
        let mut len = end - begin;
        while len > 0 {
            let mut last = 0;
            for n in 1..len {
                if stack[begin + n - 1].1 < stack[begin + n].1 {
                    stack.swap(begin + n - 1, begin + n);
                    last = n;
                }
            }
            len = last;
        }
    }

    /// Replicate `dav1d_refmvs_find` for refpair `{0,-1}` on a still
    /// (no temporal, no gmv, `mf = 0` everywhere) in full: primary top
    /// row and left column, the above-right neighbour when the edge flag
    /// allows it, the above-left cell when a primary scan ran and the
    /// cell is in-frame, then the secondary 8x8-resolution rows/columns
    /// for `n = 2..=3` gated on the accumulated scan returns, then
    /// dav1d's two weight bubble sorts (`[0, nearest_cnt)` and the tail;
    /// the +640 boost is uniform within region 1, hence order-neutral).
    /// Returns the decoder's predictor: first nonzero
    /// stack entry, else the position-dependent default — plus whether
    /// entry 0 is usable. An empty search falls back to the default
    /// (deterministic in every conformant decoder: libaom computes
    /// `av1_find_ref_dv` with identical constants); a present-but-zero
    /// entry 0 can only come from a `(0, 0)` MV this encoder never emits,
    /// so it alone gates IntraBC off.
    pub fn predictor(
        &self,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
        top_has_right: bool,
    ) -> ((i32, i32), bool) {
        let mut stack: Vec<((i32, i32), u32)> = Vec::new();
        let w4 = bw4.min(16).min(self.cols.saturating_sub(c));
        let h4 = bh4.min(16).min(self.rows.saturating_sub(r));
        let mut n_rows: u32 = u32::MAX;
        let mut max_rows: u32 = 0;
        if r > 0 {
            max_rows = (((r + 1) >> 1).min(3)) as u32;
            let step = if bw4 >= 16 { 4 } else { 1 };
            n_rows = self.scan_row_at(&mut stack, r - 1, c, bw4, w4, max_rows, step);
        }
        let mut n_cols: u32 = u32::MAX;
        let mut max_cols: u32 = 0;
        if c > 0 {
            max_cols = (((c + 1) >> 1).min(3)) as u32;
            let step = if bh4 >= 16 { 4 } else { 1 };
            n_cols = self.scan_col_at(&mut stack, r, c - 1, bh4, h4, max_cols, step);
        }
        if r > 0 && top_has_right && bw4.max(bh4) <= 16 && c + bw4 < self.cols {
            self.consider(&mut stack, r - 1, c + bw4, 4);
        }
        let nearest_cnt = stack.len();
        // Top-left corner cell (`b_top[-1]`): only when some primary scan
        // ran (else dav1d's pointer is uninitialized) and fully in-frame.
        if (n_rows | n_cols) != u32::MAX && r > 0 && c > 0 {
            self.consider(&mut stack, r - 1, c - 1, 4);
        }
        // Secondary 8x8-resolution rows/columns.
        for n in 2u32..=3 {
            if n > n_rows && n <= max_rows {
                let rr = ((r as i32 - 2 * n as i32 + 1) | 1).max(0) as usize;
                let cc = (c as i32 | 1).max(0) as usize;
                n_rows += self.scan_row_at(
                    &mut stack,
                    rr,
                    cc,
                    bw4,
                    w4,
                    1 + max_rows - n,
                    if bw4 >= 16 { 4 } else { 2 },
                );
            }
            if n > n_cols && n <= max_cols {
                let rr = (r as i32 | 1).max(0) as usize;
                let cc = ((c as i32 - 2 * n as i32 + 1) | 1).max(0) as usize;
                n_cols += self.scan_col_at(
                    &mut stack,
                    rr,
                    cc,
                    bh4,
                    h4,
                    1 + max_cols - n,
                    if bh4 >= 16 { 4 } else { 2 },
                );
            }
        }
        // dav1d's two bubble sorts: [0, nearest_cnt), then the tail.
        // Only relative order within each region matters for entry 0.
        let split = nearest_cnt.min(stack.len());
        let total = stack.len();
        Self::sort_region(&mut stack, 0, split);
        Self::sort_region(&mut stack, split, total);
        let first = stack.first().map(|(m, _)| *m).unwrap_or((0, 0));
        if first != (0, 0) {
            return (first, true);
        }
        // Empty search with no usable entry 0: every conformant decoder
        // falls back to the position-dependent default (libaom computes
        // `av1_find_ref_dv` explicitly — same constants; dav1d reads
        // benign stack garbage that lands at zero in practice, and aomenc
        // itself ships such blocks). A present-but-zero entry 0 can only
        // come from a `(0, 0)` MV this encoder never emits, so it stays a
        // palette fallback out of paranoia.
        let fallback = if (r as i32) < 16 {
            (0, -2560)
        } else {
            (-512, 0)
        };
        if stack.is_empty() {
            return (fallback, true);
        }
        (fallback, false)
    }

    /// Code the residual `M - P` (both integer-pel, i.e. multiples of 8),
    /// mirroring `read_mv_residual`: joint symbol selecting nonzero
    /// components, then one integer-only component diff each.
    pub fn encode_mvd(
        &mut self,
        sym: &mut SymbolEncoder,
        my: i32,
        mx: i32,
        py: i32,
        px: i32,
    ) {
        let dy = my - py;
        let dx = mx - px;
        debug_assert!(dy % 8 == 0 && dx % 8 == 0);
        let joint = match (dx == 0, dy == 0) {
            (true, true) => 0,
            (false, true) => 1,
            (true, false) => 2,
            (false, false) => 3,
        };
        self.joint.encode(sym, joint);
        if dy != 0 {
            self.comp[0].encode(sym, dy);
        }
        if dx != 0 {
            self.comp[1].encode(sym, dx);
        }
    }

    /// Rect fully inside the frame and marked decoded (all rects here are
    /// MI-aligned: 4px-grid search, MI-sized blocks).
    fn available(&self, x0: i32, y0: i32, wpx: i32, hpx: i32, w: i32, h: i32) -> bool {
        if x0 < 0 || y0 < 0 || x0 + wpx > w || y0 + hpx > h {
            return false;
        }
        for y in (y0 / 4)..((y0 + hpx) / 4) {
            for x in (x0 / 4)..((x0 + wpx) / 4) {
                if !self.decoded[y as usize * self.cols + x as usize] {
                    return false;
                }
            }
        }
        true
    }

    /// Pixels of the candidate source rect equal the block's pixels.
    fn matches(
        &self,
        px: &[u8],
        img_w: usize,
        sx: i32,
        sy: i32,
        x0: i32,
        y0: i32,
        wpx: i32,
        hpx: i32,
    ) -> bool {
        for dy in 0..hpx {
            let a = ((sy + dy) as usize) * img_w + sx as usize;
            let b = ((y0 + dy) as usize) * img_w + x0 as usize;
            if px[a..a + wpx as usize] != px[b..b + wpx as usize] {
                return false;
            }
        }
        true
    }

    /// Find an exact-matching source rect for the block at MI `(r, c)`
    /// (pixel size `bw4*4`), or `None`. Candidates spiral outward from the
    /// predictor (cheapest residuals first) on the 4px grid. Validity is
    /// conservative against dav1d's clip/overlap rules: fully above the
    /// current 64px SB row, or same-row slab strictly left of the SB —
    /// both fully decoded — so the decoder's adjustments never trigger
    /// and erroring overlap is impossible. Returns the MV in 1/8-pel units.
    pub fn find_match(
        &self,
        px: &[u8],
        img_w: usize,
        img_h: usize,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
        top_has_right: bool,
    ) -> Option<(i32, i32)> {
        let (wpx, hpx) = (bw4 as i32 * 4, bh4 as i32 * 4);
        let (sx, sy) = (c as i32 * 4, r as i32 * 4);
        let (sbx, sby) = (
            (c as i32 / 16) * 64,
            (r as i32 / 16) * 64,
        );
        // Gate: without a usable replicated entry 0 the decoder would
        // draw on the unreplicated tail (corner-garbage/secondary rows),
        // so fall back to palette instead of risking divergence.
        let ((py, px_), ok) = self.predictor(r, c, bw4, bh4, top_has_right);
        if !ok {
            return None;
        }
        // Residual rings (square, 4px steps): ring 0 is the predictor
        // itself (zero residual, cheapest possible).
        for k in 0..=SEARCH_RINGS {
            let mut ring: Vec<(i32, i32)> = Vec::new();
            if k == 0 {
                ring.push((0, 0));
            } else {
                let s = k * MATCH_STEP;
                for i in -k..=k {
                    ring.push((i * MATCH_STEP, -s));
                    ring.push((i * MATCH_STEP, s));
                    if i != -k && i != k {
                        ring.push((-s, i * MATCH_STEP));
                        ring.push((s, i * MATCH_STEP));
                    }
                }
            }
            for (rdx, rdy) in ring {
                // Decoder convention (dav1d): source = current + MV,
                // so MV = source - current, in 1/8-pel units.
                let my = py + rdy * 8;
                let mx = px_ + rdx * 8;
                let x0 = sx + mx / 8;
                let y0 = sy + my / 8;
                let ok_pos = x0 >= 0
                    && y0 >= 0
                    && x0 + wpx <= img_w as i32
                    && y0 + hpx <= img_h as i32;
                if !ok_pos {
                    continue;
                }
                let above = y0 + hpx <= sby;
                let left_slab =
                    y0 >= sby && y0 + hpx <= sby + 64 && x0 + wpx <= sbx;
                if !(above || left_slab) {
                    continue;
                }
                if !self.available(x0, y0, wpx, hpx, img_w as i32, img_h as i32) {
                    continue;
                }
                if self.matches(px, img_w, sx, sy, x0, y0, wpx, hpx) {
                    return Some((my, mx));
                }
            }
        }
        None
    }

    /// Record a coded block over its MI footprint (both paths: palette
    /// blocks splat non-intrabc records exactly like dav1d's
    /// `splat_intraref`, intrabc blocks their MV) and mark it decoded.
    /// Call sites follow decode order (the partition recursion), so the
    /// decoded bitmap always matches what the decoder has available.
    pub fn record(
        &mut self,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
        mv: Option<(i32, i32)>,
    ) {
        for y in 0..bh4 {
            for x in 0..bw4 {
                let i = (r + y) * self.cols + (c + x);
                self.decoded[i] = true;
                self.rec[i] = MvRec {
                    intrabc: mv.is_some(),
                    my: mv.map(|m| m.0).unwrap_or(0),
                    mx: mv.map(|m| m.1).unwrap_or(0),
                    bw4,
                    bh4,
                };
            }
        }
    }
}
