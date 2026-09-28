//! AV1 coded-lossless 4×4 luma residual primitives.
//!
//! This module contains the reversible WHT required when `base_q_idx == 0` and the
//! coefficient syntax for one `TX_4X4` luma transform block. It does not emit block
//! mode or transform-size syntax. An integrating caller must code the block's mode,
//! provide the `txb_skip` and `dc_sign` contexts from its current tile state, and
//! save/restore [`CoeffCdfs`] together with the range coder during speculative search.
//!
//! Coefficient CDF literals are the AV1 specification's qctx-0, luma (`ptype = 0`)
//! rows for `TX_4X4` (§9.4). Lossless qindex 0 selects qctx 0. Adaptation is performed
//! by `SymbolEncoder::encode_symbol_adapt`, matching `disable_cdf_update = 0`.

use gamut_bitstream::SymbolEncoder;
use std::io;

const SCAN_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const SIG_REF_DIFF_OFFSET_2D: [(usize, usize); 5] = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)];
const MAG_REF_OFFSET_2D: [(usize, usize); 3] = [(0, 1), (1, 0), (1, 1)];
const COEFF_BASE_CTX_OFFSET_4X4: [[u8; 5]; 5] = [
    [0, 1, 6, 6, 0],
    [1, 6, 6, 21, 0],
    [6, 6, 21, 21, 0],
    [6, 21, 21, 21, 0],
    [0, 0, 0, 0, 0],
];

// AV1 §9.4 Default_*_Cdf rows for TX_4X4, qctx 0, luma. Each row is a
// cumulative CDF whose last entry is 32768.
const TXB_SKIP_CDF: [[u16; 2]; 13] = [
    [31849, 32768],
    [5892, 32768],
    [12112, 32768],
    [21935, 32768],
    [20289, 32768],
    [27473, 32768],
    [32487, 32768],
    [7654, 32768],
    [19473, 32768],
    [29984, 32768],
    [9961, 32768],
    [30242, 32768],
    [32117, 32768],
];
const EOB_PT_16_CDF: [u16; 5] = [840, 1039, 1980, 4895, 32768];
const EOB_EXTRA_CDF: [[u16; 2]; 9] = [
    [16961, 32768],
    [17223, 32768],
    [7621, 32768],
    [16384, 32768],
    [16384, 32768],
    [16384, 32768],
    [16384, 32768],
    [16384, 32768],
    [16384, 32768],
];
const COEFF_BASE_EOB_CDF: [[u16; 3]; 4] = [
    [17837, 29055, 32768],
    [29600, 31446, 32768],
    [30844, 31878, 32768],
    [24926, 28948, 32768],
];
const COEFF_BASE_CDF: [[u16; 4]; 42] = [
    [4034, 8930, 12727, 32768],
    [18082, 29741, 31877, 32768],
    [12596, 26124, 30493, 32768],
    [9446, 21118, 27005, 32768],
    [6308, 15141, 21279, 32768],
    [2463, 6357, 9783, 32768],
    [20667, 30546, 31929, 32768],
    [13043, 26123, 30134, 32768],
    [8151, 18757, 24778, 32768],
    [5255, 12839, 18632, 32768],
    [2820, 7206, 11161, 32768],
    [8192, 16384, 24576, 32768],
    [8192, 16384, 24576, 32768],
    [8192, 16384, 24576, 32768],
    [8192, 16384, 24576, 32768],
    [8192, 16384, 24576, 32768],
    [8192, 16384, 24576, 32768],
    [8192, 16384, 24576, 32768],
    [8192, 16384, 24576, 32768],
    [8192, 16384, 24576, 32768],
    [8192, 16384, 24576, 32768],
    [15736, 27553, 30604, 32768],
    [11210, 23794, 28787, 32768],
    [5947, 13874, 19701, 32768],
    [4215, 9323, 13891, 32768],
    [2833, 6462, 10059, 32768],
    [19605, 30393, 31582, 32768],
    [13523, 26252, 30248, 32768],
    [8446, 18622, 24512, 32768],
    [3818, 10343, 15974, 32768],
    [1481, 4117, 6796, 32768],
    [22649, 31302, 32190, 32768],
    [14829, 27127, 30449, 32768],
    [8313, 17702, 23304, 32768],
    [3022, 8301, 12786, 32768],
    [1536, 4412, 7184, 32768],
    [22354, 29774, 31372, 32768],
    [14723, 25472, 29214, 32768],
    [6673, 13745, 18662, 32768],
    [2068, 5766, 9322, 32768],
    [8192, 16384, 24576, 32768],
    [8192, 16384, 24576, 32768],
];
const COEFF_BR_CDF: [[u16; 4]; 21] = [
    [14298, 20718, 24174, 32768],
    [12536, 19601, 23789, 32768],
    [8712, 15051, 19503, 32768],
    [6170, 11327, 15434, 32768],
    [4742, 8926, 12538, 32768],
    [3803, 7317, 10546, 32768],
    [1696, 3317, 4871, 32768],
    [14392, 19951, 22756, 32768],
    [15978, 23218, 26818, 32768],
    [12187, 19474, 23889, 32768],
    [9176, 15640, 20259, 32768],
    [7068, 12655, 17028, 32768],
    [5656, 10442, 14472, 32768],
    [2580, 4992, 7244, 32768],
    [12136, 18049, 21426, 32768],
    [13784, 20721, 24481, 32768],
    [10836, 17621, 21900, 32768],
    [8372, 14444, 18847, 32768],
    [6523, 11779, 16000, 32768],
    [5337, 9898, 13760, 32768],
    [3034, 5860, 8462, 32768],
];
const DC_SIGN_CDF: [[u16; 2]; 3] = [[16000, 32768], [13056, 32768], [18816, 32768]];

const NUM_BASE_LEVELS: i32 = 2;
const COEFF_BASE_RANGE: i32 = 3;
const BR_SYMBOLS: usize = 4;
const COEFF_BASE_PLUS_RANGE: i32 = NUM_BASE_LEVELS + COEFF_BASE_RANGE * BR_SYMBOLS as i32;
const MAX_8BIT_LOSSLESS_COEFF: i32 = 4 * 4 * 255;

/// AV1 1-D inverse WHT butterfly (§7.13.2.10), with the specified input shift.
fn inverse_wht_1d(input: [i64; 4], shift: u32) -> [i64; 4] {
    let mut a = input[0] >> shift;
    let mut c = input[1] >> shift;
    let mut d = input[2] >> shift;
    let mut b = input[3] >> shift;
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    [a, b, c, d]
}

/// Inverts the unshifted inverse-WHT butterfly to obtain encoder coefficients.
fn forward_wht_1d(output: [i64; 4]) -> [i64; 4] {
    let a1 = output[0] + output[1];
    let d1 = output[3] - output[2];
    let e = (a1 - d1) >> 1;
    let input3 = e - output[1];
    let input1 = e - output[2];
    let input0 = a1 - input1;
    let input2 = d1 + input3;
    [input0, input1, input2, input3]
}

/// Forward lossless AV1 `TX_4X4` Walsh–Hadamard transform (§7.13.2.10).
///
/// `residual` contains signed 8-bit-plane residuals (normally in `-255..=255`).
/// The paired [`inverse_wht4x4`] reproduces every input sample exactly.
pub(super) fn forward_wht4x4(residual: &[i32; 16]) -> [i32; 16] {
    let mut columns = [[0i64; 4]; 4];
    for col in 0..4 {
        let source = [
            i64::from(residual[col]),
            i64::from(residual[4 + col]),
            i64::from(residual[8 + col]),
            i64::from(residual[12 + col]),
        ];
        columns[col] = forward_wht_1d(source);
    }

    let mut coefficients = [0i32; 16];
    for row in 0..4 {
        let source = [
            columns[0][row],
            columns[1][row],
            columns[2][row],
            columns[3][row],
        ];
        let transformed = forward_wht_1d(source);
        for col in 0..4 {
            coefficients[row * 4 + col] = transformed[col] as i32;
        }
    }
    coefficients
}

/// Reconstructs an 8-bit coded-lossless AV1 `TX_4X4` coefficient block (§7.12.3 and §7.13.3).
#[must_use]
pub(super) fn inverse_wht4x4(coefficients: &[i32; 16]) -> [i32; 16] {
    let mut rows = [[0i64; 4]; 4];
    for row in 0..4 {
        let input = [
            i64::from(coefficients[row * 4]) * 4,
            i64::from(coefficients[row * 4 + 1]) * 4,
            i64::from(coefficients[row * 4 + 2]) * 4,
            i64::from(coefficients[row * 4 + 3]) * 4,
        ];
        rows[row] = inverse_wht_1d(input, 2);
    }

    let mut residual = [0i32; 16];
    for col in 0..4 {
        let input = [rows[0][col], rows[1][col], rows[2][col], rows[3][col]];
        let reconstructed = inverse_wht_1d(input, 0);
        for row in 0..4 {
            residual[row * 4 + col] = reconstructed[row] as i32;
        }
    }
    residual
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AdaptiveCdf<const N: usize> {
    row: [u16; N],
    count: u16,
}

impl<const N: usize> AdaptiveCdf<N> {
    fn new(row: [u16; N]) -> Self {
        Self { row, count: 0 }
    }

    fn encode(&mut self, sym: Option<&mut SymbolEncoder>, cost: Option<&mut f64>, symbol: usize) {
        if let Some(cost) = cost {
            *cost += crate::mono::adapt_bits(&self.row, symbol);
        }
        if let Some(sym) = sym {
            sym.encode_symbol_adapt(symbol, &mut self.row, &mut self.count);
        } else {
            self.update(symbol);
        }
    }

    fn update(&mut self, symbol: usize) {
        let n = N;
        let rate = 3
            + u32::from(self.count > 15)
            + u32::from(self.count > 31)
            + (31 - (n as u32).leading_zeros()).min(2);
        let (_, body) = self.row.split_last_mut().expect("CDF row is non-empty");
        for value in &mut body[..symbol] {
            *value -= *value >> rate;
        }
        for value in &mut body[symbol..] {
            *value += ((1u16 << 15) - *value) >> rate;
        }
        self.count = (self.count + 1).min(32);
    }
}

/// Adaptive CDF state for qindex-0 luma `TX_4X4` coefficient coding.
///
/// Clone this as part of every speculative encoder snapshot. Rows are per context;
/// costs must be measured from the current row and state, never reused from a different
/// `CoeffCdfs` snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CoeffCdfs {
    txb_skip: [AdaptiveCdf<2>; 13],
    eob_pt: AdaptiveCdf<5>,
    eob_extra: [AdaptiveCdf<2>; 9],
    coeff_base_eob: [AdaptiveCdf<3>; 4],
    coeff_base: [AdaptiveCdf<4>; 42],
    coeff_br: [AdaptiveCdf<4>; 21],
    dc_sign: [AdaptiveCdf<2>; 3],
}

impl Default for CoeffCdfs {
    fn default() -> Self {
        Self {
            txb_skip: std::array::from_fn(|ctx| AdaptiveCdf::new(TXB_SKIP_CDF[ctx])),
            eob_pt: AdaptiveCdf::new(EOB_PT_16_CDF),
            eob_extra: std::array::from_fn(|ctx| AdaptiveCdf::new(EOB_EXTRA_CDF[ctx])),
            coeff_base_eob: std::array::from_fn(|ctx| AdaptiveCdf::new(COEFF_BASE_EOB_CDF[ctx])),
            coeff_base: std::array::from_fn(|ctx| AdaptiveCdf::new(COEFF_BASE_CDF[ctx])),
            coeff_br: std::array::from_fn(|ctx| AdaptiveCdf::new(COEFF_BR_CDF[ctx])),
            dc_sign: std::array::from_fn(|ctx| AdaptiveCdf::new(DC_SIGN_CDF[ctx])),
        }
    }
}

/// The coefficient summary written to AV1's above/left context arrays after one transform block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct CoeffContext {
    /// `culLevel`, the sum of coefficient magnitudes, capped at 63.
    pub(super) cul_level: u8,
    /// `dcCategory`: 0 for zero, 1 for negative, and 2 for positive DC.
    pub(super) dc_category: u8,
}

/// Emits the normative coefficient syntax for one 8-bit luma `TX_4X4` block at qindex 0.
///
/// `txb_skip_ctx` is the AV1 `txb_skip` context (0..13), and `dc_sign_ctx` is the
/// `dc_sign` context (0..3), both computed from the encoder's causal above/left state.
/// The returned summary must be written to that state even when the coefficient block is
/// all-zero. Coefficients outside the exact range possible for 8-bit source residuals are
/// rejected rather than silently truncated.
#[cfg(test)]
pub(super) fn encode_tx4x4_coefficients(
    sym: &mut SymbolEncoder,
    cdfs: &mut CoeffCdfs,
    coefficients: &[i32; 16],
    txb_skip_ctx: usize,
    dc_sign_ctx: usize,
) -> io::Result<CoeffContext> {
    encode_tx4x4_coefficients_with_cost(
        Some(sym),
        cdfs,
        coefficients,
        txb_skip_ctx,
        dc_sign_ctx,
        None,
    )
}

/// Cost-only counterpart to [`encode_tx4x4_coefficients`]. When `sym` is `None`,
/// only the adaptive CDF state and optional fractional-bit estimate are updated. This
/// lets speculative RDO trials price coefficients without mutating the live range coder.
pub(super) fn encode_tx4x4_coefficients_with_cost(
    sym: Option<&mut SymbolEncoder>,
    cdfs: &mut CoeffCdfs,
    coefficients: &[i32; 16],
    txb_skip_ctx: usize,
    dc_sign_ctx: usize,
    cost: Option<&mut f64>,
) -> io::Result<CoeffContext> {
    if txb_skip_ctx >= cdfs.txb_skip.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("TXB_SKIP context {txb_skip_ctx} is outside 0..13"),
        ));
    }
    if dc_sign_ctx >= cdfs.dc_sign.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("DC_SIGN context {dc_sign_ctx} is outside 0..3"),
        ));
    }
    if coefficients
        .iter()
        .any(|&coefficient| coefficient.unsigned_abs() > MAX_8BIT_LOSSLESS_COEFF as u32)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "TX_4X4 coefficient exceeds the 8-bit lossless residual bound",
        ));
    }

    let mut eob = 0usize;
    for (scan_index, &position) in SCAN_4X4.iter().enumerate() {
        if coefficients[position] != 0 {
            eob = scan_index + 1;
        }
    }
    let mut sym = sym;
    let mut cost = cost;
    cdfs.txb_skip[txb_skip_ctx].encode(
        sym.as_deref_mut(),
        cost.as_deref_mut(),
        usize::from(eob == 0),
    );
    if eob == 0 {
        return Ok(CoeffContext::default());
    }

    let eob_pt = eob_pt_from_eob(eob);
    cdfs.eob_pt
        .encode(sym.as_deref_mut(), cost.as_deref_mut(), eob_pt - 1);
    if eob_pt >= 3 {
        let nbits = eob_pt - 2;
        let base_eob = (1usize << (eob_pt - 2)) + 1;
        let extra = eob - base_eob;
        cdfs.eob_extra[eob_pt - 3].encode(
            sym.as_deref_mut(),
            cost.as_deref_mut(),
            (extra >> (nbits - 1)) & 1,
        );
        for bit in (0..nbits - 1).rev() {
            encode_literal(&mut sym, &mut cost, ((extra >> bit) & 1) as u32, 1);
        }
    }

    let mut levels = [0i32; 16];
    for scan_index in (0..eob).rev() {
        let position = SCAN_4X4[scan_index];
        let level = coefficients[position].abs();
        if scan_index == eob - 1 {
            cdfs.coeff_base_eob[coeff_base_eob_ctx(scan_index)].encode(
                sym.as_deref_mut(),
                cost.as_deref_mut(),
                (level.min(3) - 1) as usize,
            );
        } else {
            cdfs.coeff_base[coeff_base_ctx(position, &levels)].encode(
                sym.as_deref_mut(),
                cost.as_deref_mut(),
                level.min(3) as usize,
            );
        }
        if level > NUM_BASE_LEVELS {
            let mut remainder = level - (NUM_BASE_LEVELS + 1);
            let context = coeff_br_ctx(position, &levels);
            for _ in 0..BR_SYMBOLS {
                let br_value = remainder.min(COEFF_BASE_RANGE);
                cdfs.coeff_br[context].encode(
                    sym.as_deref_mut(),
                    cost.as_deref_mut(),
                    br_value as usize,
                );
                remainder -= br_value;
                if br_value < COEFF_BASE_RANGE {
                    break;
                }
            }
        }
        levels[position] = level;
    }

    for (scan_index, &position) in SCAN_4X4.iter().enumerate().take(eob) {
        let level = coefficients[position].abs();
        if level == 0 {
            continue;
        }
        let negative = coefficients[position] < 0;
        if scan_index == 0 {
            cdfs.dc_sign[dc_sign_ctx].encode(
                sym.as_deref_mut(),
                cost.as_deref_mut(),
                usize::from(negative),
            );
        } else {
            encode_literal(&mut sym, &mut cost, u32::from(negative), 1);
        }
        if level > COEFF_BASE_PLUS_RANGE {
            encode_golomb(&mut sym, &mut cost, (level - COEFF_BASE_PLUS_RANGE) as u32);
        }
    }

    let cul_level = levels.iter().sum::<i32>().min(63) as u8;
    let dc_category = match coefficients[0].cmp(&0) {
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 2,
    };
    Ok(CoeffContext {
        cul_level,
        dc_category,
    })
}

fn eob_pt_from_eob(eob: usize) -> usize {
    if eob <= 1 {
        eob
    } else {
        (usize::BITS - (eob - 1).leading_zeros()) as usize + 1
    }
}

fn coeff_base_eob_ctx(scan_index: usize) -> usize {
    if scan_index == 0 {
        0
    } else if scan_index <= 2 {
        1
    } else if scan_index <= 4 {
        2
    } else {
        3
    }
}

fn coeff_base_ctx(position: usize, levels: &[i32; 16]) -> usize {
    let row = position >> 2;
    let col = position & 3;
    let mut magnitude = 0i32;
    for &(dr, dc) in &SIG_REF_DIFF_OFFSET_2D {
        let (neighbor_row, neighbor_col) = (row + dr, col + dc);
        if neighbor_row < 4 && neighbor_col < 4 {
            magnitude += levels[neighbor_row * 4 + neighbor_col].min(3);
        }
    }
    let context = ((magnitude + 1) >> 1).min(4) as usize;
    if row == 0 && col == 0 {
        0
    } else {
        context + usize::from(COEFF_BASE_CTX_OFFSET_4X4[row.min(4)][col.min(4)])
    }
}

fn coeff_br_ctx(position: usize, levels: &[i32; 16]) -> usize {
    let row = position >> 2;
    let col = position & 3;
    let mut magnitude = 0i32;
    for &(dr, dc) in &MAG_REF_OFFSET_2D {
        let (neighbor_row, neighbor_col) = (row + dr, col + dc);
        if neighbor_row < 4 && neighbor_col < 4 {
            magnitude += levels[neighbor_row * 4 + neighbor_col].min(15);
        }
    }
    let magnitude = (((magnitude + 1) >> 1).min(6)) as usize;
    if position == 0 {
        magnitude
    } else if row < 2 && col < 2 {
        magnitude + 7
    } else {
        magnitude + 14
    }
}

fn encode_literal(
    sym: &mut Option<&mut SymbolEncoder>,
    cost: &mut Option<&mut f64>,
    value: u32,
    bits: u32,
) {
    if let Some(cost) = cost.as_deref_mut() {
        *cost += f64::from(bits);
    }
    if let Some(sym) = sym.as_deref_mut() {
        sym.encode_literal(value, bits);
    }
}

fn encode_golomb(sym: &mut Option<&mut SymbolEncoder>, cost: &mut Option<&mut f64>, value: u32) {
    debug_assert_ne!(value, 0);
    let bit_len = u32::BITS - value.leading_zeros();
    for _ in 0..(bit_len - 1) {
        encode_literal(sym, cost, 0, 1);
    }
    encode_literal(sym, cost, 1, 1);
    for bit in (0..bit_len - 1).rev() {
        encode_literal(sym, cost, (value >> bit) & 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_residual_uses_all_zero_token_and_zero_context() {
        let mut sym = SymbolEncoder::new();
        let mut cdfs = CoeffCdfs::default();
        let context = encode_tx4x4_coefficients(&mut sym, &mut cdfs, &[0; 16], 1, 0).unwrap();
        assert_eq!(context, CoeffContext::default());
        assert_eq!(cdfs.txb_skip[1].count, 1);
        assert_eq!(cdfs.eob_pt.count, 0);
        assert!(!sym.finish().is_empty());
    }

    #[test]
    fn constant_residual_has_only_dc_coefficient_and_roundtrips() {
        let residual = [17; 16];
        let coefficients = forward_wht4x4(&residual);
        assert_eq!(coefficients[0], 17 * 4);
        assert!(coefficients[1..]
            .iter()
            .all(|&coefficient| coefficient == 0));
        assert_eq!(inverse_wht4x4(&coefficients), residual);

        let mut sym = SymbolEncoder::new();
        let mut cdfs = CoeffCdfs::default();
        let context = encode_tx4x4_coefficients(&mut sym, &mut cdfs, &coefficients, 0, 0).unwrap();
        assert_eq!(context.cul_level, 63);
        assert_eq!(context.dc_category, 2);
        assert_eq!(cdfs.txb_skip[0].count, 1);
        assert_eq!(cdfs.eob_pt.count, 1);
        assert!(!sym.finish().is_empty());
    }

    #[test]
    fn impulse_residual_exercises_full_scan_signs_and_golomb_tail() {
        let mut residual = [0; 16];
        residual[5] = -255;
        let coefficients = forward_wht4x4(&residual);
        assert!(
            coefficients.iter().all(|&coefficient| coefficient != 0),
            "unexpected impulse coefficients: {coefficients:?}"
        );
        assert_eq!(inverse_wht4x4(&coefficients), residual);

        let mut sym = SymbolEncoder::new();
        let mut cdfs = CoeffCdfs::default();
        let context = encode_tx4x4_coefficients(&mut sym, &mut cdfs, &coefficients, 6, 2).unwrap();
        assert_eq!(context.cul_level, 63);
        assert_eq!(context.dc_category, 1);
        assert_eq!(cdfs.txb_skip[6].count, 1);
        assert_eq!(cdfs.eob_pt.count, 1);
        assert!(cdfs.coeff_base.iter().any(|cdf| cdf.count != 0));
        assert!(cdfs.coeff_br.iter().any(|cdf| cdf.count != 0));
        assert!(cdfs.dc_sign[2].count > 0);
        assert!(!sym.finish().is_empty());
    }

    #[test]
    fn reversible_wht_roundtrips_bounded_residuals() {
        let cases = [
            [0; 16],
            [
                255, -255, 0, 17, 85, -1, 2, -10, 44, 31, -70, 0, 9, 13, 21, -33,
            ],
            std::array::from_fn(|i| if i % 2 == 0 { 255 } else { -255 }),
        ];
        for residual in cases {
            assert_eq!(inverse_wht4x4(&forward_wht4x4(&residual)), residual);
        }
    }

    #[test]
    fn coefficient_cdfs_clone_for_state_replay() {
        let coefficients = forward_wht4x4(&[5; 16]);
        let initial = CoeffCdfs::default();

        let mut first_cdfs = initial.clone();
        let mut first_sym = SymbolEncoder::new();
        encode_tx4x4_coefficients(&mut first_sym, &mut first_cdfs, &coefficients, 0, 0).unwrap();
        let first_bytes = first_sym.finish();

        let mut replay_cdfs = initial;
        let mut replay_sym = SymbolEncoder::new();
        encode_tx4x4_coefficients(&mut replay_sym, &mut replay_cdfs, &coefficients, 0, 0).unwrap();
        assert_eq!(replay_sym.finish(), first_bytes);
        assert_eq!(replay_cdfs, first_cdfs);
    }

    #[test]
    fn cost_only_coefficients_replay_adaptive_state_without_a_coder() {
        let mut coefficients = [0; 16];
        coefficients[0] = 384;
        coefficients[5] = -48;
        let mut emitting_cdfs = CoeffCdfs::default();
        let mut emitting_sym = SymbolEncoder::new();
        let emitted =
            encode_tx4x4_coefficients(&mut emitting_sym, &mut emitting_cdfs, &coefficients, 4, 2)
                .unwrap();

        let mut trial_cdfs = CoeffCdfs::default();
        let mut cost = 0.0;
        let trial = encode_tx4x4_coefficients_with_cost(
            None,
            &mut trial_cdfs,
            &coefficients,
            4,
            2,
            Some(&mut cost),
        )
        .unwrap();
        assert_eq!(trial, emitted);
        assert_eq!(trial_cdfs, emitting_cdfs);
        assert!(cost > 0.0);
    }

    #[test]
    fn rejects_invalid_contexts_and_out_of_profile_coefficients() {
        let mut sym = SymbolEncoder::new();
        let mut cdfs = CoeffCdfs::default();
        assert_eq!(
            encode_tx4x4_coefficients(&mut sym, &mut cdfs, &[0; 16], 13, 0)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            encode_tx4x4_coefficients(&mut sym, &mut cdfs, &[0; 16], 0, 3)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        let mut too_large = [0; 16];
        too_large[0] = MAX_8BIT_LOSSLESS_COEFF + 1;
        assert_eq!(
            encode_tx4x4_coefficients(&mut sym, &mut cdfs, &too_large, 0, 0)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }
}
