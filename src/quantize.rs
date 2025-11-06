/// Quantizes a float in [0..1] range into an n-bit fixed point unorm value.
///
/// Assumes reconstruction function `q / (2^n-1)`, which is the case for fixed-function normalized fixed point conversion.
///
/// Maximum reconstruction error: `1/2^(n+1)`
pub fn quantize_unorm(mut v: f32, n: i32) -> i32 {
    let scale = ((1 << n) - 1) as f32;

    v = if v >= 0.0 { v } else { 0.0 };
    v = if v <= 1.0 { v } else { 1.0 };

    (v * scale + 0.5) as i32
}

/// Quantizes a float in [-1..1] range into an n-bit fixed point snorm value.
///
/// Assumes reconstruction function `q / (2^(n-1)-1)`, which is the case for fixed-function normalized fixed point conversion (except early OpenGL versions).
///
/// Maximum reconstruction error: `1/2^n`
pub fn quantize_snorm(mut v: f32, n: u32) -> i32 {
    let scale = ((1 << (n - 1)) - 1) as f32;

    let round = if v >= 0.0 { 0.5 } else { -0.5 };

    v = if v >= -1.0 { v } else { -1.0 };
    v = if v <= 1.0 { v } else { 1.0 };

    (v * scale + round) as i32
}

/// Quantizes a float into half-precision floating point value.
///
/// Generates +-inf for overflow, preserves NaN, flushes denormals to zero, rounds to nearest.
///
/// Representable magnitude range: `[6e-5; 65504]`
///
/// Maximum relative reconstruction error: `5e-4`
pub fn quantize_half(v: f32) -> u16 {
    let ui: u32 = f32::to_bits(v);

    let s = (ui >> 16) & 0x8000;
    let em = ui & 0x7fffffff;

    /* bias exponent and round to nearest; 112 is relative exponent bias (127-15) */
    let mut h = (em.wrapping_sub(112 << 23).wrapping_add(1 << 12)) >> 13;

    /* underflow: flush to zero; 113 encodes exponent -14 */
    h = if em < (113 << 23) { 0 } else { h };

    /* overflow: infinity; 143 encodes exponent 16 */
    h = if em >= (143 << 23) { 0x7c00 } else { h };

    /* NaN; note that we convert all types of NaN to qNaN */
    h = if em > (255 << 23) { 0x7e00 } else { h };

    (s | h) as u16
}

/// Quantizes a float into a floating point value with a limited number of significant mantissa bits.
///
/// Generates +-inf for overflow, preserves NaN, flushes denormals to zero, rounds to nearest.
///
/// Assumes `n` is in a valid mantissa precision range, which is 1..23
pub fn quantize_float(v: f32, n: i32) -> f32 {
    let mut ui: u32 = f32::to_bits(v);

    let mask: u32 = (1 << (23 - n)) - 1;
    let round = (1 << (23 - n)) >> 1;

    let e = ui & 0x7f800000;
    let rui = (ui + round) & (!mask);

    // round all numbers except inf/nan; this is important to make sure nan doesn't overflow into -0
    ui = if e == 0x7f800000 { ui } else { rui };

    // flush denormals to zero
    ui = if e == 0 { 0 } else { ui };

    f32::from_bits(ui)
}
