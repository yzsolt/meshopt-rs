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

/// Quantizes a float into half-precision (as defined by IEEE-754 fp16) floating point value.
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

/// Quantizes a float into a floating point value with a limited number of significant mantissa bits, preserving the IEEE-754 fp32 binary representation
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

/// Reverse quantization of a half-precision (as defined by IEEE-754 fp16) floating point value
///
/// Preserves Inf/NaN, flushes denormals to zero
pub fn dequantize_half(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let em = (h & 0x7fff) as u32;

    // bias exponent and pad mantissa with 0; 112 is relative exponent bias (127-15)
    let mut r = (em + (112 << 10)) << 13;

    // denormal: flush to zero
    r = if em < (1 << 10) { 0 } else { r };

    // infinity/NaN; note that we preserve NaN payload as a byproduct of unifying inf/nan cases
    // 112 is an exponent bias fixup; since we already applied it once, applying it twice converts 31 to 255
    r += if em >= (31 << 10) { 112 << 23 } else { 0 };

    let ui = s | r;

    f32::from_bits(ui)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_quantize_float() {
        assert_eq!(quantize_float(1.2345, 23), 1.2345);

        assert_eq!(quantize_float(1.2345, 16), 1.2344971);
        assert_eq!(quantize_float(1.2345, 8), 1.2343750);
        assert_eq!(quantize_float(1.2345, 4), 1.25);
        assert_eq!(quantize_float(1.2345, 1), 1.0);

        assert_eq!(quantize_float(1.0, 0), 1.0);

        assert_eq!(quantize_float(1.0 / 0.0, 0), 1.0 / 0.0);
        assert_eq!(quantize_float(-1.0 / 0.0, 0), -1.0 / 0.0);

        let nanf = quantize_float(0.0 / 0.0, 8);
        assert!(nanf.is_nan());
    }

    #[test]
    fn test_quantize_half() {
        // normal
        assert_eq!(quantize_half(1.2345), 0x3cf0);

        // overflow
        assert_eq!(quantize_half(65535.0), 0x7c00);
        assert_eq!(quantize_half(-65535.0), 0xfc00);

        // large
        assert_eq!(quantize_half(65000.0), 0x7bef);
        assert_eq!(quantize_half(-65000.0), 0xfbef);

        // small
        assert_eq!(quantize_half(0.125), 0x3000);
        assert_eq!(quantize_half(-0.125), 0xb000);

        // very small
        assert_eq!(quantize_half(1e-4), 0x068e);
        assert_eq!(quantize_half(-1e-4), 0x868e);

        // underflow
        assert_eq!(quantize_half(1e-5), 0x0000);
        assert_eq!(quantize_half(-1e-5), 0x8000);

        // exponent underflow
        assert_eq!(quantize_half(1e-20), 0x0000);
        assert_eq!(quantize_half(-1e-20), 0x8000);

        // exponent overflow
        assert_eq!(quantize_half(1e20), 0x7c00);
        assert_eq!(quantize_half(-1e20), 0xfc00);

        // inf
        assert_eq!(quantize_half(1.0 / 0.0), 0x7c00);
        assert_eq!(quantize_half(-1.0 / 0.0), 0xfc00);

        // nan
        let nanh = quantize_half(0.0 / 0.0);
        assert!(nanh == 0x7e00 || nanh == 0xfe00);
    }

    #[test]
    fn test_dequantize_half() {
        // normal
        assert_eq!(dequantize_half(0x3cf0), 1.234375);

        // large
        assert_eq!(dequantize_half(0x7bef), 64992.0);
        assert_eq!(dequantize_half(0xfbef), -64992.0);

        // small
        assert_eq!(dequantize_half(0x3000), 0.125);
        assert_eq!(dequantize_half(0xb000), -0.125);

        // very small
        assert_eq!(dequantize_half(0x068e), 1.00016594e-4);
        assert_eq!(dequantize_half(0x868e), -1.00016594e-4);

        // denormal
        assert_eq!(dequantize_half(0x00ff), 0.0);
        assert_eq!(dequantize_half(0x80ff), 0.0); // actually this is -0.0
        assert_eq!(1.0 / dequantize_half(0x80ff), -1.0 / 0.0);

        // inf
        assert_eq!(dequantize_half(0x7c00), 1.0 / 0.0);
        assert_eq!(dequantize_half(0xfc00), -1.0 / 0.0);

        // nan
        let nanf = dequantize_half(0x7e00);
        assert!(nanf.is_nan());
    }
}
