//! Vertex buffer filters
//!
//! These functions can be used to filter output of [decode_vertex_buffer](crate::vertex::buffer::decode_vertex_buffer) in-place.
//!
//! Mainly useful in combination with the `EXT_meshopt_compression` glTF extension.

use crate::{quantize::quantize_snorm, util::zero_inverse};

macro_rules! decode_filter_oct_scalar_impl {
    ($dest:ty, $signed_dest:ty, $data:expr) => {
        const MAX: f32 = ((1 << (std::mem::size_of::<$dest>() * 8 - 1)) - 1) as f32;

        for v in $data {
            // convert x and y to floats and reconstruct z; this assumes zf encodes 1.0 at the same bit count
            let (mut x, mut y, mut z) = (
                v[0] as $signed_dest as f32,
                v[1] as $signed_dest as f32,
                v[2] as $signed_dest as f32,
            );

            z = z - x.abs() - y.abs();

            // fixup octahedral coordinates for z<0
            let t = z.min(0.0);

            x += if x >= 0.0 { t } else { -t };
            y += if y >= 0.0 { t } else { -t };

            // compute normal length & scale
            let l = (x * x + y * y + z * z).sqrt();
            let s = MAX / l;

            // rounded signed float->int
            let xf = (x * s + if x >= 0.0 { 0.5 } else { -0.5 }) as i32;
            let yf = (y * s + if y >= 0.0 { 0.5 } else { -0.5 }) as i32;
            let zf = (z * s + if z >= 0.0 { 0.5 } else { -0.5 }) as i32;

            v[0] = xf as $dest;
            v[1] = yf as $dest;
            v[2] = zf as $dest;
        }
    };
}

fn decode_filter_quat_scalar<'a>(data: impl Iterator<Item = &'a mut [u16; 4]>) {
    let scale = 1.0 / 2.0f32.sqrt();

    for q in data {
        // recover scale from the high byte of the component
        let sf = q[3] | 3;
        let ss = scale / sf as f32;

        // convert x/y/z to [-1..1] (scaled...)
        let x = q[0] as f32 * ss;
        let y = q[1] as f32 * ss;
        let z = q[2] as f32 * ss;

        // reconstruct w as a square root; we clamp to 0.0 to avoid NaN due to precision errors
        let ww = 1.0 - x * x - y * y - z * z;
        let w = ww.max(0.0).sqrt();

        // rounded signed float->int
        let xf = (x * 32767.0 + if x >= 0.0 { 0.5 } else { -0.5 }) as i32;
        let yf = (y * 32767.0 + if y >= 0.0 { 0.5 } else { -0.5 }) as i32;
        let zf = (z * 32767.0 + if z >= 0.0 { 0.5 } else { -0.5 }) as i32;
        let wf = (w * 32767.0 + 0.5) as i32;

        let qc = (q[3] & 3) as usize;

        // output order is dictated by input index
        q[(qc + 1) & 3] = (xf as i16) as u16;
        q[(qc + 2) & 3] = (yf as i16) as u16;
        q[(qc + 3) & 3] = (zf as i16) as u16;
        q[(qc + 0) & 3] = (wf as i16) as u16;
    }
}

fn decode_filter_exp_scalar<'a>(data: impl Iterator<Item = &'a mut u32>) {
    for v in data {
        // decode mantissa and exponent
        let m = (*v << 8) as i32 >> 8;
        let e = *v as i32 >> 24;

        union U {
            f: f32,
            ui: u32,
        }

        // optimized version of ldexp(float(m), e)
        let ui = ((e + 127) as u32) << 23;
        let mut u = U { ui };
        u.f = unsafe { u.f } * m as f32;

        *v = unsafe { u.ui };
    }
}

/// Decodes octahedral encoding of a unit vector with K-bit (K <= 16) signed X/Y as an input; Z must store 1.0.
///
/// Each component is stored as an 8-bit normalized integer. W is preserved as is.
pub fn decode_filter_oct_8<'a>(data: impl Iterator<Item = &'a mut [u8; 4]>) {
    decode_filter_oct_scalar_impl!(u8, i8, data);
}

/// Decodes octahedral encoding of a unit vector with K-bit (K <= 16) signed X/Y as an input; Z must store 1.0.
///
/// Each component is stored as a 16-bit normalized integer. W is preserved as is.
pub fn decode_filter_oct_16<'a>(data: impl Iterator<Item = &'a mut [u16; 4]>) {
    decode_filter_oct_scalar_impl!(u16, i16, data);
}

/// Decodes 3-component quaternion encoding with K-bit (4 <= K <= 16) component encoding and a 2-bit component index indicating which component to reconstruct.
pub fn decode_filter_quat<'a>(data: impl Iterator<Item = &'a mut [u16; 4]>) {
    decode_filter_quat_scalar(data);
}

/// Decodes exponential encoding of floating-point data with 8-bit exponent and 24-bit integer mantissa as 2^E*M.
///
/// Each 32-bit component is decoded in isolation.
pub fn decode_filter_exp<'a>(data: impl Iterator<Item = &'a mut u32>) {
    decode_filter_exp_scalar(data);
}

macro_rules! encode_filter_oct_scalar_impl {
    ($data_type:ty, $signed_data_type:ty, $bits: expr, $data:expr, $dest:expr) => {
        assert!((1..=16).contains(&$bits));

        const STRIDE: usize = std::mem::size_of::<$data_type>() * 4;
        const BYTE_BITS: u32 = (STRIDE * 2) as u32;

        for (n, dest) in $data.zip($dest) {
            // octahedral encoding of a unit vector
            let mut nx = n[0];
            let mut ny = n[1];
            let nz = n[2];
            let nw = n[3];
            let nl = nx.abs() + ny.abs() + nz.abs();
            let ns = zero_inverse(nl);

            nx *= ns;
            ny *= ns;

            let u = if nz >= 0.0 {
                nx
            } else {
                (1.0 - ny.abs()) * if nx >= 0.0 { 1.0 } else { -1.0 }
            };
            let v = if nz >= 0.0 {
                ny
            } else {
                (1.0 - nx.abs()) * if ny >= 0.0 { 1.0 } else { -1.0 }
            };

            let fu = quantize_snorm(u, $bits);
            let fv = quantize_snorm(v, $bits);
            let fo = quantize_snorm(1.0, $bits);
            let fw = quantize_snorm(nw, BYTE_BITS);

            dest[0] = (fu as $signed_data_type) as $data_type;
            dest[1] = (fv as $signed_data_type) as $data_type;
            dest[2] = (fo as $signed_data_type) as $data_type;
            dest[3] = (fw as $signed_data_type) as $data_type;
        }
    };
}

/// Encodes unit vectors with K-bit (K <= 16) signed X/Y as an output.
///
/// Each component is stored as an 8-bit normalized integer. W is preserved as is.
pub fn encode_filter_oct_8<'a>(
    destination: impl Iterator<Item = &'a mut [u8; 4]>,
    bits: u32,
    data: impl Iterator<Item = &'a [f32; 4]>,
) {
    encode_filter_oct_scalar_impl!(u8, i8, bits, data, destination);
}

/// Encodes unit vectors with K-bit (K <= 16) signed X/Y as an output.
///
/// Each component is stored as a 16-bit normalized integer. W is preserved as is.
pub fn encode_filter_oct_16<'a>(
    destination: impl Iterator<Item = &'a mut [u16; 4]>,
    bits: u32,
    data: impl Iterator<Item = &'a [f32; 4]>,
) {
    encode_filter_oct_scalar_impl!(u16, i16, bits, data, destination);
}

/// Encodes unit quaternions with K-bit (4 <= K <= 16) component encoding.
///
/// Each component is stored as an 16-bit integer.
pub fn encode_filter_quat<'a>(
    destination: impl Iterator<Item = &'a mut [u16; 4]>,
    bits: u32,
    data: impl Iterator<Item = &'a [f32; 4]>,
) {
    assert!((4..=16).contains(&bits));

    let scaler = 2.0f32.sqrt();

    for (q, d) in data.zip(destination) {
        // establish maximum quaternion component
        let mut qc = 0;
        qc = if q[1].abs() > q[qc].abs() { 1 } else { qc };
        qc = if q[2].abs() > q[qc].abs() { 2 } else { qc };
        qc = if q[3].abs() > q[qc].abs() { 3 } else { qc };

        // we use double-cover properties to discard the sign
        let sign = if q[qc] < 0.0 { -1.0 } else { 1.0 };

        // note: we always encode a cyclical swizzle to be able to recover the order via rotation
        d[0] = quantize_snorm(q[(qc + 1) & 3] * scaler * sign, bits) as u16;
        d[1] = quantize_snorm(q[(qc + 2) & 3] * scaler * sign, bits) as u16;
        d[2] = quantize_snorm(q[(qc + 3) & 3] * scaler * sign, bits) as u16;
        d[3] = ((quantize_snorm(1.0, bits) & !3) | (qc as i32)) as u16;
    }
}

// optimized variant of frexp
fn optlog2(v: f32) -> i32 {
    let u = v.to_bits();
    // +1 accounts for implicit 1. in mantissa; denormalized numbers will end up clamped to min_exp by calling code
    if u == 0 { 0 } else { ((u >> 23) & 0xff) as i32 - 127 + 1 }
}

// optimized variant of ldexp
fn optexp2(e: i32) -> f32 {
    let u = ((e + 127) as u32) << 23;
    f32::from_bits(u)
}

#[derive(Debug, PartialEq, Eq)]
pub enum EncodeExpMode {
    /// When encoding exponents, use separate values for each component (maximum quality)
    Separate,
    /// When encoding exponents, use shared value for all components of each vector (better compression)
    SharedVector,
    /// When encoding exponents, use shared value for each component of all vectors (best compression)
    SharedComponent,
}

/// Encodes arbitrary (finite) floating-point data with 8-bit exponent and K-bit integer mantissa (1 <= K <= 24).
///
/// Exponent can be shared between all components of a given vector as defined by `N` or all values of a given component.
pub fn encode_filter_exp<'a, const N: usize>(
    destination: impl Iterator<Item = &'a mut [u32; N]>,
    bits: u32,
    data: impl Iterator<Item = &'a [f32; N]> + Clone,
    mode: EncodeExpMode,
) {
    assert!((1..=24).contains(&bits));
    assert!((1..=64).contains(&N));

    const MIN_EXP: i32 = -100;
    let mut component_exp = [MIN_EXP; N];

    if mode == EncodeExpMode::SharedComponent {
        for v in data.clone() {
            // use maximum exponent to encode values; this guarantees that mantissa is [-1, 1]
            for (c, ce) in v.iter().zip(component_exp.iter_mut()) {
                let e = optlog2(*c);
                *ce = (*ce).max(e);
            }
        }
    }

    for (v, d) in data.zip(destination) {
        let mut vector_exp = MIN_EXP;

        match mode {
            EncodeExpMode::SharedVector => {
                // use maximum exponent to encode values; this guarantees that mantissa is [-1, 1]
                for c in v {
                    let e = optlog2(*c);
                    vector_exp = vector_exp.max(e);
                }
            }
            EncodeExpMode::Separate => {
                for (c, ce) in v.iter().zip(component_exp.iter_mut()) {
                    let e = optlog2(*c);
                    *ce = MIN_EXP.max(e);
                }
            }
            _ => {}
        }

        for ((vc, dc), ce) in v.iter().zip(d.iter_mut()).zip(component_exp) {
            let mut exp = if mode == EncodeExpMode::SharedVector {
                vector_exp
            } else {
                ce
            };

            // note that we additionally scale the mantissa to make it a K-bit signed integer (K-1 bits for magnitude)
            exp -= bits as i32 - 1;

            // compute renormalized rounded mantissa for each component
            let mmask = (1 << 24) - 1;

            let m = (vc * optexp2(-exp) + if *vc >= 0.0 { 0.5 } else { -0.5 }) as i32;

            *dc = ((m & mmask) as u32) | ((exp as u32) << 24);
        }
    }
}

#[cfg(test)]
mod test {
    use std::f32::consts::FRAC_1_SQRT_2;

    use super::*;

    #[test]
    fn test_decode_filter_oct_8() {
        const DATA: [[u8; 4]; 4] = [
            [0, 1, 127, 0],
            [0, 187, 127, 1],
            [255, 1, 127, 0],
            [14, 130, 127, 1],
        ];

        const EXPECTED: [[u8; 4]; 4] = [
            [0, 1, 127, 0],
            [0, 159, 82, 1],
            [255, 1, 127, 0],
            [1, 130, 241, 1],
        ];

        // Aligned by 4
        let mut full = [[0; 4]; 4];
        full.copy_from_slice(&DATA);
        decode_filter_oct_8(full.iter_mut());
        assert_eq!(full, EXPECTED);

        // Tail processing for unaligned data
        let mut tail = [[0; 4]; 3];
        tail.copy_from_slice(&DATA[0..3]);
        decode_filter_oct_8(tail.iter_mut());
        assert_eq!(tail, &EXPECTED[0..tail.len()]);
    }

    #[test]
    fn test_decode_filter_oct_12() {
        const DATA: [[u16; 4]; 4] = [
            [0, 1, 2047, 0],
            [0, 1870, 2047, 1],
            [2017, 1, 2047, 0],
            [14, 1300, 2047, 1],
        ];

        const EXPECTED: [[u16; 4]; 4] = [
            [0, 16, 32767, 0],
            [0, 32621, 3088, 1],
            [32764, 16, 471, 0],
            [307, 28541, 16093, 1],
        ];

        // Aligned by 4
        let mut full = [[0; 4]; 4];
        full.copy_from_slice(&DATA);
        decode_filter_oct_16(full.iter_mut());
        assert_eq!(full, EXPECTED);

        // Tail processing for unaligned data
        let mut tail = [[0; 4]; 3];
        tail.copy_from_slice(&DATA[0..3]);
        decode_filter_oct_16(tail.iter_mut());
        assert_eq!(tail, &EXPECTED[0..tail.len()]);
    }

    #[test]
    fn test_decode_filter_quat_12() {
        const DATA: [[u16; 4]; 4] = [
            [0, 1, 0, 0x7fc],
            [0, 1870, 0, 0x7fd],
            [2017, 1, 0, 0x7fe],
            [14, 1300, 0, 0x7ff],
        ];

        const EXPECTED: [[u16; 4]; 4] = [
            [32767, 0, 11, 0],
            [0, 25013, 0, 21166],
            [11, 0, 23504, 22830],
            [158, 14715, 0, 29277],
        ];

        // Aligned by 4
        let mut full = DATA;
        decode_filter_quat(full.iter_mut());
        assert_eq!(full, EXPECTED);

        // Tail processing for unaligned data
        let mut tail = [[0; 4]; 3];
        tail.copy_from_slice(&DATA[0..3]);
        decode_filter_quat(tail.iter_mut());
        assert_eq!(tail, &EXPECTED[0..tail.len()]);
    }

    #[test]
    fn test_decode_filter_exp() {
        const DATA: [u32; 4] = [0, 0xff000003, 0x02fffff7, 0xfe7fffff];
        const EXPECTED: [u32; 4] = [0, 0x3fc00000, 0xc2100000, 0x49fffffe];

        // Aligned by 4
        let mut full = DATA;
        decode_filter_exp(full.iter_mut());
        assert_eq!(full, EXPECTED);

        // Tail processing for unaligned data
        let mut tail = [0; 3];
        tail.copy_from_slice(&DATA[0..3]);
        decode_filter_exp(tail.iter_mut());
        assert_eq!(tail, &EXPECTED[0..tail.len()]);
    }

    #[test]
    fn test_encode_filter_oct_8() {
        const DATA: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [FRAC_1_SQRT_2, 0.0, 0.707168, 1.0],
            [-FRAC_1_SQRT_2, 0.0, -0.707168, 1.0],
        ];

        const EXPECTED: [[u8; 4]; 4] = [
            [0x7f, 0, 0x7f, 0],
            [0, 0x81, 0x7f, 0],
            [0x3f, 0, 0x7f, 0x7f],
            [0x81, 0x40, 0x7f, 0x7f],
        ];

        let mut encoded = [0u8; 4 * 4];
        encode_filter_oct_8(encoded.as_chunks_mut::<4>().0.iter_mut(), 8, DATA.iter());

        assert_eq!(encoded, *EXPECTED.iter().flatten().copied().collect::<Vec<_>>());

        let mut decoded = encoded;
        decode_filter_oct_8(decoded.as_chunks_mut::<4>().0.iter_mut());

        for (dat, dec) in DATA.iter().flatten().zip(decoded) {
            assert!((dec as i8 as f32 / 127.0 - dat).abs() < 1e-2);
        }
    }

    #[test]
    fn test_encode_filter_oct_12() {
        const DATA: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [FRAC_1_SQRT_2, 0.0, 0.707168, 1.0],
            [-FRAC_1_SQRT_2, 0.0, -0.707168, 1.0],
        ];

        const EXPECTED: [[u16; 4]; 4] = [
            [0x7ff, 0, 0x7ff, 0],
            [0, 0xf801, 0x7ff, 0],
            [0x3ff, 0, 0x7ff, 0x7fff],
            [0xf801, 0x400, 0x7ff, 0x7fff],
        ];

        let mut encoded = [0u16; 4 * 4];
        encode_filter_oct_16(encoded.as_chunks_mut::<4>().0.iter_mut(), 12, DATA.iter());

        assert_eq!(encoded, *EXPECTED.iter().flatten().copied().collect::<Vec<_>>());

        let mut decoded = encoded;
        decode_filter_oct_16(decoded.as_chunks_mut::<4>().0.iter_mut());

        for (dat, dec) in DATA.iter().flatten().zip(decoded) {
            assert!((dec as i16 as f32 / 32767.0 - dat).abs() < 1e-3);
        }
    }

    #[test]
    fn test_encode_filter_quat_12() {
        const DATA: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [FRAC_1_SQRT_2, 0.0, 0.0, 0.707168],
            [-FRAC_1_SQRT_2, 0.0, 0.0, -0.707168],
        ];

        const EXPECTED: [[u16; 4]; 4] = [
            [0, 0, 0, 0x7fc],
            [0, 0, 0, 0x7fd],
            [0x7ff, 0, 0, 0x7ff],
            [0x7ff, 0, 0, 0x7ff],
        ];

        let mut encoded = [0u16; 4 * 4];
        encode_filter_quat(encoded.as_chunks_mut::<4>().0.iter_mut(), 12, DATA.iter());

        assert_eq!(encoded, *EXPECTED.iter().flatten().copied().collect::<Vec<_>>());

        let mut decoded = encoded;
        decode_filter_quat(decoded.as_chunks_mut::<4>().0.iter_mut());

        for (dat, dec) in DATA.iter().zip(decoded.as_chunks::<4>().0) {
            let dx = dec[0] as f32 / 32767.0;
            let dy = dec[1] as f32 / 32767.0;
            let dz = dec[2] as f32 / 32767.0;
            let dw = dec[3] as f32 / 32767.0;

            let dp = dat[0] * dx + dat[1] * dy + dat[2] * dz + dat[3] * dw;

            assert!((dp.abs() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_encode_filter_exp() {
        const DATA: [[f32; 2]; 2] = [[1.0, -23.4], [-0.1, 11.0]];

        // separate exponents: each component gets its own value
        const EXPECTED1: [[u32; 2]; 2] = [[0xf3002000, 0xf7ffd133], [0xefffcccd, 0xf6002c00]];

        // shared exponents (vector): all components of each vector get the same value
        const EXPECTED2: [[u32; 2]; 2] = [[0xf7000200, 0xf7ffd133], [0xf6ffff9a, 0xf6002c00]];

        // shared exponents (component): each component gets the same value across all vectors
        const EXPECTED3: [[u32; 2]; 2] = [[0xf3002000, 0xf7ffd133], [0xf3fffccd, 0xf7001600]];

        let mut encoded1 = [[0u32; 2]; 2];
        encode_filter_exp::<2>(encoded1.iter_mut(), 15, DATA.iter(), EncodeExpMode::Separate);
        assert_eq!(encoded1, EXPECTED1);

        let mut encoded2 = [[0u32; 2]; 2];
        encode_filter_exp::<2>(encoded2.iter_mut(), 15, DATA.iter(), EncodeExpMode::SharedVector);
        assert_eq!(encoded2, EXPECTED2);

        let mut encoded3 = [[0u32; 2]; 2];
        encode_filter_exp::<2>(encoded3.iter_mut(), 15, DATA.iter(), EncodeExpMode::SharedComponent);
        assert_eq!(encoded3, EXPECTED3);

        let mut decoded1 = encoded1;
        decode_filter_exp(decoded1.iter_mut().flatten());

        let mut decoded2 = encoded2;
        decode_filter_exp(decoded2.iter_mut().flatten());

        let mut decoded3 = encoded3;
        decode_filter_exp(decoded3.iter_mut().flatten());

        for (i, data) in DATA[0].iter().enumerate() {
            assert!((f32::from_bits(decoded1[0][i]) - *data).abs() < 1e-3);
            assert!((f32::from_bits(decoded2[0][i]) - *data).abs() < 1e-3);
            assert!((f32::from_bits(decoded3[0][i]) - *data).abs() < 1e-3);
        }
    }

    #[test]
    fn test_encode_filter_exp_zero() {
        const DATA: [[f32; 1]; 1] = [[0.0]];
        const EXPECTED: [[u32; 1]; 1] = [[0xf2000000]];

        let mut encoded = [[0u32; 1]; 1];
        encode_filter_exp::<1>(encoded.iter_mut(), 15, DATA.iter(), EncodeExpMode::Separate);

        assert_eq!(encoded, EXPECTED);

        let mut decoded = encoded;
        decode_filter_exp(decoded.iter_mut().flatten());

        assert_eq!(
            decoded.iter().flatten().map(|d| f32::from_bits(*d)).collect::<Vec<_>>(),
            DATA.iter().flatten().copied().collect::<Vec<_>>(),
        );
    }
}
