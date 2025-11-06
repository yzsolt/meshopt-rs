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

fn frexp(s: f32) -> (f32, i32) {
    if 0.0 == s {
        (s, 0)
    } else {
        let lg = s.abs().log2();
        let x = (lg.fract() - 1.).exp2();
        let exp = lg.floor() + 1.0;
        (s.signum() * x, exp as i32)
    }
}

fn ldexp(a: f32, exp: i32) -> f32 {
    let f = exp as f32;
    a * f.exp2()
}

/// Encodes arbitrary (finite) floating-point data with 8-bit exponent and K-bit integer mantissa (1 <= K <= 24).
///
/// Mantissa is shared between all components of a given vector as defined by `N`.
/// When individual (scalar) encoding is desired, simply pass N=1 and adjust iterators accordingly ([core::array::from_ref], [core::array::from_mut]).
pub fn encode_filter_exp<'a, const N: usize>(
    destination: impl Iterator<Item = &'a mut [u32; N]>,
    bits: u32,
    data: impl Iterator<Item = &'a [f32; N]>,
) {
    assert!((1..=24).contains(&bits));

    for (v, d) in data.zip(destination) {
        // use maximum exponent to encode values; this guarantees that mantissa is [-1, 1]
        let mut exp = -100;

        for c in v {
            let (_, e) = frexp(*c);
            exp = exp.max(e);
        }

        // note that we additionally scale the mantissa to make it a K-bit signed integer (K-1 bits for magnitude)
        exp -= (bits - 1) as i32;

        // compute renormalized rounded mantissa for each component
        let mmask = (1 << 24) - 1;

        for (vc, dc) in v.iter().zip(d.iter_mut()) {
            let m = (ldexp(*vc, -exp) + if *vc >= 0.0 { 0.5 } else { -0.5 }) as i32;

            *dc = ((m & mmask) as u32) | ((exp as u32) << 24);
        }
    }
}

#[cfg(test)]
mod test {
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
            [0.7071068, 0.0, 0.707168, 1.0],
            [-0.7071068, 0.0, -0.707168, 1.0],
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

        let mut decoded = encoded.clone();
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
            [0.7071068, 0.0, 0.707168, 1.0],
            [-0.7071068, 0.0, -0.707168, 1.0],
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

        let mut decoded = encoded.clone();
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
            [0.7071068, 0.0, 0.0, 0.707168],
            [-0.7071068, 0.0, 0.0, -0.707168],
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

        let mut decoded = encoded.clone();
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
        const DATA: [[f32; 3]; 1] = [[1.0, -23.4, -0.1]];
        const EXPECTED: [[u32; 3]; 1] = [[0xf7000200, 0xf7ffd133, 0xf7ffffcd]];

        let mut encoded = [[0u32; 3]; 1];
        encode_filter_exp::<3>(encoded.iter_mut(), 15, DATA.iter());

        assert_eq!(encoded, EXPECTED);

        let mut decoded = encoded.clone();
        decode_filter_exp(decoded.iter_mut().flatten());

        for (dat, dec) in DATA.iter().flatten().zip(decoded.iter().flatten()) {
            assert!((f32::from_bits(*dec) - *dat).abs() < 1e-3);
        }
    }
}
