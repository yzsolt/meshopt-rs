//! Vertex buffer filters
//!
//! These functions can be used to filter output of [decode_vertex_buffer](crate::vertex::buffer::decode_vertex_buffer) in-place.
//!
//! Mainly useful in combination with the `EXT_meshopt_compression` glTF extension.

use crate::util::as_mut_bytes;

macro_rules! from_bytes_oct {
    ($e:ident, 4) => {
        (
            i8::from_ne_bytes([$e[0]]) as f32,
            i8::from_ne_bytes([$e[1]]) as f32,
            i8::from_ne_bytes([$e[2]]) as f32,
        )
    };
    ($e:ident, 8) => {
        (
            i16::from_ne_bytes([$e[0], $e[1]]) as f32,
            i16::from_ne_bytes([$e[2], $e[3]]) as f32,
            i16::from_ne_bytes([$e[4], $e[5]]) as f32,
        )
    };
}

macro_rules! to_bytes_oct {
    ($e:ident, $x:ident, $y:ident, $z:ident, 4) => {{
        $e[0] = $x.to_ne_bytes()[0];
        $e[1] = $y.to_ne_bytes()[0];
        $e[2] = $z.to_ne_bytes()[0];
    }};
    ($e:ident, $x:ident, $y:ident, $z:ident, 8) => {{
        $e[0] = $x.to_ne_bytes()[0];
        $e[1] = $x.to_ne_bytes()[1];
        $e[2] = $y.to_ne_bytes()[0];
        $e[3] = $y.to_ne_bytes()[1];
        $e[4] = $z.to_ne_bytes()[0];
        $e[5] = $z.to_ne_bytes()[1];
    }};
}

fn decode_filter_oct_scalar(data: &mut [u8], stride: usize) {
    let max = ((1 << (stride / 4 * 8 - 1)) - 1) as f32;

    for bytes in data.chunks_exact_mut(stride) {
        // convert x and y to floats and reconstruct z; this assumes zf encodes 1.0 at the same bit count
        let (mut x, mut y, mut z) = match stride {
            4 => from_bytes_oct!(bytes, 4),
            8 => from_bytes_oct!(bytes, 8),
            _ => unreachable!(""),
        };

        z = z - x.abs() - y.abs();

        // fixup octahedral coordinates for z<0
        let t = z.min(0.0);

        x += if x >= 0.0 { t } else { -t };
        y += if y >= 0.0 { t } else { -t };

        // compute normal length & scale
        let l = (x * x + y * y + z * z).sqrt();
        let s = max / l;

        // rounded signed float->int
        let xf = (x * s + if x >= 0.0 { 0.5 } else { -0.5 }) as i32;
        let yf = (y * s + if y >= 0.0 { 0.5 } else { -0.5 }) as i32;
        let zf = (z * s + if z >= 0.0 { 0.5 } else { -0.5 }) as i32;

        match stride {
            4 => to_bytes_oct!(bytes, xf, yf, zf, 4),
            8 => to_bytes_oct!(bytes, xf, yf, zf, 8),
            _ => unreachable!(""),
        }
    }
}

fn decode_filter_quat_scalar(data: &mut [[u16; 4]]) {
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

fn decode_filter_exp_scalar(data: &mut [u32]) {
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
pub fn decode_filter_oct_8(data: &mut [[u8; 4]]) {
    decode_filter_oct_scalar(as_mut_bytes(data), 4);
}

/// Decodes octahedral encoding of a unit vector with K-bit (K <= 16) signed X/Y as an input; Z must store 1.0.
///
/// Each component is stored as a 16-bit normalized integer. W is preserved as is.
pub fn decode_filter_oct_16(data: &mut [[u16; 4]]) {
    decode_filter_oct_scalar(as_mut_bytes(data), 8);
}

/// Decodes 3-component quaternion encoding with K-bit (4 <= K <= 16) component encoding and a 2-bit component index indicating which component to reconstruct.
pub fn decode_filter_quat(data: &mut [[u16; 4]]) {
    decode_filter_quat_scalar(data);
}

/// Decodes exponential encoding of floating-point data with 8-bit exponent and 24-bit integer mantissa as 2^E*M.
///
/// Each 32-bit component is decoded in isolation.
pub fn decode_filter_exp(data: &mut [u32]) {
    decode_filter_exp_scalar(data);
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
        decode_filter_oct_8(&mut full);
        assert_eq!(full, EXPECTED);

        // Tail processing for unaligned data
        let mut tail = [[0; 4]; 3];
        tail.copy_from_slice(&DATA[0..3]);
        decode_filter_oct_8(&mut tail);
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
        decode_filter_oct_16(&mut full);
        assert_eq!(full, EXPECTED);

        // Tail processing for unaligned data
        let mut tail = [[0; 4]; 3];
        tail.copy_from_slice(&DATA[0..3]);
        decode_filter_oct_16(&mut tail);
        assert_eq!(tail, &EXPECTED[0..tail.len()]);
    }

    #[test]
    fn decode_filter_quat_12() {
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
        decode_filter_quat(&mut full);
        assert_eq!(full, EXPECTED);

        // Tail processing for unaligned data
        let mut tail = [[0; 4]; 3];
        tail.copy_from_slice(&DATA[0..3]);
        decode_filter_quat(&mut tail);
        assert_eq!(tail, &EXPECTED[0..tail.len()]);
    }

    #[test]
    fn test_decode_filter_exp() {
        const DATA: [u32; 4] = [0, 0xff000003, 0x02fffff7, 0xfe7fffff];

        const EXPECTED: [u32; 4] = [0, 0x3fc00000, 0xc2100000, 0x49fffffe];

        // Aligned by 4
        let mut full = DATA;
        decode_filter_exp(&mut full);
        assert_eq!(full, EXPECTED);

        // Tail processing for unaligned data
        let mut tail = [0; 3];
        tail.copy_from_slice(&DATA[0..3]);
        decode_filter_exp(&mut tail);
        assert_eq!(tail, &EXPECTED[0..tail.len()]);
    }
}
