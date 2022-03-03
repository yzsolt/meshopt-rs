//! Vertex buffer encoding and decoding

use crate::util::{as_bytes, as_mut_bytes, read_byte, write_byte, write_exact};

use std::io::{Cursor, Read, Write};

use super::{DecodeError, VertexEncodingVersion};

const VERTEX_HEADER: u8 = 0xa0;

const VERTEX_BLOCK_SIZE_BYTES: usize = 8192;
const VERTEX_BLOCK_MAX_SIZE: usize = 256;
const BYTE_GROUP_SIZE: usize = 16;
const BYTE_GROUP_DECODE_LIMIT: usize = 24;
const TAIL_MAX_SIZE: usize = 32;

fn get_vertex_block_size(vertex_size: usize) -> usize {
    // make sure the entire block fits into the scratch buffer
    let mut result = VERTEX_BLOCK_SIZE_BYTES / vertex_size;

    // align to byte group size; we encode each byte as a byte group
    // if vertex block is misaligned, it results in wasted bytes, so just truncate the block size
    result &= !(BYTE_GROUP_SIZE - 1);

    result.min(VERTEX_BLOCK_MAX_SIZE)
}

fn zigzag8(v: u8) -> u8 {
    ((v as i8) >> 7) as u8 ^ (v << 1)
}

fn unzigzag8(v: u8) -> u8 {
    (v & 1).wrapping_neg() ^ (v >> 1)
}

fn encode_bytes_group_zero(buffer: &[u8]) -> bool {
    !buffer[0..BYTE_GROUP_SIZE].iter().any(|b| *b > 0)
}

fn encode_bytes_group_measure(buffer: &[u8], bits: usize) -> usize {
    assert!(bits >= 1 && bits <= 8);

    match bits {
        1 => {
            if encode_bytes_group_zero(buffer) {
                0
            } else {
                usize::MAX
            }
        }
        8 => BYTE_GROUP_SIZE,
        _ => {
            let mut result = BYTE_GROUP_SIZE * bits / 8;

            let sentinel = (1 << bits) - 1;

            result += &buffer[0..BYTE_GROUP_SIZE]
                .iter()
                .map(|b| (*b >= sentinel) as usize)
                .sum();

            result
        }
    }
}

fn encode_bytes_group<W: Write>(data: &mut W, buffer: &[u8], bits: usize) -> Option<usize> {
    assert!(bits >= 1 && bits <= 8);

    match bits {
        1 => Some(0),
        8 => write_exact(data, &buffer[0..BYTE_GROUP_SIZE]),
        _ => {
            let byte_size = 8 / bits;
            assert!(BYTE_GROUP_SIZE % byte_size == 0);

            // fixed portion: bits bits for each value
            // variable portion: full byte for each out-of-range value (using 1...1 as sentinel)
            let sentinel = (1 << bits) - 1;

            let mut written = 0;

            for i in (0..BYTE_GROUP_SIZE).step_by(byte_size) {
                let mut byte = 0;

                for k in 0..byte_size {
                    let enc = if buffer[i + k] >= sentinel {
                        sentinel
                    } else {
                        buffer[i + k]
                    };

                    byte <<= bits;
                    byte |= enc;
                }

                written += write_byte(data, byte);
            }

            for i in 0..BYTE_GROUP_SIZE {
                if buffer[i] >= sentinel {
                    written += write_byte(data, buffer[i]);
                }
            }

            Some(written)
        }
    }
}

fn encode_bytes(data: &mut [u8], buffer: &[u8]) -> Option<usize> {
    assert!(buffer.len() % BYTE_GROUP_SIZE == 0);

    // round number of groups to 4 to get number of header bytes
    let header_size = (buffer.len() / BYTE_GROUP_SIZE + 3) / 4;

    if data.len() < header_size {
        return None;
    }

    let (header, mut data) = data.split_at_mut(header_size);

    header.fill(0);

    let mut written = header_size;

    for i in (0..buffer.len()).step_by(BYTE_GROUP_SIZE) {
        if data.len() < BYTE_GROUP_DECODE_LIMIT {
            return None;
        }

        let mut best_bits = 8;
        let mut best_size = encode_bytes_group_measure(&buffer[i..], 8);

        for bits in [1, 2, 4, 8].iter() {
            let size = encode_bytes_group_measure(&buffer[i..], *bits);

            if size < best_size {
                best_bits = *bits;
                best_size = size;
            }
        }

        let bitslog2 = match best_bits {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => unreachable!(),
        };
        assert!((1 << bitslog2) == best_bits);

        let header_offset = i / BYTE_GROUP_SIZE;

        header[header_offset / 4] |= bitslog2 << ((header_offset % 4) * 2);

        let group_written = encode_bytes_group(&mut data, &buffer[i..], best_bits)?;

        assert!(group_written == best_size);

        written += group_written;
    }

    Some(written)
}

fn encode_vertex_block(
    mut data: &mut [u8],
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &mut [u8; 256],
) -> Option<usize> {
    assert!(vertex_count > 0 && vertex_count <= VERTEX_BLOCK_MAX_SIZE);

    // we sometimes encode elements we didn't fill when rounding to BYTE_GROUP_SIZE
    let mut buffer = [0u8; VERTEX_BLOCK_MAX_SIZE];
    assert!(VERTEX_BLOCK_MAX_SIZE % BYTE_GROUP_SIZE == 0);

    let mut written_sum = 0;

    for k in 0..vertex_size {
        let mut vertex_offset = k;
        let mut p = last_vertex[k];

        for i in 0..vertex_count {
            buffer[i] = zigzag8(vertex_data[vertex_offset].wrapping_sub(p));

            p = vertex_data[vertex_offset];

            vertex_offset += vertex_size;
        }

        let written = encode_bytes(
            data,
            &buffer[0..(vertex_count + BYTE_GROUP_SIZE - 1) & !(BYTE_GROUP_SIZE - 1)],
        )?;
        data = &mut data[written..];

        written_sum += written;
    }

    let offset = vertex_size * (vertex_count - 1);
    last_vertex[0..vertex_size].copy_from_slice(&vertex_data[offset..offset + vertex_size]);

    Some(written_sum)
}

fn decode_bytes_group<W>(data: &mut Cursor<&[u8]>, buffer: &mut W, bitslog2: i32)
where
    W: Write,
{
    let mut byte = 0;

    let mut data_pos = data.position();

    let read = |data: &mut Cursor<&[u8]>, byte: &mut u8, data_pos: &mut u64| {
        data.set_position(*data_pos);
        *byte = read_byte(data);
        *data_pos += 1;
    };

    let mut next = |bits, data: &mut Cursor<&[u8]>, byte: &mut u8, data_var_pos: &mut u64| {
        let enc = *byte >> (8 - bits);
        *byte <<= bits;
        data.set_position(*data_var_pos);
        let encv = read_byte(data);
        write_byte(buffer, if enc == (1 << bits) as u8 - 1 { encv } else { enc });
        *data_var_pos += (enc == (1 << bits) as u8 - 1) as u64;
    };

    let mut buf = [0; BYTE_GROUP_SIZE];

    match bitslog2 {
        0 => {
            buffer.write(&[0; BYTE_GROUP_SIZE]).unwrap();
        }
        1 => {
            let mut data_var_pos = data_pos + 4;

            // 4 groups with 4 2-bit values in each byte
            for _ in 0..4 {
                read(data, &mut byte, &mut data_pos);
                next(2, data, &mut byte, &mut data_var_pos);
                next(2, data, &mut byte, &mut data_var_pos);
                next(2, data, &mut byte, &mut data_var_pos);
                next(2, data, &mut byte, &mut data_var_pos);
            }

            data.set_position(data_var_pos);
        }
        2 => {
            let mut data_var_pos = data_pos + 8;

            // 8 groups with 2 4-bit values in each byte
            for _ in 0..8 {
                read(data, &mut byte, &mut data_pos);
                next(4, data, &mut byte, &mut data_var_pos);
                next(4, data, &mut byte, &mut data_var_pos);
            }

            data.set_position(data_var_pos);
        }
        3 => {
            data.read(&mut buf).unwrap();
            buffer.write(&buf).unwrap();
        }
        _ => unreachable!("Unexpected bit length"), // unreachable since bitslog2 is a 2-bit value
    }
}

fn decode_bytes(data: &mut Cursor<&[u8]>, buffer: &mut [u8]) -> Result<(), DecodeError> {
    assert!(buffer.len() % BYTE_GROUP_SIZE == 0);

    // round number of groups to 4 to get number of header bytes
    let header_size = (buffer.len() / BYTE_GROUP_SIZE + 3) / 4;

    let raw_data = &data.get_ref()[data.position() as usize..];

    if raw_data.len() < header_size {
        return Err(DecodeError::UnexpectedEof);
    }

    let header = &raw_data[0..header_size];

    data.set_position(data.position() + header_size as u64);

    for i in (0..buffer.len()).step_by(BYTE_GROUP_SIZE) {
        let raw_data = &data.get_ref()[data.position() as usize..];

        if raw_data.len() < BYTE_GROUP_DECODE_LIMIT {
            return Err(DecodeError::UnexpectedEof);
        }

        let header_offset = i / BYTE_GROUP_SIZE;

        let bitslog2 = (header[header_offset / 4] >> ((header_offset % 4) * 2)) & 3;

        let mut b = &mut buffer[i..];

        decode_bytes_group(data, &mut b, bitslog2 as i32);
    }

    Ok(())
}

fn decode_vertex_block(
    data: &mut Cursor<&[u8]>,
    vertex_data: &mut [u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &mut [u8; 256],
) -> Result<(), DecodeError> {
    assert!(vertex_count > 0 && vertex_count <= VERTEX_BLOCK_MAX_SIZE);

    let mut buffer = [0; VERTEX_BLOCK_MAX_SIZE];
    let mut transposed = [0; VERTEX_BLOCK_SIZE_BYTES];

    let vertex_count_aligned = (vertex_count + BYTE_GROUP_SIZE - 1) & !(BYTE_GROUP_SIZE - 1);

    for k in 0..vertex_size {
        decode_bytes(data, &mut buffer[0..vertex_count_aligned])?;

        let mut vertex_offset = k;

        let mut p = last_vertex[k];

        for i in 0..vertex_count {
            let v = unzigzag8(buffer[i]).wrapping_add(p);

            transposed[vertex_offset] = v;
            p = v;

            vertex_offset += vertex_size;
        }
    }

    vertex_data[0..vertex_count * vertex_size].copy_from_slice(&transposed[0..vertex_count * vertex_size]);

    let offset = vertex_size * (vertex_count - 1);
    last_vertex[0..vertex_size].copy_from_slice(&transposed[offset..offset + vertex_size]);

    Ok(())
}

/// Encodes vertex data into an array of bytes that is generally smaller and compresses better compared to original.
///
/// Returns encoded data size on success, `None` on error; the only error condition is if buffer doesn't have enough space.
///
/// This function works for a single vertex stream; for multiple vertex streams, call [encode_vertex_buffer] for each stream.
///
/// Note that all bytes of each vertex are encoded verbatim, including padding which should be zero-initialized.
///
/// # Arguments
///
/// * `buffer`: must contain enough space for the encoded vertex buffer (use [encode_vertex_buffer_bound] to compute worst case size)
pub fn encode_vertex_buffer<Vertex>(
    buffer: &mut [u8],
    vertices: &[Vertex],
    version: VertexEncodingVersion,
) -> Option<usize> {
    let vertex_size = std::mem::size_of::<Vertex>();

    assert!(vertex_size > 0 && vertex_size <= 256);
    assert!(vertex_size % 4 == 0);

    let vertex_data = as_bytes(vertices);

    let mut data = buffer;

    if data.len() < 1 + vertex_size {
        return None;
    }

    let version: u8 = version.into();

    let mut written_sum = write_byte(&mut data, (VERTEX_HEADER | version) as u8);

    let mut first_vertex = [0; 256];
    if !vertices.is_empty() {
        first_vertex[0..vertex_size].copy_from_slice(&vertex_data[0..vertex_size]);
    }

    let mut last_vertex = [0; 256];
    last_vertex[0..vertex_size].copy_from_slice(&first_vertex[0..vertex_size]);

    let vertex_block_size = get_vertex_block_size(vertex_size);

    let mut vertex_offset = 0;

    while vertex_offset < vertices.len() {
        let block_size = if vertex_offset + vertex_block_size < vertices.len() {
            vertex_block_size
        } else {
            vertices.len() - vertex_offset
        };

        let written = encode_vertex_block(
            &mut data,
            &vertex_data[vertex_offset * vertex_size..],
            block_size,
            vertex_size,
            &mut last_vertex,
        )?;
        data = &mut data[written..];

        written_sum += written;

        vertex_offset += block_size;
    }

    let tail_size = vertex_size.max(TAIL_MAX_SIZE);

    if data.len() < tail_size {
        return None;
    }

    // write first vertex to the end of the stream and pad it to 32 bytes; this is important to simplify bounds checks in decoder
    if vertex_size < TAIL_MAX_SIZE {
        let written = TAIL_MAX_SIZE - vertex_size;
        data[0..written].fill(0);
        data = &mut data[written..];
        written_sum += written;
    }

    written_sum += write_exact(&mut data, &first_vertex[0..vertex_size])?;

    //assert!(data >= buffer + tail_size);

    Some(written_sum)
}

/// Returns worst case size requirement for [encode_vertex_buffer].
pub fn encode_vertex_buffer_bound(vertex_count: usize, vertex_size: usize) -> usize {
    assert!(vertex_size > 0 && vertex_size <= 256);
    assert!(vertex_size % 4 == 0);

    let vertex_block_size = get_vertex_block_size(vertex_size);
    let vertex_block_count = (vertex_count + vertex_block_size - 1) / vertex_block_size;

    let vertex_block_header_size = (vertex_block_size / BYTE_GROUP_SIZE + 3) / 4;
    let vertex_block_data_size = vertex_block_size;

    let tail_size = vertex_size.max(TAIL_MAX_SIZE);

    1 + vertex_block_count * vertex_size * (vertex_block_header_size + vertex_block_data_size) + tail_size
}

/// Decodes vertex data from an array of bytes generated by [encode_vertex_buffer].
///
/// Returns `Ok` if decoding was successful, and an error otherwise.
///
/// The decoder is safe to use for untrusted input, but it may produce garbage data.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting vertex buffer (vertex_count * vertex_size bytes)
pub fn decode_vertex_buffer<Vertex>(destination: &mut [Vertex], buffer: &[u8]) -> Result<(), DecodeError> {
    let vertex_size = std::mem::size_of::<Vertex>();
    let vertex_count = destination.len();

    assert!(vertex_size > 0 && vertex_size <= 256);
    assert!(vertex_size % 4 == 0);

    let vertex_data = as_mut_bytes(destination);

    if buffer.len() < 1 + vertex_size {
        return Err(DecodeError::UnexpectedEof);
    }

    let mut data = Cursor::new(buffer);

    let data_header = read_byte(&mut data);

    if (data_header & 0xf0) != VERTEX_HEADER {
        return Err(DecodeError::InvalidHeader);
    }

    let version = data_header & 0x0f;
    if version > 0 {
        return Err(DecodeError::UnsupportedVersion);
    }

    let mut last_vertex = [0; 256];
    last_vertex[0..vertex_size].copy_from_slice(&buffer[buffer.len() - vertex_size..]);

    let vertex_block_size = get_vertex_block_size(vertex_size);

    let mut vertex_offset = 0;

    while vertex_offset < vertex_count {
        let block_size = if vertex_offset + vertex_block_size < vertex_count {
            vertex_block_size
        } else {
            vertex_count - vertex_offset
        };

        decode_vertex_block(
            &mut data,
            &mut vertex_data[vertex_offset * vertex_size..],
            block_size,
            vertex_size,
            &mut last_vertex,
        )?;

        vertex_offset += block_size;
    }

    let tail_size = vertex_size.max(TAIL_MAX_SIZE);

    if buffer.len() - data.position() as usize != tail_size {
        return Err(DecodeError::UnexpectedEof);
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    #[repr(C)]
    struct PackedVertexOct {
        p: [u16; 3],
        n: [i8; 2], // octahedron encoded normal, aliases .pw
        t: [u16; 2],
    }

    const VERTEX_BUFFER: [PackedVertexOct; 4] = [
        PackedVertexOct {
            p: [0, 0, 0],
            n: [0, 0],
            t: [0, 0],
        },
        PackedVertexOct {
            p: [300, 0, 0],
            n: [0, 0],
            t: [500, 0],
        },
        PackedVertexOct {
            p: [0, 300, 0],
            n: [0, 0],
            t: [0, 500],
        },
        PackedVertexOct {
            p: [300, 300, 0],
            n: [0, 0],
            t: [500, 500],
        },
    ];

    const VERTEX_DATA_V0: [u8; 85] = [
        0xa0, 0x01, 0x3f, 0x00, 0x00, 0x00, 0x58, 0x57, 0x58, 0x01, 0x26, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x00, 0x00,
        0x00, 0x58, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x3f, 0x00, 0x00, 0x00, 0x17, 0x18,
        0x17, 0x01, 0x26, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x00, 0x00, 0x00, 0x17, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn test_decode_vertex_v0() {
        let mut decoded = [PackedVertexOct::default(); VERTEX_BUFFER.len()];

        assert!(decode_vertex_buffer(&mut decoded, &VERTEX_DATA_V0).is_ok());
        assert_eq!(decoded, VERTEX_BUFFER);
    }

    fn encode_test_vertex() -> Vec<u8> {
        let mut buffer =
            vec![0; encode_vertex_buffer_bound(VERTEX_BUFFER.len(), std::mem::size_of::<PackedVertexOct>())];

        let written = encode_vertex_buffer(&mut buffer, &VERTEX_BUFFER, VertexEncodingVersion::default()).unwrap();
        buffer.resize(written, 0);

        buffer
    }

    #[test]
    fn test_encode_vertex_memory_safe() {
        let mut buffer = encode_test_vertex();

        // check that encode is memory-safe
        for i in buffer.len()..=buffer.len() {
            let result = encode_vertex_buffer(&mut buffer[0..i], &VERTEX_BUFFER, VertexEncodingVersion::default());

            if i == buffer.len() {
                assert_eq!(result, Some(buffer.len()));
            } else {
                assert_eq!(result, None);
            }
        }
    }

    #[test]
    fn test_decode_vertex_memory_safe() {
        let buffer = encode_test_vertex();

        // check that decode is memory-safe
        let mut decoded = vec![PackedVertexOct::default(); VERTEX_BUFFER.len()];

        for i in buffer.len()..=buffer.len() {
            let result = decode_vertex_buffer(&mut decoded, &buffer[0..i]);

            if i == buffer.len() {
                assert!(result.is_ok());
            } else {
                assert!(result.is_err());
            }
        }
    }

    #[test]
    fn test_decode_vertex_reject_extra_bytes() {
        let mut buffer = encode_test_vertex();

        // check that decoder doesn't accept extra bytes after a valid stream
        buffer.push(0);

        let mut decoded = vec![PackedVertexOct::default(); VERTEX_BUFFER.len()];
        assert!(decode_vertex_buffer(&mut decoded, &buffer).is_err());
    }

    #[test]
    fn test_decode_vertex_reject_malformed_headers() {
        let mut buffer = encode_test_vertex();

        // check that decoder doesn't accept malformed headers
        buffer[0] = 0;

        let mut decoded = vec![PackedVertexOct::default(); VERTEX_BUFFER.len()];
        assert!(decode_vertex_buffer(&mut decoded, &buffer).is_err());
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    #[repr(C)]
    struct Vertex([u8; 4]);

    #[test]
    fn test_decode_vertex_bit_groups() {
        let mut data = [Vertex::default(); 16];

        // this tests 0/2/4/8 bit groups in one stream
        for (i, v) in data.iter_mut().enumerate() {
            let i = i as u8;
            v.0 = [i * 0, i * 1, i * 2, i * 8];
        }

        let mut buffer = vec![0; encode_vertex_buffer_bound(data.len(), std::mem::size_of::<Vertex>())];

        let written = encode_vertex_buffer(&mut buffer, &data, VertexEncodingVersion::default()).unwrap();
        buffer.resize(written, 0);

        let mut decoded = [Vertex::default(); 16];
        assert!(decode_vertex_buffer(&mut decoded, &buffer).is_ok());
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_decode_vertex_bit_group_sentinels() {
        let mut data = [Vertex::default(); 16];

        // this tests 0/2/4/8 bit groups and sentinels in one stream
        for (i, v) in data.iter_mut().enumerate() {
            let i = i as u8;

            if i == 7 || i == 13 {
                v.0 = [42; 4];
            } else {
                v.0 = [i * 0, i * 1, i * 2, i * 8];
            }
        }

        let mut buffer = vec![0; encode_vertex_buffer_bound(data.len(), std::mem::size_of::<Vertex>())];

        let written = encode_vertex_buffer(&mut buffer, &data, VertexEncodingVersion::default()).unwrap();
        buffer.resize(written, 0);

        let mut decoded = [Vertex::default(); 16];
        assert!(decode_vertex_buffer(&mut decoded, &buffer).is_ok());
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_decode_vertex_large() {
        let mut data = [Vertex::default(); 128];

        // this tests 0/2/4/8 bit groups in one stream
        for (i, v) in data.iter_mut().enumerate() {
            let i = i as u8;
            v.0 = [i * 0, i * 1, i.wrapping_mul(2), i.wrapping_mul(8)];
        }

        let mut buffer = vec![0; encode_vertex_buffer_bound(data.len(), std::mem::size_of::<Vertex>())];

        let written = encode_vertex_buffer(&mut buffer, &data, VertexEncodingVersion::default()).unwrap();
        buffer.resize(written, 0);

        let mut decoded = [Vertex::default(); 128];
        assert!(decode_vertex_buffer(&mut decoded, &buffer).is_ok());
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_encode_vertex_empty() {
        let mut buffer = vec![0; encode_vertex_buffer_bound(0, 16)];
        let size = encode_vertex_buffer::<PackedVertexOct>(&mut buffer, &[], VertexEncodingVersion::default()).unwrap();
        buffer.resize(size, 0);

        assert!(decode_vertex_buffer::<PackedVertexOct>(&mut [], &buffer).is_ok());
    }
}
