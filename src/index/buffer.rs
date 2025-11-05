//! Index buffer encoding and decoding

use crate::INVALID_INDEX;

use std::io::{Read, Write};

use super::{DecodeError, IndexEncodingVersion, decode_v_byte, encode_v_byte, read_byte, write_byte};

const INDEX_HEADER: u8 = 0xe0;

type VertexFifo = [u32; 16];

const DEFAULT_VERTEX_FIFO: VertexFifo = [INVALID_INDEX; 16];

type EdgeFifo = [[u32; 2]; 16];

const DEFAULT_EDGE_FIFO: EdgeFifo = [[INVALID_INDEX; 2]; 16];

const TRIANGLE_INDEX_ORDER: [[usize; 3]; 3] = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];

const CODE_AUX_ENCODING_TABLE: [u8; 16] = [
    0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9, 0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0,
    0, // last two entries aren't used for encoding
];

fn rotate_triangle(b: u32, c: u32, next: u32) -> i32 {
    if b == next {
        1
    } else if c == next {
        2
    } else {
        0
    }
}

fn get_edge_fifo(fifo: &EdgeFifo, a: u32, b: u32, c: u32, offset: usize) -> i32 {
    for i in 0..16 {
        let index = (offset.wrapping_sub(i + 1)) & 15;

        let e0 = fifo[index][0];
        let e1 = fifo[index][1];

        if e0 == a && e1 == b {
            return ((i << 2) | 0) as i32;
        }

        if e0 == b && e1 == c {
            return ((i << 2) | 1) as i32;
        }

        if e0 == c && e1 == a {
            return ((i << 2) | 2) as i32;
        }
    }

    -1
}

#[inline(always)]
fn push_edge_fifo(fifo: &mut EdgeFifo, a: u32, b: u32, offset: &mut usize) {
    fifo[*offset][0] = a;
    fifo[*offset][1] = b;
    *offset = (*offset + 1) & 15;
}

fn get_vertex_fifo(fifo: &VertexFifo, v: u32, offset: usize) -> i32 {
    for i in 0..16 {
        let index = (offset.wrapping_sub(i + 1)) & 15;

        if fifo[index] == v {
            return i as i32;
        }
    }

    -1
}

#[inline(always)]
fn push_vertex_fifo(fifo: &mut VertexFifo, v: u32, offset: &mut usize, cond: Option<bool>) {
    fifo[*offset] = v;
    *offset = (*offset + cond.unwrap_or(true) as usize) & 15;
}

fn encode_index<W: Write>(data: &mut W, index: u32, last: u32) {
    let d = index.wrapping_sub(last);
    let v = (d << 1) ^ (((d as i32) >> 31) as u32);

    encode_v_byte(data, v);
}

fn decode_index<R: Read>(data: &mut R, last: u32) -> u32 {
    let v = decode_v_byte(data);
    let d = (v >> 1) ^ (-((v & 1) as i32) as u32);

    last.wrapping_add(d)
}

fn get_code_aux_index(v: u8, table: &[u8]) -> i32 {
    table[0..16]
        .iter()
        .position(|t| *t == v)
        .map(|t| t as i32)
        .unwrap_or(-1)
}

fn write_triangle<T>(destination: &mut [T], a: u32, b: u32, c: u32)
where
    T: Copy + From<u32>,
{
    destination.copy_from_slice(&[T::from(a), T::from(b), T::from(c)]);
}

/// Encodes index data into an array of bytes that is generally much smaller (<1.5 bytes/triangle) and compresses better (<1 bytes/triangle) compared to original.
///
/// Input index buffer must represent a triangle list.
///
/// Returns encoded data size on success, `None` on error; the only error condition is if `buffer` doesn't have enough space.
///
/// For maximum efficiency the index buffer being encoded has to be optimized for vertex cache and vertex fetch first.
///
/// # Arguments
///
/// * `buffer`: must contain enough space for the encoded index buffer (use [encode_index_buffer_bound] to compute worst case size)
pub fn encode_index_buffer(mut buffer: &mut [u8], indices: &[u32], version: IndexEncodingVersion) -> Option<usize> {
    assert!(indices.len().is_multiple_of(3));

    let buffer_len = buffer.len();

    // the minimum valid encoding is header, 1 byte per triangle and a 16-byte codeaux table
    if buffer_len < 1 + indices.len() / 3 + 16 {
        return None;
    }

    let version: u8 = version.into();

    write_byte(&mut buffer, INDEX_HEADER | version);

    let mut edgefifo = DEFAULT_EDGE_FIFO;
    let mut vertexfifo = DEFAULT_VERTEX_FIFO;

    let mut edgefifooffset = 0;
    let mut vertexfifooffset = 0;

    let mut next = 0;
    let mut last = 0;

    let (mut code, mut data) = buffer.split_at_mut(indices.len() / 3);

    let fecmax = if version >= 1 { 13 } else { 15 };

    // use static encoding table; it's possible to pack the result and then build an optimal table and repack
    // for now we keep it simple and use the table that has been generated based on symbol frequency on a training mesh set
    let codeaux_table = CODE_AUX_ENCODING_TABLE;

    for i in (0..indices.len()).step_by(3) {
        // make sure we have enough space to write a triangle
        // each triangle writes at most 16 bytes: 1b for codeaux and 5b for each free index
        // after this we can be sure we can write without extra bounds checks
        if data.len() < 16 {
            return None;
        }

        let fer = get_edge_fifo(
            &edgefifo,
            indices[i + 0],
            indices[i + 1],
            indices[i + 2],
            edgefifooffset,
        );

        if fer >= 0 && (fer >> 2) < 15 {
            let order = TRIANGLE_INDEX_ORDER[(fer & 3) as usize];

            let a = indices[i + order[0]];
            let b = indices[i + order[1]];
            let c = indices[i + order[2]];

            // encode edge index and vertex fifo index, next or free index
            let fe = fer >> 2;
            let fc = get_vertex_fifo(&vertexfifo, c, vertexfifooffset);

            let mut fec = if fc >= 1 && fc < fecmax {
                fc
            } else if c == next {
                next += 1;
                0
            } else {
                15
            };

            if fec == 15 && version >= 1 {
                // encode last-1 and last+1 to optimize strip-like sequences
                if c + 1 == last {
                    fec = 13;
                    last = c;
                }
                if c == last + 1 {
                    fec = 14;
                    last = c;
                }
            }

            write_byte(&mut code, ((fe << 4) | fec) as u8);

            // note that we need to update the last index since free indices are delta-encoded
            if fec == 15 {
                encode_index(&mut data, c, last);
                last = c;
            }

            // we only need to push third vertex since first two are likely already in the vertex fifo
            if fec == 0 || fec >= fecmax {
                push_vertex_fifo(&mut vertexfifo, c, &mut vertexfifooffset, None);
            }

            // we only need to push two new edges to edge fifo since the third one is already there
            push_edge_fifo(&mut edgefifo, c, b, &mut edgefifooffset);
            push_edge_fifo(&mut edgefifo, a, c, &mut edgefifooffset);
        } else {
            let rotation = rotate_triangle(indices[i + 1], indices[i + 2], next);
            let order = TRIANGLE_INDEX_ORDER[rotation as usize];

            let a = indices[i + order[0]];
            let b = indices[i + order[1]];
            let c = indices[i + order[2]];

            // if a/b/c are 0/1/2, we emit a reset code
            let mut reset = false;

            if a == 0 && b == 1 && c == 2 && next > 0 && version >= 1 {
                reset = true;
                next = 0;

                // reset vertex fifo to make sure we don't accidentally reference vertices from that in the future
                // this makes sure next continues to get incremented instead of being stuck
                vertexfifo = DEFAULT_VERTEX_FIFO;
            }

            let fb = get_vertex_fifo(&vertexfifo, b, vertexfifooffset);
            let fc = get_vertex_fifo(&vertexfifo, c, vertexfifooffset);

            // after rotation, a is almost always equal to next, so we don't waste bits on FIFO encoding for a
            let fea = if a == next {
                next += 1;
                0
            } else {
                15
            };
            let feb = if (0..14).contains(&fb) {
                fb + 1
            } else if b == next {
                next += 1;
                0
            } else {
                15
            };
            let fec = if (0..14).contains(&fc) {
                fc + 1
            } else if c == next {
                next += 1;
                0
            } else {
                15
            };

            // we encode feb & fec in 4 bits using a table if possible, and as a full byte otherwise
            let codeaux = ((feb << 4) | fec) as u8;
            let codeauxindex = get_code_aux_index(codeaux, &codeaux_table);

            // <14 encodes an index into codeaux table, 14 encodes fea=0, 15 encodes fea=15
            if fea == 0 && (0..14).contains(&codeauxindex) && !reset {
                write_byte(&mut code, ((15 << 4) | codeauxindex) as u8);
            } else {
                write_byte(&mut code, ((15 << 4) | 14 | fea) as u8);
                write_byte(&mut data, codeaux);
            }

            // note that we need to update the last index since free indices are delta-encoded
            if fea == 15 {
                encode_index(&mut data, a, last);
                last = a;
            }

            if feb == 15 {
                encode_index(&mut data, b, last);
                last = b;
            }

            if fec == 15 {
                encode_index(&mut data, c, last);
                last = c;
            }

            // only push vertices that weren't already in fifo
            if fea == 0 || fea == 15 {
                push_vertex_fifo(&mut vertexfifo, a, &mut vertexfifooffset, None);
            }

            if feb == 0 || feb == 15 {
                push_vertex_fifo(&mut vertexfifo, b, &mut vertexfifooffset, None);
            }

            if fec == 0 || fec == 15 {
                push_vertex_fifo(&mut vertexfifo, c, &mut vertexfifooffset, None);
            }

            // all three edges aren't in the fifo; pushing all of them is important so that we can match them for later triangles
            push_edge_fifo(&mut edgefifo, b, a, &mut edgefifooffset);
            push_edge_fifo(&mut edgefifo, c, b, &mut edgefifooffset);
            push_edge_fifo(&mut edgefifo, a, c, &mut edgefifooffset);
        }
    }

    // make sure we have enough space to write codeaux table
    if data.len() < 16 {
        return None;
    }

    // add codeaux encoding table to the end of the stream; this is used for decoding codeaux *and* as padding
    // we need padding for decoding to be able to assume that each triangle is encoded as <= 16 bytes of extra data
    // this is enough space for aux byte + 5 bytes per varint index which is the absolute worst case for any input
    for value in &codeaux_table {
        // decoder assumes that table entries never refer to separately encoded indices
        assert!((value & 0xf) != 0xf && (value >> 4) != 0xf);

        write_byte(&mut data, *value);
    }

    // since we encode restarts as codeaux without a table reference, we need to make sure 00 is encoded as a table reference
    assert_eq!(codeaux_table[0], 0);

    //assert!(data >= buffer + indices.len() / 3 + 16);
    //assert!(data <= buffer + buffer.len());

    Some(buffer_len - data.len())
}

/// Returns worst case size requirement for [encode_index_buffer].
pub fn encode_index_buffer_bound(index_count: usize, vertex_count: usize) -> usize {
    assert!(index_count.is_multiple_of(3));

    // compute number of bits required for each index
    let mut vertex_bits: usize = 1;

    while vertex_bits < 32 && vertex_count > (1 << vertex_bits) {
        vertex_bits += 1;
    }

    // worst-case encoding is 2 header bytes + 3 varint-7 encoded index deltas
    let vertex_groups = (vertex_bits + 1).div_ceil(7);

    1 + (index_count / 3) * (2 + 3 * vertex_groups) + 16
}

/// Decodes index data from an array of bytes generated by [encode_index_buffer].
///
/// Returns `Ok` if decoding was successful, and an error otherwise.
/// The decoder is safe to use for untrusted input, but it may produce garbage data (e.g. out of range indices).
///
/// # Arguments
///
/// * `destination`: must contain the exact space for the resulting index buffer
pub fn decode_index_buffer<T>(destination: &mut [T], buffer: &[u8]) -> Result<(), DecodeError>
where
    T: Copy + From<u32>,
{
    assert_eq!(destination.len() % 3, 0);
    //assert!(index_size == 2 || index_size == 4);

    // the minimum valid encoding is header, 1 byte per triangle and a 16-byte codeaux table
    if buffer.len() < 1 + destination.len() / 3 + 16 {
        return Err(DecodeError::UnexpectedEof);
    }

    if (buffer[0] & 0xf0) != INDEX_HEADER {
        return Err(DecodeError::InvalidHeader);
    }

    let version = buffer[0] & 0x0f;
    if version > 1 {
        return Err(DecodeError::UnsupportedVersion);
    }

    let mut edgefifo = DEFAULT_EDGE_FIFO;
    let mut vertexfifo = DEFAULT_VERTEX_FIFO;

    let mut edgefifooffset: usize = 0;
    let mut vertexfifooffset: usize = 0;

    let mut next: u32 = 0;
    let mut last: u32 = 0;

    let fecmax = if version >= 1 { 13 } else { 15 };

    // since we store 16-byte codeaux table at the end, triangle data has to begin before data_safe_end
    let (mut code, mut data, codeaux_table, data_safe_end) = {
        let (code, data) = buffer[1..].split_at(destination.len() / 3);
        let data_safe_end = data.len() - 16;
        let codeaux_table = &data[data_safe_end..];
        (code, std::io::Cursor::new(data), codeaux_table, data_safe_end)
    };

    for dst in destination.chunks_exact_mut(3) {
        // make sure we have enough data to read for a triangle
        // each triangle reads at most 16 bytes of data: 1b for codeaux and 5b for each free index
        // after this we can be sure we can read without extra bounds checks
        if data.position() > data_safe_end as u64 {
            return Err(DecodeError::UnexpectedEof);
        }

        let codetri = read_byte(&mut code) as usize;

        if codetri < 0xf0 {
            let fe = codetri >> 4;

            // fifo reads are wrapped around 16 entry buffer
            let a = edgefifo[(edgefifooffset.wrapping_sub(fe + 1)) & 15][0];
            let b = edgefifo[(edgefifooffset.wrapping_sub(fe + 1)) & 15][1];

            let fec = codetri & 15;

            // note: this is the most common path in the entire decoder
            // inside this if we try to stay branchless (by using cmov/etc.) since these aren't predictable
            if fec < fecmax {
                // fifo reads are wrapped around 16 entry buffer
                let cf = vertexfifo[(vertexfifooffset.wrapping_sub(fec + 1)) & 15];
                let c = if fec == 0 { next } else { cf };

                let fec0 = fec == 0;
                next += fec0 as u32;

                // output triangle
                write_triangle(dst, a, b, c);

                // push vertex/edge fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
                push_vertex_fifo(&mut vertexfifo, c, &mut vertexfifooffset, Some(fec0));

                push_edge_fifo(&mut edgefifo, c, b, &mut edgefifooffset);
                push_edge_fifo(&mut edgefifo, a, c, &mut edgefifooffset);
            } else {
                // fec - (fec ^ 3) decodes 13, 14 into -1, 1
                // note that we need to update the last index since free indices are delta-encoded
                let c = if fec != 15 {
                    last.wrapping_add((fec.wrapping_sub(fec ^ 3)) as u32)
                } else {
                    decode_index(&mut data, last)
                };
                last = c;

                // output triangle
                write_triangle(dst, a, b, c);

                // push vertex/edge fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
                push_vertex_fifo(&mut vertexfifo, c, &mut vertexfifooffset, None);

                push_edge_fifo(&mut edgefifo, c, b, &mut edgefifooffset);
                push_edge_fifo(&mut edgefifo, a, c, &mut edgefifooffset);
            }
        } else {
            // fast path: read codeaux from the table
            if codetri < 0xfe {
                let codeaux = codeaux_table[codetri & 15];

                // note: table can't contain feb/fec=15
                let feb = codeaux >> 4;
                let fec = codeaux & 15;

                // fifo reads are wrapped around 16 entry buffer
                // also note that we increment next for all three vertices before decoding indices - this matches encoder behavior
                let a = next;
                next += 1;

                let bf = vertexfifo[(vertexfifooffset.wrapping_sub(feb as usize)) & 15];
                let b = if feb == 0 { next } else { bf };

                let feb0 = feb == 0;
                next += feb0 as u32;

                let cf = vertexfifo[(vertexfifooffset.wrapping_sub(fec as usize)) & 15];
                let c = if fec == 0 { next } else { cf };

                let fec0 = fec == 0;
                next += fec0 as u32;

                // output triangle
                write_triangle(dst, a, b, c);

                // push vertex/edge fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
                push_vertex_fifo(&mut vertexfifo, a, &mut vertexfifooffset, None);
                push_vertex_fifo(&mut vertexfifo, b, &mut vertexfifooffset, Some(feb0));
                push_vertex_fifo(&mut vertexfifo, c, &mut vertexfifooffset, Some(fec0));

                push_edge_fifo(&mut edgefifo, b, a, &mut edgefifooffset);
                push_edge_fifo(&mut edgefifo, c, b, &mut edgefifooffset);
                push_edge_fifo(&mut edgefifo, a, c, &mut edgefifooffset);
            } else {
                // slow path: read a full byte for codeaux instead of using a table lookup
                let codeaux = read_byte(&mut data);

                let fea = if codetri == 0xfe { 0 } else { 15 };
                let feb = codeaux >> 4;
                let fec = codeaux & 15;

                // reset: codeaux is 0 but encoded as not-a-table
                if codeaux == 0 {
                    next = 0;
                }

                // fifo reads are wrapped around 16 entry buffer
                // also note that we increment next for all three vertices before decoding indices - this matches encoder behavior
                let mut a = if fea == 0 {
                    let n = next;
                    next += 1;
                    n
                } else {
                    0
                };
                let mut b = if feb == 0 {
                    let n = next;
                    next += 1;
                    n
                } else {
                    vertexfifo[(vertexfifooffset.wrapping_sub(feb as usize)) & 15]
                };
                let mut c = if fec == 0 {
                    let n = next;
                    next += 1;
                    n
                } else {
                    vertexfifo[(vertexfifooffset.wrapping_sub(fec as usize)) & 15]
                };

                // note that we need to update the last index since free indices are delta-encoded
                if fea == 15 {
                    a = decode_index(&mut data, last);
                    last = a;
                }

                if feb == 15 {
                    b = decode_index(&mut data, last);
                    last = b;
                }

                if fec == 15 {
                    c = decode_index(&mut data, last);
                    last = c;
                }

                // output triangle
                write_triangle(dst, a, b, c);

                // push vertex/edge fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
                push_vertex_fifo(&mut vertexfifo, a, &mut vertexfifooffset, None);
                push_vertex_fifo(
                    &mut vertexfifo,
                    b,
                    &mut vertexfifooffset,
                    Some((feb == 0) | (feb == 15)),
                );
                push_vertex_fifo(
                    &mut vertexfifo,
                    c,
                    &mut vertexfifooffset,
                    Some((fec == 0) | (fec == 15)),
                );

                push_edge_fifo(&mut edgefifo, b, a, &mut edgefifooffset);
                push_edge_fifo(&mut edgefifo, c, b, &mut edgefifooffset);
                push_edge_fifo(&mut edgefifo, a, c, &mut edgefifooffset);
            }
        }
    }

    if data.position() == data_safe_end as u64 {
        Ok(())
    } else {
        // we should've read all data bytes and stopped at the boundary between data and codeaux table
        Err(DecodeError::ExtraBytes)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // note: 4 6 5 triangle here is a combo-breaker:
    // we encode it without rotating, a=next, c=next - this means we do *not* bump next to 6
    // which means that the next triangle can't be encoded via next sequencing!
    const INDEX_BUFFER: [u32; 12] = [0, 1, 2, 2, 1, 3, 4, 6, 5, 7, 8, 9];

    const INDEX_DATA_V0: [u8; 27] = [
        0xe0, 0xf0, 0x10, 0xfe, 0xff, 0xf0, 0x0c, 0xff, 0x02, 0x02, 0x02, 0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9,
        0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0x00, 0x00,
    ];

    // note: this exercises two features of v1 format, restarts (0 1 2) and last
    const INDEX_BUFFER_TRICKY: [u32; 15] = [0, 1, 2, 2, 1, 3, 0, 1, 2, 2, 1, 5, 2, 1, 4];

    const INDEX_DATA_V1: [u8; 24] = [
        0xe1, 0xf0, 0x10, 0xfe, 0x1f, 0x3d, 0x00, 0x0a, 0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9, 0x86, 0x65, 0x89,
        0x68, 0x98, 0x01, 0x69, 0x00, 0x00,
    ];

    #[test]
    fn test_decode_index_v0() {
        let mut decoded: [u32; INDEX_BUFFER.len()] = Default::default();

        assert!(decode_index_buffer(&mut decoded, &INDEX_DATA_V0).is_ok());
        assert_eq!(decoded, INDEX_BUFFER);
    }

    #[test]
    fn test_decode_index_v1() {
        let mut decoded: [u32; INDEX_BUFFER_TRICKY.len()] = Default::default();

        assert!(decode_index_buffer(&mut decoded, &INDEX_DATA_V1).is_ok());
        assert_eq!(decoded, INDEX_BUFFER_TRICKY);
    }

    #[test]
    fn test_decode_index_16() {
        let buffer = encode_test_index();

        #[derive(Clone, Copy, Default)]
        struct U16(u16);

        impl From<u32> for U16 {
            fn from(index: u32) -> Self {
                Self(index as u16)
            }
        }

        let mut decoded = [U16::default(); INDEX_BUFFER.len()];
        assert!(decode_index_buffer(&mut decoded, &buffer).is_ok());

        assert!(decoded.iter().enumerate().all(|(i, v)| v.0 as u32 == INDEX_BUFFER[i]));
    }

    #[test]
    fn test_encode_index_memory_safe() {
        let mut buffer = encode_test_index();

        // check that encode is memory-safe
        for i in 0..=buffer.len() {
            let result = encode_index_buffer(&mut buffer[0..i], &INDEX_BUFFER, IndexEncodingVersion::default());

            if i == buffer.len() {
                assert_eq!(result, Some(buffer.len()));
            } else {
                assert_eq!(result, None);
            }
        }
    }

    fn encode_test_index() -> Vec<u8> {
        let mut buffer = vec![0; encode_index_buffer_bound(INDEX_BUFFER.len(), 10)];

        let written = encode_index_buffer(&mut buffer, &INDEX_BUFFER, IndexEncodingVersion::default()).unwrap();
        buffer.resize_with(written, Default::default);

        buffer
    }

    #[test]
    fn test_decode_index_memory_safe() {
        let buffer = encode_test_index();

        // check that decode is memory-safe
        let mut decoded: [u32; INDEX_BUFFER.len()] = Default::default();

        for i in 0..=buffer.len() {
            let result = decode_index_buffer(&mut decoded, &buffer[0..i]);

            if i == buffer.len() {
                assert!(result.is_ok());
            } else {
                assert!(result.is_err());
            }
        }
    }

    #[test]
    fn test_decode_index_reject_extra_bytes() {
        let mut buffer = encode_test_index();

        // check that decoder doesn't accept extra bytes after a valid stream
        buffer.push(0);

        let mut decoded: [u32; INDEX_BUFFER.len()] = Default::default();
        assert!(decode_index_buffer(&mut decoded, &buffer).is_err());
    }

    #[test]
    fn test_decode_index_reject_malformed_headers() {
        let mut buffer = encode_test_index();

        // check that decoder doesn't accept malformed headers
        buffer[0] = 0;

        let mut decoded: [u32; INDEX_BUFFER.len()] = Default::default();
        assert!(decode_index_buffer(&mut decoded, &buffer).is_err());
    }

    #[test]
    fn test_decode_index_reject_invalid_version() {
        let mut buffer = encode_test_index();

        // check that decoder doesn't accept invalid version
        buffer[0] |= 0x0f;

        let mut decoded: [u32; INDEX_BUFFER.len()] = Default::default();
        assert!(decode_index_buffer(&mut decoded, &buffer).is_err());
    }

    #[test]
    fn test_decode_index_malformed_v_byte() {
        #[rustfmt::skip]
        let input = [
            0xe1, 0x20, 0x20, 0x20, 0xff, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
            0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
            0xff, 0xff, 0xff, 0xff, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
            0x20, 0x20, 0x20,
        ];

        let mut decoded = [0u32; 66];
        assert!(decode_index_buffer(&mut decoded, &input).is_err());
    }

    #[test]
    fn test_roundtrip_index_tricky() {
        let mut buffer = Vec::new();
        buffer.resize_with(
            encode_index_buffer_bound(INDEX_BUFFER_TRICKY.len(), 6),
            Default::default,
        );

        let written = encode_index_buffer(&mut buffer, &INDEX_BUFFER_TRICKY, IndexEncodingVersion::default()).unwrap();
        buffer.resize_with(written, Default::default);

        let mut decoded: [u32; INDEX_BUFFER_TRICKY.len()] = Default::default();
        assert!(decode_index_buffer(&mut decoded, &buffer).is_ok());
        assert_eq!(decoded, INDEX_BUFFER_TRICKY);
    }

    #[test]
    fn test_encode_index_empty() {
        let mut buffer = Vec::new();
        buffer.resize_with(encode_index_buffer_bound(0, 0), Default::default);

        let written = encode_index_buffer(&mut buffer, &[], IndexEncodingVersion::default()).unwrap();
        buffer.resize_with(written, Default::default);

        assert!(decode_index_buffer::<u32>(&mut [], &buffer).is_ok());
    }
}
