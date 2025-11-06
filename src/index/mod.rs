pub mod buffer;
pub mod generator;
pub mod sequence;

use crate::util::{read_byte, write_byte};

use std::io::{Read, Write};

#[derive(Debug)]
pub enum DecodeError {
    InvalidHeader,
    UnsupportedVersion,
    ExtraBytes,
    UnexpectedEof,
}

#[derive(Default)]
pub enum IndexEncodingVersion {
    /// Decodable by all versions
    V0,
    /// Decodable by 0.14+
    #[default]
    V1,
}

impl From<IndexEncodingVersion> for u8 {
    fn from(value: IndexEncodingVersion) -> u8 {
        match value {
            IndexEncodingVersion::V0 => 0,
            IndexEncodingVersion::V1 => 1,
        }
    }
}

fn encode_v_byte<W: Write>(data: &mut W, mut v: u32) {
    // encode 32-bit value in up to 5 7-bit groups
    loop {
        write_byte(data, ((v & 127) | (if v > 127 { 128 } else { 0 })) as u8);
        v >>= 7;

        if v == 0 {
            break;
        }
    }
}

fn decode_v_byte<R: Read>(data: &mut R) -> u32 {
    let lead = read_byte(data) as u32;

    // fast path: single byte
    if lead < 128 {
        return lead;
    }

    // slow path: up to 4 extra bytes
    // note that this loop always terminates, which is important for malformed data
    let mut result = lead & 127;
    let mut shift: u32 = 7;

    for _ in 0..4 {
        let group = read_byte(data) as u32;
        result |= (group & 127) << shift;
        shift += 7;

        if group < 128 {
            break;
        }
    }

    result
}
