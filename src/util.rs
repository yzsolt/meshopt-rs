#![doc(hidden)]

use std::io::{Read, Write};

#[inline(always)]
pub(crate) fn zero_inverse(value: f32) -> f32 {
    if value != 0.0 { 1.0 / value } else { 0.0 }
}

pub(crate) fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

pub(crate) fn as_mut_bytes<T>(data: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, std::mem::size_of_val(data)) }
}

pub(crate) fn read_byte<R: Read>(data: &mut R) -> u8 {
    let mut byte = [0];
    data.read_exact(&mut byte).unwrap();
    byte[0]
}

pub(crate) fn write_byte<W: Write>(data: &mut W, byte: u8) -> usize {
    data.write(&[byte]).unwrap() // `write` always succeeds on slices
}

pub(crate) fn write_exact<W: Write>(data: &mut W, bytes: &[u8]) -> Option<usize> {
    match data.write(bytes) {
        Ok(written) => {
            if written == bytes.len() {
                Some(written)
            } else {
                None
            }
        }
        Err(_) => None,
    }
}
