//! meshopt-rs
//!
//! # Features
//!
//! * `experimental`: Enables experimental APIs which have unstable interface and might have implementation that's not fully tested or optimized

#![allow(clippy::identity_op)]
#![allow(clippy::erasing_op)]

pub mod cluster;
mod hash;
pub mod index;
pub mod overdraw;
pub mod quantize;
pub mod simplify;
#[cfg(feature = "experimental")]
pub mod spatial_order;
pub mod stripify;
pub mod util;
pub mod vertex;

use std::ops::Range;

use crate::vertex::Position;

pub const INVALID_INDEX: u32 = u32::MAX;

/// A stream of value groups which are meant to be used together (e.g. 3 floats representing a vertex position).
pub struct Stream<'a> {
    data: &'a [u8],
    stride: usize,
    subset: Range<usize>,
}

impl<'a> Stream<'a> {
    /// Creates a stream from a slice.
    ///
    /// # Example
    ///
    /// ```
    /// use meshopt_rs::Stream;
    ///
    /// let positions = vec![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]];
    /// let stream = Stream::from_slice(&positions);
    ///
    /// assert_eq!(stream.len(), positions.len());
    /// ```
    pub fn from_slice<T>(slice: &'a [T]) -> Self {
        let value_size = std::mem::size_of::<T>();

        let data = util::as_bytes(slice);

        Self::from_bytes(data, value_size, 0..value_size)
    }

    /// Creates a stream from a slice with the given byte subset.
    ///
    /// # Arguments
    ///
    /// * `subset`: subset of data to use inside a `T`
    ///
    /// # Example
    ///
    /// ```
    /// use meshopt_rs::Stream;
    ///
    /// #[derive(Clone, Default)]
    /// #[repr(C)]
    /// struct Vertex {
    ///     position: [f32; 3],
    ///     normal: [f32; 3],
    ///     uv: [f32; 2],
    /// }
    ///
    /// let normals_offset = std::mem::size_of::<f32>() * 3;
    /// let normals_size = std::mem::size_of::<f32>() * 3;
    ///
    /// let vertices = vec![Vertex::default(); 1];
    /// let normal_stream = Stream::from_slice_with_subset(&vertices, normals_offset..normals_offset+normals_size);
    ///
    /// assert_eq!(normal_stream.len(), 1);
    /// ```
    pub fn from_slice_with_subset<T>(slice: &'a [T], subset: Range<usize>) -> Self {
        let value_size = std::mem::size_of::<T>();

        let data = util::as_bytes(slice);

        Self::from_bytes(data, value_size, subset)
    }

    /// Creates a stream from raw bytes.
    ///
    /// # Arguments
    ///
    /// * `stride`: stride between value groups
    /// * `subset`: subset of data to use inside a value group
    pub fn from_bytes<T>(slice: &'a [T], stride: usize, subset: Range<usize>) -> Self {
        assert!(subset.end <= stride);

        let value_size = std::mem::size_of::<T>();

        let stride = stride * value_size;
        let subset = subset.start * value_size..subset.end * value_size;

        let data = util::as_bytes(slice);

        Self { data, stride, subset }
    }

    fn get(&self, index: usize) -> &'a [u8] {
        let i = index * self.stride;
        &self.data[i + self.subset.start..i + self.subset.end]
    }

    /// Returns length of the stream in value groups.
    pub fn len(&self) -> usize {
        self.data.len() / self.stride
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct Vector3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vector3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Self { x, y, z }
    }

    pub fn normalize(&mut self) -> f32 {
        let length = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();

        if length > 0.0 {
            self.x /= length;
            self.y /= length;
            self.z /= length;
        }

        length
    }
}

impl Position for Vector3 {
    fn pos(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}
