//! meshopt-rs
//!
//! # Features
//!
//! * `experimental`: Enables experimental APIs which have unstable interface and might have implementation that's not fully tested or optimized

#[cfg(feature = "experimental")]
pub mod cluster;
pub mod index;
pub mod overdraw;
#[cfg(feature = "experimental")]
pub mod simplify;
#[cfg(feature = "experimental")]
pub mod spatial_order;
pub mod stripify;
pub mod quantize;
pub mod util;
pub mod vertex;

#[derive(Clone, Copy, Default)]
struct Vector3 {
    x: f32,
    y: f32,
    z: f32
}

impl Vector3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Self {
            x, y, z
        }
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
