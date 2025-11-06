use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub struct NoopHasher {
    state: u32,
}

impl Hasher for NoopHasher {
    fn write(&mut self, bytes: &[u8]) {
        debug_assert_eq!(bytes.len(), 4);
        self.state = u32::from_ne_bytes(bytes.try_into().unwrap());
    }

    fn finish(&self) -> u64 {
        self.state as u64
    }
}

pub type BuildNoopHasher = BuildHasherDefault<NoopHasher>;
