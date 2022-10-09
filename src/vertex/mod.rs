pub mod buffer;
pub mod cache;
pub mod fetch;
pub mod filter;

#[derive(Debug)]
pub enum DecodeError {
    InvalidHeader,
    UnsupportedVersion,
    ExtraBytes,
    UnexpectedEof,
}

pub enum VertexEncodingVersion {
    /// Decodable by all versions
    V0,
}

impl Default for VertexEncodingVersion {
    fn default() -> Self {
        Self::V0
    }
}

impl Into<u8> for VertexEncodingVersion {
    fn into(self) -> u8 {
        match self {
            Self::V0 => 0,
        }
    }
}

pub trait Position {
    fn pos(&self) -> [f32; 3];
}

impl Position for [f32; 3] {
    #[inline]
    fn pos(&self) -> [f32; 3] {
        *self
    }
}

pub(crate) fn calc_pos_extents<Vertex>(vertices: &[Vertex]) -> ([f32; 3], f32)
where
    Vertex: Position,
{
    let mut minv = [f32::MAX; 3];
    let mut maxv = [-f32::MAX; 3];

    for vertex in vertices {
        let v = vertex.pos();

        for j in 0..3 {
            minv[j] = minv[j].min(v[j]);
            maxv[j] = maxv[j].max(v[j]);
        }
    }

    let extent = (maxv[0] - minv[0]).max(maxv[1] - minv[1]).max(maxv[2] - minv[2]);

    (minv, extent)
}
