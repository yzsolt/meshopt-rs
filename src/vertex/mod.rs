pub mod buffer;
pub mod cache;
pub mod fetch;
#[cfg(feature = "experimental")]
pub mod filter;

#[derive(Debug)]
pub enum DecodeError {
    InvalidHeader,
    UnsupportedVersion,
    ExtraBytes,
    UnexpectedEof,
}

#[derive(Default)]
pub enum VertexEncodingVersion {
    /// Decodable by all versions
    #[default]
    V0,
}

impl From<VertexEncodingVersion> for u8 {
    fn from(value: VertexEncodingVersion) -> Self {
        match value {
            VertexEncodingVersion::V0 => 0,
        }
    }
}

pub trait Vertex<const ATTR_COUNT: usize = 0> {
    const HAS_COLORS: bool = false;

    fn pos(&self) -> [f32; 3];

    fn attrs(&self) -> [f32; ATTR_COUNT] {
        [0f32; ATTR_COUNT]
    }

    fn colors(&self) -> [f32; 3] {
        [0f32; 3]
    }
}

impl Vertex for [f32; 3] {
    #[inline]
    fn pos(&self) -> [f32; 3] {
        *self
    }
}

pub(crate) fn calc_pos_extents<V, const ATTR_COUNT: usize>(vertices: &[V]) -> ([f32; 3], f32)
where
    V: Vertex<ATTR_COUNT>,
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

#[derive(Default)]
pub(crate) struct TriangleAdjacency {
    pub counts: Vec<u32>,
    pub offsets: Vec<u32>,
    pub data: Vec<u32>,
}

pub(crate) fn build_triangle_adjacency(adjacency: &mut TriangleAdjacency, indices: &[u32], vertex_count: usize) {
    let face_count = indices.len() / 3;

    // allocate arrays
    adjacency.counts = vec![0; vertex_count];
    adjacency.offsets = vec![0; vertex_count];
    adjacency.data = vec![0; indices.len()];

    // fill triangle counts
    for index in indices {
        let index = *index as usize;
        adjacency.counts[index] += 1;
    }

    // fill offset table
    let mut offset = 0;

    for i in 0..vertex_count {
        adjacency.offsets[i] = offset;
        offset += adjacency.counts[i];
    }

    assert!(offset as usize == indices.len());

    // fill triangle data
    for i in 0..face_count {
        for j in 0..3 {
            let a = indices[i * 3 + j] as usize;
            let o = &mut adjacency.offsets[a];
            adjacency.data[*o as usize] = i as u32;
            *o += 1;
        }
    }

    // fix offsets that have been disturbed by the previous pass
    for i in 0..vertex_count {
        assert!(adjacency.offsets[i] >= adjacency.counts[i]);

        adjacency.offsets[i] -= adjacency.counts[i];
    }
}
