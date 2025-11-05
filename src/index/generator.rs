//! Index buffer generation and index/vertex buffer remapping

// This work is based on:
// John McDonald, Mark Kilgard. Crack-Free Point-Normal Triangles using Adjacent Edge Normals. 2010

use crate::{INVALID_INDEX, Stream};

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::{BuildHasherDefault, Hash, Hasher};

#[derive(Default)]
struct VertexHasher {
    state: u32,
}

impl Hasher for VertexHasher {
    fn write(&mut self, bytes: &[u8]) {
        // MurmurHash2
        const M: u32 = 0x5bd1e995;
        const R: i32 = 24;

        let mut h = self.state;

        for k4 in bytes.chunks_exact(4) {
            let mut k = u32::from_ne_bytes(k4.try_into().unwrap());

            k = k.wrapping_mul(M);
            k ^= k >> R;
            k = k.wrapping_mul(M);

            h = h.wrapping_mul(M);
            h ^= k;
        }

        self.state = h;
    }

    fn finish(&self) -> u64 {
        self.state as u64
    }
}

type BuildVertexHasher = BuildHasherDefault<VertexHasher>;

#[derive(PartialEq, Eq)]
struct Edge((u32, u32));

impl Hash for Edge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        const M: u32 = 0x5bd1e995;

        let edge = self.0;
        let mut h1 = edge.0;
        let mut h2 = edge.1;

        // MurmurHash64B finalizer
        h1 ^= h2 >> 18;
        h1 = h1.wrapping_mul(M);
        h2 ^= h1 >> 22;
        h2 = h2.wrapping_mul(M);
        h1 ^= h2 >> 17;
        h1 = h1.wrapping_mul(M);
        h2 ^= h1 >> 19;
        h2 = h2.wrapping_mul(M);

        state.write_u32(h2);
    }
}

#[derive(Default)]
struct NoopEdgeHasher {
    state: u32,
}

impl Hasher for NoopEdgeHasher {
    fn write(&mut self, bytes: &[u8]) {
        debug_assert_eq!(bytes.len(), 4);
        self.state = u32::from_ne_bytes(bytes.try_into().unwrap());
    }

    fn finish(&self) -> u64 {
        self.state as u64
    }
}

type BuildNoopEdgeHasher = BuildHasherDefault<NoopEdgeHasher>;

fn generate_vertex_remap_inner<Vertex, Lookup>(
    destination: &mut [u32],
    indices: Option<&[u32]>,
    vertex_count: usize,
    lookup: Lookup,
) -> usize
where
    Lookup: Fn(usize) -> Vertex,
    Vertex: Eq + std::hash::Hash,
{
    let index_count = match indices {
        Some(buffer) => buffer.len(),
        None => vertex_count,
    };
    assert_eq!(index_count % 3, 0);

    destination.fill(INVALID_INDEX);

    let mut table = HashMap::with_capacity_and_hasher(vertex_count, BuildVertexHasher::default());

    let mut next_vertex = 0;

    for i in 0..index_count {
        let index = match indices {
            Some(buffer) => buffer[i] as usize,
            None => i,
        };
        assert!(index < vertex_count);

        if destination[index] == INVALID_INDEX {
            match table.entry(lookup(index)) {
                Entry::Occupied(entry) => {
                    let value = *entry.get() as usize;
                    assert!(destination[value] != INVALID_INDEX);
                    destination[index] = destination[value];
                }
                Entry::Vacant(entry) => {
                    entry.insert(index);
                    destination[index] = next_vertex as u32;
                    next_vertex += 1;
                }
            }
        }
    }

    assert!(next_vertex <= vertex_count);

    next_vertex
}

/// Generates a vertex remap table from the vertex buffer and an optional index buffer and returns number of unique vertices.
///
/// As a result, all vertices that are binary equivalent map to the same (new) location, with no gaps in the resulting sequence.
/// Resulting remap table maps old vertices to new vertices and can be used in [remap_vertex_buffer]/[remap_index_buffer].
///
/// Note that binary equivalence considers all `Stream::subset` bytes, including padding which should be zero-initialized.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting remap table (`vertex_count` elements defined by `vertices`)
/// * `indices`: can be `None` if the input is unindexed
pub fn generate_vertex_remap(destination: &mut [u32], indices: Option<&[u32]>, vertices: &Stream) -> usize {
    generate_vertex_remap_inner(destination, indices, vertices.len(), |index| vertices.get(index))
}

struct StreamVertex<'a> {
    streams: &'a [Stream<'a>],
    index: usize,
}

impl<'a> PartialEq for StreamVertex<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.streams
            .iter()
            .zip(other.streams)
            .all(|(s1, s2)| s1.get(self.index) == s2.get(other.index))
    }
}

impl<'a> Eq for StreamVertex<'a> {}

impl<'a> std::hash::Hash for StreamVertex<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for stream in self.streams {
            state.write(stream.get(self.index));
        }
    }
}

/// Generates a vertex remap table from multiple vertex streams and an optional index buffer and returns number of unique vertices.
///
/// As a result, all vertices that are binary equivalent map to the same (new) location, with no gaps in the resulting sequence.
/// Resulting remap table maps old vertices to new vertices and can be used in [remap_vertex_buffer]/[remap_index_buffer].
///
/// To remap vertex buffers, you will need to call [remap_vertex_buffer] for each vertex stream.
///
/// Note that binary equivalence considers all `Stream::subset` bytes, including padding which should be zero-initialized.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting remap table (`vertex_count` elements defined by `streams`)
/// * `indices`: can be `None` if the input is unindexed
pub fn generate_vertex_remap_multi(destination: &mut [u32], indices: Option<&[u32]>, streams: &[Stream]) -> usize {
    let vertex_count = streams[0].len();
    assert!(&streams[1..].iter().all(|s| s.len() == vertex_count));

    generate_vertex_remap_inner(destination, indices, vertex_count, |index| StreamVertex {
        streams,
        index: index as usize,
    })
}

/// Generates vertex buffer from the source vertex buffer and remap table generated by [generate_vertex_remap].
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting vertex buffer (`unique_vertex_count` elements, returned by [generate_vertex_remap])
/// * `vertices`: should have the initial vertex count and not the value returned by [generate_vertex_remap]
pub fn remap_vertex_buffer<Vertex>(destination: &mut [Vertex], vertices: &[Vertex], remap: &[u32])
where
    Vertex: Copy,
{
    remap
        .iter()
        .filter(|dst| **dst != INVALID_INDEX)
        .enumerate()
        .for_each(|(src, dst)| destination[*dst as usize] = vertices[src]);
}

/// Remaps indices in-place based on the remap table generated by [generate_vertex_remap].
pub fn remap_index_buffer(indices: &mut [u32], remap: &[u32]) {
    assert_eq!(indices.len() % 3, 0);

    for v in indices {
        assert!(*v != INVALID_INDEX);

        *v = remap[*v as usize];
    }
}

fn generate_shadow_index_buffer_inner<Vertex, Lookup>(
    destination: &mut [u32],
    indices: &[u32],
    vertex_count: usize,
    lookup: Lookup,
) where
    Lookup: Fn(usize) -> Vertex,
    Vertex: Eq + std::hash::Hash,
{
    assert_eq!(indices.len() % 3, 0);

    let mut remap: Vec<u32> = vec![INVALID_INDEX; vertex_count];

    let mut table = HashMap::with_capacity_and_hasher(vertex_count, BuildVertexHasher::default());

    for (i, index) in indices.iter().enumerate() {
        let index = *index as usize;

        if remap[index] == INVALID_INDEX {
            remap[index] = match table.entry(lookup(index)) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    entry.insert(index as u32);
                    index as u32
                }
            };
        }

        destination[i] = remap[index];
    }
}

/// Generates index buffer that can be used for more efficient rendering when only a subset of the vertex attributes is necessary.
///
/// All vertices that are binary equivalent map to the first vertex in the original vertex buffer.
/// This makes it possible to use the index buffer for Z pre-pass or shadowmap rendering, while using the original index buffer for regular rendering.
///
/// Note that binary equivalence considers all `Stream::subset` bytes, including padding which should be zero-initialized.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting index buffer (`indices.len()` elements)
pub fn generate_shadow_index_buffer(destination: &mut [u32], indices: &[u32], vertices: &Stream) {
    generate_shadow_index_buffer_inner(destination, indices, vertices.len(), |index| vertices.get(index))
}

/// Generates index buffer that can be used for more efficient rendering when only a subset of the vertex attributes is necessary.
///
/// All vertices that are binary equivalent map to the first vertex in the original vertex buffer.
/// This makes it possible to use the index buffer for Z pre-pass or shadowmap rendering, while using the original index buffer for regular rendering.
///
/// Note that binary equivalence considers all `Stream::subset` bytes, including padding which should be zero-initialized.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting index buffer (`indices.len()` elements)
pub fn generate_shadow_index_buffer_multi(destination: &mut [u32], indices: &[u32], streams: &[Stream]) {
    let vertex_count = streams[0].len();
    assert!(&streams[1..].iter().all(|s| s.len() == vertex_count));

    generate_shadow_index_buffer_inner(destination, indices, vertex_count, |index| StreamVertex {
        streams,
        index: index as usize,
    })
}

fn build_position_remap(indices: &[u32], vertices: &Stream) -> Vec<u32> {
    let mut table = HashMap::with_capacity_and_hasher(vertices.len(), BuildVertexHasher::default());
    let mut remap = vec![INVALID_INDEX; vertices.len()];

    for index in indices.iter() {
        remap[*index as usize] = match table.entry(vertices.get(*index as usize)) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                entry.insert(*index);
                *index
            }
        };
    }

    remap
}

/// Generate index buffer that can be used for PN-AEN tessellation with crack-free displacement
///
/// Each triangle is converted into a 12-vertex patch with the following layout:
/// - 0, 1, 2: original triangle vertices
/// - 3, 4: opposing edge for edge 0, 1
/// - 5, 6: opposing edge for edge 1, 2
/// - 7, 8: opposing edge for edge 2, 0
/// - 9, 10, 11: dominant vertices for corners 0, 1, 2
/// The resulting patch can be rendered with hardware tessellation using PN-AEN and displacement mapping.
/// See "Tessellation on Any Budget" (John McDonald, GDC 2011) for implementation details.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting index buffer (`indices.len() * 4` elements)
#[cfg(feature = "experimental")]
pub fn generate_tessellation_index_buffer(destination: &mut [u32], indices: &[u32], vertices: &Stream) {
    assert_eq!(indices.len() % 3, 0);
    assert!(destination.len() >= indices.len() * 4);

    const NEXT: [usize; 3] = [1, 2, 0];

    // build position remap: for each vertex, which other (canonical) vertex does it map to?
    let remap = build_position_remap(indices, vertices);

    // build edge set; this stores all triangle edges but we can look these up by any other wedge
    let mut edge_table = HashMap::with_capacity_and_hasher(indices.len(), BuildNoopEdgeHasher::default());

    for i in indices.chunks_exact(3) {
        for e in 0..3 {
            let i0 = i[e];
            let i1 = i[NEXT[e]];
            assert!((i0 as usize) < vertices.len() && (i1 as usize) < vertices.len());

            edge_table.insert(Edge((remap[i0 as usize], remap[i1 as usize])), Edge((i0, i1)));
        }
    }

    // build resulting index buffer: 12 indices for each input triangle
    for (t, i) in indices.chunks_exact(3).enumerate() {
        let mut patch = [0u32; 12];

        for e in 0..3 {
            let i0 = i[e];
            let i1 = i[NEXT[e]];

            // note: this refers to the opposite edge!
            let edge = Edge((i1, i0));

            // use the same edge if opposite edge doesn't exist (border)
            let oppe = edge_table
                .get(&Edge((remap[edge.0.0 as usize], remap[edge.0.1 as usize])))
                .unwrap_or(&edge);

            // triangle index (0, 1, 2)
            patch[e] = i0;

            // opposite edge (3, 4; 5, 6; 7, 8)
            patch[3 + e * 2 + 0] = oppe.0.1;
            patch[3 + e * 2 + 1] = oppe.0.0;

            // dominant vertex (9, 10, 11)
            patch[9 + e] = remap[i0 as usize];
        }

        let offset = t * 3 * 4;
        destination[offset..offset + patch.len()].copy_from_slice(&patch);
    }
}

/// Generate index buffer that can be used as a geometry shader input with triangle adjacency topology
///
/// Each triangle is converted into a 6-vertex patch with the following layout:
/// - 0, 2, 4: original triangle vertices
/// - 1, 3, 5: vertices adjacent to edges 02, 24 and 40
/// The resulting patch can be rendered with geometry shaders using e.g. `VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY``.
/// This can be used to implement algorithms like silhouette detection/expansion and other forms of GS-driven rendering.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting index buffer (`indices.len() * 2` elements)
#[cfg(feature = "experimental")]
pub fn generate_adjacency_index_buffer(destination: &mut [u32], indices: &[u32], vertices: &Stream) {
    assert_eq!(indices.len() % 3, 0);
    assert!(destination.len() >= indices.len() * 2);

    const NEXT: [usize; 4] = [1, 2, 0, 1];

    // build position remap: for each vertex, which other (canonical) vertex does it map to?
    let remap = build_position_remap(indices, vertices);

    // build edge set; this stores all triangle edges but we can look these up by any other wedge
    let mut edge_vertex_table = HashMap::with_capacity_and_hasher(indices.len(), BuildNoopEdgeHasher::default());

    for i in indices.chunks_exact(3) {
        for e in 0..3 {
            let i0 = i[e];
            let i1 = i[NEXT[e]];
            let i2 = i[NEXT[e + 1]];
            assert!((i0 as usize) < vertices.len() && (i1 as usize) < vertices.len() && (i2 as usize) < vertices.len());

            match edge_vertex_table.entry(Edge((remap[i0 as usize], remap[i1 as usize]))) {
                Entry::Vacant(entry) => {
                    // store vertex opposite to the edge
                    entry.insert(i2);
                }
                _ => {}
            }
        }
    }

    // build resulting index buffer: 6 indices for each input triangle
    for (t, i) in indices.chunks_exact(3).enumerate() {
        let mut patch = [0u32; 6];

        for e in 0..3 {
            let i0 = i[e];
            let i1 = i[NEXT[e]];

            // note: this refers to the opposite edge!
            let edge = Edge((i1, i0));
            let oppe = edge_vertex_table.get(&Edge((remap[edge.0.0 as usize], remap[edge.0.1 as usize])));

            patch[e * 2 + 0] = i0;
            patch[e * 2 + 1] = if let Some(vertex) = oppe { *vertex } else { i0 };
        }

        let offset = t * 3 * 2;
        destination[offset..offset + patch.len()].copy_from_slice(&patch);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tesselation() {
        // 0 1/4
        // 2/5 3
        const VB: [[f32; 3]; 6] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        const IB: [u32; 6] = [0, 1, 2, 5, 4, 3];

        let mut tessib = [0u32; 24];
        generate_tessellation_index_buffer(&mut tessib, &IB, &Stream::from_slice(&VB));

        #[rustfmt::skip]
        let expected = [
            // patch 0
            0, 1, 2,
            0, 1,
            4, 5,
            2, 0,
            0, 1, 2,

            // patch 1
            5, 4, 3,
            2, 1,
            4, 3,
            3, 5,
            2, 1, 3,
        ];

        assert_eq!(tessib, expected);
    }

    #[test]
    fn test_adjacency() {
        // 0 1/4
        // 2/5 3
        const VB: [[f32; 3]; 6] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        const IB: [u32; 6] = [0, 1, 2, 5, 4, 3];

        let mut adjib = [0u32; 12];
        generate_adjacency_index_buffer(&mut adjib, &IB, &Stream::from_slice(&VB));

        #[rustfmt::skip]
        let expected = [
            // patch 0
            0, 0,
            1, 3,
            2, 2,

            // patch 1
            5, 0,
            4, 4,
            3, 3,
        ];

        assert_eq!(adjib, expected);
    }
}
