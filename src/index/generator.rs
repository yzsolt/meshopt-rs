//! Index buffer generation and index/vertex buffer remapping

// This work is based on:
// John McDonald, Mark Kilgard. Crack-Free Point-Normal Triangles using Adjacent Edge Normals. 2010
// John Hable. Variable Rate Shading with Visibility Buffer Rendering. 2024

use crate::hash::BuildNoopHasher;
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

        for k4 in bytes.as_chunks().0 {
            let mut k = u32::from_ne_bytes(*k4);

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
pub struct Edge(pub (u32, u32));

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
                    let value = *entry.get();
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
        index,
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
        index,
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
///
/// The resulting patch can be rendered with hardware tessellation using PN-AEN and displacement mapping.
/// See "Tessellation on Any Budget" (John McDonald, GDC 2011) for implementation details.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting index buffer (`indices.len() * 4` elements)
pub fn generate_tessellation_index_buffer(destination: &mut [u32], indices: &[u32], vertices: &Stream) {
    assert_eq!(indices.len() % 3, 0);
    assert!(destination.len() >= indices.len() * 4);

    const NEXT: [usize; 3] = [1, 2, 0];

    // build position remap: for each vertex, which other (canonical) vertex does it map to?
    let remap = build_position_remap(indices, vertices);

    // build edge set; this stores all triangle edges but we can look these up by any other wedge
    let mut edge_table = HashMap::with_capacity_and_hasher(indices.len(), BuildNoopHasher::default());

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

/// Generates index buffer that can be used for visibility buffer rendering and returns the size of the reorder table
///
/// Each triangle's provoking vertex index is equal to primitive id; this allows passing it to the fragment shader using nointerpolate attribute.
/// This is important for performance on hardware where primitive id can't be accessed efficiently in fragment shader.
/// The reorder table stores the original vertex id for each vertex in the new index buffer, and should be used in the vertex shader to load vertex data.
/// The provoking vertex is assumed to be the first vertex in the triangle; if this is not the case (OpenGL), rotate each triangle (abc -> bca) before rendering.
/// For maximum efficiency the input index buffer should be optimized for vertex cache first.
///
/// # Arguments
///
/// - `destination`: must contain enough space for the resulting index buffer (`indices.len()` elements)
/// - `reorder`: must contain enough space for the worst case reorder table (`vertex_count` + `indices.len()`/3 elements)
#[cfg(feature = "experimental")]
pub fn generate_provoking_index_buffer(
    destination: &mut [u32],
    reorder: &mut [u32],
    indices: &[u32],
    vertex_count: usize,
) -> usize {
    assert!(indices.len().is_multiple_of(3));

    let mut remap = vec![INVALID_INDEX; vertex_count];

    // compute vertex valence; this is used to prioritize least used corner
    // note: we use 8-bit counters for performance; for outlier vertices the valence is incorrect but that just affects the heuristic
    let mut valence = vec![0u8; vertex_count];

    for index in indices.iter().map(|i| *i as usize) {
        assert!(index < vertex_count);

        valence[index] += 1;
    }

    let mut reorder_offset: usize = 0;

    // assign provoking vertices; leave the rest for the next pass
    for (abc, dst) in indices
        .as_chunks::<3>()
        .0
        .iter()
        .zip(destination.as_chunks_mut::<3>().0.iter_mut())
    {
        assert!(abc.iter().all(|i| (*i as usize) < vertex_count));
        let [a, b, c] = abc;
        let [mut a, mut b, mut c] = [*a as usize, *b as usize, *c as usize];

        // try to rotate triangle such that provoking vertex hasn't been seen before
        // if multiple vertices are new, prioritize the one with least valence
        // this reduces the risk that a future triangle will have all three vertices seen
        let va = if remap[a] == INVALID_INDEX {
            valence[a] as u32
        } else {
            INVALID_INDEX
        };
        let vb = if remap[b] == INVALID_INDEX {
            valence[b] as u32
        } else {
            INVALID_INDEX
        };
        let vc = if remap[c] == INVALID_INDEX {
            valence[c] as u32
        } else {
            INVALID_INDEX
        };

        if vb != INVALID_INDEX && vb <= va && vb <= vc {
            // abc -> bca
            let t = a;
            a = b;
            b = c;
            c = t;
        } else if vc != INVALID_INDEX && vc <= va && vc <= vb {
            // abc -> cab
            let t = c;
            c = b;
            b = a;
            a = t;
        }

        let newidx = reorder_offset as u32;

        // now remap[a] = INVALID_INDEX or all three vertices are old
        // recording remap[a] makes it possible to remap future references to the same index, conserving space
        if remap[a] == INVALID_INDEX {
            remap[a] = newidx;
        }

        // we need to clone the provoking vertex to get a unique index
        // if all three are used the choice is arbitrary since no future triangle will be able to reuse any of these
        reorder[reorder_offset] = a as u32;
        reorder_offset += 1;

        // note: first vertex is final, the other two will be fixed up in next pass
        dst[0] = newidx;
        dst[1] = b as u32;
        dst[2] = c as u32;

        // update vertex valences for corner heuristic
        valence[a] -= 1;
        valence[b] -= 1;
        valence[c] -= 1;
    }

    // remap or clone non-provoking vertices (iterating to skip provoking vertices)
    let mut step = 1;
    let mut i = 1;

    while i < indices.len() {
        let index = destination[i] as usize;

        if remap[index] == INVALID_INDEX {
            // we haven't seen the vertex before as a provoking vertex
            // to maintain the reference to the original vertex we need to clone it
            let newidx = reorder_offset as u32;

            remap[index] = newidx;
            reorder[reorder_offset] = index as u32;
            reorder_offset += 1;
        }

        destination[i] = remap[index];

        i += step;
        step ^= 3;
    }

    assert!(reorder_offset <= vertex_count + indices.len() / 3);

    reorder_offset
}

/// Generate index buffer that can be used as a geometry shader input with triangle adjacency topology
///
/// Each triangle is converted into a 6-vertex patch with the following layout:
/// - 0, 2, 4: original triangle vertices
/// - 1, 3, 5: vertices adjacent to edges 02, 24 and 40
///
/// The resulting patch can be rendered with geometry shaders using e.g. `VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY`.
/// This can be used to implement algorithms like silhouette detection/expansion and other forms of GS-driven rendering.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting index buffer (`indices.len() * 2` elements)
pub fn generate_adjacency_index_buffer(destination: &mut [u32], indices: &[u32], vertices: &Stream) {
    assert_eq!(indices.len() % 3, 0);
    assert!(destination.len() >= indices.len() * 2);

    const NEXT: [usize; 4] = [1, 2, 0, 1];

    // build position remap: for each vertex, which other (canonical) vertex does it map to?
    let remap = build_position_remap(indices, vertices);

    // build edge set; this stores all triangle edges but we can look these up by any other wedge
    let mut edge_vertex_table = HashMap::with_capacity_and_hasher(indices.len(), BuildNoopHasher::default());

    for i in indices.chunks_exact(3) {
        for e in 0..3 {
            let i0 = i[e];
            let i1 = i[NEXT[e]];
            let i2 = i[NEXT[e + 1]];
            assert!((i0 as usize) < vertices.len() && (i1 as usize) < vertices.len() && (i2 as usize) < vertices.len());

            if let Entry::Vacant(entry) = edge_vertex_table.entry(Edge((remap[i0 as usize], remap[i1 as usize]))) {
                // store vertex opposite to the edge
                entry.insert(i2);
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

    #[test]
    #[cfg(feature = "experimental")]
    fn test_provoking() {
        // 0 1 2
        // 3 4 5
        #[rustfmt::skip]
        let ib = [
            0, 1, 3,
            3, 1, 4,
            1, 2, 4,
            4, 2, 5,
            0, 2, 4,
        ];

        let mut pib = [0u32; 15];
        let mut pre = [0u32; 6 + 5]; // limit is vertex count + triangle count
        let res = generate_provoking_index_buffer(&mut pib, &mut pre, &ib, 6);

        #[rustfmt::skip]
        let expectedib = [
            0, 5, 1,
            1, 4, 0,
            2, 4, 1,
            3, 4, 2,
            4, 5, 2,
        ];

        let expectedre = [3, 1, 2, 5, 4, 0];

        assert_eq!(6, res);
        assert_eq!(expectedib, pib);
        assert_eq!(expectedre, &pre[..expectedre.len()]);
    }
}
