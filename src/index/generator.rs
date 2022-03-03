//! Index buffer generation and index/vertex buffer remapping

use crate::{Stream, INVALID_INDEX};

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

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
