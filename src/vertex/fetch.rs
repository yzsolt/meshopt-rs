//! Vertex fetch analysis and optimization

use crate::util::fill_slice;

use super::Position;

#[derive(Default)]
pub struct VertexFetchStatistics {
    pub bytes_fetched: u32,
    /// Fetched bytes / vertex buffer size
    ///
    /// Best case is 1.0 (each byte is fetched once)
	pub overfetch: f32,
}

/// Returns cache hit statistics using a simplified direct mapped model.
///
/// Results may not match actual GPU performance.
pub fn analyze_vertex_fetch(indices: &[u32], vertex_count: usize, vertex_size: usize) -> VertexFetchStatistics {
	assert!(indices.len() % 3 == 0);
	assert!(vertex_size > 0 && vertex_size <= 256);

	let mut result = VertexFetchStatistics::default();

	let mut vertex_visited = vec![false; vertex_count];

	const CACHE_LINE: usize = 64;
	const CACHE_SIZE: usize = 128 * 1024;

	// simple direct mapped cache; on typical mesh data this is close to 4-way cache, and this model is a gross approximation anyway
	let mut cache = [0usize; CACHE_SIZE / CACHE_LINE];

	for index in indices {
        let index = *index as usize;

		assert!(index < vertex_count);

		vertex_visited[index] = true;

		let start_address = index * vertex_size;
		let end_address = start_address + vertex_size;

		let start_tag = start_address / CACHE_LINE;
		let end_tag = (end_address + CACHE_LINE - 1) / CACHE_LINE;

		assert!(start_tag < end_tag);

		for tag in start_tag..end_tag {
			let line = tag % (CACHE_SIZE / CACHE_LINE);

			// we store +1 since cache is filled with 0 by default
			result.bytes_fetched += (cache[line] != tag + 1) as u32 * CACHE_LINE as u32;
			cache[line] = tag + 1;
		}
	}

	let unique_vertex_count: usize = vertex_visited.iter().map(|v| *v as usize).sum();

	result.overfetch = if unique_vertex_count == 0 { 
        0.0 
    } else { 
        result.bytes_fetched as f32 / (unique_vertex_count * vertex_size) as f32 
    };

	result
}

/// Generates vertex remap to reduce the amount of GPU memory fetches during vertex processing.
///
/// Returns the number of unique vertices, which is the same as input vertex count unless some vertices are unused.
/// The resulting remap table should be used to reorder vertex/index buffers using [remap_vertex_buffer](crate::index::generator::remap_vertex_buffer)/[remap_index_buffer](crate::index::generator::remap_index_buffer).
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting remap table (`vertex_count` elements)
pub fn optimize_vertex_fetch_remap(destination: &mut [u32], indices: &[u32], vertex_count: usize) -> usize {
	assert!(indices.len() % 3 == 0);

    fill_slice(&mut destination[0..vertex_count], u32::MAX);

	let mut next_vertex = 0;

	for index in indices {
        let index = *index as usize;

		assert!(index < vertex_count);

		if destination[index] == u32::MAX {
            destination[index] = next_vertex as u32;
            next_vertex += 1;
		}
	}

	assert!(next_vertex <= vertex_count);

	next_vertex
}

/// Reorders vertices and changes indices to reduce the amount of GPU memory fetches during vertex processing.
///
/// Returns the number of unique vertices, which is the same as input vertex count unless some vertices are unused.
/// This functions works for a single vertex stream; for multiple vertex streams, use [optimize_vertex_fetch_remap] + [remap_vertex_buffer](crate::index::generator::remap_vertex_buffer) for each stream.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting vertex buffer (`vertices.len()` elements)
pub fn optimize_vertex_fetch<Vertex>(destination: &mut [Vertex], indices: &mut [u32], vertices: &[Vertex])  -> usize
where
    Vertex: Position + Copy
{
	assert!(indices.len() % 3 == 0);

	// build vertex remap table
	let mut vertex_remap = vec![u32::MAX; vertices.len()];

	let mut next_vertex = 0;

	for index in indices.iter_mut() {
        let idx = *index as usize;

		assert!(idx < vertices.len());

		let remap = &mut vertex_remap[idx];

		if *remap == u32::MAX { // vertex was not added to destination VB
			// add vertex
            destination[next_vertex] = vertices[idx];

            *remap = next_vertex as u32;
            next_vertex += 1;
		}

		// modify indices in place
		*index = *remap;
	}

	assert!(next_vertex <= vertices.len());

	next_vertex
}
