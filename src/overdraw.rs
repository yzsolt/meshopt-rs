//! Overdraw analysis and optimization

use crate::Vector3;
use crate::quantize::quantize_unorm;
use crate::util::zero_inverse;
use crate::vertex::{Position, calc_pos_extents};

const VIEWPORT: usize = 256;

#[derive(Default)]
pub struct OverdrawStatistics {
    pub pixels_covered: u32,
    pub pixels_shaded: u32,
    /// Shaded pixels / covered pixels; best case is 1.0
    pub overdraw: f32,
}

struct OverdrawBuffer {
    z: [[[f32; 2]; VIEWPORT]; VIEWPORT],
    overdraw: [[[u32; 2]; VIEWPORT]; VIEWPORT],
}

impl Default for OverdrawBuffer {
    fn default() -> Self {
        Self {
            z: [[[0.0; 2]; VIEWPORT]; VIEWPORT],
            overdraw: [[[0; 2]; VIEWPORT]; VIEWPORT],
        }
    }
}

fn compute_depth_gradients(v1: Vector3, v2: Vector3, v3: Vector3) -> (f32, f32, f32) {
    // z2 = z1 + dzdx * (x2 - x1) + dzdy * (y2 - y1)
    // z3 = z1 + dzdx * (x3 - x1) + dzdy * (y3 - y1)
    // (x2-x1 y2-y1)(dzdx) = (z2-z1)
    // (x3-x1 y3-y1)(dzdy)   (z3-z1)
    // we'll solve it with Cramer's rule
    let det = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x);
    let invdet = zero_inverse(det);

    let dzdx = (v2.z - v1.z) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.z - v1.z) * invdet;
    let dzdy = (v2.x - v1.x) * (v3.z - v1.z) - (v2.z - v1.z) * (v3.x - v1.x) * invdet;

    (det, dzdx, dzdy)
}

// half-space fixed point triangle rasterizer
fn rasterize(buffer: &mut OverdrawBuffer, mut v1: Vector3, mut v2: Vector3, mut v3: Vector3) {
    // compute depth gradients
    let (det, mut dzx, mut dzy) = compute_depth_gradients(v1, v2, v3);
    let sign = det > 0.0;

    // flip backfacing triangles to simplify rasterization logic
    if sign {
        // flipping v2 & v3 preserves depth gradients since they're based on v1
        std::mem::swap(&mut v2, &mut v3);

        // flip depth since we rasterize backfacing triangles to second buffer with reverse Z; only v1z is used below
        v1.z = VIEWPORT as f32 - v1.z;
        dzx = -dzx;
        dzy = -dzy;
    }

    // coordinates, 28.4 fixed point
    let x1 = (16.0 * v1.x + 0.5) as i32;
    let x2 = (16.0 * v2.x + 0.5) as i32;
    let x3 = (16.0 * v3.x + 0.5) as i32;

    let y1 = (16.0 * v1.y + 0.5) as i32;
    let y2 = (16.0 * v2.y + 0.5) as i32;
    let y3 = (16.0 * v3.y + 0.5) as i32;

    // bounding rectangle, clipped against viewport
    // since we rasterize pixels with covered centers, min >0.5 should round up
    // as for max, due to top-left filling convention we will never rasterize right/bottom edges
    // so max >= 0.5 should round down
    let minx = ((x1.min(x2.min(x3)) + 7) >> 4).max(0);
    let maxx = ((x1.max(x2.max(x3)) + 7) >> 4).min(VIEWPORT as i32);
    let miny = ((y1.min(y2.min(y3)) + 7) >> 4).max(0);
    let maxy = ((y1.max(y2.max(y3)) + 7) >> 4).min(VIEWPORT as i32);

    // deltas, 28.4 fixed point
    let dx12 = x1 - x2;
    let dx23 = x2 - x3;
    let dx31 = x3 - x1;

    let dy12 = y1 - y2;
    let dy23 = y2 - y3;
    let dy31 = y3 - y1;

    // fill convention correction
    let tl1 = (dy12 < 0 || (dy12 == 0 && dx12 > 0)) as i32;
    let tl2 = (dy23 < 0 || (dy23 == 0 && dx23 > 0)) as i32;
    let tl3 = (dy31 < 0 || (dy31 == 0 && dx31 > 0)) as i32;

    // half edge equations, 24.8 fixed point
    // note that we offset minx/miny by half pixel since we want to rasterize pixels with covered centers
    let fx = (minx << 4) + 8;
    let fy = (miny << 4) + 8;
    let mut cy1 = dx12 * (fy - y1) - dy12 * (fx - x1) + tl1 - 1;
    let mut cy2 = dx23 * (fy - y2) - dy23 * (fx - x2) + tl2 - 1;
    let mut cy3 = dx31 * (fy - y3) - dy31 * (fx - x3) + tl3 - 1;
    let mut zy = v1.z + (dzx * (fx - x1) as f32 + dzy * (fy - y1) as f32) * (1.0 / 16.0);

    for y in miny..maxy {
        let y = y as usize;

        let mut cx1 = cy1;
        let mut cx2 = cy2;
        let mut cx3 = cy3;
        let mut zx = zy;

        for x in minx..maxx {
            let x = x as usize;
            let sign = sign as usize;

            // check if all CXn are non-negative
            if cx1 | cx2 | cx3 >= 0 {
                if zx >= buffer.z[y][x][sign] {
                    buffer.z[y][x][sign] = zx;
                    buffer.overdraw[y][x][sign] += 1;
                }
            }

            // signed left shift is UB for negative numbers so use unsigned-signed casts
            // FIXME: in Rust too?
            cx1 -= ((dy12 as u32) << 4) as i32;
            cx2 -= ((dy23 as u32) << 4) as i32;
            cx3 -= ((dy31 as u32) << 4) as i32;
            zx += dzx;
        }

        // signed left shift is UB for negative numbers so use unsigned-signed casts
        // FIXME: in Rust too?
        cy1 += ((dx12 as u32) << 4) as i32;
        cy2 += ((dx23 as u32) << 4) as i32;
        cy3 += ((dx31 as u32) << 4) as i32;
        zy += dzy;
    }
}

/// Returns overdraw statistics using a software rasterizer.
///
/// Results may not match actual GPU performance.
pub fn analyze_overdraw<Vertex>(indices: &[u32], vertices: &[Vertex]) -> OverdrawStatistics
where
    Vertex: Position,
{
    assert!(indices.len() % 3 == 0);

    let mut result = OverdrawStatistics::default();

    let (minv, extent) = calc_pos_extents(vertices);

    let scale = VIEWPORT as f32 / extent;

    let mut triangles = vec![0.0; indices.len() * 3];

    for (i, index) in indices.iter().enumerate() {
        let index = *index as usize;

        let v = vertices[index].pos();

        triangles[i * 3 + 0] = (v[0] - minv[0]) * scale;
        triangles[i * 3 + 1] = (v[1] - minv[1]) * scale;
        triangles[i * 3 + 2] = (v[2] - minv[2]) * scale;
    }

    let mut buffer = Box::default();

    for axis in 0..3 {
        *buffer = OverdrawBuffer::default();

        for vn in triangles.chunks_exact(9) {
            let vn0 = &vn[0..3];
            let vn1 = &vn[3..6];
            let vn2 = &vn[6..9];

            match axis {
                0 => rasterize(
                    &mut buffer,
                    Vector3::new(vn0[2], vn0[1], vn0[0]),
                    Vector3::new(vn1[2], vn1[1], vn1[0]),
                    Vector3::new(vn2[2], vn2[1], vn2[0]),
                ),
                1 => rasterize(
                    &mut buffer,
                    Vector3::new(vn0[0], vn0[2], vn0[1]),
                    Vector3::new(vn1[0], vn1[2], vn1[1]),
                    Vector3::new(vn2[0], vn2[2], vn2[1]),
                ),
                2 => rasterize(
                    &mut buffer,
                    Vector3::new(vn0[1], vn0[0], vn0[2]),
                    Vector3::new(vn1[1], vn1[0], vn1[2]),
                    Vector3::new(vn2[1], vn2[0], vn2[2]),
                ),
                _ => unreachable!(),
            }
        }

        for y in 0..VIEWPORT {
            for x in 0..VIEWPORT {
                for s in 0..2 {
                    let overdraw = buffer.overdraw[y][x][s];

                    result.pixels_covered += (overdraw > 0) as u32;
                    result.pixels_shaded += overdraw;
                }
            }
        }
    }

    result.overdraw = if result.pixels_covered > 0 {
        result.pixels_shaded as f32 / result.pixels_covered as f32
    } else {
        0.0
    };

    result
}

fn calculate_sort_data<Vertex>(sort_data: &mut [f32], indices: &[u32], vertices: &[Vertex], clusters: &[u32])
where
    Vertex: Position,
{
    let mut mesh_centroid = [0.0; 3];

    for index in indices {
        let p = vertices[*index as usize].pos();

        mesh_centroid[0] += p[0];
        mesh_centroid[1] += p[1];
        mesh_centroid[2] += p[2];
    }

    mesh_centroid[0] /= indices.len() as f32;
    mesh_centroid[1] /= indices.len() as f32;
    mesh_centroid[2] /= indices.len() as f32;

    for (cluster_idx, cluster_begin) in clusters.iter().enumerate() {
        let cluster_end = if let Some(begin) = clusters.get(cluster_idx + 1) {
            (*begin) as usize * 3
        } else {
            indices.len()
        };
        let cluster = (*cluster_begin as usize) * 3..cluster_end;
        assert!(!cluster.is_empty());

        let mut cluster_area = 0.0;
        let mut cluster_centroid = [0.0; 3];
        let mut cluster_normal = [0.0f32; 3];

        for i in indices[cluster].chunks_exact(3) {
            let p0 = vertices[i[0] as usize].pos();
            let p1 = vertices[i[1] as usize].pos();
            let p2 = vertices[i[2] as usize].pos();

            let p10 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let p20 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

            let normalx = p10[1] * p20[2] - p10[2] * p20[1];
            let normaly = p10[2] * p20[0] - p10[0] * p20[2];
            let normalz = p10[0] * p20[1] - p10[1] * p20[0];

            let area = (normalx * normalx + normaly * normaly + normalz * normalz).sqrt();

            cluster_centroid[0] += (p0[0] + p1[0] + p2[0]) * (area / 3.0);
            cluster_centroid[1] += (p0[1] + p1[1] + p2[1]) * (area / 3.0);
            cluster_centroid[2] += (p0[2] + p1[2] + p2[2]) * (area / 3.0);
            cluster_normal[0] += normalx;
            cluster_normal[1] += normaly;
            cluster_normal[2] += normalz;
            cluster_area += area;
        }

        let inv_cluster_area = zero_inverse(cluster_area);

        cluster_centroid[0] *= inv_cluster_area;
        cluster_centroid[1] *= inv_cluster_area;
        cluster_centroid[2] *= inv_cluster_area;

        let cluster_normal_length = (cluster_normal[0] * cluster_normal[0]
            + cluster_normal[1] * cluster_normal[1]
            + cluster_normal[2] * cluster_normal[2])
            .sqrt();
        let inv_cluster_normal_length = zero_inverse(cluster_normal_length);

        cluster_normal[0] *= inv_cluster_normal_length;
        cluster_normal[1] *= inv_cluster_normal_length;
        cluster_normal[2] *= inv_cluster_normal_length;

        let centroid_vector = [
            cluster_centroid[0] - mesh_centroid[0],
            cluster_centroid[1] - mesh_centroid[1],
            cluster_centroid[2] - mesh_centroid[2],
        ];

        sort_data[cluster_idx] = centroid_vector[0] * cluster_normal[0]
            + centroid_vector[1] * cluster_normal[1]
            + centroid_vector[2] * cluster_normal[2];
    }
}

fn calculate_sort_order_radix(sort_order: &mut [u32], sort_data: &[f32], sort_keys: &mut [u16]) {
    // compute sort data bounds and renormalize, using fixed point snorm
    let mut sort_data_max: f32 = 0.001;

    for data in sort_data {
        sort_data_max = sort_data_max.max(data.abs());
    }

    const SORT_BITS: i32 = 11;

    for (data, key) in sort_data.iter().zip(sort_keys.iter_mut()) {
        // note that we flip distribution since high dot product should come first
        let sort_key = 0.5 - 0.5 * (data / sort_data_max);

        *key = (quantize_unorm(sort_key, SORT_BITS) & ((1 << SORT_BITS) - 1)) as u16;
    }

    // fill histogram for counting sort
    let mut histogram = [0; 1 << SORT_BITS];

    for key in sort_keys.iter() {
        histogram[*key as usize] += 1;
    }

    // compute offsets based on histogram data
    let mut histogram_sum = 0;

    for i in 0..histogram.len() {
        let count = histogram[i];
        histogram[i] = histogram_sum;
        histogram_sum += count;
    }

    assert_eq!(histogram_sum, sort_keys.len());

    // compute sort order based on offsets
    for i in 0..sort_keys.len() {
        let idx = &mut histogram[sort_keys[i] as usize];
        sort_order[*idx as usize] = i as u32;
        *idx += 1;
    }
}

fn update_cache(a: u32, b: u32, c: u32, cache_size: u32, cache_timestamps: &mut [u32], timestamp: &mut u32) -> u32 {
    let mut cache_misses = 0;

    // if vertex is not in cache, put it in cache
    if *timestamp - cache_timestamps[a as usize] > cache_size {
        cache_timestamps[a as usize] = *timestamp;
        *timestamp += 1;
        cache_misses += 1;
    }

    if *timestamp - cache_timestamps[b as usize] > cache_size {
        cache_timestamps[b as usize] = *timestamp;
        *timestamp += 1;
        cache_misses += 1;
    }

    if *timestamp - cache_timestamps[c as usize] > cache_size {
        cache_timestamps[c as usize] = *timestamp;
        *timestamp += 1;
        cache_misses += 1;
    }

    cache_misses
}

fn generate_hard_boundaries(
    destination: &mut [u32],
    indices: &[u32],
    cache_size: u32,
    cache_timestamps: &mut [u32],
) -> usize {
    let mut timestamp = cache_size as u32 + 1;

    let face_count = indices.len() / 3;

    let mut result = 0;

    for i in 0..face_count {
        let m = update_cache(
            indices[i * 3 + 0],
            indices[i * 3 + 1],
            indices[i * 3 + 2],
            cache_size,
            cache_timestamps,
            &mut timestamp,
        );

        // when all three vertices are not in the cache it's usually relatively safe to assume that this is a new patch in the mesh
        // that is disjoint from previous vertices; sometimes it might come back to reference existing vertices but that frequently
        // suggests an inefficiency in the vertex cache optimization algorithm
        // usually the first triangle has 3 misses unless it's degenerate - thus we make sure the first cluster always starts with 0
        if i == 0 || m == 3 {
            destination[result] = i as u32;
            result += 1;
        }
    }

    assert!(result <= indices.len() / 3);

    result
}

fn generate_soft_boundaries(
    destination: &mut [u32],
    indices: &[u32],
    clusters: &[u32],
    cache_size: u32,
    threshold: f32,
    cache_timestamps: &mut [u32],
) -> usize {
    let mut timestamp = 0;

    let mut result = 0;

    for (cluster_idx, start) in clusters.iter().enumerate() {
        let end = if let Some(begin) = clusters.get(cluster_idx + 1) {
            *begin as usize
        } else {
            indices.len() / 3
        };
        let cluster = (*start as usize)..end;
        assert!(!cluster.is_empty());

        // reset cache
        timestamp += cache_size + 1;

        // measure cluster ACMR
        let mut cluster_misses = 0;

        for i in cluster.clone() {
            let m = update_cache(
                indices[i * 3 + 0],
                indices[i * 3 + 1],
                indices[i * 3 + 2],
                cache_size,
                cache_timestamps,
                &mut timestamp,
            );

            cluster_misses += m;
        }

        let cluster_threshold = threshold * (cluster_misses as f32 / (end - *start as usize) as f32);

        // first cluster always starts from the hard cluster boundary
        destination[result] = *start;
        result += 1;

        // reset cache
        timestamp += cache_size + 1;

        let mut running_misses = 0;
        let mut running_faces = 0;

        for i in cluster {
            let m = update_cache(
                indices[i * 3 + 0],
                indices[i * 3 + 1],
                indices[i * 3 + 2],
                cache_size,
                cache_timestamps,
                &mut timestamp,
            );

            running_misses += m;
            running_faces += 1;

            if running_misses as f32 / running_faces as f32 <= cluster_threshold {
                // we have reached the target ACMR with the current triangle so we need to start a new cluster on the next one
                // note that this may mean that we add 'end` to destination for the last triangle, which will imply that the last
                // cluster is empty; however, the 'pop_back' after the loop will clean it up
                destination[result] = i as u32 + 1;
                result += 1;

                // reset cache
                timestamp += cache_size + 1;

                running_misses = 0;
                running_faces = 0;
            }
        }

        // each time we reach the target ACMR we flush the cluster
        // this means that the last cluster is by definition not very good - there are frequent cases where we are left with a few triangles
        // in the last cluster, producing a very bad ACMR and significantly penalizing the overall results
        // thus we remove the last cluster boundary, merging the last complete cluster with the last incomplete one
        // there are sometimes cases when the last cluster is actually good enough - in which case the code above would have added 'end'
        // to the cluster boundary array which we need to remove anyway - this code will do that automatically
        if destination[result - 1] != *start {
            result -= 1;
        }
    }

    assert!(result >= clusters.len());
    assert!(result <= indices.len() / 3);

    result
}

/// Reorders indices to reduce the number of GPU vertex shader invocations and the pixel overdraw.
///
/// If index buffer contains multiple ranges for multiple draw calls, this functions needs to be called on each range individually.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting index buffer (`indices.len()` elements)
/// * `indices`: must contain index data that is the result of [optimize_vertex_cache](crate::vertex::cache::optimize_vertex_cache) (**not** the original mesh indices!)
/// * `threshold`: indicates how much the overdraw optimizer can degrade vertex cache efficiency (1.05 = up to 5%) to reduce overdraw more efficiently
pub fn optimize_overdraw<Vertex>(destination: &mut [u32], indices: &[u32], vertices: &[Vertex], threshold: f32)
where
    Vertex: Position,
{
    assert_eq!(indices.len() % 3, 0);

    // guard for empty meshes
    if indices.is_empty() || vertices.is_empty() {
        return;
    }

    const CACHE_SIZE: u32 = 16;

    let mut cache_timestamps = vec![0u32; vertices.len()];

    // generate hard boundaries from full-triangle cache misses
    let mut hard_clusters = vec![0u32; indices.len() / 3];
    let hard_cluster_count = generate_hard_boundaries(&mut hard_clusters, indices, CACHE_SIZE, &mut cache_timestamps);

    // generate soft boundaries
    cache_timestamps.fill(0);
    let mut soft_clusters = vec![0u32; indices.len() / 3 + 1];
    let soft_cluster_count = generate_soft_boundaries(
        &mut soft_clusters,
        indices,
        &hard_clusters[0..hard_cluster_count],
        CACHE_SIZE,
        threshold,
        &mut cache_timestamps,
    );

    let clusters = &soft_clusters[0..soft_cluster_count];

    // fill sort data
    let mut sort_data = vec![0.0; clusters.len()];
    calculate_sort_data(&mut sort_data, indices, vertices, clusters);

    // sort clusters using sort data
    let mut sort_keys = vec![0u16; clusters.len()];
    let mut sort_order = vec![0u32; clusters.len()];
    calculate_sort_order_radix(&mut sort_order, &mut sort_data, &mut sort_keys);

    // fill output buffer
    let mut offset = 0;

    for cluster in &sort_order {
        let cluster = *cluster as usize;

        let cluster_begin = (clusters[cluster] * 3) as usize;
        let cluster_end = if let Some(begin) = clusters.get(cluster + 1) {
            (begin * 3) as usize
        } else {
            indices.len()
        };
        assert!(cluster_begin < cluster_end);

        let cluster_size = cluster_end - cluster_begin;

        destination[offset..offset + cluster_size].copy_from_slice(&indices[cluster_begin..cluster_end]);

        offset += cluster_size;
    }

    assert_eq!(offset, indices.len());
}

#[cfg(test)]
mod test {
    use super::*;

    struct DummyVertex;

    impl Position for DummyVertex {
        fn pos(&self) -> [f32; 3] {
            [0.0; 3]
        }
    }

    #[test]
    fn test_empty() {
        optimize_overdraw(&mut [], &[], &[DummyVertex], 1.0);
    }
}
