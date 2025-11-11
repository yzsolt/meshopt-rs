//! **Experimental** spatial sorting

use crate::Vector3;
use crate::util::zero_inverse;
use crate::vertex::{Vertex, calc_pos_extents};

// "Insert" two 0 bits after each of the 10 low bits of x
#[inline(always)]
fn part_1_by_2(mut x: u32) -> u32 {
    x &= 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x << 8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x << 4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x << 2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x
}

fn compute_order<V>(result: &mut [u32], vertices: &[V])
where
    V: Vertex,
{
    let (minv, extent) = calc_pos_extents(vertices);

    let scale = zero_inverse(extent);

    // generate Morton order based on the position inside a unit cube
    for (i, vertex) in vertices.iter().enumerate() {
        let v = vertex.pos();

        let x = ((v[0] - minv[0]) * scale * 1023.0 + 0.5) as i32;
        let y = ((v[1] - minv[1]) * scale * 1023.0 + 0.5) as i32;
        let z = ((v[2] - minv[2]) * scale * 1023.0 + 0.5) as i32;

        result[i] = part_1_by_2(x as u32) | (part_1_by_2(y as u32) << 1) | (part_1_by_2(z as u32) << 2);
    }
}

fn compute_histogram(hist: &mut [[u32; 3]; 1024], data: &[u32]) {
    // compute 3 10-bit histograms in parallel
    for id in data {
        let id = *id as usize;

        hist[(id >> 0) & 1023][0] += 1;
        hist[(id >> 10) & 1023][1] += 1;
        hist[(id >> 20) & 1023][2] += 1;
    }

    let mut sum = [0; 3];

    // replace histogram data with prefix histogram sums in-place
    for h in hist {
        let sav = *h;

        h.copy_from_slice(&sum);

        sum[0] += sav[0];
        sum[1] += sav[1];
        sum[2] += sav[2];
    }

    assert!(sum.iter().all(|s| *s as usize == data.len()));
}

fn radix_pass(destination: &mut [u32], source: &[u32], keys: &[u32], hist: &mut [[u32; 3]; 1024], pass: i32) {
    let bitoff = pass * 10;

    for s in source {
        let id = (keys[*s as usize] >> bitoff) & 1023;

        let h = &mut hist[id as usize][pass as usize];
        destination[*h as usize] = *s;
        *h += 1;
    }
}

/// Generates a remap table that can be used to reorder points for spatial locality.
///
/// Resulting remap table maps old vertices to new vertices and can be used in [remap_vertex_buffer](crate::index::generator::remap_vertex_buffer).
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting remap table (`vertices.len()` elements)
pub fn spatial_sort_remap<V>(destination: &mut [u32], vertices: &[V])
where
    V: Vertex,
{
    let mut keys = vec![0; vertices.len()];
    compute_order(&mut keys, vertices);

    let mut hist = [[0u32; 3]; 1024];
    compute_histogram(&mut hist, &keys);

    let mut scratch = vec![0; vertices.len()];

    for (i, d) in destination[0..vertices.len()].iter_mut().enumerate() {
        *d = i as u32;
    }

    // 3-pass radix sort computes the resulting order into scratch
    radix_pass(&mut scratch, destination, &keys, &mut hist, 0);
    radix_pass(destination, &scratch, &keys, &mut hist, 1);
    radix_pass(&mut scratch, destination, &keys, &mut hist, 2);

    // since our remap table is mapping old=>new, we need to reverse it
    for (i, s) in scratch.iter().enumerate() {
        destination[*s as usize] = i as u32;
    }
}

/// Reorders triangles for spatial locality, and generates a new index buffer.
///
/// The resulting index buffer can be used with other functions like [optimize_vertex_cache](crate::vertex::cache::optimize_vertex_cache).
///
/// # Arguments
///
/// * `destination`: must contain enough space for the resulting index buffer (`indices.len()` elements)
pub fn spatial_sort_triangles<V>(destination: &mut [u32], indices: &[u32], vertices: &[V])
where
    V: Vertex,
{
    assert!(indices.len().is_multiple_of(3));

    let face_count = indices.len() / 3;

    let mut centroids = vec![Vector3::default(); face_count];

    for i in 0..face_count {
        let a = indices[i * 3 + 0] as usize;
        let b = indices[i * 3 + 1] as usize;
        let c = indices[i * 3 + 2] as usize;

        assert!(a < vertices.len() && b < vertices.len() && c < vertices.len());

        let va = vertices[a].pos();
        let vb = vertices[b].pos();
        let vc = vertices[c].pos();

        centroids[i].x = (va[0] + vb[0] + vc[0]) / 3.0;
        centroids[i].y = (va[1] + vb[1] + vc[1]) / 3.0;
        centroids[i].z = (va[2] + vb[2] + vc[2]) / 3.0;
    }

    let mut remap = vec![0; face_count];

    spatial_sort_remap(&mut remap, &centroids);

    for i in 0..face_count {
        let abc = &indices[i * 3..i * 3 + 3];
        let r = remap[i] as usize;

        destination[r * 3..r * 3 + 3].copy_from_slice(abc);
    }
}
