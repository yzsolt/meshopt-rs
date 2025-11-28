//! This is a playground for experimenting with algorithms necessary for Nanite like hierarchical clustering
//! The code is not optimized, not robust, and not intended for production use.

// For reference, see the original Nanite paper:
// Brian Karis. Nanite: A Deep Dive. 2021

use meshopt_rs::{
    Stream,
    cluster::{Meshlet, build_meshlets, build_meshlets_bound, compute_cluster_bounds, optimize_meshlet},
    index::generator::generate_vertex_remap_multi,
    simplify::{SimplificationOptions, simplify, simplify_with_attributes},
    vertex::Vertex,
};

#[derive(Default, Clone)]
struct LodBounds {
    center: [f32; 3],
    radius: f32,
    error: f32,
}

#[derive(Default, Clone)]
struct Cluster {
    indices: Vec<u32>,

    self_: LodBounds,
    parent: LodBounds,
}

const CLUSTER_SIZE: usize = 128;
const GROUP_SIZE: usize = 8;
const USE_LOCKS: bool = true;

fn bounds(vertices: &[impl Vertex], indices: &[u32], error: f32) -> LodBounds {
    let bounds = compute_cluster_bounds(indices, vertices);

    LodBounds {
        center: bounds.center,
        radius: bounds.radius,
        error,
    }
}

fn bounds_merge(clusters: &[Cluster], group: &[i32]) -> LodBounds {
    let mut result = LodBounds::default();

    // we approximate merged bounds center as weighted average of cluster centers
    // (could also use bounds() center, but we can't use bounds() radius so might as well just merge manually)
    let mut weight = 0.0;
    for g in group.iter().map(|g| *g as usize) {
        result.center[0] += clusters[g].self_.center[0] * clusters[g].self_.radius;
        result.center[1] += clusters[g].self_.center[1] * clusters[g].self_.radius;
        result.center[2] += clusters[g].self_.center[2] * clusters[g].self_.radius;
        weight += clusters[g].self_.radius;
    }

    if weight > 0.0 {
        result.center[0] /= weight;
        result.center[1] /= weight;
        result.center[2] /= weight;
    }

    // merged bounds must strictly contain all cluster bounds
    result.radius = 0.0;
    for g in group.iter().map(|g| *g as usize) {
        let dx = clusters[g].self_.center[0] - result.center[0];
        let dy = clusters[g].self_.center[1] - result.center[1];
        let dz = clusters[g].self_.center[2] - result.center[2];
        result.radius = result
            .radius
            .max(clusters[g].self_.radius + (dx * dx + dy * dy + dz * dz).sqrt());
    }

    // merged bounds error must be conservative wrt cluster errors
    result.error = 0.0;
    for g in group.iter().map(|g| *g as usize) {
        result.error = result.error.max(clusters[g].self_.error);
    }

    result
}

fn bounds_error(bounds: &LodBounds, x: f32, y: f32, z: f32) -> f32 {
    let dx = bounds.center[0] - x;
    let dy = bounds.center[1] - y;
    let dz = bounds.center[2] - z;
    let d = (dx * dx + dy * dy + dz * dz).sqrt() - bounds.radius;

    if d <= 0.0 { f32::MAX } else { bounds.error / d }
}

fn clusterize(vertices: &[impl Vertex], indices: &[u32]) -> Vec<Cluster> {
    const MAX_VERTICES: usize = 192; // TODO: depends on CLUSTER_SIZE, also may want to dial down for mesh shaders
    const MAX_TRIANGLES: usize = CLUSTER_SIZE;

    let max_meshlets = build_meshlets_bound(indices.len(), MAX_VERTICES, MAX_TRIANGLES);

    let mut meshlets = vec![Meshlet::default(); max_meshlets];
    let mut meshlet_vertices = vec![0u32; max_meshlets * MAX_VERTICES];
    let mut meshlet_triangles = vec![0u8; max_meshlets * MAX_TRIANGLES * 3];

    let count = build_meshlets(
        &mut meshlets,
        &mut meshlet_vertices,
        &mut meshlet_triangles,
        indices,
        vertices,
        MAX_VERTICES,
        MAX_TRIANGLES,
        0.0,
    );
    meshlets.truncate(count);

    let mut clusters = vec![Cluster::default(); meshlets.len()];

    for (meshlet, cluster) in meshlets.iter().zip(clusters.iter_mut()) {
        optimize_meshlet(
            &mut meshlet_vertices
                [meshlet.vertex_offset as usize..(meshlet.vertex_offset + meshlet.vertex_count) as usize],
            &mut meshlet_triangles
                [meshlet.triangle_offset as usize..(meshlet.triangle_offset + meshlet.triangle_count * 3) as usize],
        );

        // note: for now we discard meshlet-local indices; they are valuable for shader code so in the future we should bring them back
        let index_count = meshlet.triangle_count as usize * 3;
        cluster.indices = (0..index_count)
            .map(|j| {
                meshlet_vertices
                    [(meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset as usize + j] as u32) as usize]
            })
            .collect();

        cluster.parent.error = f32::MAX;
    }

    clusters
}

fn partition(clusters: &[Cluster], pending: &[i32], _remap: &[u32]) -> Vec<Vec<i32>> {
    let mut result = Vec::new();

    let mut last_indices = 0;

    // rough merge; while clusters are approximately spatially ordered, this should use a proper partitioning algorithm
    for p in pending {
        if result.is_empty() || last_indices + clusters[*p as usize].indices.len() > CLUSTER_SIZE * GROUP_SIZE * 3 {
            result.push(Vec::new());
            last_indices = 0;
        }

        result.last_mut().unwrap().push(*p);
        last_indices += clusters[*p as usize].indices.len();
    }

    result
}

fn lock_boundary(locks: &mut [u8], groups: &Vec<Vec<i32>>, clusters: &[Cluster], remap: &[u32]) {
    let mut groupmap = vec![-1i32; locks.len()];

    for i in 0..groups.len() {
        for j in 0..groups[i].len() {
            let cluster = &clusters[groups[i][j] as usize];

            for k in 0..cluster.indices.len() {
                let v = cluster.indices[k];
                let r = remap[v as usize] as usize;

                let gr = &mut groupmap[r];
                if *gr == -1 || *gr == i as i32 {
                    *gr = i as i32;
                } else {
                    *gr = -2;
                }
            }
        }
    }

    // note: we need to consistently lock all vertices with the same position to avoid holes
    for i in 0..locks.len() {
        let r = remap[i];

        locks[i] = (groupmap[r as usize] == -2) as u8;
    }
}

fn simplify2(
    vertices: &[impl Vertex],
    indices: &[u32],
    locks: Option<&[u8]>,
    target_count: usize,
    error: Option<&mut f32>,
) -> Vec<u32> {
    if target_count > indices.len() {
        return indices.to_vec();
    }

    let mut lod = vec![0u32; indices.len()];
    let options = SimplificationOptions::SimplifySparse | SimplificationOptions::SimplifyErrorAbsolute;
    let size = if let Some(locks) = locks {
        let locks = locks.iter().map(|l| *l != 0).collect::<Vec<_>>();

        simplify_with_attributes(
            &mut lod,
            indices,
            vertices,
            &[],
            Some(&locks),
            target_count,
            f32::MAX,
            options,
            error,
        )
    } else {
        simplify(
            &mut lod,
            indices,
            vertices,
            target_count,
            f32::MAX,
            options | SimplificationOptions::SimplifyLockBorder,
            error,
        )
    };
    lod.truncate(size);

    lod
}

pub fn nanite(vertices: &[impl Vertex], indices: &[u32]) {
    // initial clusterization splits the original mesh
    let mut clusters = clusterize(vertices, indices);
    for cluster in &mut clusters {
        cluster.self_ = bounds(vertices, &cluster.indices, 0.0);
    }

    println!("lod 0: {} clusters, {} triangles", clusters.len(), indices.len() / 3);

    let mut pending: Vec<_> = (0..clusters.len() as i32).collect();

    let mut depth = 0;
    let mut locks = vec![0u8; vertices.len()];

    // for cluster connectivity, we need a position-only remap that maps vertices with the same position to the same index
    // it's more efficient to build it once; unfortunately, meshopt_generateVertexRemap doesn't support stride so we need to use *Multi version
    let mut remap = vec![0u32; vertices.len()];
    let position = Stream::from_slice(&vertices);
    generate_vertex_remap_multi(&mut remap, Some(&indices), &[position]);

    // merge and simplify clusters until we can't merge anymore
    while pending.len() > 1 {
        let groups = partition(&clusters, &pending, &remap);
        pending.clear();

        let mut retry = Vec::new();

        let mut triangles = 0;
        let mut stuck_triangles = 0;
        let mut single_clusters = 0;
        let mut stuck_clusters = 0;
        let mut full_clusters = 0;

        if USE_LOCKS {
            lock_boundary(&mut locks, &groups, &clusters, &remap);
        }

        // every group needs to be simplified now
        for i in 0..groups.len() {
            if groups[i].is_empty() {
                continue; // metis shortcut
            }

            if groups[i].len() == 1 {
                #[cfg(feature = "trace")]
                println!(
                    "stuck cluster: singleton with {} triangles",
                    clusters[groups[i][0] as usize].indices.len() / 3
                );

                single_clusters += 1;
                stuck_clusters += 1;
                stuck_triangles += clusters[groups[i][0] as usize].indices.len() / 3;
                retry.push(groups[i][0]);
                continue;
            }

            let mut merged = Vec::new();
            for j in 0..groups[i].len() {
                merged.extend_from_slice(clusters[groups[i][j] as usize].indices.as_slice());
            }

            let target_size = ((groups[i].len() + 1) / 2) * CLUSTER_SIZE * 3;
            let mut error = 0.0;
            let simplified = simplify2(
                vertices,
                &merged,
                if USE_LOCKS { Some(&locks) } else { None },
                target_size,
                Some(&mut error),
            );
            if simplified.len() as f64 > merged.len() as f64 * 0.85
                || simplified.len() / (CLUSTER_SIZE * 3) >= merged.len() / (CLUSTER_SIZE * 3)
            {
                #[cfg(feature = "trace")]
                println!(
                    "stuck cluster: simplified {} => {} over threshold",
                    merged.len() / 3,
                    simplified.len() / 3
                );

                stuck_clusters += 1;
                stuck_triangles += merged.len() / 3;
                for g in &groups[i] {
                    retry.push(*g);
                }
                continue; // simplification is stuck; abandon the merge
            }

            // enforce bounds and error monotonicity
            // note: it is incorrect to use the precise bounds of the merged or simplified mesh, because this may violate monotonicity
            let mut groupb = bounds_merge(&clusters, &groups[i]);
            groupb.error += error; // this may overestimate the error, but we are starting from the simplified mesh so this is a little more correct

            let mut split = clusterize(vertices, &simplified);

            // update parent bounds and error for all clusters in the group
            // note that all clusters in the group need to switch simultaneously so they have the same bounds
            for j in 0..groups[i].len() {
                assert_eq!(clusters[groups[i][j] as usize].parent.error, f32::MAX);
                clusters[groups[i][j] as usize].parent = groupb.clone();
            }

            for s in &mut split {
                s.self_ = groupb.clone();

                clusters.push(s.clone());
                pending.push(clusters.len() as i32 - 1);

                triangles += s.indices.len() / 3;
                full_clusters += (s.indices.len() == CLUSTER_SIZE * 3) as usize;
            }
        }

        depth += 1;
        println!(
            "lod {}: simplified {} clusters ({} full, {:.1} tri/cl), {} triangles; stuck {} clusters ({} single), {} triangles",
            depth,
            pending.len(),
            full_clusters,
            if pending.is_empty() {
                0.0
            } else {
                triangles as f64 / pending.len() as f64
            },
            triangles,
            stuck_clusters,
            single_clusters,
            stuck_triangles
        );

        if triangles < stuck_triangles / 3 {
            break;
        }

        pending.extend_from_slice(&retry);
    }

    let mut total_triangles = 0;
    let mut lowest_triangles = 0;
    for c in &clusters {
        total_triangles += c.indices.len() / 3;
        if c.parent.error == f32::MAX {
            lowest_triangles += c.indices.len() / 3;
        }
    }

    println!("total: {total_triangles} triangles in {} clusters", clusters.len());
    println!("lowest lod: {} triangles", lowest_triangles);

    // for testing purposes, we can compute a DAG cut from a given viewpoint and dump it as an OBJ
    let mut maxx: f32 = 0.0;
    let mut maxy: f32 = 0.0;
    let mut maxz: f32 = 0.0;
    for v in vertices {
        maxx = maxx.max(v.pos()[0] * 2.0);
        maxy = maxy.max(v.pos()[1] * 2.0);
        maxz = maxz.max(v.pos()[2] * 2.0);
    }
    let threshold = 3e-3;

    let mut cut = Vec::new();
    for c in &clusters {
        if bounds_error(&c.self_, maxx, maxy, maxz) <= threshold
            && bounds_error(&c.parent, maxx, maxy, maxz) > threshold
        {
            cut.extend_from_slice(&c.indices);
        }
    }

    println!("cut ({:.3}): {} triangles", threshold, cut.len() / 3);
}
