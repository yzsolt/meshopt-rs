//! **Experimental** meshlet building and cluster bounds generation

use crate::quantize::quantize_snorm;
use crate::util::zero_inverse;
use crate::vertex::{build_triangle_adjacency, Position, TriangleAdjacency};

const UNUSED: u8 = 0xff;

/// Bounds returned by `compute_cluster/meshlet_bounds`.
///
/// `cone_axis_s8` and `cone_cutoff_s8` are stored in 8-bit SNORM format; decode them using `x/127.0`.
///
/// * Bounding sphere: useful for backface culling
/// * Normal cone: useful for backface culling
#[derive(Default)]
pub struct Bounds {
    /// Bounding sphere center
    pub center: [f32; 3],
    /// Bounding sphere radius
    pub radius: f32,
    /// Normal cone apex
    pub cone_apex: [f32; 3],
    /// Normal cone axis
    pub cone_axis: [f32; 3],
    /// Normal cone cutoff
    ///
    /// Can be calculated from angle using `cos(angle/2)`.
    pub cone_cutoff: f32,
    /// Normal cone axis
    pub cone_axis_s8: [i8; 3],
    /// Normal cone cutoff
    pub cone_cutoff_s8: i8,
}

#[derive(Clone)]
pub struct Meshlet {
    pub vertices: [i32; 64],
    pub indices: [[u8; 3]; 126],
    pub triangle_count: u8,
    pub vertex_count: u8,
}

impl Meshlet {
    pub const VERTICES_COUNT: usize = 64;
    pub const TRIANGLES_COUNT: usize = 126;
}

impl Default for Meshlet {
    fn default() -> Self {
        Meshlet {
            vertices: [0; Self::VERTICES_COUNT],
            indices: [Default::default(); Self::TRIANGLES_COUNT],
            triangle_count: Default::default(),
            vertex_count: Default::default(),
        }
    }
}

fn compute_bounding_sphere(points: &[[f32; 3]]) -> [f32; 4] {
    assert!(!points.is_empty());

    // find extremum points along all 3 axes; for each axis we get a pair of points with min/max coordinates
    let mut pmin = [[f32::MAX; 3]; 3];
    let mut pmax = [[f32::MIN; 3]; 3];

    for p in points {
        for axis in 0..3 {
            if p[axis] < pmin[axis][axis] {
                pmin[axis] = *p;
            }
            if p[axis] > pmax[axis][axis] {
                pmax[axis] = *p;
            }
        }
    }

    // find the pair of points with largest distance
    let mut paxisd2 = 0.0;
    let mut paxis = 0;

    for axis in 0..3 {
        let p1 = pmin[axis];
        let p2 = pmax[axis];

        let d2 =
            (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]) + (p2[2] - p1[2]) * (p2[2] - p1[2]);

        if d2 > paxisd2 {
            paxisd2 = d2;
            paxis = axis;
        }
    }

    // use the longest segment as the initial sphere diameter
    let p1 = pmin[paxis];
    let p2 = pmax[paxis];

    let mut center: [f32; 3] = [
        (p1[0] + p2[0]) / 2.0,
        (p1[1] + p2[1]) / 2.0,
        (p1[2] + p2[2]) / 2.0,
    ];
    let mut radius = paxisd2.sqrt() / 2.0;

    // iteratively adjust the sphere up until all points fit
    for p in points {
        let d2 = (p[0] - center[0]) * (p[0] - center[0])
            + (p[1] - center[1]) * (p[1] - center[1])
            + (p[2] - center[2]) * (p[2] - center[2]);

        if d2 > radius * radius {
            let d = d2.sqrt();
            assert!(d > 0.0);

            let k = 0.5 + (radius / d) / 2.0;

            center[0] = center[0] * k + p[0] * (1.0 - k);
            center[1] = center[1] * k + p[1] * (1.0 - k);
            center[2] = center[2] * k + p[2] * (1.0 - k);
            radius = (radius + d) / 2.0;
        }
    }

    [center[0], center[1], center[2], radius]
}

#[derive(Clone, Default)]
struct Cone {
    px: f32,
    py: f32,
    pz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
}

impl Position for Cone {
    fn pos(&self) -> [f32; 3] {
        [self.px, self.py, self.pz]
    }
}

fn get_meshlet_score(distance2: f32, spread: f32, cone_weight: f32, expected_radius: f32) -> f32 {
    let cone = 1.0 - spread * cone_weight;
    let cone_clamped = cone.max(1e-3);

    (1.0 + distance2.sqrt() / expected_radius * (1.0 - cone_weight)) * cone_clamped
}

fn get_meshlet_cone(acc: &Cone, triangle_count: u32) -> Cone {
    let mut result = acc.clone();

    let center_scale = zero_inverse(triangle_count as f32);

    result.px *= center_scale;
    result.py *= center_scale;
    result.pz *= center_scale;

    let axis_length = result.nx * result.nx + result.ny * result.ny + result.nz * result.nz;
    let axis_scale = if axis_length == 0.0 {
        0.0
    } else {
        1.0 / axis_length.sqrt()
    };

    result.nx *= axis_scale;
    result.ny *= axis_scale;
    result.nz *= axis_scale;

    result
}

fn compute_triangle_cones<Vertex>(triangles: &mut [Cone], indices: &[u32], vertices: &[Vertex]) -> f32
where
    Vertex: Position,
{
    let face_count = indices.len() / 3;

    let mut mesh_area = 0.0;

    for i in 0..face_count {
        let a = indices[i * 3 + 0];
        let b = indices[i * 3 + 1];
        let c = indices[i * 3 + 2];

        let p0 = vertices[a as usize].pos();
        let p1 = vertices[b as usize].pos();
        let p2 = vertices[c as usize].pos();

        let p10 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let p20 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        let normalx = p10[1] * p20[2] - p10[2] * p20[1];
        let normaly = p10[2] * p20[0] - p10[0] * p20[2];
        let normalz = p10[0] * p20[1] - p10[1] * p20[0];

        let area = (normalx * normalx + normaly * normaly + normalz * normalz).sqrt();
        let invarea = zero_inverse(area);

        triangles[i].px = (p0[0] + p1[0] + p2[0]) / 3.0;
        triangles[i].py = (p0[1] + p1[1] + p2[1]) / 3.0;
        triangles[i].pz = (p0[2] + p1[2] + p2[2]) / 3.0;

        triangles[i].nx = normalx * invarea;
        triangles[i].ny = normaly * invarea;
        triangles[i].nz = normalz * invarea;

        mesh_area += area;
    }

    mesh_area
}

fn append_meshlet(
    meshlet: &mut Meshlet,
    a: u32,
    b: u32,
    c: u32,
    used: &mut [u8],
    destination: &mut [Meshlet],
    offset: usize,
    max_vertices: usize,
    max_triangles: usize,
) -> bool {
    let mut av = used[a as usize];
    let mut bv = used[b as usize];
    let mut cv = used[c as usize];

    let used_extra = [av, bv, cv].iter().filter(|v| **v == UNUSED).count();

    let mut result = false;

    if meshlet.vertex_count as usize + used_extra > max_vertices || meshlet.triangle_count as usize >= max_triangles {
        destination[offset] = meshlet.clone();

        for j in 0..meshlet.vertex_count {
            used[meshlet.vertices[j as usize] as usize] = UNUSED;
        }

        *meshlet = Meshlet::default();

        result = true;
    }

    if av == UNUSED {
        av = meshlet.vertex_count;
        used[a as usize] = av;
        meshlet.vertices[meshlet.vertex_count as usize] = a.try_into().unwrap();
        meshlet.vertex_count += 1;
    }

    if bv == UNUSED {
        bv = meshlet.vertex_count;
        used[b as usize] = bv;
        meshlet.vertices[meshlet.vertex_count as usize] = b.try_into().unwrap();
        meshlet.vertex_count += 1;
    }

    if cv == UNUSED {
        cv = meshlet.vertex_count;
        used[c as usize] = cv;
        meshlet.vertices[meshlet.vertex_count as usize] = c.try_into().unwrap();
        meshlet.vertex_count += 1;
    }

    meshlet.indices[meshlet.triangle_count as usize][..].copy_from_slice(&[av, bv, cv]);

    meshlet.triangle_count += 1;

    result
}

#[derive(Clone, Default)]
struct KdNode {
    node_type: KdNodeType,
    children: u32,
}

#[derive(Clone)]
enum KdNodeType {
    Branch { axis: u8, split: f32 },
    Leaf { index: u32 },
}

impl Default for KdNodeType {
    fn default() -> Self {
        Self::Leaf { index: !0u32 }
    }
}

fn kd_tree_partition<Point>(indices: &mut [u32], points: &[Point], axis: u32, pivot: f32) -> usize
where
    Point: Position,
{
    let mut m = 0;

    // invariant: elements in range [0, m) are < pivot, elements in range [m, i) are >= pivot
    for i in 0..indices.len() {
        let v = points[indices[i] as usize].pos()[axis as usize];

        // swap(m, i) unconditionally
        let t = indices[m];
        indices[m] = indices[i];
        indices[i] = t;

        // when v >= pivot, we swap i with m without advancing it, preserving invariants
        m += (v < pivot) as usize;
    }

    m
}

fn kd_tree_build_leaf(offset: usize, nodes: &mut [KdNode], indices: &[u32]) -> usize {
    let result = &mut nodes[offset];

    *result = KdNode {
        node_type: KdNodeType::Leaf { index: indices[0] },
        children: (indices.len() - 1) as u32,
    };

    // all remaining points are stored in nodes immediately following the leaf
    for i in 1..indices.len() {
        let tail = &mut nodes[offset + i];

        *tail = KdNode {
            node_type: KdNodeType::Leaf { index: indices[i] },
            children: !0u32, // bogus value to prevent misuse
        };
    }

    offset + indices.len()
}

fn kd_tree_build<Point>(
    offset: usize,
    nodes: &mut [KdNode],
    points: &[Point],
    indices: &mut [u32],
    leaf_size: usize,
) -> usize
where
    Point: Position,
{
    assert!(!indices.is_empty());

    if indices.len() <= leaf_size {
        return kd_tree_build_leaf(offset, nodes, indices);
    }

    let mut mean = [0f32; 3];
    let mut vars = [0f32; 3];
    let mut runc = 1.0;
    let mut runs = 1.0;

    // gather statistics on the points in the subtree using Welford's algorithm
    for i in 0..indices.len() {
        let point = points[indices[i] as usize].pos();

        for k in 0..3 {
            let delta = point[k] - mean[k];
            mean[k] += delta * runs;
            vars[k] += delta * (point[k] - mean[k]);
        }

        runc += 1.0;
        runs = 1.0 / runc;
    }

    // split axis is one where the variance is largest
    let axis: u32 = if vars[0] >= vars[1] && vars[0] >= vars[2] {
        0
    } else {
        if vars[1] >= vars[2] {
            1
        } else {
            2
        }
    };

    let split = mean[axis as usize];
    let middle = kd_tree_partition(indices, points, axis, split);

    // when the partition is degenerate simply consolidate the points into a single node
    if middle <= leaf_size / 2 || middle >= indices.len() - leaf_size / 2 {
        return kd_tree_build_leaf(offset, nodes, indices);
    }

    {
        let result = &mut nodes[offset];
        result.node_type = KdNodeType::Branch {
            axis: axis as u8,
            split,
        };
    }

    // left subtree is right after our node
    let next_offset = kd_tree_build(offset + 1, nodes, points, &mut indices[0..middle], leaf_size);

    // distance to the right subtree is represented explicitly
    {
        let result = &mut nodes[offset];
        result.children = (next_offset - offset - 1) as u32;
    }

    kd_tree_build(next_offset, nodes, points, &mut indices[middle..], leaf_size)
}

fn kd_tree_nearest<Point>(
    nodes: &[KdNode],
    root: u32,
    points: &[Point],
    emitted_flags: &[bool],
    position: &[f32; 3],
    result: &mut u32,
    limit: &mut f32,
) where
    Point: Position,
{
    let node = &nodes[root as usize];

    match node.node_type {
        KdNodeType::Leaf { .. } => {
            for i in 0..=node.children {
                match nodes[root as usize + i as usize].node_type {
                    KdNodeType::Leaf { index } => {
                        if emitted_flags[index as usize] {
                            continue;
                        }

                        let point: [f32; 3] = points[index as usize].pos();

                        let distance2 = (point[0] - position[0]) * (point[0] - position[0])
                            + (point[1] - position[1]) * (point[1] - position[1])
                            + (point[2] - position[2]) * (point[2] - position[2]);
                        let distance = distance2.sqrt();

                        if distance < *limit {
                            *result = index;
                            *limit = distance;
                        }
                    }
                    KdNodeType::Branch { .. } => panic!(),
                }
            }
        }
        KdNodeType::Branch { axis, split } => {
            // branch; we order recursion to process the node that search position is in first
            let delta = position[axis as usize] - split;
            let first = if delta <= 0.0 { 0 } else { node.children };
            let second = first ^ node.children;

            kd_tree_nearest(nodes, root + 1 + first, points, emitted_flags, position, result, limit);

            // only process the other node if it can have a match based on closest distance so far
            if delta.abs() <= *limit {
                kd_tree_nearest(nodes, root + 1 + second, points, emitted_flags, position, result, limit);
            }
        }
    }
}

/// Returns worst case size requirement for [build_meshlets].
pub fn build_meshlets_bound(index_count: usize, max_vertices: usize, max_triangles: usize) -> usize {
    assert!(index_count % 3 == 0);
    assert!(max_vertices >= 3);
    assert!(max_triangles >= 1);

    // meshlet construction is limited by max vertices and max triangles per meshlet
    // the worst case is that the input is an unindexed stream since this equally stresses both limits
    // note that we assume that in the worst case, we leave 2 vertices unpacked in each meshlet - if we have space for 3 we can pack any triangle
    let max_vertices_conservative = max_vertices - 2;
    let meshlet_limit_vertices = (index_count + max_vertices_conservative - 1) / max_vertices_conservative;
    let meshlet_limit_triangles = (index_count / 3 + max_triangles - 1) / max_triangles;

    if meshlet_limit_vertices > meshlet_limit_triangles {
        meshlet_limit_vertices
    } else {
        meshlet_limit_triangles
    }
}

/// Splits the mesh into a set of meshlets where each meshlet has a micro index buffer indexing into meshlet vertices that refer to the original vertex buffer.
///
/// The resulting data can be used to render meshes using NVidia programmable mesh shading pipeline, or in other cluster-based renderers.
/// For maximum efficiency the index buffer being converted has to be optimized for vertex cache first.
///
/// # Arguments
///
/// * `destination`: must contain enough space for all meshlets, worst case size can be computed with [build_meshlets_bound]
/// * `max_vertices` and `max_triangles`: can't exceed limits statically declared in [Meshlet] (`VERTICES_COUNT` and `TRIANGLES_COUNT`)
pub fn build_meshlets_scan(
    destination: &mut [Meshlet],
    indices: &[u32],
    vertex_count: usize,
    max_vertices: usize,
    max_triangles: usize,
) -> usize {
    assert!(indices.len() % 3 == 0);
    assert!(max_vertices >= 3);
    assert!(max_triangles >= 1);

    let mut meshlet = Meshlet::default();

    assert!(max_vertices <= Meshlet::VERTICES_COUNT);
    assert!(max_triangles <= Meshlet::TRIANGLES_COUNT);

    // index of the vertex in the meshlet, `UNUSED` if the vertex isn't used
    let mut used = vec![UNUSED; vertex_count];

    let mut offset = 0;

    for abc in indices.chunks_exact(3) {
        let (a, b, c) = (abc[0], abc[1], abc[2]);

        // appends triangle to the meshlet and writes previous meshlet to the output if full
        offset += append_meshlet(
            &mut meshlet,
            a,
            b,
            c,
            &mut used,
            destination,
            offset,
            max_vertices,
            max_triangles,
        ) as usize;
    }

    if meshlet.triangle_count > 0 {
        destination[offset] = meshlet;
        offset += 1;
    }

    assert!(offset <= build_meshlets_bound(indices.len(), max_vertices, max_triangles));

    offset
}

/// Splits the mesh into a set of meshlets where each meshlet has a micro index buffer indexing into meshlet vertices that refer to the original vertex buffer.
///
/// The resulting data can be used to render meshes using NVidia programmable mesh shading pipeline, or in other cluster-based renderers.
/// For maximum efficiency the index buffer being converted has to be optimized for vertex cache first.
///
/// # Arguments
///
/// * `destination`: must contain enough space for all meshlets, worst case size can be computed with [build_meshlets_bound]
/// * `max_vertices` and `max_triangles`: can't exceed limits statically declared in [Meshlet] (`VERTICES_COUNT` and `TRIANGLES_COUNT`)
/// * `cone_weight`: should be set to 0 when cone culling is not used, and a value between 0 and 1 otherwise to balance between cluster size and cone culling efficiency
pub fn build_meshlets<Vertex>(
    destination: &mut [Meshlet],
    indices: &[u32],
    vertices: &[Vertex],
    max_vertices: usize,
    max_triangles: usize,
    cone_weight: f32,
) -> usize
where
    Vertex: Position,
{
    assert!(indices.len() % 3 == 0);
    assert!(max_vertices >= 3);
    assert!(max_triangles >= 1);

    let mut meshlet = Meshlet::default();

    assert!(max_vertices <= Meshlet::VERTICES_COUNT);
    assert!(max_triangles <= Meshlet::TRIANGLES_COUNT);

    let mut adjacency = TriangleAdjacency::default();
    build_triangle_adjacency(&mut adjacency, indices, vertices.len());

    let mut live_triangles = adjacency.counts.clone();

    let face_count = indices.len() / 3;

    let mut emitted_flags = vec![false; face_count];

    // for each triangle, precompute centroid & normal to use for scoring
    let mut triangles = vec![Cone::default(); face_count];
    let mesh_area = compute_triangle_cones(&mut triangles, indices, &vertices);

    // assuming each meshlet is a square patch, expected radius is sqrt(expected area)
    let triangle_area_avg = if face_count == 0 {
        0.0
    } else {
        mesh_area / face_count as f32 * 0.5
    };
    let meshlet_expected_radius = (triangle_area_avg * max_triangles as f32).sqrt() * 0.5;

    // build a kd-tree for nearest neighbor lookup
    let mut kdindices = vec![0u32; face_count];
    for i in 0..face_count {
        kdindices[i] = i as u32;
    }

    let mut nodes = vec![KdNode::default(); face_count * 2];
    kd_tree_build(0, &mut nodes, &triangles, &mut kdindices, /* leaf_size= */ 8);

    // index of the vertex in the meshlet, `UNUSED` if the vertex isn't used
    let mut used = vec![UNUSED; vertices.len()];

    let mut offset = 0;

    let mut meshlet_cone_acc = Cone::default();

    loop {
        let mut best_triangle = None;
        let mut best_extra = 5;
        let mut best_score = f32::MAX;

        let meshlet_cone = get_meshlet_cone(&meshlet_cone_acc, meshlet.triangle_count as u32);

        for i in 0..meshlet.vertex_count {
            let index = meshlet.vertices[i as usize] as usize;

            let neighbours_size = adjacency.counts[index] as usize;
            let neighbouts_offset = adjacency.offsets[index] as usize;
            let neighbours = &adjacency.data[neighbouts_offset..neighbouts_offset + neighbours_size];

            for j in 0..neighbours_size {
                let triangle = neighbours[j] as usize;
                assert!(!emitted_flags[triangle]);

                let a = indices[triangle * 3 + 0] as usize;
                let b = indices[triangle * 3 + 1] as usize;
                let c = indices[triangle * 3 + 2] as usize;
                assert!(a < vertices.len() && b < vertices.len() && c < vertices.len());

                let mut extra = [used[a], used[b], used[c]].iter().filter(|v| **v == UNUSED).count();

                // triangles that don't add new vertices to meshlets are max. priority
                if extra != 0 {
                    // artificially increase the priority of dangling triangles as they're expensive to add to new meshlets
                    if live_triangles[a] == 1 || live_triangles[b] == 1 || live_triangles[c] == 1 {
                        extra = 0;
                    }

                    extra += 1;
                }

                // since topology-based priority is always more important than the score, we can skip scoring in some cases
                if extra > best_extra {
                    continue;
                }

                let tri_cone = triangles[triangle].clone();

                let distance2 = (tri_cone.px - meshlet_cone.px) * (tri_cone.px - meshlet_cone.px)
                    + (tri_cone.py - meshlet_cone.py) * (tri_cone.py - meshlet_cone.py)
                    + (tri_cone.pz - meshlet_cone.pz) * (tri_cone.pz - meshlet_cone.pz);

                let spread =
                    tri_cone.nx * meshlet_cone.nx + tri_cone.ny * meshlet_cone.ny + tri_cone.nz * meshlet_cone.nz;

                let score = get_meshlet_score(distance2, spread, cone_weight, meshlet_expected_radius);

                // note that topology-based priority is always more important than the score
                // this helps maintain reasonable effectiveness of meshlet data and reduces scoring cost
                if extra < best_extra || score < best_score {
                    best_triangle = Some(triangle);
                    best_extra = extra;
                    best_score = score;
                }
            }
        }

        if best_triangle.is_none() {
            let position = [meshlet_cone.px, meshlet_cone.py, meshlet_cone.pz];
            let mut index = !0u32;
            let mut limit = f32::MAX;

            kd_tree_nearest(&nodes, 0, &triangles, &emitted_flags, &position, &mut index, &mut limit);

            if index != !0u32 {
                best_triangle = Some(index as usize);
            }
        }

        if let Some(best_triangle) = best_triangle {
            let abc = &indices[best_triangle * 3..best_triangle * 3 + 3];
            let (a, b, c) = (abc[0], abc[1], abc[2]);

            // add meshlet to the output; when the current meshlet is full we reset the accumulated bounds
            if append_meshlet(
                &mut meshlet,
                a,
                b,
                c,
                &mut used,
                destination,
                offset,
                max_vertices,
                max_triangles,
            ) {
                offset += 1;
                meshlet_cone_acc = Cone::default();
            }

            live_triangles[a as usize] -= 1;
            live_triangles[b as usize] -= 1;
            live_triangles[c as usize] -= 1;

            // remove emitted triangle from adjacency data
            // this makes sure that we spend less time traversing these lists on subsequent iterations
            for index in abc {
                let index = *index as usize;

                let neighbours_offset = adjacency.offsets[index] as usize;
                let neighbours_size = adjacency.counts[index] as usize;
                let neighbours = &mut adjacency.data[neighbours_offset..neighbours_offset + neighbours_size];

                for i in 0..neighbours_size {
                    let tri = neighbours[i] as usize;

                    if tri == best_triangle {
                        neighbours[i] = neighbours[neighbours_size - 1];
                        adjacency.counts[index] -= 1;
                        break;
                    }
                }
            }

            // update aggregated meshlet cone data for scoring subsequent triangles
            meshlet_cone_acc.px += triangles[best_triangle].px;
            meshlet_cone_acc.py += triangles[best_triangle].py;
            meshlet_cone_acc.pz += triangles[best_triangle].pz;
            meshlet_cone_acc.nx += triangles[best_triangle].nx;
            meshlet_cone_acc.ny += triangles[best_triangle].ny;
            meshlet_cone_acc.nz += triangles[best_triangle].nz;

            emitted_flags[best_triangle] = true;
        } else {
            break;
        }
    }

    if meshlet.triangle_count > 0 {
        destination[offset] = meshlet;
        offset += 1;
    }

    assert!(offset <= build_meshlets_bound(indices.len(), max_vertices, max_triangles));

    offset
}

/// Creates bounding volumes that can be used for frustum, backface and occlusion culling.
///
/// For backface culling with orthographic projection, use the following formula to reject backfacing clusters:
/// ```glsl
/// dot(view, cone_axis) >= cone_cutoff
/// ```
///
/// For perspective projection, you can the formula that needs cone apex in addition to axis & cutoff:
/// ```glsl
/// dot(normalize(cone_apex - camera_position), cone_axis) >= cone_cutoff
/// ```
///
/// Alternatively, you can use the formula that doesn't need cone apex and uses bounding sphere instead:
/// ```glsl
/// dot(normalize(center - camera_position), cone_axis) >= cone_cutoff + radius / length(center - camera_position)
/// ```
/// or an equivalent formula that doesn't have a singularity at center = camera_position:
/// ```glsl
/// dot(center - camera_position, cone_axis) >= cone_cutoff * length(center - camera_position) + radius
/// ```
///
/// The formula that uses the apex is slightly more accurate but needs the apex; if you are already using bounding sphere
/// to do frustum/occlusion culling, the formula that doesn't use the apex may be preferable.
///
/// # Arguments
///
/// * `indices`: should be smaller than or equal to 256*3 (the function assumes clusters of limited size)
pub fn compute_cluster_bounds<Vertex>(indices: &[u32], vertices: &[Vertex]) -> Bounds
where
    Vertex: Position,
{
    assert!(indices.len() % 3 == 0);
    assert!(indices.len() / 3 <= 256);

    // compute triangle normals and gather triangle corners
    let mut normals: [[f32; 3]; 256] = [Default::default(); 256];
    let mut corners: [[[f32; 3]; 3]; 256] = [Default::default(); 256];
    let mut triangles = 0;

    let vertex_count = vertices.len();

    for abc in indices.chunks_exact(3) {
        let (a, b, c) = (abc[0] as usize, abc[1] as usize, abc[2] as usize);

        assert!(a < vertex_count && b < vertex_count && c < vertex_count);

        let p0 = vertices[a].pos();
        let p1 = vertices[b].pos();
        let p2 = vertices[c].pos();

        let p10 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let p20 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        let normalx = p10[1] * p20[2] - p10[2] * p20[1];
        let normaly = p10[2] * p20[0] - p10[0] * p20[2];
        let normalz = p10[0] * p20[1] - p10[1] * p20[0];

        let area = (normalx * normalx + normaly * normaly + normalz * normalz).sqrt();

        // no need to include degenerate triangles - they will be invisible anyway
        if area == 0.0 {
            continue;
        }

        // record triangle normals & corners for future use; normal and corner 0 define a plane equation
        normals[triangles][0] = normalx / area;
        normals[triangles][1] = normaly / area;
        normals[triangles][2] = normalz / area;

        corners[triangles][0].copy_from_slice(&p0);
        corners[triangles][1].copy_from_slice(&p1);
        corners[triangles][2].copy_from_slice(&p2);

        triangles += 1;
    }

    let mut bounds = Bounds::default();

    // degenerate cluster, no valid triangles => trivial reject (cone data is 0)
    if triangles == 0 {
        return bounds;
    }

    // compute cluster bounding sphere; we'll use the center to determine normal cone apex as well
    let psphere = compute_bounding_sphere(unsafe {
        let x: &[[f32; 3]] = std::mem::transmute(&corners[0..triangles]);
        std::slice::from_raw_parts(x.as_ptr(), triangles * 3)
    });

    let center = [psphere[0], psphere[1], psphere[2]];

    // treating triangle normals as points, find the bounding sphere - the sphere center determines the optimal cone axis
    let nsphere = compute_bounding_sphere(&normals[0..triangles]);

    let mut axis = [nsphere[0], nsphere[1], nsphere[2]];
    let axislength = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let invaxislength = zero_inverse(axislength);

    axis[0] *= invaxislength;
    axis[1] *= invaxislength;
    axis[2] *= invaxislength;

    // compute a tight cone around all normals, mindp = cos(angle/2)
    let mut mindp = 1.0;

    for normal in &normals[0..triangles] {
        let dp = normal[0] * axis[0] + normal[1] * axis[1] + normal[2] * axis[2];

        mindp = dp.min(mindp);
    }

    // fill bounding sphere info; note that below we can return bounds without cone information for degenerate cones
    bounds.center = center;
    bounds.radius = psphere[3];

    // degenerate cluster, normal cone is larger than a hemisphere => trivial accept
    // note that if mindp is positive but close to 0, the triangle intersection code below gets less stable
    // we arbitrarily decide that if a normal cone is ~168 degrees wide or more, the cone isn't useful
    if mindp <= 0.1 {
        bounds.cone_cutoff = 1.0;
        bounds.cone_cutoff_s8 = 127;
        return bounds;
    }

    let mut maxt = 0.0;

    // we need to find the point on center-t*axis ray that lies in negative half-space of all triangles
    for i in 0..triangles {
        // dot(center-t*axis-corner, trinormal) = 0
        // dot(center-corner, trinormal) - t * dot(axis, trinormal) = 0
        let corner = corners[i][0];
        let cx = center[0] - corner[0];
        let cy = center[1] - corner[1];
        let cz = center[2] - corner[2];

        let normal = normals[i];
        let dc = cx * normal[0] + cy * normal[1] + cz * normal[2];
        let dn = axis[0] * normal[0] + axis[1] * normal[1] + axis[2] * normal[2];

        // dn should be larger than mindp cutoff above
        assert!(dn > 0.0);
        let t = dc / dn;

        maxt = t.max(maxt);
    }

    // cone apex should be in the negative half-space of all cluster triangles by construction
    bounds.cone_apex[0] = center[0] - axis[0] * maxt;
    bounds.cone_apex[1] = center[1] - axis[1] * maxt;
    bounds.cone_apex[2] = center[2] - axis[2] * maxt;

    // note: this axis is the axis of the normal cone, but our test for perspective camera effectively negates the axis
    bounds.cone_axis[0] = axis[0];
    bounds.cone_axis[1] = axis[1];
    bounds.cone_axis[2] = axis[2];

    // cos(a) for normal cone is mindp; we need to add 90 degrees on both sides and invert the cone
    // which gives us -cos(a+90) = -(-sin(a)) = sin(a) = sqrt(1 - cos^2(a))
    bounds.cone_cutoff = (1.0 - mindp * mindp).sqrt();

    // quantize axis & cutoff to 8-bit SNORM format
    bounds.cone_axis_s8[0] = quantize_snorm(bounds.cone_axis[0], 8).try_into().unwrap();
    bounds.cone_axis_s8[1] = quantize_snorm(bounds.cone_axis[1], 8).try_into().unwrap();
    bounds.cone_axis_s8[2] = quantize_snorm(bounds.cone_axis[2], 8).try_into().unwrap();

    // for the 8-bit test to be conservative, we need to adjust the cutoff by measuring the max. error
    let cone_axis_s8_e0 = (bounds.cone_axis_s8[0] as f32 / 127.0 - bounds.cone_axis[0]).abs();
    let cone_axis_s8_e1 = (bounds.cone_axis_s8[1] as f32 / 127.0 - bounds.cone_axis[1]).abs();
    let cone_axis_s8_e2 = (bounds.cone_axis_s8[2] as f32 / 127.0 - bounds.cone_axis[2]).abs();

    // note that we need to round this up instead of rounding to nearest, hence +1
    let cone_cutoff_s8 =
        (127.0 * (bounds.cone_cutoff + cone_axis_s8_e0 + cone_axis_s8_e1 + cone_axis_s8_e2) + 1.0) as i32;

    bounds.cone_cutoff_s8 = if cone_cutoff_s8 > 127 {
        127
    } else {
        cone_cutoff_s8.try_into().unwrap()
    };

    bounds
}

/// Creates bounding volumes that can be used for frustum, backface and occlusion culling.
///
/// Same as [compute_cluster_bounds] but with meshlets as input.
pub fn compute_meshlet_bounds<Vertex>(meshlet: &Meshlet, vertices: &[Vertex]) -> Bounds
where
    Vertex: Position,
{
    let mut indices = [0; Meshlet::TRIANGLES_COUNT * 3];

    for i in 0..meshlet.triangle_count as usize {
        let triangle = meshlet.indices[i];

        let a = meshlet.vertices[triangle[0] as usize] as u32;
        let b = meshlet.vertices[triangle[1] as usize] as u32;
        let c = meshlet.vertices[triangle[2] as usize] as u32;

        // note: `compute_cluster_bounds` checks later if a/b/c are in range, no need to do it here

        indices[i * 3..i * 3 + 3].copy_from_slice(&[a, b, c]);
    }

    compute_cluster_bounds(&indices[0..meshlet.triangle_count as usize * 3], vertices)
}

#[cfg(test)]
mod test {
    use super::*;

    struct Vertex {
        x: f32,
        y: f32,
        z: f32,
    }

    impl Position for Vertex {
        fn pos(&self) -> [f32; 3] {
            [self.x, self.y, self.z]
        }
    }

    fn vb_from_slice(slice: &[f32]) -> Vec<Vertex> {
        slice
            .chunks_exact(3)
            .map(|v| Vertex {
                x: v[0],
                y: v[1],
                z: v[2],
            })
            .collect()
    }

    #[test]
    fn test_cluster_bounds_degenerate() {
        let vbd = vb_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let ibd = [0, 0, 0];
        let ib1 = [0, 1, 2];

        // all of the bounds below are degenerate as they use 0 triangles, one topology-degenerate triangle and one position-degenerate triangle respectively
        let bounds0 = compute_cluster_bounds::<Vertex>(&[], &[]);
        let boundsd = compute_cluster_bounds(&ibd, &vbd);
        let bounds1 = compute_cluster_bounds(&ib1, &vbd);

        assert!(bounds0.center == [0.0, 0.0, 0.0] && bounds0.radius == 0.0);
        assert!(boundsd.center == [0.0, 0.0, 0.0] && boundsd.radius == 0.0);
        assert!(bounds1.center == [0.0, 0.0, 0.0] && bounds1.radius == 0.0);

        let vb1 = vb_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let ib2 = [0, 1, 2, 0, 2, 1];

        // these bounds have a degenerate cone since the cluster has two triangles with opposite normals
        let bounds2 = compute_cluster_bounds(&ib2, &vb1);

        assert!(bounds2.cone_apex == [0.0, 0.0, 0.0]);
        assert!(bounds2.cone_axis == [0.0, 0.0, 0.0]);
        assert!(bounds2.cone_cutoff == 1.0);
        assert!(bounds2.cone_axis_s8 == [0, 0, 0]);
        assert!(bounds2.cone_cutoff_s8 == 127);

        // however, the bounding sphere needs to be in tact (here we only check bbox for simplicity)
        assert!(bounds2.center[0] - bounds2.radius <= 0.0 && bounds2.center[0] + bounds2.radius >= 1.0);
        assert!(bounds2.center[1] - bounds2.radius <= 0.0 && bounds2.center[1] + bounds2.radius >= 1.0);
        assert!(bounds2.center[2] - bounds2.radius <= 0.0 && bounds2.center[2] + bounds2.radius >= 1.0);
    }
}
