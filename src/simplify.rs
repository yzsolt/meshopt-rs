//! **Experimental** mesh and point cloud simplification

use crate::util::{fill_slice, zero_inverse};
use crate::vertex::{calc_pos_extents, Position};
use crate::Vector3;
use crate::INVALID_INDEX;

use std::collections::{hash_map::Entry, HashMap};
use std::ops::AddAssign;

#[derive(Default)]
struct EdgeAdjacency {
    counts: Vec<u32>,
    offsets: Vec<u32>,
    data: Vec<u32>,
}

fn build_edge_adjacency(adjacency: &mut EdgeAdjacency, indices: &[u32], vertex_count: usize) {
    let face_count = indices.len() / 3;

    // allocate arrays
    adjacency.counts = vec![0; vertex_count];
    adjacency.offsets = vec![0; vertex_count];
    adjacency.data = vec![0; indices.len()];

    // fill edge counts
    for index in indices {
        adjacency.counts[*index as usize] += 1;
    }

    // fill offset table
    let mut offset = 0;

    for (o, count) in adjacency.offsets.iter_mut().zip(adjacency.counts.iter()) {
        *o = offset;
        offset += *count;
    }

    assert_eq!(offset as usize, indices.len());

    // fill edge data
    for i in 0..face_count {
        let a = indices[i * 3 + 0] as usize;
        let b = indices[i * 3 + 1] as usize;
        let c = indices[i * 3 + 2] as usize;

        adjacency.data[adjacency.offsets[a] as usize] = b as u32;
        adjacency.data[adjacency.offsets[b] as usize] = c as u32;
        adjacency.data[adjacency.offsets[c] as usize] = a as u32;

        adjacency.offsets[a] += 1;
        adjacency.offsets[b] += 1;
        adjacency.offsets[c] += 1;
    }

    // fix offsets that have been disturbed by the previous pass
    for (offset, count) in adjacency.offsets.iter_mut().zip(adjacency.counts.iter()) {
        assert!(*offset >= *count);

        *offset -= *count;
    }
}

mod hash {
    use std::hash::{BuildHasherDefault, Hash, Hasher};

    #[derive(Clone)]
    pub struct VertexPosition(pub [f32; 3]);

    impl VertexPosition {
        const BYTES: usize = 3 * std::mem::size_of::<f32>();

        fn as_bytes(&self) -> &[u8] {
            let bytes: &[u8; Self::BYTES] = unsafe { std::mem::transmute(&self.0) };
            bytes
        }
    }

    impl Hash for VertexPosition {
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write(self.as_bytes());
        }
    }

    impl PartialEq for VertexPosition {
        fn eq(&self, other: &Self) -> bool {
            self.as_bytes() == other.as_bytes()
        }
    }

    impl Eq for VertexPosition {}

    #[derive(Default)]
    pub struct PositionHasher {
        state: u64,
    }

    impl Hasher for PositionHasher {
        fn write(&mut self, bytes: &[u8]) {
            assert!(bytes.len() == VertexPosition::BYTES);

            let a = u32::from_ne_bytes((&bytes[0..4]).try_into().unwrap());
            let b = u32::from_ne_bytes((&bytes[4..8]).try_into().unwrap());
            let c = u32::from_ne_bytes((&bytes[8..12]).try_into().unwrap());

            // Optimized Spatial Hashing for Collision Detection of Deformable Objects
            self.state = ((a.wrapping_mul(73856093)) ^ (b.wrapping_mul(19349663)) ^ (c.wrapping_mul(83492791))) as u64;
        }

        fn finish(&self) -> u64 {
            self.state
        }
    }

    pub type BuildPositionHasher = BuildHasherDefault<PositionHasher>;

    #[derive(Default)]
    pub struct IdHasher {
        state: u64,
    }

    impl Hasher for IdHasher {
        fn write(&mut self, bytes: &[u8]) {
            assert!(bytes.len() == std::mem::size_of::<u32>());

            let mut h = u32::from_ne_bytes((&bytes[0..4]).try_into().unwrap());

            // MurmurHash2 finalizer
            h ^= h >> 13;
            h = h.wrapping_mul(0x5bd1e995);
            h ^= h >> 15;

            self.state = h as u64;
        }

        fn finish(&self) -> u64 {
            self.state
        }
    }

    pub type BuildIdHasher = BuildHasherDefault<IdHasher>;
}

fn build_position_remap<Vertex>(remap: &mut [u32], wedge: &mut [u32], vertices: &[Vertex])
where
    Vertex: Position,
{
    let mut table = HashMap::with_capacity_and_hasher(vertices.len(), hash::BuildPositionHasher::default());

    // build forward remap: for each vertex, which other (canonical) vertex does it map to?
    // we use position equivalence for this, and remap vertices to other existing vertices
    for (index, vertex) in vertices.iter().enumerate() {
        remap[index] = match table.entry(hash::VertexPosition(vertex.pos())) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                entry.insert(index as u32);
                index as u32
            }
        };
    }

    // build wedge table: for each vertex, which other vertex is the next wedge that also maps to the same vertex?
    // entries in table form a (cyclic) wedge loop per vertex; for manifold vertices, wedge[i] == remap[i] == i
    for (i, w) in wedge.iter_mut().enumerate() {
        *w = i as u32;
    }

    for (i, ri) in remap.iter().enumerate() {
        let ri = *ri as usize;

        if ri != i {
            let r = ri;

            wedge[i] = wedge[r];
            wedge[r] = i as u32;
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum VertexKind {
    Manifold, // not on an attribute seam, not on any boundary
    Border,   // not on an attribute seam, has exactly two open edges
    Seam,     // on an attribute seam with exactly two attribute seam edges
    Complex,  // none of the above; these vertices can move as long as all wedges move to the target vertex
    Locked,   // none of the above; these vertices can't move
}

impl VertexKind {
    pub fn index(&self) -> usize {
        match *self {
            VertexKind::Manifold => 0,
            VertexKind::Border => 1,
            VertexKind::Seam => 2,
            VertexKind::Complex => 3,
            VertexKind::Locked => 4,
        }
    }
}

const KIND_COUNT: usize = 5;

// manifold vertices can collapse onto anything
// border/seam vertices can only be collapsed onto border/seam respectively
// complex vertices can collapse onto complex/locked
// a rule of thumb is that collapsing kind A into kind B preserves the kind B in the target vertex
// for example, while we could collapse Complex into Manifold, this would mean the target vertex isn't Manifold anymore
const CAN_COLLAPSE: [[bool; KIND_COUNT]; KIND_COUNT] = [
    [true, true, true, true, true],
    [false, true, false, false, false],
    [false, false, true, false, false],
    [false, false, false, true, true],
    [false, false, false, false, false],
];

// if a vertex is manifold or seam, adjoining edges are guaranteed to have an opposite edge
// note that for seam edges, the opposite edge isn't present in the attribute-based topology
// but is present if you consider a position-only mesh variant
const HAS_OPPOSITE: [[bool; KIND_COUNT]; KIND_COUNT] = [
    [true, true, true, false, true],
    [true, false, true, false, false],
    [true, true, true, false, true],
    [false, false, false, false, false],
    [true, false, true, false, false],
];

fn has_edge(adjacency: &EdgeAdjacency, a: u32, b: u32) -> bool {
    let count = adjacency.counts[a as usize] as usize;
    let offset = adjacency.offsets[a as usize] as usize;

    adjacency.data[offset..offset + count].iter().any(|d| *d == b)
}

fn classify_vertices(
    result: &mut [VertexKind],
    loop_: &mut [u32],
    loopback: &mut [u32],
    vertex_count: usize,
    adjacency: &EdgeAdjacency,
    remap: &[u32],
    wedge: &[u32],
) {
    // incoming & outgoing open edges: `INVALID_INDEX` if no open edges, i if there are more than 1
    // note that this is the same data as required in loop[] arrays; loop[] data is only valid for border/seam
    // but here it's okay to fill the data out for other types of vertices as well
    let openinc = loopback;
    let openout = loop_;

    for vertex in 0..vertex_count {
        let offset = adjacency.offsets[vertex] as usize;
        let count = adjacency.counts[vertex] as usize;

        let data = &adjacency.data[offset..offset + count];

        for target in data {
            if !has_edge(adjacency, *target, vertex as u32) {
                openinc[*target as usize] = if openinc[*target as usize] == INVALID_INDEX {
                    vertex as u32
                } else {
                    *target
                };
                openout[vertex] = if openout[vertex] == INVALID_INDEX {
                    *target
                } else {
                    vertex as u32
                };
            }
        }
    }

    for i in 0..vertex_count {
        if remap[i] == i as u32 {
            if wedge[i] == i as u32 {
                // no attribute seam, need to check if it's manifold
                let openi = openinc[i];
                let openo = openout[i];

                // note: we classify any vertices with no open edges as manifold
                // this is technically incorrect - if 4 triangles share an edge, we'll classify vertices as manifold
                // it's unclear if this is a problem in practice
                if openi == INVALID_INDEX && openo == INVALID_INDEX {
                    result[i] = VertexKind::Manifold;
                } else if openi != i as u32 && openo != i as u32 {
                    result[i] = VertexKind::Border;
                } else {
                    result[i] = VertexKind::Locked;
                }
            } else if wedge[wedge[i] as usize] == i as u32 {
                // attribute seam; need to distinguish between Seam and Locked
                let w = wedge[i] as usize;
                let openiv = openinc[i] as usize;
                let openov = openout[i] as usize;
                let openiw = openinc[w] as usize;
                let openow = openout[w] as usize;

                // seam should have one open half-edge for each vertex, and the edges need to "connect" - point to the same vertex post-remap
                if openiv != INVALID_INDEX as usize
                    && openiv != i
                    && openov != INVALID_INDEX as usize
                    && openov != i
                    && openiw != INVALID_INDEX as usize
                    && openiw != w
                    && openow != INVALID_INDEX as usize
                    && openow != w
                {
                    if remap[openiv] == remap[openow] && remap[openov] == remap[openiw] {
                        result[i] = VertexKind::Seam;
                    } else {
                        result[i] = VertexKind::Locked;
                    }
                } else {
                    result[i] = VertexKind::Locked;
                }
            } else {
                // more than one vertex maps to this one; we don't have classification available
                result[i] = VertexKind::Locked;
            }
        } else {
            assert!(remap[i] < i as u32);

            result[i] = result[remap[i] as usize];
        }
    }
}

fn rescale_positions<Vertex>(result: &mut [Vector3], vertices: &[Vertex])
where
    Vertex: Position,
{
    let (minv, extent) = calc_pos_extents(vertices);

    for (i, vertex) in vertices.iter().enumerate() {
        let v = vertex.pos();

        result[i] = Vector3 {
            x: v[0],
            y: v[1],
            z: v[2],
        };
    }

    let scale = zero_inverse(extent);

    for pos in result {
        pos.x = (pos.x - minv[0]) * scale;
        pos.y = (pos.y - minv[1]) * scale;
        pos.z = (pos.z - minv[2]) * scale;
    }
}

#[derive(Clone, Copy, Default)]
struct Quadric {
    a00: f32,
    a11: f32,
    a22: f32,
    a10: f32,
    a20: f32,
    a21: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    c: f32,
    w: f32,
}

union CollapseUnion {
    bidi: u32,
    error: f32,
    errorui: u32,
}

impl Clone for CollapseUnion {
    fn clone(&self) -> Self {
        Self {
            bidi: unsafe { self.bidi },
        }
    }
}

impl Default for CollapseUnion {
    fn default() -> Self {
        Self { bidi: 0 }
    }
}

#[derive(Clone, Default)]
struct Collapse {
    v0: u32,
    v1: u32,
    u: CollapseUnion,
}

impl AddAssign for Quadric {
    fn add_assign(&mut self, other: Self) {
        self.a00 += other.a00;
        self.a11 += other.a11;
        self.a22 += other.a22;
        self.a10 += other.a10;
        self.a20 += other.a20;
        self.a21 += other.a21;
        self.b0 += other.b0;
        self.b1 += other.b1;
        self.b2 += other.b2;
        self.c += other.c;
        self.w += other.w;
    }
}

impl Quadric {
    pub fn from_point(x: f32, y: f32, z: f32, w: f32) -> Self {
        // we need to encode (x - X) ^ 2 + (y - Y)^2 + (z - Z)^2 into the quadric
        Self {
            a00: w,
            a11: w,
            a22: w,
            a10: 0.0,
            a20: 0.0,
            a21: 0.0,
            b0: -2.0 * x * w,
            b1: -2.0 * y * w,
            b2: -2.0 * z * w,
            c: (x * x + y * y + z * z) * w,
            w,
        }
    }

    fn from_plane(a: f32, b: f32, c: f32, d: f32, w: f32) -> Self {
        let aw = a * w;
        let bw = b * w;
        let cw = c * w;
        let dw = d * w;

        Self {
            a00: a * aw,
            a11: b * bw,
            a22: c * cw,
            a10: a * bw,
            a20: a * cw,
            a21: b * cw,
            b0: a * dw,
            b1: b * dw,
            b2: c * dw,
            c: d * dw,
            w,
        }
    }

    pub fn from_triangle(p0: &Vector3, p1: &Vector3, p2: &Vector3, weight: f32) -> Self {
        let p10 = Vector3::new(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
        let p20 = Vector3::new(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);

        // normal = cross(p1 - p0, p2 - p0)
        let mut normal = Vector3::new(
            p10.y * p20.z - p10.z * p20.y,
            p10.z * p20.x - p10.x * p20.z,
            p10.x * p20.y - p10.y * p20.x,
        );
        let area = normal.normalize();

        let distance = normal.x * p0.x + normal.y * p0.y + normal.z * p0.z;

        // we use sqrtf(area) so that the error is scaled linearly; this tends to improve silhouettes
        Self::from_plane(normal.x, normal.y, normal.z, -distance, area.sqrt() * weight)
    }

    pub fn from_triangle_edge(p0: &Vector3, p1: &Vector3, p2: &Vector3, weight: f32) -> Self {
        let mut p10 = Vector3::new(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
        let length = p10.normalize();

        // p20p = length of projection of p2-p0 onto normalize(p1 - p0)
        let p20 = Vector3::new(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
        let p20p = p20.x * p10.x + p20.y * p10.y + p20.z * p10.z;

        // normal = altitude of triangle from point p2 onto edge p1-p0
        let mut normal = Vector3::new(p20.x - p10.x * p20p, p20.y - p10.y * p20p, p20.z - p10.z * p20p);
        normal.normalize();

        let distance = normal.x * p0.x + normal.y * p0.y + normal.z * p0.z;

        // note: the weight is scaled linearly with edge length; this has to match the triangle weight
        Self::from_plane(normal.x, normal.y, normal.z, -distance, length * weight)
    }

    pub fn error(&self, v: &Vector3) -> f32 {
        let mut rx = self.b0;
        let mut ry = self.b1;
        let mut rz = self.b2;

        rx += self.a10 * v.y;
        ry += self.a21 * v.z;
        rz += self.a20 * v.x;

        rx *= 2.0;
        ry *= 2.0;
        rz *= 2.0;

        rx += self.a00 * v.x;
        ry += self.a11 * v.y;
        rz += self.a22 * v.z;

        let mut r = self.c;
        r += rx * v.x;
        r += ry * v.y;
        r += rz * v.z;

        let s = zero_inverse(self.w);

        r.abs() * s
    }
}

fn fill_face_quadrics(vertex_quadrics: &mut [Quadric], indices: &[u32], vertex_positions: &[Vector3], remap: &[u32]) {
    for i in indices.chunks_exact(3) {
        let (i0, i1, i2) = (i[0] as usize, i[1] as usize, i[2] as usize);

        let q = Quadric::from_triangle(&vertex_positions[i0], &vertex_positions[i1], &vertex_positions[i2], 1.0);

        vertex_quadrics[remap[i0] as usize] += q;
        vertex_quadrics[remap[i1] as usize] += q;
        vertex_quadrics[remap[i2] as usize] += q;
    }
}

fn fill_edge_quadrics(
    vertex_quadrics: &mut [Quadric],
    indices: &[u32],
    vertex_positions: &[Vector3],
    remap: &[u32],
    vertex_kind: &[VertexKind],
    loop_: &[u32],
    loopback: &[u32],
) {
    for i in indices.chunks_exact(3) {
        const NEXT: [usize; 3] = [1, 2, 0];

        for e in 0..3 {
            let i0 = i[e] as usize;
            let i1 = i[NEXT[e]] as usize;

            let k0 = vertex_kind[i0];
            let k1 = vertex_kind[i1];

            // check that either i0 or i1 are border/seam and are on the same edge loop
            // note that we need to add the error even for edged that connect e.g. border & locked
            // if we don't do that, the adjacent border->border edge won't have correct errors for corners
            if k0 != VertexKind::Border && k0 != VertexKind::Seam && k1 != VertexKind::Border && k1 != VertexKind::Seam
            {
                continue;
            }

            if (k0 == VertexKind::Border || k0 == VertexKind::Seam) && loop_[i0] != i1 as u32 {
                continue;
            }

            if (k1 == VertexKind::Border || k1 == VertexKind::Seam) && loopback[i1] != i0 as u32 {
                continue;
            }

            // seam edges should occur twice (i0->i1 and i1->i0) - skip redundant edges
            if HAS_OPPOSITE[k0.index()][k1.index()] && remap[i1] > remap[i0] {
                continue;
            }

            let i2 = i[NEXT[NEXT[e]]] as usize;

            // we try hard to maintain border edge geometry; seam edges can move more freely
            // due to topological restrictions on collapses, seam quadrics slightly improves collapse structure but aren't critical
            const EDGE_WEIGHT_SEAM: f32 = 1.0;
            const EDGE_WEIGHT_BORDER: f32 = 10.0;

            let edge_weight = if k0 == VertexKind::Border || k1 == VertexKind::Border {
                EDGE_WEIGHT_BORDER
            } else {
                EDGE_WEIGHT_SEAM
            };

            let q = Quadric::from_triangle_edge(
                &vertex_positions[i0],
                &vertex_positions[i1],
                &vertex_positions[i2],
                edge_weight,
            );

            vertex_quadrics[remap[i0] as usize] += q;
            vertex_quadrics[remap[i1] as usize] += q;
        }
    }
}

fn pick_edge_collapses(
    collapses: &mut [Collapse],
    indices: &[u32],
    remap: &[u32],
    vertex_kind: &[VertexKind],
    loop_: &[u32],
) -> usize {
    let mut collapse_count = 0;

    for i in indices.chunks_exact(3) {
        const NEXT: [usize; 3] = [1, 2, 0];

        for e in 0..3 {
            let i0 = i[e] as usize;
            let i1 = i[NEXT[e]] as usize;

            // this can happen either when input has a zero-length edge, or when we perform collapses for complex
            // topology w/seams and collapse a manifold vertex that connects to both wedges onto one of them
            // we leave edges like this alone since they may be important for preserving mesh integrity
            if remap[i0] == remap[i1] {
                continue;
            }

            let k0 = vertex_kind[i0];
            let k1 = vertex_kind[i1];

            // the edge has to be collapsible in at least one direction
            if !(CAN_COLLAPSE[k0.index()][k1.index()] || CAN_COLLAPSE[k1.index()][k0.index()]) {
                continue;
            }

            // manifold and seam edges should occur twice (i0->i1 and i1->i0) - skip redundant edges
            if HAS_OPPOSITE[k0.index()][k1.index()] && remap[i1] > remap[i0] {
                continue;
            }

            // two vertices are on a border or a seam, but there's no direct edge between them
            // this indicates that they belong to two different edge loops and we should not collapse this edge
            // loop[] tracks half edges so we only need to check i0->i1
            if k0 == k1 && (k0 == VertexKind::Border || k0 == VertexKind::Seam) && loop_[i0] != i1 as u32 {
                continue;
            }

            // edge can be collapsed in either direction - we will pick the one with minimum error
            // note: we evaluate error later during collapse ranking, here we just tag the edge as bidirectional
            if CAN_COLLAPSE[k0.index()][k1.index()] & CAN_COLLAPSE[k1.index()][k0.index()] {
                let c = Collapse {
                    v0: i0 as u32,
                    v1: i1 as u32,
                    u: CollapseUnion { bidi: 1 },
                };
                collapses[collapse_count] = c;
                collapse_count += 1;
            } else {
                // edge can only be collapsed in one direction
                let e0 = if CAN_COLLAPSE[k0.index()][k1.index()] { i0 } else { i1 };
                let e1 = if CAN_COLLAPSE[k0.index()][k1.index()] { i1 } else { i0 };

                let c = Collapse {
                    v0: e0 as u32,
                    v1: e1 as u32,
                    u: CollapseUnion { bidi: 0 },
                };
                collapses[collapse_count] = c;
                collapse_count += 1;
            }
        }
    }

    collapse_count
}

fn rank_edge_collapses(
    collapses: &mut [Collapse],
    vertex_positions: &[Vector3],
    vertex_quadrics: &[Quadric],
    remap: &[u32],
) {
    for c in collapses {
        let i0 = c.v0;
        let i1 = c.v1;

        // most edges are bidirectional which means we need to evaluate errors for two collapses
        // to keep this code branchless we just use the same edge for unidirectional edges
        let j0 = unsafe {
            if c.u.bidi != 0 {
                i1
            } else {
                i0
            }
        };
        let j1 = unsafe {
            if c.u.bidi != 0 {
                i0
            } else {
                i1
            }
        };

        let qi = vertex_quadrics[remap[i0 as usize] as usize];
        let qj = vertex_quadrics[remap[j0 as usize] as usize];

        let ei = qi.error(&vertex_positions[i1 as usize]);
        let ej = qj.error(&vertex_positions[j1 as usize]);

        // pick edge direction with minimal error
        c.v0 = if ei <= ej { i0 } else { j0 };
        c.v1 = if ei <= ej { i1 } else { j1 };
        c.u.error = ei.min(ej);
    }
}

fn sort_edge_collapses(sort_order: &mut [u32], collapses: &[Collapse]) {
    const SORT_BITS: usize = 11;

    // fill histogram for counting sort
    let mut histogram = [0u32; 1 << SORT_BITS];

    for c in collapses {
        // skip sign bit since error is non-negative
        let key = unsafe { (c.u.errorui << 1) >> (32 - SORT_BITS) };

        histogram[key as usize] += 1;
    }

    // compute offsets based on histogram data
    let mut histogram_sum = 0;

    for i in 0..(1 << SORT_BITS) {
        let count = histogram[i];
        histogram[i] = histogram_sum;
        histogram_sum += count;
    }

    assert_eq!(histogram_sum as usize, collapses.len());

    // compute sort order based on offsets
    for (i, c) in collapses.iter().enumerate() {
        // skip sign bit since error is non-negative
        let key = unsafe { ((c.u.errorui << 1) >> (32 - SORT_BITS)) as usize };

        sort_order[histogram[key] as usize] = i as u32;
        histogram[key] += 1;
    }
}

fn perform_edge_collapses(
    collapse_remap: &mut [u32],
    collapse_locked: &mut [bool],
    vertex_quadrics: &mut [Quadric],
    collapses: &[Collapse],
    collapse_order: &[u32],
    remap: &[u32],
    wedge: &[u32],
    vertex_kind: &[VertexKind],
    triangle_collapse_goal: usize,
    error_goal: f32,
    error_limit: f32,
) -> usize {
    let mut edge_collapses = 0;
    let mut triangle_collapses = 0;

    for order in collapse_order {
        let c = collapses[*order as usize].clone();

        let error = unsafe { c.u.error };

        if error > error_limit {
            break;
        }

        if error > error_goal && triangle_collapses > triangle_collapse_goal / 10 {
            break;
        }

        if triangle_collapses >= triangle_collapse_goal {
            break;
        }

        let i0 = c.v0 as usize;
        let i1 = c.v1 as usize;

        let r0 = remap[i0] as usize;
        let r1 = remap[i1] as usize;

        // we don't collapse vertices that had source or target vertex involved in a collapse
        // it's important to not move the vertices twice since it complicates the tracking/remapping logic
        // it's important to not move other vertices towards a moved vertex to preserve error since we don't re-rank collapses mid-pass
        if collapse_locked[r0] || collapse_locked[r1] {
            continue;
        }

        assert_eq!(collapse_remap[r0] as usize, r0);
        assert_eq!(collapse_remap[r1] as usize, r1);

        vertex_quadrics[r1] += vertex_quadrics[r0];

        match vertex_kind[i0] {
            VertexKind::Complex => {
                let mut v = i0;

                loop {
                    collapse_remap[v] = r1 as u32;
                    v = wedge[v] as usize;

                    if v == i0 {
                        break;
                    }
                }
            }
            VertexKind::Seam => {
                // remap v0 to v1 and seam pair of v0 to seam pair of v1
                let s0 = wedge[i0] as usize;
                let s1 = wedge[i1] as usize;

                assert!(s0 != i0 && s1 != i1);
                assert!(wedge[s0] as usize == i0 && wedge[s1] as usize == i1);

                collapse_remap[i0] = i1 as u32;
                collapse_remap[s0] = s1 as u32;
            }
            _ => {
                assert_eq!(wedge[i0] as usize, i0);

                collapse_remap[i0] = i1 as u32;
            }
        }

        collapse_locked[r0] = true;
        collapse_locked[r1] = true;

        // border edges collapse 1 triangle, other edges collapse 2 or more
        triangle_collapses += if vertex_kind[i0] == VertexKind::Border { 1 } else { 2 };
        edge_collapses += 1;
    }

    edge_collapses
}

fn remap_index_buffer(indices: &mut [u32], collapse_remap: &[u32]) -> usize {
    let mut write = 0;

    for i in (0..indices.len()).step_by(3) {
        let v0 = collapse_remap[indices[i + 0] as usize];
        let v1 = collapse_remap[indices[i + 1] as usize];
        let v2 = collapse_remap[indices[i + 2] as usize];

        // we never move the vertex twice during a single pass
        assert_eq!(collapse_remap[v0 as usize], v0);
        assert_eq!(collapse_remap[v1 as usize], v1);
        assert_eq!(collapse_remap[v2 as usize], v2);

        if v0 != v1 && v0 != v2 && v1 != v2 {
            indices[write + 0] = v0;
            indices[write + 1] = v1;
            indices[write + 2] = v2;
            write += 3;
        }
    }

    write
}

fn remap_edge_loops(loop_: &mut [u32], collapse_remap: &[u32]) {
    for i in 0..loop_.len() {
        if loop_[i] != INVALID_INDEX {
            let l = loop_[i];
            let r = collapse_remap[l as usize];

            // i == r is a special case when the seam edge is collapsed in a direction opposite to where loop goes
            loop_[i] = if i == r as usize { loop_[l as usize] } else { r };
        }
    }
}

fn compute_vertex_ids(vertex_ids: &mut [u32], vertex_positions: &[Vector3], grid_size: i32) {
    assert!(grid_size >= 1 && grid_size <= 1024);
    let cell_scale = (grid_size - 1) as f32;

    for (pos, id) in vertex_positions.iter().zip(vertex_ids.iter_mut()) {
        let xi = (pos.x * cell_scale + 0.5) as i32;
        let yi = (pos.y * cell_scale + 0.5) as i32;
        let zi = (pos.z * cell_scale + 0.5) as i32;

        *id = ((xi << 20) | (yi << 10) | zi) as u32;
    }
}

fn count_triangles(vertex_ids: &[u32], indices: &[u32]) -> usize {
    let mut result = 0;

    for abc in indices.chunks_exact(3) {
        let id0 = vertex_ids[abc[0] as usize];
        let id1 = vertex_ids[abc[1] as usize];
        let id2 = vertex_ids[abc[2] as usize];

        result += ((id0 != id1) && (id0 != id2) && (id1 != id2)) as usize;
    }

    result
}

fn fill_vertex_cells(
    table: &mut HashMap<u32, u32, hash::BuildIdHasher>,
    vertex_cells: &mut [u32],
    vertex_ids: &[u32],
) -> usize {
    let mut result: usize = 0;

    for i in 0..vertex_ids.len() {
        vertex_cells[i] = match table.entry(vertex_ids[i]) {
            Entry::Occupied(entry) => vertex_cells[*entry.get() as usize],
            Entry::Vacant(entry) => {
                entry.insert(i as u32);
                result += 1;
                result as u32 - 1
            }
        };
    }

    result
}

fn count_vertex_cells(table: &mut HashMap<u32, u32, hash::BuildIdHasher>, vertex_ids: &[u32]) -> usize {
    table.clear();

    let mut result = 0;

    for id in vertex_ids {
        result += match table.entry(*id) {
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = *id;
                0
            }
            Entry::Vacant(entry) => {
                entry.insert(*id as u32);
                1
            }
        };
    }

    result
}

fn fill_cell_quadrics(
    cell_quadrics: &mut [Quadric],
    indices: &[u32],
    vertex_positions: &[Vector3],
    vertex_cells: &[u32],
) {
    for abc in indices.chunks_exact(3) {
        let i0 = abc[0] as usize;
        let i1 = abc[1] as usize;
        let i2 = abc[2] as usize;

        let c0 = vertex_cells[i0] as usize;
        let c1 = vertex_cells[i1] as usize;
        let c2 = vertex_cells[i2] as usize;

        let single_cell = (c0 == c1) & (c0 == c2);

        let q = Quadric::from_triangle(
            &vertex_positions[i0],
            &vertex_positions[i1],
            &vertex_positions[i2],
            if single_cell { 3.0 } else { 1.0 },
        );

        if single_cell {
            cell_quadrics[c0] += q;
        } else {
            cell_quadrics[c0] += q;
            cell_quadrics[c1] += q;
            cell_quadrics[c2] += q;
        }
    }
}

fn fill_cell_quadrics2(cell_quadrics: &mut [Quadric], vertex_positions: &[Vector3], vertex_cells: &[u32]) {
    for (c, v) in vertex_cells.iter().zip(vertex_positions.iter()) {
        let q = Quadric::from_point(v.x, v.y, v.z, 1.0);

        cell_quadrics[*c as usize] += q;
    }
}

fn fill_cell_remap(
    cell_remap: &mut [u32],
    cell_errors: &mut [f32],
    vertex_cells: &[u32],
    cell_quadrics: &[Quadric],
    vertex_positions: &[Vector3],
) {
    for ((i, c), v) in vertex_cells.iter().enumerate().zip(vertex_positions.iter()) {
        let cell = *c as usize;
        let error = cell_quadrics[cell].error(v);

        if cell_remap[cell] == INVALID_INDEX || cell_errors[cell] > error {
            cell_remap[cell] = i as u32;
            cell_errors[cell] = error;
        }
    }
}

fn filter_triangles(
    destination: &mut [u32],
    tritable: &mut HashMap<hash::VertexPosition, u32, hash::BuildPositionHasher>,
    indices: &[u32],
    vertex_cells: &[u32],
    cell_remap: &[u32],
) -> usize {
    let mut result = 0;

    for idx in indices.chunks_exact(3) {
        let c0 = vertex_cells[idx[0] as usize] as usize;
        let c1 = vertex_cells[idx[1] as usize] as usize;
        let c2 = vertex_cells[idx[2] as usize] as usize;

        if c0 != c1 && c0 != c2 && c1 != c2 {
            let a = cell_remap[c0];
            let b = cell_remap[c1];
            let c = cell_remap[c2];

            let mut abc = [a, b, c];

            if b < a && b < c {
                abc.rotate_left(1);
            } else if c < a && c < b {
                abc.rotate_right(1);
            };

            destination[result * 3..result * 3 + 3].copy_from_slice(&abc);

            let p = hash::VertexPosition(unsafe { std::mem::transmute(abc) });

            match tritable.entry(p) {
                Entry::Occupied(_entry) => {}
                Entry::Vacant(entry) => {
                    entry.insert(result as u32);
                    result += 1;
                }
            }
        }
    }

    result * 3
}

fn interpolate(y: f32, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    // three point interpolation from "revenge of interpolation search" paper
    let num = (y1 - y) * (x1 - x2) * (x1 - x0) * (y2 - y0);
    let den = (y2 - y) * (x1 - x2) * (y0 - y1) + (y0 - y) * (x1 - x0) * (y1 - y2);
    x1 + num / den
}

/// Reduces the number of triangles in the mesh, attempting to preserve mesh appearance as much as possible.
///
/// The algorithm tries to preserve mesh topology and can stop short of the target goal based on topology constraints or target error.
/// If not all attributes from the input mesh are required, it's recommended to reindex the mesh using [generate_shadow_index_buffer](crate::index::generator::generate_shadow_index_buffer) prior to simplification.
///
/// Returns the number of indices after simplification, with destination containing new index data.
/// The resulting index buffer references vertices from the original vertex buffer.
/// If the original vertex data isn't required, creating a compact vertex buffer using [optimize_vertex_fetch](crate::vertex::fetch::optimize_vertex_fetch) is recommended.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the **source** index buffer (since optimization is iterative, this means `indices.len()` elements - **not** `target_index_count`!)
pub fn simplify<Vertex>(
    destination: &mut [u32],
    indices: &[u32],
    vertices: &[Vertex],
    target_index_count: usize,
    target_error: f32,
) -> usize
where
    Vertex: Position,
{
    assert_eq!(indices.len() % 3, 0);
    assert!(target_index_count <= indices.len());

    let result = &mut destination[0..indices.len()];

    // build adjacency information
    let mut adjacency = EdgeAdjacency::default();
    build_edge_adjacency(&mut adjacency, indices, vertices.len());

    // build position remap that maps each vertex to the one with identical position
    let mut remap = vec![0u32; vertices.len()];
    let mut wedge = vec![0u32; vertices.len()];
    build_position_remap(&mut remap, &mut wedge, vertices);

    // classify vertices; vertex kind determines collapse rules, see `CAN_COLLAPSE`
    let mut vertex_kind = vec![VertexKind::Manifold; vertices.len()];
    let mut loop_ = vec![INVALID_INDEX; vertices.len()];
    let mut loopback = vec![INVALID_INDEX; vertices.len()];
    classify_vertices(
        &mut vertex_kind,
        &mut loop_,
        &mut loopback,
        vertices.len(),
        &adjacency,
        &remap,
        &wedge,
    );

    let mut vertex_positions = vec![Vector3::default(); vertices.len()]; // TODO: spare init?
    rescale_positions(&mut vertex_positions, vertices);

    let mut vertex_quadrics = vec![Quadric::default(); vertices.len()];
    fill_face_quadrics(&mut vertex_quadrics, indices, &vertex_positions, &remap);
    fill_edge_quadrics(
        &mut vertex_quadrics,
        indices,
        &vertex_positions,
        &remap,
        &vertex_kind,
        &loop_,
        &loopback,
    );

    result.copy_from_slice(indices);

    // TODO: skip init?
    let mut edge_collapses = vec![Collapse::default(); indices.len()];
    let mut collapse_order = vec![0u32; indices.len()];
    let mut collapse_remap = vec![0u32; vertices.len()];

    let mut collapse_locked = vec![false; vertices.len()];

    let mut result_count = indices.len();

    // `target_error` input is linear; we need to adjust it to match `Quadric::error` units
    let error_limit = target_error * target_error;

    while result_count > target_index_count {
        let edge_collapse_count = pick_edge_collapses(
            &mut edge_collapses,
            &result[0..result_count],
            &remap,
            &vertex_kind,
            &loop_,
        );

        // no edges can be collapsed any more due to topology restrictions
        if edge_collapse_count == 0 {
            break;
        }

        rank_edge_collapses(
            &mut edge_collapses[0..edge_collapse_count],
            &vertex_positions,
            &vertex_quadrics,
            &remap,
        );

        sort_edge_collapses(&mut collapse_order, &edge_collapses[0..edge_collapse_count]);

        // most collapses remove 2 triangles; use this to establish a bound on the pass in terms of error limit
        // note that edge_collapse_goal is an estimate; triangle_collapse_goal will be used to actually limit collapses
        let triangle_collapse_goal = (result_count - target_index_count) / 3;
        let edge_collapse_goal = triangle_collapse_goal / 2;

        // we limit the error in each pass based on the error of optimal last collapse; since many collapses will be locked
        // as they will share vertices with other successfull collapses, we need to increase the acceptable error by this factor
        const PASS_ERROR_BOUND: f32 = 1.5;

        let error_goal = if edge_collapse_goal < edge_collapse_count {
            unsafe { edge_collapses[collapse_order[edge_collapse_goal] as usize].u.error * PASS_ERROR_BOUND }
        } else {
            f32::MAX
        };

        for (i, r) in collapse_remap.iter_mut().enumerate() {
            *r = i as u32;
        }

        fill_slice(&mut collapse_locked, false);

        let collapses = perform_edge_collapses(
            &mut collapse_remap,
            &mut collapse_locked,
            &mut vertex_quadrics,
            &edge_collapses,
            &collapse_order,
            &remap,
            &wedge,
            &vertex_kind,
            triangle_collapse_goal,
            error_goal,
            error_limit,
        );

        // no edges can be collapsed any more due to hitting the error limit or triangle collapse limit
        if collapses == 0 {
            break;
        }

        remap_edge_loops(&mut loop_, &collapse_remap);
        remap_edge_loops(&mut loopback, &collapse_remap);

        let new_count = remap_index_buffer(&mut result[0..result_count], &collapse_remap);
        assert!(new_count < result_count);

        result_count = new_count;
    }

    result_count
}

/// Reduces the number of triangles in the mesh, sacrificing mesh apperance for simplification performance.
///
/// The algorithm doesn't preserve mesh topology but is always able to reach target triangle count.
///
/// Returns the number of indices after simplification, with destination containing new index data
/// The resulting index buffer references vertices from the original vertex buffer.
/// If the original vertex data isn't required, creating a compact vertex buffer using [optimize_vertex_fetch](crate::vertex::fetch::optimize_vertex_fetch) is recommended.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the target index buffer
pub fn simplify_sloppy<Vertex>(
    destination: &mut [u32],
    indices: &[u32],
    vertices: &[Vertex],
    target_index_count: usize,
) -> usize
where
    Vertex: Position,
{
    assert_eq!(indices.len() % 3, 0);
    assert!(target_index_count <= indices.len());

    // we expect to get ~2 triangles/vertex in the output
    let target_cell_count = target_index_count / 6;

    if target_cell_count == 0 {
        return 0;
    }

    let mut vertex_positions = vec![Vector3::default(); vertices.len()];
    rescale_positions(&mut vertex_positions, vertices);

    // find the optimal grid size using guided binary search

    let mut vertex_ids = vec![0; vertices.len()];

    const INTERPOLATION_PASSES: i32 = 5;

    // invariant: # of triangles in min_grid <= target_count
    let mut min_grid: i32 = 0;
    let mut max_grid: i32 = 1025;
    let mut min_triangles = 0;
    let mut max_triangles = indices.len() / 3;

    // instead of starting in the middle, let's guess as to what the answer might be! triangle count usually grows as a square of grid size...
    let mut next_grid_size = ((target_cell_count as f32).sqrt() + 0.5) as i32;

    for pass in 0..10 + INTERPOLATION_PASSES {
        assert!(min_triangles < target_index_count / 3);
        assert!(max_grid - min_grid > 1);

        // we clamp the prediction of the grid size to make sure that the search converges
        let mut grid_size = next_grid_size;
        grid_size = if grid_size <= min_grid {
            min_grid + 1
        } else {
            if grid_size >= max_grid {
                max_grid - 1
            } else {
                grid_size
            }
        };

        compute_vertex_ids(&mut vertex_ids, &vertex_positions, grid_size);
        let triangles = count_triangles(&vertex_ids, &indices);

        let tip = interpolate(
            (target_index_count / 3) as f32,
            min_grid as f32,
            min_triangles as f32,
            grid_size as f32,
            triangles as f32,
            max_grid as f32,
            max_triangles as f32,
        );

        if triangles <= target_index_count / 3 {
            min_grid = grid_size;
            min_triangles = triangles;
        } else {
            max_grid = grid_size;
            max_triangles = triangles;
        }

        if triangles == target_index_count / 3 || max_grid - min_grid <= 1 {
            break;
        }

        // we start by using interpolation search - it usually converges faster
        // however, interpolation search has a worst case of O(N) so we switch to binary search after a few iterations which converges in O(logN)
        next_grid_size = if pass < INTERPOLATION_PASSES {
            (tip + 0.5) as i32
        } else {
            (min_grid + max_grid) / 2
        };
    }

    if min_triangles == 0 {
        return 0;
    }

    // build vertex->cell association by mapping all vertices with the same quantized position to the same cell
    let mut table = HashMap::with_capacity_and_hasher(vertices.len(), hash::BuildIdHasher::default());

    let mut vertex_cells = vec![0; vertices.len()];

    compute_vertex_ids(&mut vertex_ids, &vertex_positions, min_grid);
    let cell_count = fill_vertex_cells(&mut table, &mut vertex_cells, &vertex_ids);

    // build a quadric for each target cell
    let mut cell_quadrics = vec![Quadric::default(); cell_count];

    fill_cell_quadrics(&mut cell_quadrics, &indices, &vertex_positions, &vertex_cells);

    // for each target cell, find the vertex with the minimal error
    let mut cell_remap = vec![INVALID_INDEX; cell_count];
    let mut cell_errors = vec![0.0; cell_count];

    fill_cell_remap(
        &mut cell_remap,
        &mut cell_errors,
        &vertex_cells,
        &cell_quadrics,
        &vertex_positions,
    );

    // collapse triangles!
    // note that we need to filter out triangles that we've already output because we very frequently generate redundant triangles between cells :(
    let mut tritable = HashMap::with_capacity_and_hasher(min_triangles, hash::BuildPositionHasher::default());

    let write = filter_triangles(destination, &mut tritable, &indices, &vertex_cells, &cell_remap);
    assert!(write <= target_index_count);

    write
}

/// Reduces the number of points in the cloud to reach the given target.
///
/// Returns the number of points after simplification, with destination containing new index data.
/// The resulting index buffer references vertices from the original vertex buffer.
/// If the original vertex data isn't required, creating a compact vertex buffer using [optimize_vertex_fetch](crate::vertex::fetch::optimize_vertex_fetch) is recommended.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the target index buffer
pub fn simplify_points<Vertex>(destination: &mut [u32], vertices: &[Vertex], target_vertex_count: usize) -> usize
where
    Vertex: Position,
{
    assert!(target_vertex_count <= vertices.len());

    let target_cell_count = target_vertex_count;

    if target_cell_count == 0 {
        return 0;
    }

    let mut vertex_positions = vec![Vector3::default(); vertices.len()];
    rescale_positions(&mut vertex_positions, vertices);

    // find the optimal grid size using guided binary search

    let mut vertex_ids = vec![0; vertices.len()];

    let mut table = HashMap::with_capacity_and_hasher(vertices.len(), hash::BuildIdHasher::default());

    const INTERPOLATION_PASSES: i32 = 5;

    // invariant: # of vertices in min_grid <= target_count
    let mut min_grid: i32 = 0;
    let mut max_grid: i32 = 1025;
    let mut min_vertices = 0;
    let mut max_vertices = vertices.len();

    // instead of starting in the middle, let's guess as to what the answer might be! triangle count usually grows as a square of grid size...
    let mut next_grid_size = ((target_cell_count as f32).sqrt() + 0.5) as i32;

    for pass in 0..10 + INTERPOLATION_PASSES {
        assert!(min_vertices < target_vertex_count);
        assert!(max_grid - min_grid > 1);

        // we clamp the prediction of the grid size to make sure that the search converges
        let mut grid_size = next_grid_size;
        grid_size = if grid_size <= min_grid {
            min_grid + 1
        } else {
            if grid_size >= max_grid {
                max_grid - 1
            } else {
                grid_size
            }
        };

        compute_vertex_ids(&mut vertex_ids, &vertex_positions, grid_size);
        let vertices = count_vertex_cells(&mut table, &vertex_ids);

        let tip = interpolate(
            target_vertex_count as f32,
            min_grid as f32,
            min_vertices as f32,
            grid_size as f32,
            vertices as f32,
            max_grid as f32,
            max_vertices as f32,
        );

        if vertices <= target_vertex_count {
            min_grid = grid_size;
            min_vertices = vertices;
        } else {
            max_grid = grid_size;
            max_vertices = vertices;
        }

        if vertices == target_vertex_count || max_grid - min_grid <= 1 {
            break;
        }

        // we start by using interpolation search - it usually converges faster
        // however, interpolation search has a worst case of O(N) so we switch to binary search after a few iterations which converges in O(logN)
        next_grid_size = if pass < INTERPOLATION_PASSES {
            (tip + 0.5) as i32
        } else {
            (min_grid + max_grid) / 2
        };
    }

    if min_vertices == 0 {
        return 0;
    }

    // build vertex->cell association by mapping all vertices with the same quantized position to the same cell
    let mut vertex_cells = vec![0; vertices.len()];

    compute_vertex_ids(&mut vertex_ids, &vertex_positions, min_grid);
    table.clear();
    let cell_count = fill_vertex_cells(&mut table, &mut vertex_cells, &vertex_ids);

    // build a quadric for each target cell
    let mut cell_quadrics = vec![Quadric::default(); cell_count];

    fill_cell_quadrics2(&mut cell_quadrics, &vertex_positions, &vertex_cells);

    // for each target cell, find the vertex with the minimal error
    let mut cell_remap = vec![INVALID_INDEX; cell_count];
    let mut cell_errors = vec![0.0; cell_count];

    fill_cell_remap(
        &mut cell_remap,
        &mut cell_errors,
        &vertex_cells,
        &cell_quadrics,
        &vertex_positions,
    );

    // copy results to the output
    assert!(cell_count <= target_vertex_count);
    destination[0..cell_count].copy_from_slice(&cell_remap);

    cell_count
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
    fn test_simplify_stuck() {
        let mut dst = vec![0; 16];

        // tetrahedron can't be simplified due to collapse error restrictions
        let vb1 = vb_from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let ib1 = [0, 1, 2, 0, 2, 3, 0, 3, 1, 2, 1, 3];

        assert_eq!(simplify(&mut dst, &ib1, &vb1, 6, 0.001), 12);

        // 5-vertex strip can't be simplified due to topology restriction since middle triangle has flipped winding
        let vb2 = vb_from_slice(&[
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.5, 1.0, 0.0,
        ]);
        let ib2 = [0, 1, 3, 3, 1, 4, 1, 2, 4]; // ok
        let ib3 = [0, 1, 3, 1, 3, 4, 1, 2, 4]; // flipped

        assert_eq!(simplify(&mut dst, &ib2, &vb2, 6, 0.001), 6);
        assert_eq!(simplify(&mut dst, &ib3, &vb2, 6, 0.001), 9);

        // 4-vertex quad with a locked corner can't be simplified due to border error-induced restriction
        let vb4 = vb_from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let ib4 = [0, 1, 3, 0, 3, 2];

        assert_eq!(simplify(&mut dst, &ib4, &vb4, 3, 0.001), 6);

        // 4-vertex quad with a locked corner can't be simplified due to border error-induced restriction
        let vb5 = vb_from_slice(&[
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        ]);
        let ib5 = [0, 1, 4, 0, 3, 2];

        assert_eq!(simplify(&mut dst, &ib5, &vb5, 3, 0.001), 6);
    }

    #[test]
    fn test_simplify_sloppy_stuck() {
        let mut dst = vec![0; 16];

        let vb = vb_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let ib = [0, 1, 2, 0, 1, 2];

        // simplifying down to 0 triangles results in 0 immediately
        assert_eq!(simplify_sloppy(&mut dst, &ib[0..3], &vb, 0), 0);

        // simplifying down to 2 triangles given that all triangles are degenerate results in 0 as well
        assert_eq!(simplify_sloppy(&mut dst, &ib, &vb, 6), 0);
    }

    #[test]
    fn test_simplify_points_stuck() {
        let mut dst = vec![0; 16];

        let vb = vb_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // simplifying down to 0 points results in 0 immediately
        assert_eq!(simplify_points(&mut dst, &vb, 0), 0);
    }
}
