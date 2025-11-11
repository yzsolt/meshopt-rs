//! Mesh and point cloud simplification

// This work is based on:
// Michael Garland and Paul S. Heckbert. Surface simplification using quadric error metrics. 1997
// Michael Garland. Quadric-based polygonal surface simplification. 1999
// Peter Lindstrom. Out-of-Core Simplification of Large Polygonal Models. 2000
// Matthias Teschner, Bruno Heidelberger, Matthias Mueller, Danat Pomeranets, Markus Gross. Optimized Spatial Hashing for Collision Detection of Deformable Objects. 2003
// Peter Van Sandt, Yannis Chronis, Jignesh M. Patel. Efficiently Searching In-Memory Sorted Arrays: Revenge of the Interpolation Search? 2019
// Hugues Hoppe. New Quadric Metric for Simplifying Meshes with Appearance Attributes. 1999

use bitflags::bitflags;

use crate::INVALID_INDEX;
use crate::Vector3;
use crate::hash::BuildNoopHasher;
use crate::util::zero_inverse;
#[cfg(feature = "experimental")]
use crate::vertex::{Vertex, calc_pos_extents};

use std::collections::{HashMap, hash_map::Entry};
use std::fmt::Debug;
use std::ops::AddAssign;

const MAX_ATTRIBUTES: usize = 16;

#[derive(Clone, Default)]
struct Edge {
    next: u32,
    prev: u32,
}

#[derive(Default)]
struct EdgeAdjacency {
    counts: Vec<u32>,
    offsets: Vec<u32>,
    data: Vec<Edge>,
}

fn prepare_edge_adjacency(adjacency: &mut EdgeAdjacency, index_count: usize, vertex_count: usize) {
    adjacency.counts = vec![0; vertex_count];
    adjacency.offsets = vec![0; vertex_count];
    adjacency.data = vec![Edge::default(); index_count];
}

fn update_edge_adjacency(adjacency: &mut EdgeAdjacency, indices: &[u32], remap: Option<&[u32]>) {
    let face_count = indices.len() / 3;

    // fill edge counts
    adjacency.counts.fill(0);

    for index in indices {
        let v = if let Some(r) = remap {
            r[*index as usize] as usize
        } else {
            *index as usize
        };

        adjacency.counts[v] += 1;
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
        let mut a = indices[i * 3 + 0];
        let mut b = indices[i * 3 + 1];
        let mut c = indices[i * 3 + 2];

        if let Some(r) = remap {
            a = r[a as usize];
            b = r[b as usize];
            c = r[c as usize];
        }

        adjacency.data[adjacency.offsets[a as usize] as usize].next = b;
        adjacency.data[adjacency.offsets[a as usize] as usize].prev = c;
        adjacency.offsets[a as usize] += 1;

        adjacency.data[adjacency.offsets[b as usize] as usize].next = c;
        adjacency.data[adjacency.offsets[b as usize] as usize].prev = a;
        adjacency.offsets[b as usize] += 1;

        adjacency.data[adjacency.offsets[c as usize] as usize].next = a;
        adjacency.data[adjacency.offsets[c as usize] as usize].prev = b;
        adjacency.offsets[c as usize] += 1;
    }

    // fix offsets that have been disturbed by the previous pass
    for (offset, count) in adjacency.offsets.iter_mut().zip(adjacency.counts.iter()) {
        assert!(*offset >= *count);

        *offset -= *count;
    }
}

mod hash {
    use std::hash::{Hash, Hasher};

    #[derive(Clone)]
    pub struct VertexPosition(pub [f32; 3]);

    impl VertexPosition {
        const BYTES: usize = std::mem::size_of::<Self>();

        fn as_bytes(&self) -> &[u8] {
            let bytes: &[u8; Self::BYTES] = unsafe { std::mem::transmute(&self.0) };
            bytes
        }
    }

    impl Hash for VertexPosition {
        fn hash<H: Hasher>(&self, state: &mut H) {
            let [x, y, z] = self.0;
            let [x, y, z] = [f32::to_bits(x), f32::to_bits(y), f32::to_bits(z)];

            // scramble bits to make sure that integer coordinates have entropy in lower bits
            let x = x ^ (x >> 17);
            let y = y ^ (y >> 17);
            let z = z ^ (z >> 17);

            // Optimized Spatial Hashing for Collision Detection of Deformable Objects
            state.write_u32((x.wrapping_mul(73856093)) ^ (y.wrapping_mul(19349663)) ^ (z.wrapping_mul(83492791)));
        }
    }

    impl PartialEq for VertexPosition {
        fn eq(&self, other: &Self) -> bool {
            self.as_bytes() == other.as_bytes()
        }
    }

    impl Eq for VertexPosition {}
}

fn build_position_remap<V, const ATTR_COUNT: usize>(remap: &mut [u32], wedge: &mut [u32], vertices: &[V])
where
    V: Vertex<ATTR_COUNT>,
{
    let mut table = HashMap::with_capacity_and_hasher(vertices.len(), BuildNoopHasher::default());

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
    #[allow(unused)]
    Complex, // none of the above; these vertices can move as long as all wedges move to the target vertex
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

    let edges = &adjacency.data[offset..offset + count];

    edges.iter().any(|d| d.next == b)
}

#[allow(clippy::too_many_arguments)]
fn classify_vertices(
    result: &mut [VertexKind],
    loop_: &mut [u32],
    loopback: &mut [u32],
    vertex_count: usize,
    adjacency: &EdgeAdjacency,
    remap: &[u32],
    wedge: &[u32],
    options: SimplificationOptions,
) {
    // incoming & outgoing open edges: `INVALID_INDEX` if no open edges, i if there are more than 1
    // note that this is the same data as required in loop[] arrays; loop[] data is only valid for border/seam
    // but here it's okay to fill the data out for other types of vertices as well
    let openinc = loopback;
    let openout = loop_;

    #[allow(clippy::needless_range_loop)]
    for vertex in 0..vertex_count {
        let offset = adjacency.offsets[vertex] as usize;
        let count = adjacency.counts[vertex] as usize;

        let edges = &adjacency.data[offset..offset + count];

        for edge in edges {
            let target = edge.next;

            if target as usize == vertex {
                // degenerate triangles have two distinct edges instead of three, and the self edge
                // is bi-directional by definition; this can break border/seam classification by "closing"
                // the open edge from another triangle and falsely marking the vertex as manifold
                // instead we mark the vertex as having >1 open edges which turns it into locked/complex
                openinc[vertex] = vertex as u32;
                openout[vertex] = vertex as u32;
            } else if !has_edge(adjacency, target, vertex as u32) {
                openinc[target as usize] = if openinc[target as usize] == INVALID_INDEX {
                    vertex as u32
                } else {
                    target
                };
                openout[vertex] = if openout[vertex] == INVALID_INDEX {
                    target
                } else {
                    vertex as u32
                };
            }
        }
    }

    #[cfg(feature = "trace")]
    let mut stats = [0usize; 4];

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

                    #[cfg(feature = "trace")]
                    {
                        stats[0] += 1;
                    }
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

                        #[cfg(feature = "trace")]
                        {
                            stats[1] += 1;
                        }
                    }
                } else {
                    result[i] = VertexKind::Locked;

                    #[cfg(feature = "trace")]
                    {
                        stats[2] += 1;
                    }
                }
            } else {
                // more than one vertex maps to this one; we don't have classification available
                result[i] = VertexKind::Locked;

                #[cfg(feature = "trace")]
                {
                    stats[3] += 1;
                }
            }
        } else {
            assert!(remap[i] < i as u32);

            result[i] = result[remap[i] as usize];
        }
    }

    if options.contains(SimplificationOptions::SimplifyLockBorder) {
        for r in &mut result[0..vertex_count] {
            if *r == VertexKind::Border {
                *r = VertexKind::Locked;
            }
        }
    }

    #[cfg(feature = "trace")]
    println!(
        "locked: many open edges {}, disconnected seam {}, many seam edges {}, many wedges {}",
        stats[0], stats[1], stats[2], stats[3]
    );
}

fn rescale_positions<V, const ATTR_COUNT: usize>(result: &mut [Vector3], vertices: &[V]) -> f32
where
    V: Vertex<ATTR_COUNT>,
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

    extent
}

fn rescale_attributes<V, const ATTR_COUNT: usize>(
    vertices: &[V],
    attribute_weights: &[f32; ATTR_COUNT],
) -> Vec<[f32; ATTR_COUNT]>
where
    V: Vertex<ATTR_COUNT>,
{
    let mut vertex_weighted_attrs = vec![[0f32; ATTR_COUNT]; vertices.len()];

    for (weighted_attrs, vertex) in vertex_weighted_attrs.iter_mut().zip(vertices.iter()) {
        for (weighted_attr, (attr, attr_weight)) in weighted_attrs
            .iter_mut()
            .zip(vertex.attrs().iter().zip(attribute_weights.iter()))
        {
            *weighted_attr = attr * attr_weight;
        }
    }

    vertex_weighted_attrs
}

union CollapseUnion {
    bidi: u32,
    error: f32,
    errorui: u32,
}

impl Debug for CollapseUnion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollapseUnion")
            .field("bidi/errorui", unsafe { &self.bidi })
            .field("error", unsafe { &self.error })
            .finish()
    }
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

#[derive(Clone, Default, Debug)]
struct Collapse {
    v0: u32,
    v1: u32,
    u: CollapseUnion,
}

#[derive(Clone, Copy, Default, Debug)]
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
    #[cfg(feature = "experimental")]
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

    fn from_attributes<const ATTR_COUNT: usize>(
        g: &mut [QuadricGrad; ATTR_COUNT],
        p0: &Vector3,
        p1: &Vector3,
        p2: &Vector3,
        va0: &[f32; ATTR_COUNT],
        va1: &[f32; ATTR_COUNT],
        va2: &[f32; ATTR_COUNT],
    ) -> Self {
        // for each attribute we want to encode the following function into the quadric:
        // (eval(pos) - attr)^2
        // where eval(pos) interpolates attribute across the triangle like so:
        // eval(pos) = pos.x * gx + pos.y * gy + pos.z * gz + gw
        // where gx/gy/gz/gw are gradients
        let p10 = *p1 - *p0;
        let p20 = *p2 - *p0;

        // weight is scaled linearly with edge length
        let normal = Vector3 {
            x: p10.y * p20.z - p10.z * p20.y,
            y: p10.z * p20.x - p10.x * p20.z,
            z: p10.x * p20.y - p10.y * p20.x,
        };
        let area = normal.length();
        let w = area.sqrt(); // TODO this needs more experimentation

        // we compute gradients using barycentric coordinates; barycentric coordinates can be computed as follows:
        // v = (d11 * d20 - d01 * d21) / denom
        // w = (d00 * d21 - d01 * d20) / denom
        // u = 1 - v - w
        // here v0, v1 are triangle edge vectors, v2 is a vector from point to triangle corner, and dij = dot(vi, vj)
        let v0 = &p10;
        let v1 = &p20;
        let d00 = v0.length_squared();
        let d01 = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
        let d11 = v1.length_squared();
        let denom = d00 * d11 - d01 * d01;
        let denomr = zero_inverse(denom);

        // precompute gradient factors
        // these are derived by directly computing derivative of eval(pos) = a0 * u + a1 * v + a2 * w and factoring out common factors that are shared between attributes
        let gx1 = (d11 * v0.x - d01 * v1.x) * denomr;
        let gx2 = (d00 * v1.x - d01 * v0.x) * denomr;
        let gy1 = (d11 * v0.y - d01 * v1.y) * denomr;
        let gy2 = (d00 * v1.y - d01 * v0.y) * denomr;
        let gz1 = (d11 * v0.z - d01 * v1.z) * denomr;
        let gz2 = (d00 * v1.z - d01 * v0.z) * denomr;

        let mut q = Quadric::default();
        q.w = w;

        for (k, gg) in g.iter_mut().enumerate() {
            let a0 = va0[k];
            let a1 = va1[k];
            let a2 = va2[k];

            // compute gradient of eval(pos) for x/y/z/w
            // the formulas below are obtained by directly computing derivative of eval(pos) = a0 * u + a1 * v + a2 * w
            let gx = gx1 * (a1 - a0) + gx2 * (a2 - a0);
            let gy = gy1 * (a1 - a0) + gy2 * (a2 - a0);
            let gz = gz1 * (a1 - a0) + gz2 * (a2 - a0);
            let gw = a0 - p0.x * gx - p0.y * gy - p0.z * gz;

            // quadric encodes (eval(pos)-attr)^2; this means that the resulting expansion needs to compute, for example, pos.x * pos.y * K
            // since quadrics already encode factors for pos.x * pos.y, we can accumulate almost everything in basic quadric fields
            q.a00 += w * (gx * gx);
            q.a11 += w * (gy * gy);
            q.a22 += w * (gz * gz);

            q.a10 += w * (gy * gx);
            q.a20 += w * (gz * gx);
            q.a21 += w * (gz * gy);

            q.b0 += w * (gx * gw);
            q.b1 += w * (gy * gw);
            q.b2 += w * (gz * gw);

            q.c += w * (gw * gw);

            // the only remaining sum components are ones that depend on attr; these will be addded during error evaluation, see quadricError
            gg.gx = w * gx;
            gg.gy = w * gy;
            gg.gz = w * gz;
            gg.gw = w * gw;
        }

        q
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

    pub fn error_grad<const ATTR_COUNT: usize>(
        &self,
        g: &[QuadricGrad; ATTR_COUNT],
        v: &Vector3,
        va: &[f32; ATTR_COUNT],
    ) -> f32 {
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

        // see quadricFromAttributes for general derivation; here we need to add the parts of (eval(pos) - attr)^2 that depend on attr
        for (a, gg) in va.iter().zip(g.iter()) {
            let g = v.x * gg.gx + v.y * gg.gy + v.z * gg.gz + gg.gw;

            r += a * a * self.w;
            r -= 2.0 * a * g;
        }

        // TODO: weight normalization is breaking attribute error somehow
        let s = 1.0; // q.w == zero_inverse(q.w);

        r.abs() * s
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct QuadricGrad {
    gx: f32,
    gy: f32,
    gz: f32,
    gw: f32,
}

impl AddAssign for QuadricGrad {
    fn add_assign(&mut self, other: Self) {
        self.gx += other.gx;
        self.gy += other.gy;
        self.gz += other.gz;
        self.gw += other.gw;
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

fn add_grads<const ATTR_COUNT: usize>(g: &mut [QuadricGrad; ATTR_COUNT], r: &[QuadricGrad; ATTR_COUNT]) {
    for (gg, rr) in g.iter_mut().zip(r.iter()) {
        *gg += *rr;
    }
}

fn fill_attribute_quadrics<const ATTR_COUNT: usize>(
    attribute_quadrics: &mut [Quadric],
    attribute_gradients: &mut [[QuadricGrad; ATTR_COUNT]],
    indices: &[u32],
    vertex_positions: &[Vector3],
    vertex_attributes: &[[f32; ATTR_COUNT]],
    remap: &[u32],
) {
    for i in indices.as_chunks::<3>().0 {
        let [i0, i1, i2] = i;
        let [i0, i1, i2] = [*i0 as usize, *i1 as usize, *i2 as usize];

        let mut g = [QuadricGrad::default(); ATTR_COUNT];
        let qa = Quadric::from_attributes(
            &mut g,
            &vertex_positions[i0],
            &vertex_positions[i1],
            &vertex_positions[i2],
            &vertex_attributes[i0],
            &vertex_attributes[i1],
            &vertex_attributes[i2],
        );

        // TODO: This blends together attribute weights across attribute discontinuities, which is probably not a great idea
        attribute_quadrics[remap[i0] as usize] += qa;
        attribute_quadrics[remap[i1] as usize] += qa;
        attribute_quadrics[remap[i2] as usize] += qa;

        add_grads(&mut attribute_gradients[remap[i0] as usize], &g);
        add_grads(&mut attribute_gradients[remap[i1] as usize], &g);
        add_grads(&mut attribute_gradients[remap[i2] as usize], &g);
    }
}

// does triangle ABC flip when C is replaced with D?
fn has_triangle_flip(a: &Vector3, b: &Vector3, c: &Vector3, d: &Vector3) -> bool {
    let eb = Vector3::new(b.x - a.x, b.y - a.y, b.z - a.z);
    let ec = Vector3::new(c.x - a.x, c.y - a.y, c.z - a.z);
    let ed = Vector3::new(d.x - a.x, d.y - a.y, d.z - a.z);

    let nbc = Vector3::new(
        eb.y * ec.z - eb.z * ec.y,
        eb.z * ec.x - eb.x * ec.z,
        eb.x * ec.y - eb.y * ec.x,
    );
    let nbd = Vector3::new(
        eb.y * ed.z - eb.z * ed.y,
        eb.z * ed.x - eb.x * ed.z,
        eb.x * ed.y - eb.y * ed.x,
    );

    nbc.x * nbd.x + nbc.y * nbd.y + nbc.z * nbd.z <= 0.0
}

fn has_triangle_flips(
    adjacency: &EdgeAdjacency,
    vertex_positions: &[Vector3],
    collapse_remap: &[u32],
    i0: usize,
    i1: usize,
) -> bool {
    assert_eq!(collapse_remap[i0] as usize, i0);
    assert_eq!(collapse_remap[i1] as usize, i1);

    let v0 = vertex_positions[i0];
    let v1 = vertex_positions[i1];

    let offset = adjacency.offsets[i0] as usize;
    let count = adjacency.counts[i0] as usize;
    let edges = &adjacency.data[offset..offset + count];

    for edge in edges {
        let a = collapse_remap[edge.next as usize] as usize;
        let b = collapse_remap[edge.prev as usize] as usize;

        // skip triangles that will get collapsed by i0->i1 collapse or already got collapsed previously
        if a == i1 || b == i1 || a == b {
            continue;
        }

        // early-out when at least one triangle flips due to a collapse
        if has_triangle_flip(&vertex_positions[a], &vertex_positions[b], &v0, &v1) {
            return true;
        }
    }

    false
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

fn rank_edge_collapses<const ATTR_COUNT: usize>(
    collapses: &mut [Collapse],
    vertex_positions: &[Vector3],
    vertex_attributes: &[[f32; ATTR_COUNT]],
    vertex_quadrics: &[Quadric],
    attribute_quadrics: &[Quadric],
    attribute_gradients: &[[QuadricGrad; ATTR_COUNT]],
    remap: &[u32],
) {
    for c in collapses {
        let i0 = c.v0;
        let i1 = c.v1;

        // most edges are bidirectional which means we need to evaluate errors for two collapses
        // to keep this code branchless we just use the same edge for unidirectional edges
        let j0 = unsafe { if c.u.bidi != 0 { i1 } else { i0 } };
        let j1 = unsafe { if c.u.bidi != 0 { i0 } else { i1 } };

        let ri0 = remap[i0 as usize] as usize;
        let rj0 = remap[j0 as usize] as usize;

        let qi = vertex_quadrics[ri0];
        let qj = vertex_quadrics[rj0];

        let mut ei = qi.error(&vertex_positions[i1 as usize]);
        let mut ej = qj.error(&vertex_positions[j1 as usize]);

        if ATTR_COUNT > 0 {
            let agi = attribute_quadrics[ri0];
            let agj = attribute_quadrics[rj0];

            ei += agi.error_grad(
                &attribute_gradients[ri0],
                &vertex_positions[i1 as usize],
                &vertex_attributes[i1 as usize],
            );
            ej += agj.error_grad(
                &attribute_gradients[rj0],
                &vertex_positions[j1 as usize],
                &vertex_attributes[j1 as usize],
            );
        }

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

    for h in &mut histogram {
        let count = *h;
        *h = histogram_sum;
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

#[allow(clippy::too_many_arguments)]
fn perform_edge_collapses<const ATTR_COUNT: usize>(
    collapse_remap: &mut [u32],
    collapse_locked: &mut [bool],
    vertex_quadrics: &mut [Quadric],
    attribute_quadrics: &mut [Quadric],
    attribute_gradients: &mut [[QuadricGrad; ATTR_COUNT]],
    collapses: &[Collapse],
    collapse_order: &[u32],
    remap: &[u32],
    wedge: &[u32],
    vertex_kind: &[VertexKind],
    vertex_positions: &[Vector3],
    adjacency: &EdgeAdjacency,
    triangle_collapse_goal: usize,
    error_limit: f32,
    result_error: &mut f32,
) -> usize {
    let mut edge_collapses = 0;
    let mut triangle_collapses = 0;

    // most collapses remove 2 triangles; use this to establish a bound on the pass in terms of error limit
    // note that edge_collapse_goal is an estimate; triangle_collapse_goal will be used to actually limit collapses
    let mut edge_collapse_goal = triangle_collapse_goal / 2;

    for order in collapse_order {
        let c = collapses[*order as usize].clone();

        let error = unsafe { c.u.error };

        if error > error_limit {
            break;
        }

        if triangle_collapses >= triangle_collapse_goal {
            break;
        }

        // we limit the error in each pass based on the error of optimal last collapse; since many collapses will be locked
        // as they will share vertices with other successfull collapses, we need to increase the acceptable error by some factor
        let error_goal = if edge_collapse_goal < collapses.len() {
            1.5 * unsafe { collapses[collapse_order[edge_collapse_goal] as usize].u.error }
        } else {
            f32::MAX
        };

        // on average, each collapse is expected to lock 6 other collapses; to avoid degenerate passes on meshes with odd
        // topology, we only abort if we got over 1/6 collapses accordingly.
        if unsafe { c.u.error } > error_goal && triangle_collapses > triangle_collapse_goal / 6 {
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

        if has_triangle_flips(adjacency, vertex_positions, collapse_remap, r0, r1) {
            // adjust collapse goal since this collapse is invalid and shouldn't factor into error goal
            edge_collapse_goal += 1;
            continue;
        }

        assert_eq!(collapse_remap[r0] as usize, r0);
        assert_eq!(collapse_remap[r1] as usize, r1);

        vertex_quadrics[r1] += vertex_quadrics[r0];

        if ATTR_COUNT > 0 {
            attribute_quadrics[r1] += attribute_quadrics[r0];

            let copy = attribute_gradients[r0].clone();
            add_grads(&mut attribute_gradients[r1], &copy);
        }

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

        *result_error = result_error.max(unsafe { c.u.error });
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

#[cfg(feature = "experimental")]
mod experimental {
    use super::*;
    use std::hash::{Hash, Hasher};

    #[derive(PartialEq, Eq, Clone, Copy, Default)]
    pub struct VertexId(pub u32);

    impl Hash for VertexId {
        fn hash<H: Hasher>(&self, state: &mut H) {
            let mut h = self.0;

            // MurmurHash2 finalizer
            h ^= h >> 13;
            h = h.wrapping_mul(0x5bd1e995);
            h ^= h >> 15;

            state.write_u32(h);
        }
    }

    pub fn compute_vertex_ids(vertex_ids: &mut [VertexId], vertex_positions: &[Vector3], grid_size: i32) {
        assert!((1..=1024).contains(&grid_size));
        let cell_scale = (grid_size - 1) as f32;

        for (pos, id) in vertex_positions.iter().zip(vertex_ids.iter_mut()) {
            let xi = (pos.x * cell_scale + 0.5) as i32;
            let yi = (pos.y * cell_scale + 0.5) as i32;
            let zi = (pos.z * cell_scale + 0.5) as i32;

            *id = VertexId(((xi << 20) | (yi << 10) | zi) as u32);
        }
    }

    pub fn count_triangles(vertex_ids: &[VertexId], indices: &[u32]) -> usize {
        let mut result = 0;

        for abc in indices.chunks_exact(3) {
            let id0 = vertex_ids[abc[0] as usize];
            let id1 = vertex_ids[abc[1] as usize];
            let id2 = vertex_ids[abc[2] as usize];

            result += ((id0 != id1) && (id0 != id2) && (id1 != id2)) as usize;
        }

        result
    }

    pub fn fill_vertex_cells(
        table: &mut HashMap<VertexId, u32, BuildNoopHasher>,
        vertex_cells: &mut [u32],
        vertex_ids: &[VertexId],
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

    pub fn count_vertex_cells(table: &mut HashMap<VertexId, u32, BuildNoopHasher>, vertex_ids: &[VertexId]) -> usize {
        table.clear();

        let mut result = 0;

        for id in vertex_ids {
            result += match table.entry(*id) {
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() = id.0;
                    0
                }
                Entry::Vacant(entry) => {
                    entry.insert(id.0);
                    1
                }
            };
        }

        result
    }

    pub fn fill_cell_quadrics(
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

    pub fn fill_cell_quadrics2(cell_quadrics: &mut [Quadric], vertex_positions: &[Vector3], vertex_cells: &[u32]) {
        for (c, v) in vertex_cells.iter().zip(vertex_positions.iter()) {
            let q = Quadric::from_point(v.x, v.y, v.z, 1.0);

            cell_quadrics[*c as usize] += q;
        }
    }

    pub fn fill_cell_remap(
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

    pub fn filter_triangles(
        destination: &mut [u32],
        tritable: &mut HashMap<hash::VertexPosition, u32, BuildNoopHasher>,
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

                let p = hash::VertexPosition([
                    f32::from_bits(abc[0]),
                    f32::from_bits(abc[1]),
                    f32::from_bits(abc[2]),
                ]);

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

    pub fn interpolate(y: f32, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
        // three point interpolation from "revenge of interpolation search" paper
        let num = (y1 - y) * (x1 - x2) * (x1 - x0) * (y2 - y0);
        let den = (y2 - y) * (x1 - x2) * (y0 - y1) + (y0 - y) * (x1 - x0) * (y1 - y2);
        x1 + num / den
    }
}

bitflags! {
    pub struct SimplificationOptions: u32 {
        /// Do not move vertices that are located on the topological border (vertices on triangle edges that don't have a paired triangle). Useful for simplifying portions of the larger mesh.
        const SimplifyLockBorder = 1 << 0;
    }
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
/// * `destination`: must contain enough space for the target index buffer, worst case is `indices.len()` elements (**not** `target_index_count`)!
/// * `target_error`: represents the error relative to mesh extents that can be tolerated, e.g. 0.01 = 1% deformation; value range [0..1]
/// * `result_error`: can be None; when it's not None, it will contain the resulting (relative) error after simplification
pub fn simplify<V>(
    destination: &mut [u32],
    indices: &[u32],
    vertices: &[V],
    target_index_count: usize,
    target_error: f32,
    options: SimplificationOptions,
    result_error: Option<&mut f32>,
) -> usize
where
    V: Vertex,
{
    simplify_with_attributes::<V, 0>(
        destination,
        indices,
        vertices,
        &[],
        target_index_count,
        target_error,
        options,
        result_error,
    )
}

/// Mesh simplifier with attribute metric
///
/// The algorithm enhances [`simplify`] by incorporating attribute values into the error metric used to prioritize simplification order; see [`simplify`] documentation for details.
/// Note that the number of attributes affects memory requirements and running time; this algorithm requires ~1.5x more memory and time compared to [`simplify`] when using 4 scalar attributes.
///
/// # Arguments
///
/// * `vertex_attributes`: should have attribute_count floats for each vertex
/// * `attribute_weights`: should have attribute_count floats in total; the weights determine relative priority of attributes between each other and wrt position. The recommended weight range is [1e-3..1e-1], assuming attribute data is in [0..1] range.
///
/// TODO `target_error`/`result_error` currently use combined distance+attribute error; this may change in the future
#[cfg(feature = "experimental")]
pub fn simplify_with_attributes<V, const ATTR_COUNT: usize>(
    destination: &mut [u32],
    indices: &[u32],
    vertices: &[V],
    attribute_weights: &[f32; ATTR_COUNT],
    target_index_count: usize,
    target_error: f32,
    options: SimplificationOptions,
    result_error: Option<&mut f32>,
) -> usize
where
    V: Vertex<ATTR_COUNT>,
{
    simplify_edge(
        destination,
        indices,
        vertices,
        attribute_weights,
        target_index_count,
        target_error,
        options,
        result_error,
    )
}

fn simplify_edge<V, const ATTR_COUNT: usize>(
    destination: &mut [u32],
    indices: &[u32],
    vertices: &[V],
    attribute_weights: &[f32; ATTR_COUNT],
    target_index_count: usize,
    target_error: f32,
    options: SimplificationOptions,
    result_error: Option<&mut f32>,
) -> usize
where
    V: Vertex<ATTR_COUNT>,
{
    const {
        assert!(ATTR_COUNT < MAX_ATTRIBUTES);
    }

    assert_eq!(indices.len() % 3, 0);
    assert!(target_index_count <= indices.len());

    let result = &mut destination[0..indices.len()];

    // build adjacency information
    let mut adjacency = EdgeAdjacency::default();
    prepare_edge_adjacency(&mut adjacency, indices.len(), vertices.len());
    update_edge_adjacency(&mut adjacency, indices, None);

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
        options,
    );

    #[cfg(feature = "trace")]
    {
        let mut unique_positions = 0;
        for (i, remapped) in remap.iter().enumerate().take(vertices.len()) {
            unique_positions += (*remapped as usize == i) as u32;
        }

        println!(
            "position remap: {} vertices => {unique_positions} positions",
            vertices.len(),
        );

        let mut kinds = [0_u32; KIND_COUNT];
        for i in 0..vertices.len() {
            kinds[vertex_kind[i].index()] += (remap[i] as usize == i) as u32;
        }

        println!(
            "kinds: manifold {}, border {}, seam {}, complex {}, locked {}",
            kinds[VertexKind::Manifold.index()],
            kinds[VertexKind::Border.index()],
            kinds[VertexKind::Seam.index()],
            kinds[VertexKind::Complex.index()],
            kinds[VertexKind::Locked.index()],
        );
    }

    let mut vertex_positions = vec![Vector3::default(); vertices.len()]; // TODO: spare init?
    rescale_positions(&mut vertex_positions, vertices);

    let vertex_attributes = if ATTR_COUNT > 0 {
        rescale_attributes(vertices, attribute_weights)
    } else {
        Vec::new()
    };

    let mut vertex_quadrics = vec![Quadric::default(); vertices.len()];

    let (mut attribute_quadrics, mut attribute_gradients) = if ATTR_COUNT > 0 {
        (
            vec![Quadric::default(); vertices.len()],
            vec![[QuadricGrad::default(); ATTR_COUNT]; vertices.len()],
        )
    } else {
        (Vec::new(), Vec::new())
    };

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

    if ATTR_COUNT > 0 {
        fill_attribute_quadrics::<ATTR_COUNT>(
            &mut attribute_quadrics,
            &mut attribute_gradients,
            indices,
            &vertex_positions,
            &vertex_attributes,
            &remap,
        );
    }

    result.copy_from_slice(indices);

    // TODO: skip init?
    let mut edge_collapses = vec![Collapse::default(); indices.len()];
    let mut collapse_order = vec![0u32; indices.len()];
    let mut collapse_remap = vec![0u32; vertices.len()];
    let mut collapse_locked = vec![false; vertices.len()];

    let mut result_count = indices.len();
    let mut result_error_max = 0.0;

    // `target_error` input is linear; we need to adjust it to match `Quadric::error` units
    let error_limit = target_error * target_error;

    #[cfg(feature = "trace")]
    let mut pass_count = 0;

    while result_count > target_index_count {
        // note: throughout the simplification process adjacency structure reflects welded topology for result-in-progress
        update_edge_adjacency(&mut adjacency, &result[0..result_count], Some(&remap));

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
            &vertex_attributes,
            &vertex_quadrics,
            &attribute_quadrics,
            &attribute_gradients,
            &remap,
        );

        sort_edge_collapses(&mut collapse_order, &edge_collapses[0..edge_collapse_count]);

        let triangle_collapse_goal = (result_count - target_index_count) / 3;

        for (i, r) in collapse_remap.iter_mut().enumerate() {
            *r = i as u32;
        }

        collapse_locked.fill(false);

        #[cfg(feature = "trace")]
        {
            println!("pass: {pass_count}");
            pass_count += 1;
        }

        let collapses = perform_edge_collapses(
            &mut collapse_remap,
            &mut collapse_locked,
            &mut vertex_quadrics,
            &mut attribute_quadrics,
            &mut attribute_gradients,
            &edge_collapses,
            &collapse_order,
            &remap,
            &wedge,
            &vertex_kind,
            &vertex_positions,
            &adjacency,
            triangle_collapse_goal,
            error_limit,
            &mut result_error_max,
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

    #[cfg(feature = "trace")]
    println!(
        "result: {result_count} triangles, error: {:e}; total {pass_count} passes",
        result_error_max.sqrt()
    );

    // result_error is quadratic; we need to remap it back to linear
    if let Some(result_error) = result_error {
        *result_error = result_error_max.sqrt();
    }

    result_count
}

/// Reduces the number of triangles in the mesh, sacrificing mesh appearance for simplification performance.
///
/// The algorithm doesn't preserve mesh topology but can stop short of the target goal based on target error.
///
/// Returns the number of indices after simplification, with destination containing new index data
/// The resulting index buffer references vertices from the original vertex buffer.
/// If the original vertex data isn't required, creating a compact vertex buffer using [optimize_vertex_fetch](crate::vertex::fetch::optimize_vertex_fetch) is recommended.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the target index buffer, worst case is `indices.len()` elements (**not** `target_index_count`)!
/// * `target_error`: represents the error relative to mesh extents that can be tolerated, e.g. 0.01 = 1% deformation; value range [0..1]
/// * `result_error`: can be None; when it's not None, it will contain the resulting (relative) error after simplification
#[cfg(feature = "experimental")]
pub fn simplify_sloppy<V>(
    destination: &mut [u32],
    indices: &[u32],
    vertices: &[V],
    target_index_count: usize,
    target_error: f32,
    result_error: Option<&mut f32>,
) -> usize
where
    V: Vertex,
{
    use experimental::*;

    assert_eq!(indices.len() % 3, 0);
    assert!(target_index_count <= indices.len());

    // we expect to get ~2 triangles/vertex in the output
    let target_cell_count = target_index_count / 6;

    let mut vertex_positions = vec![Vector3::default(); vertices.len()];
    rescale_positions(&mut vertex_positions, vertices);

    // find the optimal grid size using guided binary search
    #[cfg(feature = "trace")]
    {
        println!("source: {} vertices, {} triangles", vertices.len(), indices.len() / 3,);
        println!(
            "target: {target_cell_count} cells, {} triangles",
            target_index_count / 3,
        );
    }

    let mut vertex_ids = vec![VertexId::default(); vertices.len()];

    const INTERPOLATION_PASSES: i32 = 5;

    // invariant: # of triangles in min_grid <= target_count
    let mut min_grid = (1.0 / target_error.max(1e-3)) as i32;
    let mut max_grid: i32 = 1025;
    let mut min_triangles = 0;
    let mut max_triangles = indices.len() / 3;

    // when we're error-limited, we compute the triangle count for the min. size; this accelerates convergence and provides the correct answer when we can't use a larger grid
    if min_grid > 1 {
        compute_vertex_ids(&mut vertex_ids, &vertex_positions, min_grid);
        min_triangles = count_triangles(&vertex_ids, indices);
    }

    // instead of starting in the middle, let's guess as to what the answer might be! triangle count usually grows as a square of grid size...
    let mut next_grid_size = ((target_cell_count as f32).sqrt() + 0.5) as i32;

    for pass in 0..10 + INTERPOLATION_PASSES {
        if min_triangles >= target_index_count / 3 || max_grid - min_grid <= 1 {
            break;
        }

        // we clamp the prediction of the grid size to make sure that the search converges
        let mut grid_size = next_grid_size;
        grid_size = if grid_size <= min_grid {
            min_grid + 1
        } else if grid_size >= max_grid {
            max_grid - 1
        } else {
            grid_size
        };

        compute_vertex_ids(&mut vertex_ids, &vertex_positions, grid_size);
        let triangles = count_triangles(&vertex_ids, indices);

        #[cfg(feature = "trace")]
        println!(
            "pass {pass} ({}): grid size {grid_size}, triangles {triangles}, {}",
            match pass {
                0 => "guess",
                1..=INTERPOLATION_PASSES => "lerp",
                _ => "binary",
            },
            if triangles <= target_index_count / 3 {
                "under"
            } else {
                "over"
            }
        );

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

        // we start by using interpolation search - it usually converges faster
        // however, interpolation search has a worst case of O(N) so we switch to binary search after a few iterations which converges in O(logN)
        next_grid_size = if pass < INTERPOLATION_PASSES {
            (tip + 0.5) as i32
        } else {
            (min_grid + max_grid) / 2
        };
    }

    if min_triangles == 0 {
        if let Some(result_error) = result_error {
            *result_error = 1.0;
        }

        return 0;
    }

    // build vertex->cell association by mapping all vertices with the same quantized position to the same cell
    let mut table = HashMap::with_capacity_and_hasher(vertices.len(), BuildNoopHasher::default());

    let mut vertex_cells = vec![0; vertices.len()];

    compute_vertex_ids(&mut vertex_ids, &vertex_positions, min_grid);
    let cell_count = fill_vertex_cells(&mut table, &mut vertex_cells, &vertex_ids);

    // build a quadric for each target cell
    let mut cell_quadrics = vec![Quadric::default(); cell_count];

    fill_cell_quadrics(&mut cell_quadrics, indices, &vertex_positions, &vertex_cells);

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
    let mut tritable = HashMap::with_capacity_and_hasher(min_triangles, BuildNoopHasher::default());

    let write = filter_triangles(destination, &mut tritable, indices, &vertex_cells, &cell_remap);
    assert!(write <= target_index_count);

    #[cfg(feature = "trace")]
    println!(
        "result: {cell_count} cells, {} triangles ({min_triangles} unfiltered)",
        write / 3
    );

    if let Some(result_error) = result_error {
        *result_error = cell_errors.into_iter().reduce(f32::max).unwrap().sqrt();
    }

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
/// * `destination`: must contain enough space for the target index buffer (`target_vertex_count` elements)
#[cfg(feature = "experimental")]
pub fn simplify_points<V>(destination: &mut [u32], vertices: &[V], target_vertex_count: usize) -> usize
where
    V: Vertex,
{
    use experimental::*;

    assert!(target_vertex_count <= vertices.len());

    let target_cell_count = target_vertex_count;

    if target_cell_count == 0 {
        return 0;
    }

    let mut vertex_positions = vec![Vector3::default(); vertices.len()];
    rescale_positions(&mut vertex_positions, vertices);

    // find the optimal grid size using guided binary search
    #[cfg(feature = "trace")]
    {
        println!("source: {} vertices", vertices.len());
        println!("target: {target_cell_count} cells");
    }

    let mut vertex_ids = vec![VertexId::default(); vertices.len()];

    let mut table = HashMap::with_capacity_and_hasher(vertices.len(), BuildNoopHasher::default());

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
        } else if grid_size >= max_grid {
            max_grid - 1
        } else {
            grid_size
        };

        compute_vertex_ids(&mut vertex_ids, &vertex_positions, grid_size);
        let vertices = count_vertex_cells(&mut table, &vertex_ids);

        #[cfg(feature = "trace")]
        println!(
            "pass {pass} ({}): grid size {grid_size}, triangles {vertices}, {}",
            match pass {
                0 => "guess",
                1..=INTERPOLATION_PASSES => "lerp",
                _ => "binary",
            },
            if vertices <= target_vertex_count / 3 {
                "under"
            } else {
                "over"
            }
        );

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

    #[cfg(feature = "trace")]
    println!("result: {cell_count} cells");

    cell_count
}

/// Returns the error scaling factor used by the simplifier to convert between absolute and relative extents
///
/// Absolute error must be **divided** by the scaling factor before passing it to [simplify] as `target_error`.
/// Relative error returned by [simplify] via `result_error` must be **multiplied** by the scaling factor to get absolute error.
pub fn simplify_scale<V>(vertices: &[V]) -> f32
where
    V: Vertex,
{
    let (_minv, extent) = calc_pos_extents(vertices);

    extent
}

#[cfg(test)]
mod test {
    use super::*;

    struct TestVertex {
        x: f32,
        y: f32,
        z: f32,
    }

    impl Vertex for TestVertex {
        fn pos(&self) -> [f32; 3] {
            [self.x, self.y, self.z]
        }
    }

    fn vb_from_slice(slice: &[f32]) -> Vec<TestVertex> {
        slice
            .chunks_exact(3)
            .map(|v| TestVertex {
                x: v[0],
                y: v[1],
                z: v[2],
            })
            .collect()
    }

    #[test]
    fn test_simplify() {
        // 0
        // 1 2
        // 3 4 5
        #[rustfmt::skip]
        let ib = [
            0, 2, 1,
            1, 2, 3,
            3, 2, 4,
            2, 5, 4,
        ];

        #[rustfmt::skip]
        let vb = vb_from_slice(&[
            0.0, 4.0, 0.0,
            0.0, 1.0, 0.0,
            2.0, 2.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            4.0, 0.0, 0.0,
        ]);

        let expected = [0, 5, 3];

        let mut error = 1.0;
        let mut dst = vec![0; ib.len()];
        assert_eq!(
            simplify(
                &mut dst,
                &ib,
                &vb,
                3,
                1e-2,
                SimplificationOptions::empty(),
                Some(&mut error)
            ),
            3
        );
        assert_eq!(error, 0.0);
        assert_eq!(&dst[0..expected.len()], expected);
    }

    #[test]
    fn test_simplify_stuck() {
        let mut dst = vec![0; 16];

        // tetrahedron can't be simplified due to collapse error restrictions
        let vb1 = vb_from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let ib1 = [0, 1, 2, 0, 2, 3, 0, 3, 1, 2, 1, 3];

        assert_eq!(
            simplify(&mut dst, &ib1, &vb1, 6, 0.001, SimplificationOptions::empty(), None),
            12
        );

        // 5-vertex strip can't be simplified due to topology restriction since middle triangle has flipped winding
        let vb2 = vb_from_slice(&[
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.5, 1.0, 0.0,
        ]);
        let ib2 = [0, 1, 3, 3, 1, 4, 1, 2, 4]; // ok
        let ib3 = [0, 1, 3, 1, 3, 4, 1, 2, 4]; // flipped

        assert_eq!(
            simplify(&mut dst, &ib2, &vb2, 6, 0.001, SimplificationOptions::empty(), None),
            6
        );
        assert_eq!(
            simplify(&mut dst, &ib3, &vb2, 6, 0.001, SimplificationOptions::empty(), None),
            9
        );

        // 4-vertex quad with a locked corner can't be simplified due to border error-induced restriction
        let vb4 = vb_from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let ib4 = [0, 1, 3, 0, 3, 2];

        assert_eq!(
            simplify(&mut dst, &ib4, &vb4, 3, 0.001, SimplificationOptions::empty(), None),
            6
        );

        // 4-vertex quad with a locked corner can't be simplified due to border error-induced restriction
        let vb5 = vb_from_slice(&[
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        ]);
        let ib5 = [0, 1, 4, 0, 3, 2];

        assert_eq!(
            simplify(&mut dst, &ib5, &vb5, 3, 0.001, SimplificationOptions::empty(), None),
            6
        );
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn test_simplify_sloppy_stuck() {
        let mut dst = vec![0; 16];

        let vb = vb_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let ib = [0, 1, 2, 0, 1, 2];

        // simplifying down to 0 triangles results in 0 immediately
        assert_eq!(simplify_sloppy(&mut dst, &ib[0..3], &vb, 0, 0.0, None), 0);

        // simplifying down to 2 triangles given that all triangles are degenerate results in 0 as well
        assert_eq!(simplify_sloppy(&mut dst, &ib, &vb, 6, 0.0, None), 0);
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn test_simplify_points_stuck() {
        let mut dst = vec![0; 16];

        let vb = vb_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // simplifying down to 0 points results in 0 immediately
        assert_eq!(simplify_points(&mut dst, &vb, 0), 0);
    }

    #[test]
    fn test_simplify_flip() {
        // this mesh has been constructed by taking a tessellated irregular grid with a square cutout
        // and progressively collapsing edges until the only ones left violate border or flip constraints.
        // there is only one valid non-flip collapse, so we validate that we take it; when flips are allowed,
        // the wrong collapse is picked instead.
        #[rustfmt::skip]
        let vb = vb_from_slice(&[
            1.000000, 1.000000, -1.000000, 
            1.000000, 1.000000, 1.000000, 
            1.000000, -1.000000, 1.000000, 
            1.000000, -0.200000, -0.200000, 
            1.000000, 0.200000, -0.200000, 
            1.000000, -0.200000, 0.200000, 
            1.000000, 0.200000, 0.200000, 
            1.000000, 0.500000, -0.500000, 
            1.000000, -1.000000, 0.000000,
        ]);

        // the collapse we expect is 7 -> 0
        #[rustfmt::skip]
        let ib = [
            7, 4, 3, 
            1, 2, 5, 
            7, 1, 6, 
            7, 8, 0, // gets removed
            7, 6, 4, 
            8, 5, 2, 
            8, 7, 3, 
            8, 3, 5, 
            5, 6, 1, 
            7, 0, 1, // gets removed
        ];

        #[rustfmt::skip]
        let expected = [
            0, 4, 3, 
            1, 2, 5, 
            0, 1, 6, 
            0, 6, 4, 
            8, 5, 2, 
            8, 0, 3, 
            8, 3, 5, 
            5, 6, 1,
        ];

        let mut dst = vec![0; ib.len()];

        assert_eq!(
            simplify(&mut dst, &ib, &vb, 3, 1e-3, SimplificationOptions::empty(), None),
            expected.len()
        );
        assert_eq!(&dst[0..expected.len()], expected);
    }

    #[test]
    fn test_simplify_scale() {
        let vb = vb_from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);

        assert_eq!(simplify_scale(&vb), 3.0);
    }

    #[test]
    fn test_simplify_degenerate() {
        #[rustfmt::skip]
        let vb = vb_from_slice(&[
            0.000000, 0.000000, 0.000000,
            0.000000, 1.000000, 0.000000,
            0.000000, 2.000000, 0.000000,
            1.000000, 0.000000, 0.000000,
            2.000000, 0.000000, 0.000000,
            1.000000, 1.000000, 0.000000, 
        ]);

        // 0 1 2
        // 3 5
        // 4

        #[rustfmt::skip]
        let ib = [
            0, 1, 3,
            3, 1, 5,
            1, 2, 5,
            3, 5, 4,
            1, 0, 1, // these two degenerate triangles create a fake reverse edge
            0, 3, 0, // which breaks border classification
        ];

        #[rustfmt::skip]
        let expected = [
            0, 1, 4,
		    4, 1, 2,
        ];

        let mut dst = vec![0; ib.len()];

        assert_eq!(
            simplify(&mut dst, &ib, &vb, 3, 1e-3, SimplificationOptions::empty(), None),
            expected.len()
        );
        assert_eq!(&dst[0..expected.len()], expected);
    }

    #[test]
    fn test_simplify_lock_border() {
        #[rustfmt::skip]
        let vb = vb_from_slice(&[
            0.000000, 0.000000, 0.000000,
            0.000000, 1.000000, 0.000000,
            0.000000, 2.000000, 0.000000,
            1.000000, 0.000000, 0.000000,
            1.000000, 1.000000, 0.000000,
            1.000000, 2.000000, 0.000000,
            2.000000, 0.000000, 0.000000,
            2.000000, 1.000000, 0.000000,
            2.000000, 2.000000, 0.000000,
        ]);

        // 0 1 2
        // 3 4 5
        // 6 7 8

        #[rustfmt::skip]
        let ib = [
            0, 1, 3,
            3, 1, 4,
            1, 2, 4,
            4, 2, 5,
            3, 4, 6,
            6, 4, 7,
            4, 5, 7,
            7, 5, 8,
        ];

        #[rustfmt::skip]
        let expected = [
            0, 1, 3,
            1, 2, 3,
            3, 2, 5,
            6, 3, 7,
            3, 5, 7,
            7, 5, 8,
        ];

        let mut dst = vec![0; ib.len()];

        assert_eq!(
            simplify(
                &mut dst,
                &ib,
                &vb,
                3,
                1e-3,
                SimplificationOptions::SimplifyLockBorder,
                None
            ),
            expected.len()
        );
        assert_eq!(&dst[0..expected.len()], expected);
    }

    #[test]
    fn test_simplify_attr() {
        #[derive(Default, Clone, Copy)]
        struct TestVertexWithAttributes([[f32; 3]; 2]);

        impl Vertex<3> for TestVertexWithAttributes {
            fn pos(&self) -> [f32; 3] {
                self.0[0]
            }

            fn attrs(&self) -> [f32; 3] {
                self.0[1]
            }
        }

        let mut vb = [TestVertexWithAttributes::default(); 8 * 3];

        for y in 0..8 {
            // first four rows are a blue gradient, next four rows are a yellow gradient
            let r = if y < 4 { 0.8 + y as f32 * 0.05 } else { 0.0 };
            let g = if y < 4 { 0.8 + y as f32 * 0.05 } else { 0.0 };
            let b = if y < 4 { 0.0 } else { 0.8 + (7 - y) as f32 * 0.05 };

            for x in 0..3 {
                let v = &mut vb[y * 3 + x].0;
                v[0][0] = x as f32;
                v[0][1] = y as f32;
                v[0][2] = 0.0;
                v[1][0] = r;
                v[1][1] = g;
                v[1][2] = b;
            }
        }

        let mut ib = [[0u32; 6]; 7 * 2];

        for y in 0..7 {
            for x in 0..2 {
                ib[y * 2 + x][0] = ((y + 0) * 3 + (x + 0)) as u32;
                ib[y * 2 + x][1] = ((y + 0) * 3 + (x + 1)) as u32;
                ib[y * 2 + x][2] = ((y + 1) * 3 + (x + 0)) as u32;
                ib[y * 2 + x][3] = ((y + 1) * 3 + (x + 0)) as u32;
                ib[y * 2 + x][4] = ((y + 0) * 3 + (x + 1)) as u32;
                ib[y * 2 + x][5] = ((y + 1) * 3 + (x + 1)) as u32;
            }
        }

        let ib = ib.iter().flatten().copied().collect::<Vec<_>>();

        let attr_weights = [0.01, 0.01, 0.01];

        let expected = [
            [0, 2, 9, 9, 2, 11],
            [9, 11, 12, 12, 11, 14],
            [12, 14, 21, 21, 14, 23],
        ];

        let mut actual = vec![0u32; ib.len()];

        assert_eq!(
            simplify_with_attributes::<TestVertexWithAttributes, 3>(
                &mut actual,
                &ib,
                &vb,
                &attr_weights,
                6 * 3,
                1e-2,
                SimplificationOptions::empty(),
                None
            ),
            18
        );
        assert!(actual.iter().zip(expected.iter().flatten()).all(|(a, b)| a == b));
    }
}
