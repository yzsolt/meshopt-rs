use meshopt_rs::index::buffer::*;
use meshopt_rs::index::generator::*;
use meshopt_rs::index::*;
use meshopt_rs::overdraw::*;
use meshopt_rs::quantize::*;
use meshopt_rs::stripify::*;
use meshopt_rs::vertex::buffer::*;
use meshopt_rs::vertex::cache::*;
use meshopt_rs::vertex::fetch::*;
use meshopt_rs::vertex::*;
use meshopt_rs::{INVALID_INDEX, Stream};

#[cfg(feature = "experimental")]
use meshopt_rs::{cluster::*, index::sequence::*, simplify::*, spatial_order::*};

use std::env;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
struct Vertex {
    p: [f32; 3],
    n: [f32; 3],
    t: [f32; 2],
}

impl Vertex {
    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts((self as *const Self) as *const u8, std::mem::size_of::<Self>()) }
    }
}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(self.as_bytes());
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for Vertex {}

impl Position for Vertex {
    fn pos(&self) -> [f32; 3] {
        self.p
    }
}

#[derive(Clone, Default)]
struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Mesh {
    pub fn load<P>(path: P) -> Result<Mesh, tobj::LoadError>
    where
        P: AsRef<Path> + Clone + Debug,
    {
        let start = Instant::now();

        let (models, _materials) = tobj::load_obj(
            path.clone(),
            &tobj::LoadOptions {
                triangulate: true,
                ..Default::default()
            },
        )?;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for (_, model) in models.iter().enumerate() {
            let mesh = &model.mesh;
            assert!(mesh.positions.len() % 3 == 0);

            vertices.reserve(mesh.indices.len());
            indices.extend_from_slice(&mesh.indices);

            for i in 0..mesh.indices.len() {
                let mut vertex = Vertex::default();

                let pi = mesh.indices[i] as usize;
                let ni = mesh.normal_indices[i] as usize;
                let ti = mesh.texcoord_indices[i] as usize;

                vertex.p.copy_from_slice(&mesh.positions[3 * pi..3 * (pi + 1)]);

                if !mesh.normals.is_empty() {
                    vertex.n.copy_from_slice(&mesh.normals[3 * ni..3 * (ni + 1)]);
                }

                if !mesh.texcoords.is_empty() {
                    vertex.t.copy_from_slice(&mesh.texcoords[2 * ti..2 * (ti + 1)]);
                }

                vertices.push(vertex);
            }
        }

        let read = start.elapsed();
        let start = Instant::now();

        let total_indices = indices.len();
        let mut remap = vec![0; total_indices];

        let mut result = Mesh::default();

        let total_vertices = generate_vertex_remap(&mut remap, None, &Stream::from_slice(&vertices));

        result.indices = remap;

        result.vertices.resize(total_vertices, Vertex::default());
        remap_vertex_buffer(&mut result.vertices, &vertices, &result.indices);

        let indexed = start.elapsed();

        println!(
            "# {:?}: {} vertices, {} triangles; read in {:.2} msec; indexed in {:.2} msec",
            path,
            vertices.len(),
            total_indices / 3,
            read.as_micros() as f64 / 1000.0,
            indexed.as_micros() as f64 / 1000.0,
        );

        Ok(result)
    }

    pub fn is_valid(&self) -> bool {
        if self.indices.len() % 3 != 0 {
            return false;
        }

        self.indices.iter().all(|i| (*i as usize) < self.vertices.len())
    }

    fn rotate_triangle(t: &mut [Vertex; 3]) -> bool {
        use std::cmp::Ordering;

        let c01 = t[0].as_bytes().cmp(t[1].as_bytes());
        let c02 = t[0].as_bytes().cmp(t[2].as_bytes());
        let c12 = t[1].as_bytes().cmp(t[2].as_bytes());

        if c12 == Ordering::Less && c01 == Ordering::Greater {
            // 1 is minimum, rotate 012 => 120
            t.rotate_left(1);
        } else if c02 == Ordering::Greater && c12 == Ordering::Greater {
            // 2 is minimum, rotate 012 => 201
            t.rotate_left(2);
        }

        c01 != Ordering::Equal && c02 != Ordering::Equal && c12 != Ordering::Equal
    }

    fn hash_range(bytes: &[u8]) -> u32 {
        // MurmurHash2
        const M: u32 = 0x5bd1e995;
        const R: i32 = 24;

        let mut h: u32 = 0;

        for k4 in bytes.chunks_exact(4) {
            let mut k = u32::from_ne_bytes(k4.try_into().unwrap());

            k = k.wrapping_mul(M);
            k ^= k >> R;
            k = k.wrapping_mul(M);

            h = h.wrapping_mul(M);
            h ^= k;
        }

        h
    }

    pub fn hash(&self) -> u32 {
        let triangle_count = self.indices.len() / 3;

        let mut h1 = 0;
        let mut h2: u32 = 0;

        for i in 0..triangle_count {
            let mut v = [
                self.vertices[self.indices[i * 3 + 0] as usize],
                self.vertices[self.indices[i * 3 + 1] as usize],
                self.vertices[self.indices[i * 3 + 2] as usize],
            ];

            // skip degenerate triangles since some algorithms don't preserve them
            if Self::rotate_triangle(&mut v) {
                let data = unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of::<Vertex>() * v.len())
                };

                let hash = Self::hash_range(&data);

                h1 ^= hash;
                h2 = h2.wrapping_add(hash);
            }
        }

        h1 = h1.wrapping_mul(0x5bd1e995).wrapping_add(h2);

        h1
    }
}

const CACHE_SIZE: usize = 16;

fn opt_random_shuffle(mesh: &mut Mesh) {
    let triangle_count = mesh.indices.len() / 3;

    let indices = &mut mesh.indices;

    let mut rng: u32 = 0;

    for i in (0..triangle_count).rev() {
        // Fisher-Yates shuffle
        let j = rng as usize % (i + 1);

        indices.swap(3 * j + 0, 3 * i + 0);
        indices.swap(3 * j + 1, 3 * i + 1);
        indices.swap(3 * j + 2, 3 * i + 2);

        // LCG RNG, constants from Numerical Recipes
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
    }
}

fn opt_cache(mesh: &mut Mesh) {
    let indices_copy = mesh.indices.clone();
    optimize_vertex_cache(&mut mesh.indices, &indices_copy, mesh.vertices.len());
}

fn opt_cache_fifo(mesh: &mut Mesh) {
    let indices_copy = mesh.indices.clone();
    optimize_vertex_cache_fifo(&mut mesh.indices, &indices_copy, mesh.vertices.len(), CACHE_SIZE as u32);
}

fn opt_cache_strip(mesh: &mut Mesh) {
    let indices_copy = mesh.indices.clone();
    optimize_vertex_cache_strip(&mut mesh.indices, &indices_copy, mesh.vertices.len());
}

fn opt_overdraw(mesh: &mut Mesh) {
    // use worst-case ACMR threshold so that overdraw optimizer can sort *all* triangles
    // warning: this significantly deteriorates the vertex cache efficiency so it is not advised; look at `opt_complete` for the recommended method
    const TRESHOLD: f32 = 3.0;

    let mut result = vec![0; mesh.indices.len()];
    optimize_overdraw(&mut result, &mesh.indices, &mesh.vertices, TRESHOLD);
    mesh.indices = result;
}

fn opt_fetch(mesh: &mut Mesh) {
    let vertices_copy = mesh.vertices.clone();
    optimize_vertex_fetch(&mut mesh.vertices, &mut mesh.indices, &vertices_copy);
}

fn opt_fetch_remap(mesh: &mut Mesh) {
    // this produces results equivalent to `opt_fetch`, but can be used to remap multiple vertex streams
    let mut remap = vec![0; mesh.vertices.len()];
    optimize_vertex_fetch_remap(&mut remap, &mesh.indices);

    remap_index_buffer(&mut mesh.indices, &remap);

    let vertices_copy = mesh.vertices.clone();
    remap_vertex_buffer(&mut mesh.vertices, &vertices_copy, &remap);
}

fn opt_complete(mesh: &mut Mesh) {
    // vertex cache optimization should go first as it provides starting order for overdraw
    let indices_copy = mesh.indices.clone();
    optimize_vertex_cache(&mut mesh.indices, &indices_copy, mesh.vertices.len());

    // reorder indices for overdraw, balancing overdraw and vertex cache efficiency
    const TRESHOLD: f32 = 1.01; // allow up to 1% worse ACMR to get more reordering opportunities for overdraw
    let indices_copy = mesh.indices.clone();
    optimize_overdraw(&mut mesh.indices, &indices_copy, &mesh.vertices, TRESHOLD);

    // vertex fetch optimization should go last as it depends on the final index order
    let vertices_copy = mesh.vertices.clone();
    optimize_vertex_fetch(&mut mesh.vertices, &mut mesh.indices, &vertices_copy);
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(C)]
struct PackedVertex {
    p: [u16; 4], // padded to 4b boundary
    n: [i8; 4],
    t: [u16; 2],
}

fn pack_mesh(pv: &mut [PackedVertex], vertices: &[Vertex]) {
    for i in 0..vertices.len() {
        let vi = vertices[i];
        let pvi = &mut pv[i];

        pvi.p[0] = quantize_half(vi.p[0]);
        pvi.p[1] = quantize_half(vi.p[1]);
        pvi.p[2] = quantize_half(vi.p[2]);
        pvi.p[3] = 0;

        pvi.n[0] = quantize_snorm(vi.n[0], 8) as i8;
        pvi.n[1] = quantize_snorm(vi.n[1], 8) as i8;
        pvi.n[2] = quantize_snorm(vi.n[2], 8) as i8;
        pvi.n[3] = 0;

        pvi.t[0] = quantize_half(vi.t[0]);
        pvi.t[1] = quantize_half(vi.t[1]);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(C)]
struct PackedVertexOct {
    p: [u16; 3],
    n: [i8; 2], // octahedron encoded normal, aliases .pw
    t: [u16; 2],
}

fn pack_mesh_oct(pv: &mut [PackedVertexOct], vertices: &[Vertex]) {
    for i in 0..vertices.len() {
        let vi = vertices[i];
        let pvi = &mut pv[i];

        pvi.p[0] = quantize_half(vi.p[0]);
        pvi.p[1] = quantize_half(vi.p[1]);
        pvi.p[2] = quantize_half(vi.p[2]);

        let nsum = vi.n[0].abs() + vi.n[1].abs() + vi.n[2].abs();
        let nx = vi.n[0] / nsum;
        let ny = vi.n[1] / nsum;
        let nz = vi.n[2];

        let nu = if nz >= 0.0 {
            nx
        } else {
            (1.0 - ny.abs()) * if nx >= 0.0 { 1.0 } else { -1.0 }
        };
        let nv = if nz >= 0.0 {
            ny
        } else {
            (1.0 - nx.abs()) * if ny >= 0.0 { 1.0 } else { -1.0 }
        };

        pvi.n[0] = quantize_snorm(nu, 8) as i8;
        pvi.n[1] = quantize_snorm(nv, 8) as i8;

        pvi.t[0] = quantize_half(vi.t[0]);
        pvi.t[1] = quantize_half(vi.t[1]);
    }
}

fn optimize<F>(mesh: &Mesh, name: &str, f: F)
where
    F: FnOnce(&mut Mesh),
{
    let mut copy = mesh.clone();
    let start = Instant::now();
    f(&mut copy);
    let duration = start.elapsed();

    assert!(copy.is_valid());
    assert_eq!(mesh.hash(), copy.hash());

    let vcs = analyze_vertex_cache(&copy.indices, copy.vertices.len(), CACHE_SIZE, 0, 0);
    let vfs = analyze_vertex_fetch(&copy.indices, copy.vertices.len(), std::mem::size_of::<Vertex>());
    let os = analyze_overdraw(&copy.indices, &copy.vertices);

    let vcs_nv = analyze_vertex_cache(&copy.indices, copy.vertices.len(), 32, 32, 32);
    let vcs_amd = analyze_vertex_cache(&copy.indices, copy.vertices.len(), 14, 64, 128);
    let vcs_intel = analyze_vertex_cache(&copy.indices, copy.vertices.len(), 128, 0, 0);

    println!(
        "{:9}: ACMR {:.6} ATVR {:.6} (NV {:.6} AMD {:.6} Intel {:.6}) Overfetch {:.6} Overdraw {:.6} in {:.2} msec",
        name,
        vcs.acmr,
        vcs.atvr,
        vcs_nv.atvr,
        vcs_amd.atvr,
        vcs_intel.atvr,
        vfs.overfetch,
        os.overdraw,
        duration.as_micros() as f64 / 1000.0
    );
}

fn compress_data<T>(data: &[T]) -> usize {
    use miniz_oxide::deflate::CompressionLevel;
    use miniz_oxide::deflate::core::*;

    let flags = create_comp_flags_from_zip_params(
        CompressionLevel::DefaultLevel as i32,
        15,
        CompressionStrategy::Default as i32,
    );
    let mut compressor = CompressorOxide::new(flags);

    let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of::<T>() * data.len()) };

    // Taken from miniz.c as miniz_oxide has no equivalent function
    let bound = (128 + (src.len() * 110) / 100).max(128 + src.len() + ((src.len() / (31 * 1024)) + 1) * 5);

    let mut dst = vec![0; bound];

    let (state, _cursor_src, cursor_dst) = compress(&mut compressor, src, &mut dst, TDEFLFlush::Finish);

    assert_eq!(state, TDEFLStatus::Done);

    cursor_dst
}

fn encode_index(mesh: &Mesh, desc: char) {
    // allocate result outside of the timing loop to exclude memset() from decode timing
    let mut result = vec![0; mesh.indices.len()];

    let start = Instant::now();

    let mut buffer = vec![0; encode_index_buffer_bound(mesh.indices.len(), mesh.vertices.len())];
    let size = encode_index_buffer(&mut buffer, &mesh.indices, IndexEncodingVersion::default()).unwrap();
    buffer.resize(size, 0);

    let encode = start.elapsed();
    let start = Instant::now();

    let res = decode_index_buffer(&mut result, &buffer);
    assert!(res.is_ok());

    let decode = start.elapsed();

    let csize = compress_data(&buffer);

    for (i, triangle) in mesh.indices.chunks_exact(3).enumerate() {
        let mut r = [0; 3];
        r.copy_from_slice(&result[i * 3..i * 3 + 3]);

        let mut r1 = r;
        r1.rotate_left(1);

        let mut r2 = r;
        r2.rotate_left(2);

        assert!(triangle == r || triangle == r1 || triangle == r2);
    }

    println!(
        "IdxCodec{}: {:.1} bits/triangle (post-deflate {:.1} bits/triangle); encode {:.2} msec, decode {:.2} msec ({:.2} GB/s)",
        desc,
        (buffer.len() * 8) as f64 / (mesh.indices.len() / 3) as f64,
        (csize * 8) as f64 / (mesh.indices.len() / 3) as f64,
        encode.as_micros() as f64 / 1000.0,
        decode.as_micros() as f64 / 1000.0,
        ((result.len() * 4) as f64 / (1 << 30) as f64) / decode.as_secs_f64(),
    );
}

#[cfg(feature = "experimental")]
fn encode_index_sequence1(data: &[u32], vertex_count: usize, desc: char) {
    // allocate result outside of the timing loop to exclude memset() from decode timing
    let mut result = vec![0; data.len()];

    let start = Instant::now();

    let mut buffer = vec![0; encode_index_sequence_bound(data.len(), vertex_count)];
    let size = encode_index_sequence(&mut buffer, &data, IndexEncodingVersion::default());
    buffer.resize(size, 0);

    let encode = start.elapsed();
    let start = Instant::now();

    let res = decode_index_sequence(&mut result, &buffer);
    assert!(res.is_ok());

    let decode = start.elapsed();

    let csize = compress_data(&buffer);

    assert_eq!(data, result);

    println!(
        "IdxCodec{}: {:.1} bits/index (post-deflate {:.1} bits/index); encode {:.2} msec, decode {:.2} msec ({:.2} GB/s)",
        desc,
        (buffer.len() * 8) as f64 / data.len() as f64,
        (csize * 8) as f64 / data.len() as f64,
        encode.as_micros() as f64 / 1000.0,
        decode.as_micros() as f64 / 1000.0,
        ((result.len() * 4) as f64 / (1 << 30) as f64) / decode.as_secs_f64(),
    );
}

fn pack_vertex(mesh: &Mesh) {
    let mut pv = vec![PackedVertex::default(); mesh.vertices.len()];
    pack_mesh(&mut pv, &mesh.vertices);

    let csize = compress_data(&pv);

    println!(
        "VtxPack  : {:.1} bits/vertex (post-deflate {:.1} bits/vertex)",
        (pv.len() * std::mem::size_of::<PackedVertex>() * 8) as f64 / mesh.vertices.len() as f64,
        (csize * 8) as f64 / mesh.vertices.len() as f64,
    );
}

fn encode_vertex(mesh: &Mesh) {
    let mut pv = vec![PackedVertex::default(); mesh.vertices.len()];
    pack_mesh(&mut pv, &mesh.vertices);

    encode_vertex_internal(&pv, ' ');
}

fn encode_vertex_oct(mesh: &Mesh) {
    let mut pv = vec![PackedVertexOct::default(); mesh.vertices.len()];
    pack_mesh_oct(&mut pv, &mesh.vertices);

    encode_vertex_internal(&pv, 'O');
}

fn encode_vertex_internal<PV>(pv: &[PV], pvn: char)
where
    PV: Clone + Debug + Default + PartialEq,
{
    // allocate result outside of the timing loop to exclude memset() from decode timing
    let mut result = vec![PV::default(); pv.len()];

    let start = Instant::now();

    let mut vbuf = vec![0; encode_vertex_buffer_bound(pv.len(), std::mem::size_of::<PV>())];
    let vb_size = encode_vertex_buffer(&mut vbuf, &pv, VertexEncodingVersion::default()).unwrap();
    vbuf.resize(vb_size, 0);

    let encode = start.elapsed();
    let start = Instant::now();

    decode_vertex_buffer(&mut result, &vbuf).unwrap();

    let decode = start.elapsed();

    assert_eq!(pv, result);

    let csize = compress_data(&vbuf);

    println!(
        "VtxCodec{:1}: {:.1} bits/vertex (post-deflate {:.1} bits/vertex); encode {:2} msec, decode {:2} msec ({:2} GB/s)",
        pvn,
        (vbuf.len() * 8) as f64 / pv.len() as f64,
        (csize * 8) as f64 / pv.len() as f64,
        encode.as_micros() as f64 / 1000.0,
        decode.as_micros() as f64 / 1000.0,
        ((result.len() * std::mem::size_of::<PV>()) as f64 / (1 << 30) as f64) / decode.as_secs_f64(),
    );
}

#[cfg(feature = "experimental")]
fn simplify_mesh(mesh: &Mesh) {
    let threshold = 0.2;

    let mut lod = Mesh::default();

    let start = Instant::now();

    let target_index_count = (mesh.indices.len() as f32 * threshold) as usize;
    let target_error = 0.01;
    let mut result_error = 0.0;

    lod.indices.resize(mesh.indices.len(), 0); // note: simplify needs space for index_count elements in the destination array, not `target_index_count`
    let size = simplify(
        &mut lod.indices,
        &mesh.indices,
        &mesh.vertices,
        target_index_count,
        target_error,
        Some(&mut result_error),
    );
    lod.indices.resize(size, 0);

    let size = if lod.indices.len() < mesh.vertices.len() {
        lod.indices.len()
    } else {
        mesh.vertices.len()
    };
    lod.vertices.resize(size, Vertex::default()); // note: this is just to reduce the cost of relen()
    let size = optimize_vertex_fetch(&mut lod.vertices, &mut lod.indices, &mesh.vertices);
    lod.vertices.resize(size, Vertex::default());

    let duration = start.elapsed();

    println!(
        "{:9}: {} triangles => {} triangles  ({:.2}% deviation) in {:.2} msec",
        "Simplify",
        mesh.indices.len() / 3,
        lod.indices.len() / 3,
        result_error * 100.0,
        duration.as_micros() as f64 / 1000.0
    );
}

#[cfg(feature = "experimental")]
fn simplify_mesh_sloppy(mesh: &Mesh, threshold: f32) {
    let mut lod = Mesh::default();

    let start = Instant::now();

    let target_index_count = (mesh.indices.len() as f32 * threshold) as usize;
    let target_error = 1e-1;
    let mut result_error = 0.0;

    lod.indices.resize(target_index_count, Default::default()); // note: simplify needs space for `index_count` elements in the destination array, not `target_index_count`
    let size = simplify_sloppy(
        &mut lod.indices,
        &mesh.indices,
        &mesh.vertices,
        target_index_count,
        target_error,
        Some(&mut result_error),
    );
    lod.indices.resize(size, Default::default());

    lod.vertices.resize(
        if lod.indices.len() < mesh.vertices.len() {
            lod.indices.len()
        } else {
            mesh.vertices.len()
        },
        Default::default(),
    ); // note: this is just to reduce the cost of `resize()`
    let size = optimize_vertex_fetch(&mut lod.vertices, &mut lod.indices, &mesh.vertices);
    lod.vertices.resize(size, Default::default());

    let duration = start.elapsed();

    println!(
        "{:9}: {} triangles => {} triangles ({:.2}% deviation) in {:.2} msec",
        "SimplifyS",
        mesh.indices.len() / 3,
        lod.indices.len() / 3,
        result_error * 100.0,
        duration.as_micros() as f64 / 1000.0
    );
}

#[cfg(feature = "experimental")]
fn simplify_mesh_points(mesh: &Mesh, threshold: f32) {
    let start = Instant::now();

    let target_vertex_count = (mesh.vertices.len() as f32 * threshold) as usize;

    let mut indices = vec![0; target_vertex_count];
    let size = simplify_points(&mut indices, &mesh.vertices, target_vertex_count);
    indices.resize(size, Default::default());

    let duration = start.elapsed();

    println!(
        "{:9}: {} points => {} points in {:.2} msec",
        "SimplifyP",
        mesh.vertices.len(),
        indices.len(),
        duration.as_micros() as f64 / 1000.0
    );
}

#[cfg(feature = "experimental")]
fn simplify_mesh_complete(mesh: &Mesh) {
    const LOD_COUNT: usize = 5;

    let start = Instant::now();

    // generate 4 LOD levels (1-4), with each subsequent LOD using 70% triangles
    // note that each LOD uses the same (shared) vertex buffer
    let mut lods = vec![Vec::default(); LOD_COUNT];

    lods[0] = mesh.indices.clone();

    for i in 1..LOD_COUNT {
        let (source, mut lod) = {
            let (s, l) = lods.split_at_mut(i);
            (s.last_mut().unwrap(), l.first_mut().unwrap())
        };

        let threshold = 0.7f32.powf(i as f32);
        let mut target_index_count = (mesh.indices.len() as f32 * threshold) as usize / 3 * 3;
        let target_error = 0.01;

        // we can simplify all the way from base level or from the last result
        // simplifying from the base level sometimes produces better results, but simplifying from last level is faster

        if source.len() < target_index_count {
            target_index_count = source.len();
        }

        lod.resize(source.len(), Default::default());
        let size = simplify(
            &mut lod,
            &source,
            &mesh.vertices,
            target_index_count,
            target_error,
            None,
        );
        lod.resize(size, Default::default());
    }

    let simplified = start.elapsed();
    let start = Instant::now();

    // optimize each individual LOD for vertex cache & overdraw
    for lod in lods.iter_mut() {
        let indices_copy = lod.clone();
        optimize_vertex_cache(lod, &indices_copy, mesh.vertices.len());
        let indices_copy = lod.clone();
        optimize_overdraw(lod, &indices_copy, &mesh.vertices, 1.0);
    }

    // concatenate all LODs into one IB
    // note: the order of concatenation is important - since we optimize the entire IB for vertex fetch,
    // putting coarse LODs first makes sure that the vertex range referenced by them is as small as possible
    // some GPUs process the entire range referenced by the index buffer region so doing this optimizes the vertex transform
    // cost for coarse LODs
    // this order also produces much better vertex fetch cache coherency for coarse LODs (since they're essentially optimized first)
    // somewhat surprisingly, the vertex fetch cache coherency for fine LODs doesn't seem to suffer that much.
    let mut lod_index_offsets = [0; LOD_COUNT];
    let mut lod_index_counts = [0; LOD_COUNT];
    let mut total_index_count = 0;

    for i in (0..lods.len()).rev() {
        lod_index_offsets[i] = total_index_count;
        lod_index_counts[i] = lods[i].len();

        total_index_count += lods[i].len();
    }

    let mut indices = vec![0; total_index_count];

    for i in 0..lods.len() {
        let offset = lod_index_offsets[i];
        indices[offset..offset + lods[i].len()].copy_from_slice(&lods[i]);
    }

    let mut vertices = mesh.vertices.clone();

    // vertex fetch optimization should go last as it depends on the final index order
    // note that the order of LODs above affects vertex fetch results
    optimize_vertex_fetch(&mut vertices, &mut indices, &mesh.vertices);

    let optimized = start.elapsed();

    println!(
        "{:9}: {} triangles => {} LOD levels down to {} triangles in {:.2} msec, optimized in {:.2} msec",
        "SimplifyC",
        lod_index_counts[0] / 3,
        LOD_COUNT,
        lod_index_counts[LOD_COUNT - 1] / 3,
        simplified.as_micros() as f64 / 1000.0,
        optimized.as_micros() as f64 / 1000.0,
    );

    // for using LOD data at runtime, in addition to vertices and indices you have to save lod_index_offsets/lod_index_counts.

    {
        let offset0 = lod_index_offsets[0];
        let vcs_0 = analyze_vertex_cache(
            &indices[offset0..offset0 + lod_index_counts[0]],
            vertices.len(),
            CACHE_SIZE,
            0,
            0,
        );
        let vfs_0 = analyze_vertex_fetch(
            &indices[offset0..offset0 + lod_index_counts[0]],
            vertices.len(),
            std::mem::size_of::<Vertex>(),
        );
        let offsetn = lod_index_offsets[LOD_COUNT - 1];
        let vcs_n = analyze_vertex_cache(
            &indices[offsetn..offsetn + lod_index_counts[LOD_COUNT - 1]],
            vertices.len(),
            CACHE_SIZE,
            0,
            0,
        );
        let vfs_n = analyze_vertex_fetch(
            &indices[offsetn..offsetn + lod_index_counts[LOD_COUNT - 1]],
            vertices.len(),
            std::mem::size_of::<Vertex>(),
        );

        let mut pv = vec![PackedVertexOct::default(); vertices.len()];
        pack_mesh_oct(&mut pv, &vertices);

        let mut vbuf = vec![0; encode_vertex_buffer_bound(vertices.len(), std::mem::size_of::<PackedVertexOct>())];
        let vb_size = encode_vertex_buffer(&mut vbuf, &pv, VertexEncodingVersion::default()).unwrap();
        vbuf.resize(vb_size, 0);

        let mut ibuf = vec![0; encode_index_buffer_bound(indices.len(), vertices.len())];
        let ib_size = encode_index_buffer(&mut ibuf, &indices, IndexEncodingVersion::default()).unwrap();
        ibuf.resize(ib_size, 0);

        println!(
            "{:9}  ACMR {:.6}...{:.6} Overfetch {:.6}..{:.6} Codec VB {:.1} bits/vertex IB {:.1} bits/triangle",
            "",
            vcs_0.acmr,
            vcs_n.acmr,
            vfs_0.overfetch,
            vfs_n.overfetch,
            vbuf.len() as f64 / vertices.len() as f64 * 8.0,
            ibuf.len() as f64 / (indices.len() / 3) as f64 * 8.0
        );
    }
}

fn stripify_mesh(mesh: &Mesh, use_restart: bool, desc: char) {
    let restart_index = if use_restart { INVALID_INDEX } else { 0 };

    // note: input mesh is assumed to be optimized for vertex cache and vertex fetch
    let start = Instant::now();
    let mut strip = vec![0; stripify_bound(mesh.indices.len())];
    let size = stripify(&mut strip, &mesh.indices, mesh.vertices.len(), restart_index);
    strip.resize(size, 0);
    let duration = start.elapsed();

    let mut copy = mesh.clone();
    let size = unstripify(&mut copy.indices, &strip, restart_index);
    copy.indices.resize(size, 0);
    assert!(copy.indices.len() <= unstripify_bound(strip.len()));

    assert!(copy.is_valid());
    assert_eq!(mesh.hash(), copy.hash());

    let vcs = analyze_vertex_cache(&copy.indices, mesh.vertices.len(), CACHE_SIZE, 0, 0);
    let vcs_nv = analyze_vertex_cache(&copy.indices, mesh.vertices.len(), 32, 32, 32);
    let vcs_amd = analyze_vertex_cache(&copy.indices, mesh.vertices.len(), 14, 64, 128);
    let vcs_intel = analyze_vertex_cache(&copy.indices, mesh.vertices.len(), 128, 0, 0);

    println!(
        "Stripify{}: ACMR {:.6} ATVR {:.6} (NV {:.6} AMD {:.6} Intel {:.6}); {} strip indices ({:.1}%) in {:.2} msec",
        desc,
        vcs.acmr,
        vcs.atvr,
        vcs_nv.atvr,
        vcs_amd.atvr,
        vcs_intel.atvr,
        strip.len(),
        strip.len() as f64 / mesh.indices.len() as f64 * 100.0,
        duration.as_micros() as f64 / 1000.0
    );
}

fn shadow(mesh: &Mesh) {
    // note: input mesh is assumed to be optimized for vertex cache and vertex fetch

    let start = Instant::now();
    // this index buffer can be used for position-only rendering using the same vertex data that the original index buffer uses
    let mut shadow_indices = vec![0; mesh.indices.len()];
    let position_stream = Stream::from_slice_with_subset(&mesh.vertices, 0..std::mem::size_of::<f32>() * 3);
    generate_shadow_index_buffer(&mut shadow_indices, &mesh.indices, &position_stream);
    let duration = start.elapsed();

    // while you can't optimize the vertex data after shadow IB was constructed, you can and should optimize the shadow IB for vertex cache
    // this is valuable even if the original indices array was optimized for vertex cache!
    let input = shadow_indices.clone();
    optimize_vertex_cache(&mut shadow_indices, &input, mesh.vertices.len());

    let vcs = analyze_vertex_cache(&mesh.indices, mesh.vertices.len(), CACHE_SIZE, 0, 0);
    let vcss = analyze_vertex_cache(&shadow_indices, mesh.vertices.len(), CACHE_SIZE, 0, 0);

    let mut shadow_flags = vec![0; mesh.vertices.len()];
    let mut shadow_vertices = 0;

    for index in shadow_indices {
        shadow_vertices += 1 - shadow_flags[index as usize];
        shadow_flags[index as usize] = 1;
    }

    println!(
        "ShadowIB : ACMR {:.6} ({:.2}x improvement); {} shadow vertices ({:.2}x improvement) in {:.2} msec",
        vcss.acmr,
        vcs.vertices_transformed as f64 / vcss.vertices_transformed as f64,
        shadow_vertices,
        mesh.vertices.len() as f64 / shadow_vertices as f64,
        duration.as_micros() as f64 / 1000.0,
    );
}

#[cfg(feature = "experimental")]
fn meshlets(mesh: &Mesh) {
    const MAX_VERTICES: usize = 64;
    const MAX_TRIANGLES: usize = 126;

    // note: input mesh is assumed to be optimized for vertex cache and vertex fetch
    let start = Instant::now();
    let mut meshlets = vec![Meshlet::default(); build_meshlets_bound(mesh.indices.len(), MAX_VERTICES, MAX_TRIANGLES)];
    let size = build_meshlets(
        &mut meshlets,
        &mesh.indices,
        mesh.vertices.len(),
        MAX_VERTICES,
        MAX_TRIANGLES,
    );
    meshlets.resize(size, Meshlet::default());
    let duration = start.elapsed();

    let mut vertices = 0;
    let mut triangles = 0;
    let mut not_full = 0;

    for m in &meshlets {
        vertices += m.vertex_count as usize;
        triangles += m.triangle_count as usize;
        not_full += ((m.vertex_count as usize) < MAX_VERTICES) as usize;
    }

    println!(
        "Meshlets : {} meshlets (avg vertices {:.1}, avg triangles {:.1}, not full {}) in {:.2} msec",
        meshlets.len(),
        vertices as f64 / meshlets.len() as f64,
        triangles as f64 / meshlets.len() as f64,
        not_full,
        duration.as_micros() as f64 / 1000.0
    );

    let camera = [100.0, 100.0, 100.0];

    let mut rejected = 0;
    let mut rejected_s8 = 0;
    let mut rejected_alt = 0;
    let mut rejected_alt_s8 = 0;
    let mut accepted = 0;
    let mut accepted_s8 = 0;

    let start = Instant::now();
    for meshlet in &meshlets {
        let bounds = compute_meshlet_bounds(&meshlet, &mesh.vertices);

        // trivial accept: we can't ever backface cull this meshlet
        accepted += (bounds.cone_cutoff >= 1.0) as usize;
        accepted_s8 += (bounds.cone_cutoff_s8 >= 127) as usize;

        // perspective projection: dot(normalize(cone_apex - camera_position), cone_axis) > cone_cutoff
        let mview = [
            bounds.cone_apex[0] - camera[0],
            bounds.cone_apex[1] - camera[1],
            bounds.cone_apex[2] - camera[2],
        ];
        let mviewlength = (mview[0] * mview[0] + mview[1] * mview[1] + mview[2] * mview[2]).sqrt();

        rejected += (mview[0] * bounds.cone_axis[0] + mview[1] * bounds.cone_axis[1] + mview[2] * bounds.cone_axis[2]
            >= bounds.cone_cutoff * mviewlength) as usize;
        rejected_s8 += (mview[0] * (bounds.cone_axis_s8[0] as f32 / 127.0)
            + mview[1] * (bounds.cone_axis_s8[1] as f32 / 127.0)
            + mview[2] * (bounds.cone_axis_s8[2] as f32 / 127.0)
            >= (bounds.cone_cutoff_s8 as f32 / 127.0) * mviewlength) as usize;

        // alternative formulation for perspective projection that doesn't use apex (and uses cluster bounding sphere instead):
        // dot(normalize(center - camera_position), cone_axis) > cone_cutoff + radius / length(center - camera_position)
        let cview = [
            bounds.center[0] - camera[0],
            bounds.center[1] - camera[1],
            bounds.center[2] - camera[2],
        ];
        let cviewlength = (cview[0] * cview[0] + cview[1] * cview[1] + cview[2] * cview[2]).sqrt();

        rejected_alt +=
            (cview[0] * bounds.cone_axis[0] + cview[1] * bounds.cone_axis[1] + cview[2] * bounds.cone_axis[2]
                >= bounds.cone_cutoff * cviewlength + bounds.radius) as usize;
        rejected_alt_s8 += (cview[0] * (bounds.cone_axis_s8[0] as f32 / 127.0)
            + cview[1] * (bounds.cone_axis_s8[1] as f32 / 127.0)
            + cview[2] * (bounds.cone_axis_s8[2] as f32 / 127.0)
            >= (bounds.cone_cutoff_s8 as f32 / 127.0) * cviewlength + bounds.radius)
            as usize;
    }
    let duration = start.elapsed();

    println!(
        "ConeCull : rejected apex {} ({:.1}%) / center {} ({:.1}%), trivially accepted {} ({:.1}%) in {:.2} msec",
        rejected,
        rejected as f64 / meshlets.len() as f64 * 100.0,
        rejected_alt,
        rejected_alt as f64 / meshlets.len() as f64 * 100.0,
        accepted,
        accepted as f64 / meshlets.len() as f64 * 100.0,
        duration.as_micros() as f64 / 1000.0,
    );
    println!(
        "ConeCull8: rejected apex {} ({:.1}%) / center {} ({:.1}%), trivially accepted {} ({:.1}%) in {:.2} msec",
        rejected_s8,
        rejected_s8 as f64 / meshlets.len() as f64 * 100.0,
        rejected_alt_s8,
        rejected_alt_s8 as f64 / meshlets.len() as f64 * 100.0,
        accepted_s8,
        accepted_s8 as f64 / meshlets.len() as f64 * 100.0,
        duration.as_micros() as f64 / 1000.0,
    );
}

#[cfg(feature = "experimental")]
fn spatial_sort_mesh(mesh: &Mesh) {
    let mut pv = vec![PackedVertexOct::default(); mesh.vertices.len()];
    pack_mesh_oct(&mut pv, &mesh.vertices);

    let start = Instant::now();

    let mut remap = vec![0; mesh.vertices.len()];
    spatial_sort_remap(&mut remap, &mesh.vertices);

    let duration = start.elapsed();

    let pv_copy = pv.clone();
    remap_vertex_buffer(&mut pv, &pv_copy, &remap);

    let mut vbuf = vec![0; encode_vertex_buffer_bound(mesh.vertices.len(), std::mem::size_of::<PackedVertexOct>())];
    let vb_size = encode_vertex_buffer(&mut vbuf, &pv, VertexEncodingVersion::default()).unwrap();
    vbuf.resize(vb_size, 0);

    let csize = compress_data(&vbuf);

    println!(
        "Spatial  : {:.1} bits/vertex (post-deflate {:.1} bits/vertex); sort {:.2} msec",
        (vbuf.len() * 8) as f64 / mesh.vertices.len() as f64,
        (csize * 8) as f64 / mesh.vertices.len() as f64,
        duration.as_micros() as f64 / 1000.0,
    );
}

#[cfg(feature = "experimental")]
fn spatial_sort_mesh_triangles(mesh: &Mesh) {
    let mut copy = mesh.clone();

    let start = Instant::now();

    spatial_sort_triangles(&mut copy.indices, &mesh.indices, &copy.vertices);

    let duration = start.elapsed();

    let indices_copy = copy.indices.clone();
    optimize_vertex_cache(&mut copy.indices, &indices_copy, copy.vertices.len());
    let vertices_copy = copy.vertices.clone();
    optimize_vertex_fetch(&mut copy.vertices, &mut copy.indices, &vertices_copy);

    let mut pv = vec![PackedVertexOct::default(); mesh.vertices.len()];
    pack_mesh_oct(&mut pv, &copy.vertices);

    let mut vbuf = vec![0; encode_vertex_buffer_bound(mesh.vertices.len(), std::mem::size_of::<PackedVertexOct>())];
    let vb_size = encode_vertex_buffer(&mut vbuf, &pv, VertexEncodingVersion::default()).unwrap();
    vbuf.resize(vb_size, 0);

    let mut ibuf = vec![0; encode_index_buffer_bound(mesh.indices.len(), mesh.vertices.len())];
    let ib_size = encode_index_buffer(&mut ibuf, &copy.indices, IndexEncodingVersion::default()).unwrap();
    ibuf.resize(ib_size, 0);

    let csizev = compress_data(&vbuf);
    let csizei = compress_data(&ibuf);

    println!(
        "SpatialT : {:.1} bits/vertex (post-deflate {:.1} bits/vertex); {:.1} bits/triangle (post-deflate {:.1} bits/triangle); sort {:.2} msec",
        (vbuf.len() * 8) as f64 / mesh.vertices.len() as f64,
        (csizev * 8) as f64 / mesh.vertices.len() as f64,
        (ibuf.len() * 8) as f64 / (mesh.indices.len() / 3) as f64,
        (csizei * 8) as f64 / (mesh.indices.len() / 3) as f64,
        duration.as_micros() as f64 / 1000.0,
    );
}

fn process_deinterleaved<P>(path: P) -> Result<(), tobj::LoadError>
where
    P: AsRef<Path> + Clone + Debug,
{
    // Most algorithms in the library work out of the box with deinterleaved geometry, but some require slightly special treatment;
    // this code runs a simplified version of complete opt. pipeline using deinterleaved geo. There's no compression performed but you
    // can trivially run it by quantizing all elements and running `encode_vertex_buffer` once for each vertex stream.
    let (models, _materials) = tobj::load_obj(
        path.clone(),
        &tobj::LoadOptions {
            triangulate: true,
            ..Default::default()
        },
    )?;

    let total_indices = models.iter().map(|m| m.mesh.indices.len()).sum();

    let mut unindexed_pos = vec![[0.0; 3]; total_indices];
    let mut unindexed_nrm = vec![[0.0; 3]; total_indices];
    let mut unindexed_uv = vec![[0.0; 2]; total_indices];

    let mut dst = 0;

    for model in models.iter() {
        let mesh = &model.mesh;
        assert!(mesh.positions.len() % 3 == 0);

        for i in 0..mesh.indices.len() {
            let pi = mesh.indices[i] as usize;
            unindexed_pos[dst].copy_from_slice(&mesh.positions[3 * pi..3 * (pi + 1)]);

            if !mesh.normals.is_empty() {
                let ni = mesh.normal_indices[i] as usize;
                unindexed_nrm[dst].copy_from_slice(&mesh.normals[3 * ni..3 * (ni + 1)]);
            }

            if !mesh.texcoords.is_empty() {
                let ti = mesh.texcoord_indices[i] as usize;
                unindexed_uv[dst].copy_from_slice(&mesh.texcoords[2 * ti..2 * (ti + 1)]);
            }

            dst += 1;
        }
    }

    let start = Instant::now();

    let streams = [
        Stream::from_slice(&unindexed_pos),
        Stream::from_slice(&unindexed_nrm),
        Stream::from_slice(&unindexed_uv),
    ];

    let mut remap = vec![0; total_indices];

    let total_vertices = generate_vertex_remap_multi(&mut remap, None, &streams);

    let mut indices = remap.clone();

    let mut pos = vec![[0.0; 3]; total_vertices];
    remap_vertex_buffer(&mut pos, &unindexed_pos, &remap);

    let mut nrm = vec![[0.0; 3]; total_vertices];
    remap_vertex_buffer(&mut nrm, &unindexed_nrm, &remap);

    let mut uv = vec![[0.0; 2]; total_vertices];
    remap_vertex_buffer(&mut uv, &unindexed_uv, &remap);

    let reindex = start.elapsed();
    let start = Instant::now();

    optimize_vertex_cache(&mut indices, &remap, total_vertices);

    optimize_vertex_fetch_remap(&mut remap, &indices);
    let pos_copy = pos.clone();
    remap_vertex_buffer(&mut pos, &pos_copy, &remap);
    let nrm_copy = nrm.clone();
    remap_vertex_buffer(&mut nrm, &nrm_copy, &remap);
    let uv_copy = uv.clone();
    remap_vertex_buffer(&mut uv, &uv_copy, &remap);

    let optimize = start.elapsed();
    let start = Instant::now();

    // note: since shadow index buffer is computed based on regular vertex/index buffer, the stream points at the indexed data - not `unindexed_pos`
    let shadow_stream = Stream::from_slice(&pos);

    let mut shadow_indices = vec![0; total_indices];
    generate_shadow_index_buffer_multi(&mut shadow_indices, &indices, &[shadow_stream]);

    let shadow_indices_copy = shadow_indices.clone();
    optimize_vertex_cache(&mut shadow_indices, &shadow_indices_copy, total_vertices);

    let shadow = start.elapsed();

    println!(
        "Deintrlvd: {} vertices, reindexed in {:.2} msec, optimized in {:.2} msec, generated & optimized shadow indices in {:.2} msec",
        total_vertices,
        reindex.as_micros() as f64 / 1000.0,
        optimize.as_micros() as f64 / 1000.0,
        shadow.as_micros() as f64 / 1000.0
    );

    Ok(())
}

fn process(mesh: &Mesh) {
    optimize(mesh, "Original", |_| {});
    optimize(mesh, "Random", opt_random_shuffle);
    optimize(mesh, "Cache", opt_cache);
    optimize(mesh, "CacheFifo", opt_cache_fifo);
    optimize(mesh, "CacheStrp", opt_cache_strip);
    optimize(mesh, "Overdraw", opt_overdraw);
    optimize(mesh, "Fetch", opt_fetch);
    optimize(mesh, "FetchMap", opt_fetch_remap);
    optimize(mesh, "Complete", opt_complete);

    let mut copy = mesh.clone();
    optimize_vertex_cache(&mut copy.indices, &mesh.indices, copy.vertices.len());
    let copy_vertices = copy.vertices.clone();
    optimize_vertex_fetch(&mut copy.vertices, &mut copy.indices, &copy_vertices);

    let mut copystrip = mesh.clone();
    optimize_vertex_cache_strip(&mut copystrip.indices, &mesh.indices, copystrip.vertices.len());
    let copystrip_vertices = copystrip.vertices.clone();
    optimize_vertex_fetch(&mut copystrip.vertices, &mut copystrip.indices, &copystrip_vertices);

    stripify_mesh(&copy, false, ' ');
    stripify_mesh(&copy, true, 'R');
    stripify_mesh(&copystrip, true, 'S');

    #[cfg(feature = "experimental")]
    meshlets(&copy);
    shadow(&copy);

    encode_index(&copy, ' ');
    encode_index(&copystrip, 'S');

    let mut strip = vec![0; stripify_bound(copystrip.indices.len())];
    let size = stripify(&mut strip, &copystrip.indices, copystrip.vertices.len(), 0);
    strip.resize(size, 0);

    #[cfg(feature = "experimental")]
    encode_index_sequence1(&mut strip, copystrip.vertices.len(), 'D');

    pack_vertex(&copy);
    encode_vertex(&copy);
    encode_vertex_oct(&copy);

    #[cfg(feature = "experimental")]
    {
        simplify_mesh(mesh);
        simplify_mesh_sloppy(mesh, 0.2);
        simplify_mesh_complete(mesh);
        simplify_mesh_points(mesh, 0.2);

        spatial_sort_mesh(mesh);
        spatial_sort_mesh_triangles(mesh);
    }
}

fn process_dev(mesh: &Mesh) {
    simplify_mesh(mesh);
    simplify_mesh_sloppy(mesh, 0.2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let dev_mode = match args.first() {
        Some(arg) => arg == "-d",
        None => false,
    };

    for arg in args.iter().skip(if dev_mode { 1 } else { 0 }) {
        let mesh = Mesh::load(arg.clone())?;

        if dev_mode {
            process_dev(&mesh);
        } else {
            process(&mesh);
            process_deinterleaved(arg)?;
        }
    }

    Ok(())
}
