use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use meshopt_rs::Stream;
#[cfg(feature = "experimental")]
use meshopt_rs::cluster::{Meshlet, build_meshlets, build_meshlets_bound, compute_meshlet_bounds};
use meshopt_rs::index::IndexEncodingVersion;
use meshopt_rs::index::buffer::{decode_index_buffer, encode_index_buffer, encode_index_buffer_bound};
use meshopt_rs::index::generator::{
    generate_shadow_index_buffer, generate_vertex_remap, remap_index_buffer, remap_vertex_buffer,
};
#[cfg(feature = "experimental")]
use meshopt_rs::index::sequence::{decode_index_sequence, encode_index_sequence, encode_index_sequence_bound};
use meshopt_rs::overdraw::optimize_overdraw;
#[cfg(feature = "experimental")]
use meshopt_rs::stripify::{stripify, stripify_bound};
use meshopt_rs::vertex::Position;
use meshopt_rs::vertex::cache::{optimize_vertex_cache, optimize_vertex_cache_fifo, optimize_vertex_cache_strip};
use meshopt_rs::vertex::fetch::{optimize_vertex_fetch, optimize_vertex_fetch_remap};

use std::fmt::Debug;
use std::path::Path;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Vertex {
    p: [f32; 3],
    n: [f32; 3],
    t: [f32; 2],
}

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
        let (models, _materials) = tobj::load_obj(
            path.clone(),
            &tobj::LoadOptions {
                triangulate: true,
                ..Default::default()
            },
        )?;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for model in models.iter() {
            let mesh = &model.mesh;
            assert!(mesh.positions.len().is_multiple_of(3));

            vertices.reserve(mesh.indices.len());
            indices.extend_from_slice(&mesh.indices);

            for i in 0..mesh.indices.len() {
                let mut vertex = Vertex::default();

                let pi = mesh.indices[i] as usize;
                vertex.p.copy_from_slice(&mesh.positions[3 * pi..3 * (pi + 1)]);

                if !mesh.normals.is_empty() {
                    let ni = mesh.normal_indices[i] as usize;
                    vertex.n.copy_from_slice(&mesh.normals[3 * ni..3 * (ni + 1)]);
                }

                if !mesh.texcoords.is_empty() {
                    let ti = mesh.texcoord_indices[i] as usize;
                    vertex.t.copy_from_slice(&mesh.texcoords[2 * ti..2 * (ti + 1)]);
                }

                vertices.push(vertex);
            }
        }

        let total_indices = indices.len();
        let mut remap = vec![0; total_indices];

        let mut result = Mesh::default();

        let total_vertices = generate_vertex_remap(&mut remap, None, &Stream::from_slice(&vertices));

        result.indices = remap;

        result.vertices.resize(total_vertices, Vertex::default());
        remap_vertex_buffer(&mut result.vertices, &vertices, &result.indices);

        Ok(result)
    }
}

fn with_input(c: &mut Criterion) {
    let input_name = "";
    let mesh = Mesh::load(Path::new(input_name)).unwrap();

    c.bench_with_input(
        BenchmarkId::new("optimize_vertex_cache", input_name),
        &mesh,
        |b, mesh| {
            let mut result = mesh.indices.clone();

            b.iter(|| optimize_vertex_cache(&mut result, &mesh.indices, mesh.vertices.len()));
        },
    );

    c.bench_with_input(
        BenchmarkId::new("optimize_vertex_cache_fifo", input_name),
        &mesh,
        |b, mesh| {
            const CACHE_SIZE: usize = 16;
            let mut result = mesh.indices.clone();

            b.iter(|| optimize_vertex_cache_fifo(&mut result, &mesh.indices, mesh.vertices.len(), CACHE_SIZE as u32));
        },
    );

    c.bench_with_input(
        BenchmarkId::new("optimize_vertex_cache_strip", input_name),
        &mesh,
        |b, mesh| {
            let mut result = mesh.indices.clone();

            b.iter(|| optimize_vertex_cache_strip(&mut result, &mesh.indices, mesh.vertices.len()));
        },
    );

    c.bench_with_input(BenchmarkId::new("optimize_overdraw", input_name), &mesh, |b, mesh| {
        const TRESHOLD: f32 = 3.0; // Use worst-case ACMR threshold so that overdraw optimizer can sort *all* triangles
        let mut result = vec![0; mesh.indices.len()];

        b.iter(|| optimize_overdraw(&mut result, &mesh.indices, &mesh.vertices, TRESHOLD));
    });

    c.bench_with_input(
        BenchmarkId::new("optimize_vertex_fetch", input_name),
        &mesh,
        |b, mesh| {
            let mut vertex_result = mesh.vertices.clone();

            b.iter(|| {
                let mut index_result = mesh.indices.clone(); // Must copy on every iteration because it's also an input
                optimize_vertex_fetch(&mut vertex_result, &mut index_result, &mesh.vertices);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("optimize_vertex_fetch_remap", input_name),
        &mesh,
        |b, mesh| {
            let mut remap = vec![0; mesh.vertices.len()];
            let mut index_result = mesh.indices.clone();
            let mut vertices_result = mesh.vertices.clone();

            b.iter(|| {
                optimize_vertex_fetch_remap(&mut remap, &mesh.indices);
                remap_index_buffer(&mut index_result, &remap);
                remap_vertex_buffer(&mut vertices_result, &mesh.vertices, &remap);
            });
        },
    );

    let mut copy = mesh.clone();
    optimize_vertex_cache(&mut copy.indices, &mesh.indices, copy.vertices.len());
    let copy_vertices = copy.vertices.clone();
    optimize_vertex_fetch(&mut copy.vertices, &mut copy.indices, &copy_vertices);

    let mut copystrip = mesh.clone();
    optimize_vertex_cache_strip(&mut copystrip.indices, &mesh.indices, copystrip.vertices.len());
    let copystrip_vertices = copystrip.vertices.clone();
    optimize_vertex_fetch(&mut copystrip.vertices, &mut copystrip.indices, &copystrip_vertices);

    // stripify

    #[cfg(feature = "experimental")]
    c.bench_with_input(BenchmarkId::new("build_meshlets", input_name), &copy, |b, mesh| {
        const MAX_VERTICES: usize = 64;
        const MAX_TRIANGLES: usize = 124;
        const CONE_WEIGHT: f32 = 0.5;

        let max_meshlets = build_meshlets_bound(mesh.indices.len(), MAX_VERTICES, MAX_TRIANGLES);
        let mut meshlets = vec![Meshlet::default(); max_meshlets];
        let mut meshlet_vertices = vec![0u32; max_meshlets * MAX_VERTICES];
        let mut meshlet_triangles = vec![0u8; max_meshlets * MAX_TRIANGLES * 3];

        b.iter(|| {
            build_meshlets(
                &mut meshlets,
                &mut meshlet_vertices,
                &mut meshlet_triangles,
                &mesh.indices,
                &mesh.vertices,
                MAX_VERTICES,
                MAX_TRIANGLES,
                CONE_WEIGHT,
            )
        });
    });

    #[cfg(feature = "experimental")]
    c.bench_with_input(
        BenchmarkId::new("compute_meshlet_bounds", input_name),
        &copy,
        |b, mesh| {
            const MAX_VERTICES: usize = 64;
            const MAX_TRIANGLES: usize = 124;
            const CONE_WEIGHT: f32 = 0.5;

            let max_meshlets = build_meshlets_bound(mesh.indices.len(), MAX_VERTICES, MAX_TRIANGLES);
            let mut meshlets = vec![Meshlet::default(); max_meshlets];
            let mut meshlet_vertices = vec![0u32; max_meshlets * MAX_VERTICES];
            let mut meshlet_triangles = vec![0u8; max_meshlets * MAX_TRIANGLES * 3];

            build_meshlets(
                &mut meshlets,
                &mut meshlet_vertices,
                &mut meshlet_triangles,
                &mesh.indices,
                &mesh.vertices,
                MAX_VERTICES,
                MAX_TRIANGLES,
                CONE_WEIGHT,
            );

            b.iter(|| {
                for meshlet in &meshlets {
                    compute_meshlet_bounds(
                        &meshlet_vertices[meshlet.vertex_offset as usize..],
                        &meshlet_triangles[meshlet.triangle_offset as usize
                            ..(meshlet.triangle_offset + meshlet.triangle_count * 3) as usize],
                        &mesh.vertices,
                    );
                }
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("generate_shadow_index_buffer", input_name),
        &copy,
        |b, mesh| {
            let mut shadow_indices = vec![0; mesh.indices.len()];
            let position_stream = Stream::from_slice_with_subset(&mesh.vertices, 0..std::mem::size_of::<f32>() * 3);

            b.iter(|| generate_shadow_index_buffer(&mut shadow_indices, &mesh.indices, &position_stream));
        },
    );

    let mut group = c.benchmark_group("index-buffer-encoding");
    {
        group.throughput(Throughput::Bytes(
            (mesh.indices.len() * std::mem::size_of::<u32>()) as u64,
        ));
        group.bench_with_input(BenchmarkId::new("encode_index_buffer", input_name), &copy, |b, mesh| {
            let mut result = vec![0; encode_index_buffer_bound(mesh.indices.len(), mesh.vertices.len())];

            b.iter(|| encode_index_buffer(&mut result, &mesh.indices, IndexEncodingVersion::default()));
        });

        let mut encoded_indices = vec![0; encode_index_buffer_bound(mesh.indices.len(), mesh.vertices.len())];
        let size = encode_index_buffer(&mut encoded_indices, &mesh.indices, IndexEncodingVersion::default()).unwrap();
        encoded_indices.resize(size, 0);

        let mut decoded_indices = vec![0u32; mesh.indices.len()];

        group.throughput(Throughput::Bytes(
            (mesh.indices.len() * std::mem::size_of::<u32>()) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("decode_index_buffer", input_name),
            &encoded_indices,
            |b, encoded_indices| {
                b.iter(|| decode_index_buffer(&mut decoded_indices, encoded_indices).unwrap());
            },
        );
    }
    group.finish();

    #[cfg(feature = "experimental")]
    {
        let mut group = c.benchmark_group("index-sequence-encoding");

        let mut strip = vec![0; stripify_bound(copystrip.indices.len())];
        let size = stripify(&mut strip, &copystrip.indices, copystrip.vertices.len(), 0);
        strip.resize(size, 0);

        let vertex_count = copystrip.vertices.len();

        group.throughput(Throughput::Bytes((strip.len() * std::mem::size_of::<u32>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("encode_index_sequence", input_name),
            &strip,
            |b, data| {
                let mut buffer = vec![0; encode_index_sequence_bound(data.len(), vertex_count)];

                b.iter(|| encode_index_sequence(&mut buffer, data, IndexEncodingVersion::default()));
            },
        );

        let mut buffer = vec![0; encode_index_sequence_bound(strip.len(), vertex_count)];
        let size = encode_index_sequence(&mut buffer, &strip, IndexEncodingVersion::default());
        buffer.resize(size, 0);

        let mut decoded_indices = vec![0u32; strip.len()];

        group.throughput(Throughput::Bytes((strip.len() * std::mem::size_of::<u32>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("decode_index_sequence", input_name),
            &buffer,
            |b, buffer| {
                b.iter(|| decode_index_sequence(&mut decoded_indices, buffer));
            },
        );

        group.finish();
    }
}

criterion_group!(benches, with_input);
criterion_main!(benches);
