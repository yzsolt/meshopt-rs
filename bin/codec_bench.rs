#![allow(clippy::identity_op)]

use meshopt_rs::index::IndexEncodingVersion;
use meshopt_rs::index::buffer::{decode_index_buffer, encode_index_buffer, encode_index_buffer_bound};
use meshopt_rs::vertex::Position;
use meshopt_rs::vertex::VertexEncodingVersion;
use meshopt_rs::vertex::buffer::{decode_vertex_buffer, encode_vertex_buffer, encode_vertex_buffer_bound};
use meshopt_rs::vertex::cache::{optimize_vertex_cache, optimize_vertex_cache_strip};
use meshopt_rs::vertex::fetch::optimize_vertex_fetch;
use meshopt_rs::vertex::filter::{decode_filter_exp, decode_filter_oct_8, decode_filter_oct_16, decode_filter_quat};
use std::time::Instant;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Vertex {
    data: [u16; 16],
}

impl Position for Vertex {
    fn pos(&self) -> [f32; 3] {
        let get_f32 = |start: usize| {
            let a = self.data[start].to_le_bytes();
            let b = self.data[start + 1].to_le_bytes();
            f32::from_le_bytes([a[0], a[1], b[0], b[1]])
        };

        [get_f32(0), get_f32(2), get_f32(4)]
    }
}

fn murmur3(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;

    h
}

fn bench_codecs(vertices: &[Vertex], indices: &[u32], bestvd: &mut f64, bestid: &mut f64, verbose: bool) {
    let mut vb = vec![Vertex::default(); vertices.len()];
    let mut ib = vec![0u32; indices.len()];

    let mut vc = vec![0u8; encode_vertex_buffer_bound(vertices.len(), std::mem::size_of::<Vertex>())];
    let mut ic = vec![0u8; encode_index_buffer_bound(indices.len(), vertices.len())];

    if verbose {
        println!(
            "source: vertex data {} bytes, index data {} bytes",
            std::mem::size_of_val(vertices),
            indices.len() * 4
        );
    }

    for pass in 0..if verbose { 2 } else { 1 } {
        if pass == 1 {
            optimize_vertex_cache_strip(&mut ib, indices, vertices.len());
        } else {
            optimize_vertex_cache(&mut ib, indices, vertices.len());
        }

        optimize_vertex_fetch(&mut vb, &mut ib, vertices);

        vc.resize_with(vc.capacity(), Default::default);
        let vc_size = encode_vertex_buffer(&mut vc, &vb, VertexEncodingVersion::V0).unwrap();
        vc.resize_with(vc_size, Default::default);

        ic.resize_with(ic.capacity(), Default::default);
        let ic_size = encode_index_buffer(&mut ic, &ib, IndexEncodingVersion::V1).unwrap();
        ic.resize_with(ic_size, Default::default);

        if verbose {
            println!(
                "pass {}: vertex data {} bytes, index data {} bytes",
                pass,
                vc.len(),
                ic.len()
            );
        }

        for _ in 0..10 {
            let t0 = Instant::now();

            let rv = decode_vertex_buffer(&mut vb, &vc);
            assert!(rv.is_ok());

            let t1 = Instant::now();

            let ri = decode_index_buffer(&mut ib, &ic);
            assert!(ri.is_ok());

            let t2 = Instant::now();

            const GB: f64 = 1024.0 * 1024.0 * 1024.0;

            let vertex_time = (t1 - t0).as_secs_f64();
            let index_time = (t2 - t1).as_secs_f64();

            let vertex_throughput = std::mem::size_of_val(vertices) as f64 / GB / vertex_time;
            let index_throughput = (indices.len() * 4) as f64 / GB / index_time;

            if verbose {
                println!(
                    "decode: vertex {:.2} ms ({:.2} GB/sec), index {:.2} ms ({:.2} GB/sec)",
                    vertex_time * 1_000.0,
                    vertex_throughput,
                    index_time * 1_000.0,
                    index_throughput
                );
            }

            if pass == 0 {
                *bestvd = bestvd.max(vertex_throughput);
                *bestid = bestid.max(index_throughput);
            }
        }
    }
}

fn bench_filters(
    count: usize,
    besto8: &mut f64,
    besto12: &mut f64,
    bestq12: &mut f64,
    bestexp: &mut f64,
    verbose: bool,
) {
    // note: the filters are branchless so we just run them on runs of zeroes
    let count4 = (count + 3) & !3;
    let mut d4 = vec![[0u8; 4]; count4];
    let mut d8 = vec![[0u16; 4]; count4];
    let mut d32 = vec![0u32; count4 * 2];

    let d4_size = d4.len() * std::mem::size_of::<[u8; 4]>();
    let d8_size = d4.len() * std::mem::size_of::<[u16; 4]>();

    if verbose {
        println!(
            "filters: oct8 data {} bytes, oct12/quat12 data {} bytes",
            d4_size, d8_size
        );
    }

    for _ in 0..10 {
        let t0 = Instant::now();

        decode_filter_oct_8(&mut d4);

        let t1 = Instant::now();

        decode_filter_oct_16(&mut d8);

        let t2 = Instant::now();

        decode_filter_quat(&mut d8);

        let t3 = Instant::now();

        decode_filter_exp(&mut d32);

        let t4 = Instant::now();

        const GB: f64 = 1024.0 * 1024.0 * 1024.0;

        let oct_8_time = (t1 - t0).as_secs_f64();
        let oct_16_time = (t2 - t1).as_secs_f64();
        let quat_time = (t3 - t2).as_secs_f64();
        let exp_time = (t4 - t3).as_secs_f64();

        let oct_8_throughput = d4_size as f64 / GB / oct_8_time;
        let oct_16_throughput = d8_size as f64 / GB / oct_16_time;
        let quat_throughput = d8_size as f64 / GB / quat_time;
        let exp_throughput = d8_size as f64 / GB / exp_time;

        if verbose {
            println!(
                "filter: oct8 {:.2} ms ({:.2} GB/sec), oct12 {:.2} ms ({:.2} GB/sec), quat12 {:.2} ms ({:.2} GB/sec), exp {:.2} ms ({:.2} GB/sec)",
                oct_8_time,
                oct_8_throughput,
                oct_16_time,
                oct_16_throughput,
                quat_time,
                quat_throughput,
                exp_time,
                exp_throughput
            );
        }

        *besto8 = besto8.max(oct_8_throughput);
        *besto12 = besto12.max(oct_16_throughput);
        *bestq12 = bestq12.max(quat_throughput);
        *bestexp = bestexp.max(exp_throughput);
    }
}

fn main() {
    const N: u32 = 1000;

    let mut vertices = Vec::with_capacity(((N + 1) * (N + 1)) as usize);

    let verbose = std::env::args().any(|a| a == "-v");

    for x in 0..=N {
        for y in 0..=N {
            let mut v = Vertex::default();

            for k in 0..16 {
                let h = murmur3((x * (N + 1) + y) * 16 + k);

                // use random k-bit sequence for each word to test all encoding types
                // note: this doesn't stress the sentinel logic too much but it's all branchless so it's probably fine?
                v.data[k as usize] = (h & ((1 << (k + 1)) - 1)) as u16;
            }

            vertices.push(v);
        }
    }

    let mut indices = Vec::with_capacity((N * N * 6) as usize);

    for x in 0..N {
        for y in 0..N {
            indices.push((x + 0) * N + (y + 0));
            indices.push((x + 1) * N + (y + 0));
            indices.push((x + 0) * N + (y + 1));

            indices.push((x + 0) * N + (y + 1));
            indices.push((x + 1) * N + (y + 0));
            indices.push((x + 1) * N + (y + 1));
        }
    }

    let mut bestvd = 0.0;
    let mut bestid = 0.0;
    bench_codecs(&vertices, &indices, &mut bestvd, &mut bestid, verbose);

    let mut besto8 = 0.0;
    let mut besto12 = 0.0;
    let mut bestq12 = 0.0;
    let mut bestexp = 0.0;
    bench_filters(
        8 * (N * N) as usize,
        &mut besto8,
        &mut besto12,
        &mut bestq12,
        &mut bestexp,
        verbose,
    );

    println!("Algorithm   :\tvtx\tidx\toct8\toct12\tquat12\texp");
    println!(
        "Score (GB/s):\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
        bestvd, bestid, besto8, besto12, bestq12, bestexp
    );
}
