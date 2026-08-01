//! Benchmarks over synthetic kNN-like graphs, the shape single-cell pipelines produce.
//!
//! ```text
//! cargo bench --no-default-features --bench leiden
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use single_clustering::community_search::leiden::{LeidenConfig, ObjectiveKind, leiden};
use single_clustering::network::CSRNetwork;
use std::collections::{HashMap, HashSet};
use std::hint::black_box;

/// Points drawn around well-separated centres, each joined to its `k` nearest neighbours.
/// Neighbour search is grid-bucketed so building the graph does not dominate setup.
fn knn_graph(n: usize, blocks: usize, k: usize, seed: u64) -> CSRNetwork {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let per = n / blocks;
    let centers: Vec<(f64, f64)> = (0..blocks)
        .map(|b| {
            let a = 2.0 * std::f64::consts::PI * (b as f64) / (blocks as f64);
            (60.0 * a.cos(), 60.0 * a.sin())
        })
        .collect();
    let pts: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let c = centers[(i / per).min(blocks - 1)];
            (
                c.0 + rng.random::<f64>() * 16.0 - 8.0,
                c.1 + rng.random::<f64>() * 16.0 - 8.0,
            )
        })
        .collect();

    let cell = 4.0;
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, p) in pts.iter().enumerate() {
        grid.entry(((p.0 / cell) as i64, (p.1 / cell) as i64))
            .or_default()
            .push(i);
    }

    let mut edges = HashSet::new();
    let mut cand: Vec<usize> = Vec::new();
    for i in 0..n {
        let (gx, gy) = ((pts[i].0 / cell) as i64, (pts[i].1 / cell) as i64);
        cand.clear();
        for dx in -2..=2 {
            for dy in -2..=2 {
                if let Some(bucket) = grid.get(&(gx + dx, gy + dy)) {
                    cand.extend(bucket.iter().copied().filter(|&j| j != i));
                }
            }
        }
        cand.sort_unstable_by(|&a, &b| {
            let da = (pts[i].0 - pts[a].0).powi(2) + (pts[i].1 - pts[a].1).powi(2);
            let db = (pts[i].0 - pts[b].0).powi(2) + (pts[i].1 - pts[b].1).powi(2);
            da.partial_cmp(&db).unwrap()
        });
        for &j in cand.iter().take(k) {
            edges.insert(if i < j { (i, j) } else { (j, i) });
        }
    }

    let edges: Vec<(usize, usize, f64)> = edges.into_iter().map(|(a, b)| (a, b, 1.0)).collect();
    CSRNetwork::from_edges(n, &edges).unwrap()
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("leiden/knn15");
    group.sample_size(10);

    for &n in &[1_000usize, 10_000, 100_000] {
        let graph = knn_graph(n, 10, 15, 5);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &graph, |b, graph| {
            b.iter(|| black_box(leiden(graph, &LeidenConfig::default()).unwrap()));
        });
    }
    group.finish();
}

fn bench_resolution(c: &mut Criterion) {
    let graph = knn_graph(20_000, 10, 15, 5);
    let mut group = c.benchmark_group("leiden/resolution");
    group.sample_size(10);

    for &r in &[0.25f64, 1.0, 4.0] {
        group.bench_with_input(BenchmarkId::from_parameter(r), &r, |b, &r| {
            b.iter(|| black_box(leiden(&graph, &LeidenConfig::with_resolution(r)).unwrap()));
        });
    }
    group.finish();
}

fn bench_objective(c: &mut Criterion) {
    let graph = knn_graph(20_000, 10, 15, 5);
    let mut group = c.benchmark_group("leiden/objective");
    group.sample_size(10);

    for (name, objective) in [
        ("rb", ObjectiveKind::Rb { resolution: 1.0 }),
        ("cpm", ObjectiveKind::Cpm { resolution: 0.05 }),
    ] {
        group.bench_function(name, |b| {
            let config = LeidenConfig {
                objective,
                ..Default::default()
            };
            b.iter(|| black_box(leiden(&graph, &config).unwrap()));
        });
    }
    group.finish();
}

fn bench_graph_construction(c: &mut Criterion) {
    let graph = knn_graph(100_000, 10, 15, 5);
    let edges: Vec<(usize, usize, f64)> = (0..graph.node_count())
        .flat_map(|v| {
            graph
                .neighbors(v)
                .filter(move |&(u, _)| v <= u)
                .map(move |(u, w)| (v, u, w))
        })
        .collect();

    let mut group = c.benchmark_group("graph");
    group.sample_size(10);
    group.throughput(Throughput::Elements(edges.len() as u64));
    group.bench_function("from_edges/100k", |b| {
        b.iter(|| black_box(CSRNetwork::from_edges(100_000, &edges).unwrap()));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_scaling,
    bench_resolution,
    bench_objective,
    bench_graph_construction
);
criterion_main!(benches);
