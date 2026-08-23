//! Measures how much of Leiden's runtime is node ordering rather than work.
//!
//! ```text
//! cargo run --release --no-default-features --example reorder -- 1000000
//! ```
//!
//! The inner loop is a scatter-gather over CSR, so scattered neighbour indices are cache
//! misses. Relabelling so neighbours sit near each other changes no arithmetic — the
//! clustering must come out identical up to the permutation — so the wall-time difference is
//! pure locality. Morton needs coordinates, BFS works on any graph.
//!
//! ```text
//!    n     morton   bfs    edge span (orig -> morton)
//!  200k     1.07x  1.01x    5555 -> 314
//!    1M     1.13x  1.09x   27778 -> 701
//! ```
//!
//! ~40x better locality buys 13%, and extrapolates to only ~1.2x at 8M. Real but small, which
//! is why parallelism was ranked above it. Scale-dependent, so re-measure before believing
//! this at 75M.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use single_clustering::community_search::leiden::{LeidenConfig, leiden, modularity};
use single_clustering::network::CSRNetwork;
use std::time::Instant;

/// Interleaves the low 21 bits of each coordinate: nearby points get nearby keys.
fn morton(x: u32, y: u32) -> u64 {
    fn spread(mut v: u64) -> u64 {
        v &= 0x1f_ffff;
        v = (v | (v << 32)) & 0x1f00000000ffff;
        v = (v | (v << 16)) & 0x1f0000ff0000ff;
        v = (v | (v << 8)) & 0x100f00f00f00f00f;
        v = (v | (v << 4)) & 0x10c30c30c30c30c3;
        (v | (v << 2)) & 0x1249249249249249
    }
    spread(x as u64) | (spread(y as u64) << 1)
}

type Edges = Vec<(usize, usize, f64)>;

/// Points in blobs joined to their k nearest, plus the coordinates themselves.
fn knn_graph(n: usize, blocks: usize, k: usize, seed: u64) -> (Edges, Vec<[f64; 2]>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_per = n / blocks;
    let centers: Vec<(f64, f64)> = (0..blocks)
        .map(|b| {
            let a = 2.0 * std::f64::consts::PI * (b as f64) / (blocks as f64);
            (40.0 * a.cos(), 40.0 * a.sin())
        })
        .collect();
    let pts: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let c = centers[(i / n_per).min(blocks - 1)];
            [
                c.0 + rng.random::<f64>() * 12.0 - 6.0,
                c.1 + rng.random::<f64>() * 12.0 - 6.0,
            ]
        })
        .collect();

    let cell = 3.0;
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, p) in pts.iter().enumerate() {
        grid.entry(((p[0] / cell) as i64, (p[1] / cell) as i64))
            .or_default()
            .push(i);
    }
    let mut edges = std::collections::HashSet::new();
    let mut cand = Vec::new();
    for i in 0..n {
        let (gx, gy) = ((pts[i][0] / cell) as i64, (pts[i][1] / cell) as i64);
        cand.clear();
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(b) = grid.get(&(gx + dx, gy + dy)) {
                    cand.extend(b.iter().copied().filter(|&j| j != i));
                }
            }
        }
        let d = |j: &usize| {
            let (dx, dy) = (pts[*j][0] - pts[i][0], pts[*j][1] - pts[i][1]);
            dx * dx + dy * dy
        };
        if cand.len() > k {
            cand.select_nth_unstable_by(k, |a, b| d(a).partial_cmp(&d(b)).unwrap());
            cand.truncate(k);
        }
        for &j in &cand {
            edges.insert(if i < j { (i, j) } else { (j, i) });
        }
    }
    (edges.into_iter().map(|(a, b)| (a, b, 1.0)).collect(), pts)
}

/// Applies a permutation to an edge list. `perm[old] = new`.
fn relabel(edges: &[(usize, usize, f64)], perm: &[usize]) -> Edges {
    edges
        .iter()
        .map(|&(a, b, w)| (perm[a], perm[b], w))
        .collect()
}

fn morton_permutation(pts: &[[f64; 2]]) -> Vec<usize> {
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in pts {
        for d in 0..2 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    let scale = |v: f64, d: usize| {
        let span = (hi[d] - lo[d]).max(1e-12);
        (((v - lo[d]) / span) * ((1u32 << 21) - 1) as f64) as u32
    };
    let mut order: Vec<usize> = (0..pts.len()).collect();
    order.sort_unstable_by_key(|&i| morton(scale(pts[i][0], 0), scale(pts[i][1], 1)));

    let mut perm = vec![0usize; pts.len()];
    for (new, &old) in order.iter().enumerate() {
        perm[old] = new;
    }
    perm
}

fn bfs_permutation(graph: &CSRNetwork) -> Vec<usize> {
    let n = graph.node_count();
    let mut perm = vec![usize::MAX; n];
    let mut next = 0usize;
    let mut queue = std::collections::VecDeque::new();
    for start in 0..n {
        if perm[start] != usize::MAX {
            continue;
        }
        perm[start] = next;
        next += 1;
        queue.push_back(start);
        while let Some(v) = queue.pop_front() {
            for (u, _) in graph.neighbors(v) {
                if perm[u] == usize::MAX {
                    perm[u] = next;
                    next += 1;
                    queue.push_back(u);
                }
            }
        }
    }
    perm
}

/// Mean absolute distance between an edge's endpoints in index space — the thing the
/// reordering is trying to shrink.
fn mean_edge_span(graph: &CSRNetwork) -> f64 {
    let mut total = 0.0f64;
    let mut count = 0usize;
    for v in 0..graph.node_count() {
        for (u, _) in graph.neighbors(v) {
            total += (u as f64 - v as f64).abs();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn time_clustering(graph: &CSRNetwork, label: &str, baseline: Option<f64>) -> f64 {
    let config = LeidenConfig::default();
    // One warm run, then the measured one, so allocator state is comparable.
    let _ = leiden(graph, &config).unwrap();
    let t = Instant::now();
    let c = leiden(graph, &config).unwrap();
    let secs = t.elapsed().as_secs_f64();

    let q = modularity(graph, c.labels(), 1.0);
    let speedup = baseline.map_or(String::from("—"), |b| format!("{:.2}x", b / secs));
    println!(
        "  {label:22} {:>8.3}s  {speedup:>7}   span {:>10.0}  Q {q:.4}  k {}",
        secs,
        mean_edge_span(graph),
        c.n_clusters()
    );
    secs
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    println!("building a {n}-node 15-NN graph...");
    let (edges, pts) = knn_graph(n, 12, 15, 7);
    println!("  {} edges\n", edges.len());

    println!(
        "  {:22} {:>8}  {:>7}   {:>15}",
        "ordering", "time", "vs orig", "locality"
    );
    let original = CSRNetwork::from_edges(n, &edges).unwrap();
    let base = time_clustering(&original, "as-generated", None);

    let perm = morton_permutation(&pts);
    let morton_graph = CSRNetwork::from_edges(n, &relabel(&edges, &perm)).unwrap();
    time_clustering(&morton_graph, "morton (coords)", Some(base));

    let perm = bfs_permutation(&original);
    let bfs_graph = CSRNetwork::from_edges(n, &relabel(&edges, &perm)).unwrap();
    time_clustering(&bfs_graph, "bfs (graph only)", Some(base));

    println!(
        "\nRelabelling changes no arithmetic, so any difference is locality alone.\n\
         Modularity should agree to f32 storage precision across all three."
    );
}
