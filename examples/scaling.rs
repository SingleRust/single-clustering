//! Times Leiden on synthetic kNN-like graphs of increasing size.
//!
//! ```text
//! cargo run --release --no-default-features --example scaling
//! ```
//!
//! Expect more communities than blobs — a uniform blob wired up by 15-NN is a sparse mesh,
//! not one dense community, and splitting it really does score higher than the planted
//! labelling. That's the objective working, not over-splitting.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use single_clustering::community_search::leiden::{LeidenConfig, leiden, modularity};
use single_clustering::network::CSRNetwork;
use std::time::Instant;

/// Points drawn around `blocks` well-separated centres, each joined to its `k` nearest
/// neighbours — the shape of a single-cell neighbourhood graph.
fn knn_graph(
    n_per: usize,
    blocks: usize,
    k: usize,
    seed: u64,
) -> (Vec<(usize, usize, f64)>, usize) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = n_per * blocks;
    let centers: Vec<(f64, f64)> = (0..blocks)
        .map(|b| {
            let a = 2.0 * std::f64::consts::PI * (b as f64) / (blocks as f64);
            (40.0 * a.cos(), 40.0 * a.sin())
        })
        .collect();
    let pts: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let c = centers[i / n_per];
            (
                c.0 + rng.random::<f64>() * 12.0 - 6.0,
                c.1 + rng.random::<f64>() * 12.0 - 6.0,
            )
        })
        .collect();

    // Grid-bucketed nearest neighbours, so graph construction does not dominate the timing.
    let cell = 3.0;
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, p) in pts.iter().enumerate() {
        grid.entry(((p.0 / cell) as i64, (p.1 / cell) as i64))
            .or_default()
            .push(i);
    }

    let mut edges = std::collections::HashSet::new();
    let mut cand = Vec::new();
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

    (edges.into_iter().map(|(a, b)| (a, b, 1.0)).collect(), n)
}

fn main() {
    println!(
        "{:>8}  {:>9}  {:>10}  {:>7}  {:>8}",
        "nodes", "edges", "time", "comms", "Q"
    );
    for &(n_per, blocks) in &[
        (250usize, 4usize),
        (500, 8),
        (1000, 10),
        (2000, 10),
        (10_000, 10),
    ] {
        let (edges, n) = knn_graph(n_per, blocks, 15, 5);
        let graph = CSRNetwork::from_edges(n, &edges).unwrap();

        let start = Instant::now();
        let clustering = leiden(&graph, &LeidenConfig::default()).unwrap();
        let elapsed = start.elapsed();

        println!(
            "{:>8}  {:>9}  {:>10.2?}  {:>7}  {:>8.4}",
            n,
            edges.len(),
            elapsed,
            clustering.n_clusters(),
            modularity(&graph, clustering.labels(), 1.0)
        );
    }
}
