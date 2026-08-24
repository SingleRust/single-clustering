//! Head-to-head against igraph, clustering only.
//!
//! ```text
//! cargo run --release --no-default-features --example vs_igraph -- 200000 /tmp/bench
//! uv run tools/igraph_bench.py /tmp/bench
//! ```
//!
//! Writes the graph to `<dir>/edges.tsv` so both sides cluster the identical input, and times
//! only `leiden()`. Construction is excluded on both sides — including it would measure the
//! pipeline, not the algorithm.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use single_clustering::community_search::leiden::{LeidenConfig, leiden, modularity};
use single_clustering::network::CSRNetwork;
use std::io::Write;
use std::time::Instant;

/// Ring lattice with long-range noise: bounded degree and local structure, like a kNN graph,
/// but generated in linear time so large sizes are practical.
fn synthetic(n: usize, k: usize, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let block = 2_000usize;
    let mut edges = Vec::with_capacity(n * k);
    for i in 0..n {
        for d in 1..=k / 2 {
            let j = (i + d) % n;
            if i != j {
                edges.push((i.min(j), i.max(j), 1.0));
            }
        }
        // one cross-block edge, so communities stay detectable
        let j = rng.random_range(0..n);
        let same_block = i / block == j / block;
        if i != j && !same_block {
            edges.push((i.min(j), i.max(j), 1.0));
        }
    }
    edges.sort_unstable_by_key(|&(a, b, _)| (a, b));
    edges.dedup_by_key(|e| (e.0, e.1));
    edges
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let dir = args.next().unwrap_or_else(|| "/tmp/bench".into());
    std::fs::create_dir_all(&dir).unwrap();

    let edges = synthetic(n, 14, 7);
    println!("{n} nodes, {} edges", edges.len());

    let path = format!("{dir}/edges.tsv");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    for &(a, b, _) in &edges {
        writeln!(out, "{a}\t{b}").unwrap();
    }
    out.flush().unwrap();
    drop(out);

    let graph = CSRNetwork::from_edges(n, &edges).unwrap();
    let config = LeidenConfig::default();

    // Warm once, then take the best of five: the fairest thing to compare against igraph's
    // own best, and least sensitive to a stray scheduling hiccup.
    let _ = leiden(&graph, &config).unwrap();
    let mut best = f64::INFINITY;
    let mut clustering = None;
    for _ in 0..5 {
        let t = Instant::now();
        let c = leiden(&graph, &config).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
        clustering = Some(c);
    }
    let c = clustering.unwrap();
    println!(
        "rust   {:.4}s  Q {:.4}  k {}",
        best,
        modularity(&graph, c.labels(), 1.0),
        c.n_clusters()
    );
    println!("wrote {path}");
}
