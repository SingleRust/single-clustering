//! Upper bound on what multi-threading could buy, without writing any of it.
//!
//! ```text
//! cargo run --release --no-default-features --example scaling_ceiling -- 500000
//! ```
//!
//! Runs N *independent* clusterings concurrently and measures aggregate throughput against
//! one thread. Nothing is shared, so only the memory system can stop it scaling linearly —
//! which makes it a ceiling. A real parallel local-move shares state and can't beat it.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use single_clustering::community_search::leiden::{LeidenConfig, leiden};
use single_clustering::network::CSRNetwork;
use std::time::Instant;

fn synthetic(n: usize, k: usize, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut edges = Vec::with_capacity(n * k);
    for i in 0..n {
        for d in 1..=k / 2 {
            let j = (i + d) % n;
            if i != j {
                edges.push((i.min(j), i.max(j), 1.0));
            }
        }
        let j = rng.random_range(0..n);
        if i != j && i / 2000 != j / 2000 {
            edges.push((i.min(j), i.max(j), 1.0));
        }
    }
    edges.sort_unstable_by_key(|&(a, b, _)| (a, b));
    edges.dedup_by_key(|e| (e.0, e.1));
    edges
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000);
    let cores = std::thread::available_parallelism().map_or(8, |c| c.get());

    println!("building {n}-node graph, {cores} cores available");
    let edges = synthetic(n, 14, 7);
    let graph = CSRNetwork::from_edges(n, &edges).unwrap();
    drop(edges);
    println!(
        "  {} edges, {:.0} MB\n",
        graph.edge_count(),
        graph.memory_bytes() as f64 / 1e6
    );

    let config = LeidenConfig::default();
    let _ = leiden(&graph, &config).unwrap();

    println!(
        "{:>7} {:>10} {:>12} {:>10}",
        "threads", "wall", "throughput", "vs 1"
    );
    let mut base = 0.0f64;
    let mut t = 1;
    while t <= cores {
        // Each thread gets its own copy, so nothing is shared but the memory system.
        let copies: Vec<CSRNetwork> = (0..t).map(|_| graph.clone()).collect();

        let start = Instant::now();
        std::thread::scope(|s| {
            for g in &copies {
                let cfg = config.clone();
                s.spawn(move || {
                    let _ = leiden(g, &cfg).unwrap();
                });
            }
        });
        let wall = start.elapsed().as_secs_f64();
        let throughput = t as f64 / wall;
        if t == 1 {
            base = throughput;
        }
        println!(
            "{t:>7} {:>9.3}s {:>10.2}/s {:>9.2}x",
            wall,
            throughput,
            throughput / base
        );
        t *= 2;
    }

    println!(
        "\nThis is the ceiling: independent work, zero sharing. A parallel local-move shares\n\
         partition state, so it lands below whatever this shows."
    );
}
