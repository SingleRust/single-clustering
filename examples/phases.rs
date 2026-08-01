//! Splits one Leiden pass into its three phases and times each, per aggregation level.
//!
//! ```text
//! cargo run --release --no-default-features --example phases -- 1000000
//! ```
//!
//! Replicates the driver in `leiden::run_once` using the public API, so the phase split is
//! measured on exactly the work the real driver does. Answers "what is worth parallelising".

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use single_clustering::community_search::leiden::objective::{Cpm, Objective, Rb};
use single_clustering::community_search::leiden::{
    LeidenConfig, MoveScratch, ObjectiveKind, Partition, local_move, refine,
};
use single_clustering::network::CSRNetwork;
use std::time::{Duration, Instant};

/// Same generator as `examples/scaling.rs`: blobs joined to their k nearest neighbours.
fn knn_graph(n: usize, blocks: usize, k: usize, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_per = n / blocks;
    let centers: Vec<(f64, f64)> = (0..blocks)
        .map(|b| {
            let a = 2.0 * std::f64::consts::PI * (b as f64) / (blocks as f64);
            (40.0 * a.cos(), 40.0 * a.sin())
        })
        .collect();
    let pts: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let c = centers[(i / n_per).min(blocks - 1)];
            (
                c.0 + rng.random::<f64>() * 12.0 - 6.0,
                c.1 + rng.random::<f64>() * 12.0 - 6.0,
            )
        })
        .collect();

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
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(bucket) = grid.get(&(gx + dx, gy + dy)) {
                    cand.extend(bucket.iter().copied().filter(|&j| j != i));
                }
            }
        }
        let d = |j: &usize| {
            let (dx, dy) = (pts[*j].0 - pts[i].0, pts[*j].1 - pts[i].1);
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
    edges.into_iter().map(|(a, b)| (a, b, 1.0)).collect()
}

fn pct(d: Duration, total: Duration) -> f64 {
    100.0 * d.as_secs_f64() / total.as_secs_f64().max(1e-12)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    println!("generating {n}-node 15-NN graph...");
    let t = Instant::now();
    let edges = knn_graph(n, 12, 15, 7);
    let graph = CSRNetwork::from_edges(n, &edges).unwrap();
    drop(edges);
    println!(
        "  {} nodes, {} edges, {:.2} MB, built in {:.2?}\n",
        graph.node_count(),
        graph.edge_count(),
        graph.memory_bytes() as f64 / 1e6,
        t.elapsed()
    );

    let config = LeidenConfig::default();
    let objective: Box<dyn Objective> = match config.objective {
        ObjectiveKind::Rb { resolution } => Box::new(Rb::new(resolution)),
        ObjectiveKind::Cpm { resolution } => Box::new(Cpm::new(resolution)),
    };
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed.unwrap_or(42));

    let mut level_graph = graph.clone();
    let initial: Vec<usize> = (0..n).collect();
    let mut partition = Partition::from_membership(&level_graph, &initial);
    let mut scratch = MoveScratch::with_capacity(n, partition.slots().max(n) + 1);

    let (mut t_move, mut t_refine, mut t_agg) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);

    println!(
        "{:>5} {:>10} {:>10} | {:>10} {:>10} {:>10}",
        "lvl", "nodes", "edges", "move", "refine", "aggregate"
    );
    println!("{}", "-".repeat(64));

    let whole = Instant::now();
    for level in 0..config.max_levels {
        let nodes_before = level_graph.node_count();

        let a = Instant::now();
        local_move(
            &level_graph,
            &mut partition,
            objective.as_ref(),
            &mut scratch,
            &mut rng,
            config.max_community_weight,
        );
        let d_move = a.elapsed();

        let a = Instant::now();
        let mut collapse_by = refine(
            &level_graph,
            partition.membership_raw(),
            objective.as_ref(),
            &mut scratch,
            &mut rng,
            config.max_community_weight,
            config.refine_randomness,
        );
        let d_refine = a.elapsed();

        let a = Instant::now();
        let grouping = collapse_by.renumber_into_grouping();
        let coarse_graph = level_graph.aggregate(&grouping);
        let d_agg = a.elapsed();

        t_move += d_move;
        t_refine += d_refine;
        t_agg += d_agg;

        println!(
            "{level:>5} {:>10} {:>10} | {:>10.2?} {:>10.2?} {:>10.2?}",
            nodes_before,
            level_graph.edge_count(),
            d_move,
            d_refine,
            d_agg
        );

        if coarse_graph.node_count() >= nodes_before {
            break;
        }

        let mut coarse_membership = vec![0usize; coarse_graph.node_count()];
        for v in 0..nodes_before {
            coarse_membership[collapse_by.membership(v)] = partition.membership(v);
        }
        level_graph = coarse_graph;
        partition = Partition::from_membership(&level_graph, &coarse_membership);
    }
    let total = whole.elapsed();

    println!("{}", "-".repeat(64));
    println!(
        "{:>27} | {:>10.2?} {:>10.2?} {:>10.2?}",
        "total", t_move, t_refine, t_agg
    );
    println!(
        "{:>27} | {:>9.1}% {:>9.1}% {:>9.1}%",
        "share",
        pct(t_move, total),
        pct(t_refine, total),
        pct(t_agg, total)
    );
    println!("\none pass: {total:.2?}");
}
