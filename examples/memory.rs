//! Measures peak memory and time per node/edge, so cost at sizes too large to run here can be
//! extrapolated.
//!
//! ```text
//! cargo run --release --no-default-features --example memory -- 2000000
//! ```

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use single_clustering::community_search::leiden::{LeidenConfig, leiden};
use single_clustering::network::CSRNetwork;
use std::time::Instant;

/// Peak resident set size in bytes.
#[cfg(target_os = "macos")]
fn peak_rss() -> u64 {
    // getrusage(RUSAGE_SELF).ru_maxrss, in bytes on macOS
    unsafe {
        let mut usage: libc_rusage = std::mem::zeroed();
        rusage(0, &mut usage);
        usage.ru_maxrss as u64
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Default)]
struct libc_rusage {
    ru_utime: [i64; 2],
    ru_stime: [i64; 2],
    ru_maxrss: i64,
    rest: [i64; 14],
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    #[link_name = "getrusage"]
    fn rusage(who: i32, usage: *mut libc_rusage) -> i32;
}

#[cfg(not(target_os = "macos"))]
fn peak_rss() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
                .map(|kb| kb * 1024)
        })
        .unwrap_or(0)
}

/// A ring-lattice-with-noise graph: cheap to build at scale and structurally similar to a kNN
/// graph (bounded degree, local connectivity, a few long-range edges).
fn synthetic_knn(n: usize, k: usize, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut edges = Vec::with_capacity(n * k / 2 + n);
    let block = 2_000usize; // community size
    for i in 0..n {
        for d in 1..=(k / 2) {
            let j = i + d;
            if j < n && (i / block == j / block) {
                edges.push((i, j, 1.0f64));
            }
        }
        // one cross-block edge per node, keeping communities detectable
        if rng.random::<f64>() < 0.1 {
            let j = rng.random_range(0..n);
            if j != i {
                edges.push((i.min(j), i.max(j), 1.0f64));
            }
        }
    }
    edges
}

/// Builds the same graph as `synthetic_knn`, but emits symmetric CSR arrays directly — the
/// shape a kNN connectivity matrix already has. No edge list is ever materialised.
fn synthetic_knn_csr(n: usize, k: usize, seed: u64) -> (Vec<usize>, Vec<u32>, Vec<f32>) {
    // two streaming passes, count then fill, so no edge list is ever resident; the seeded
    // RNG makes both passes emit the same edges
    let emit = |mut f: Box<dyn FnMut(usize, usize, f32) + '_>| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let block = 2_000usize;
        for i in 0..n {
            for d in 1..=(k / 2) {
                let j = i + d;
                if j < n && (i / block == j / block) {
                    f(i, j, 1.0);
                }
            }
            if rng.random::<f64>() < 0.1 {
                let j = rng.random_range(0..n);
                if j != i {
                    f(i.min(j), i.max(j), 1.0);
                }
            }
        }
    };

    let mut counts = vec![0usize; n];
    emit(Box::new(|a, b, _| {
        counts[a] += 1;
        if a != b {
            counts[b] += 1;
        }
    }));

    let mut node_ptrs = vec![0usize; n + 1];
    for v in 0..n {
        node_ptrs[v + 1] = node_ptrs[v] + counts[v];
    }
    drop(counts);

    let slots = node_ptrs[n];
    let mut neighbors = vec![0u32; slots];
    let mut weights = vec![0.0f32; slots];
    let mut cursor = node_ptrs.clone();
    emit(Box::new(|a, b, w| {
        neighbors[cursor[a]] = b as u32;
        weights[cursor[a]] = w;
        cursor[a] += 1;
        if a != b {
            neighbors[cursor[b]] = a as u32;
            weights[cursor[b]] = w;
            cursor[b] += 1;
        }
    }));
    drop(cursor);
    (node_ptrs, neighbors, weights)
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);
    let k = 15;

    // Zero-copy path: hand over CSR buffers we already hold.
    if std::env::var("CSR_PATH").is_ok() {
        let t = Instant::now();
        let (node_ptrs, neighbors, weights) = synthetic_knn_csr(n, k, 7);
        let input_bytes = (node_ptrs.len() * 8 + neighbors.len() * 4 + weights.len() * 4) as u64;
        println!(
            "built symmetric CSR ({:.2} GB) in {:.2?}",
            gb(input_bytes),
            t.elapsed()
        );
        let t = Instant::now();
        let graph = CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None).unwrap();
        println!(
            "from_csr_parts: {:.2?}, peak RSS {:.2} GB, graph itself {:.2} GB",
            t.elapsed(),
            gb(peak_rss()),
            gb(graph.memory_bytes() as u64)
        );
        let t = Instant::now();
        let c = leiden(&graph, &LeidenConfig::default()).unwrap();
        println!(
            "leiden: {:.2?}, {} clusters, peak RSS {:.2} GB",
            t.elapsed(),
            c.n_clusters(),
            gb(peak_rss())
        );
        return;
    }

    let base = peak_rss();
    let t0 = Instant::now();
    let edges = synthetic_knn(n, k, 7);
    let edge_list_bytes = (edges.len() * std::mem::size_of::<(usize, usize, f64)>()) as u64;
    println!(
        "built {} edges for {n} nodes in {:.2?} (edge list itself: {:.2} GB)",
        edges.len(),
        t0.elapsed(),
        gb(edge_list_bytes)
    );

    let t1 = Instant::now();
    let graph = CSRNetwork::from_edges(n, &edges).unwrap();
    let after_build = peak_rss();
    println!(
        "CSRNetwork::from_edges: {:.2?}, peak RSS {:.2} GB, graph itself {:.2} GB",
        t1.elapsed(),
        gb(after_build),
        gb(graph.memory_bytes() as u64)
    );

    drop(edges);

    let t2 = Instant::now();
    let clustering = leiden(&graph, &LeidenConfig::default()).unwrap();
    let after_cluster = peak_rss();
    let cluster_time = t2.elapsed();

    println!(
        "leiden: {:.2?}, {} clusters, peak RSS {:.2} GB",
        cluster_time,
        clustering.n_clusters(),
        gb(after_cluster)
    );

    let m = graph.edge_count();
    println!();
    println!("--- per-element costs (for extrapolation) ---");
    println!("nodes {n}, undirected edges {m}");
    println!(
        "peak bytes/node {:.1}, peak bytes/edge {:.1}",
        (after_cluster - base) as f64 / n as f64,
        (after_cluster - base) as f64 / m as f64
    );
    println!(
        "clustering time: {:.1} ns/edge, {:.1} ns/node",
        cluster_time.as_nanos() as f64 / m as f64,
        cluster_time.as_nanos() as f64 / n as f64
    );
}
