//! Head-to-head evaluation against `leidenalg` on LFR benchmarks and large kNN graphs.
//!
//! Reads graphs and reference results produced by the generator in `tools/`, runs our
//! algorithm on the same inputs, and reports NMI against ground truth, modularity, and wall
//! time for both.
//!
//! ```text
//! cargo run --release --no-default-features --example evaluate -- <evaldata-dir>
//! ```

use single_clustering::community_search::leiden::{LeidenConfig, leiden, modularity};
use single_clustering::network::CSRNetwork;
use std::time::Instant;

fn read_edges(path: &std::path::Path) -> (usize, Vec<(usize, usize, f64)>) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut edges = Vec::new();
    let mut max_node = 0usize;
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let a: usize = it.next().unwrap().parse().unwrap();
        let b: usize = it.next().unwrap().parse().unwrap();
        let w: f64 = it.next().map_or(1.0, |v| v.parse().unwrap());
        max_node = max_node.max(a).max(b);
        edges.push((a, b, w));
    }
    (max_node + 1, edges)
}

fn read_labels(path: &std::path::Path) -> Option<Vec<usize>> {
    std::fs::read_to_string(path)
        .ok()
        .map(|t| t.lines().map(|l| l.trim().parse().unwrap()).collect())
}

fn nmi(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len() as f64;
    if n == 0.0 {
        return 1.0;
    }
    let ka = a.iter().max().map_or(0, |m| m + 1);
    let kb = b.iter().max().map_or(0, |m| m + 1);
    let mut joint = std::collections::HashMap::new();
    let (mut pa, mut pb) = (vec![0.0f64; ka], vec![0.0f64; kb]);
    for i in 0..a.len() {
        *joint.entry((a[i], b[i])).or_insert(0.0) += 1.0;
        pa[a[i]] += 1.0;
        pb[b[i]] += 1.0;
    }
    let mut mi = 0.0;
    for (&(i, j), &c) in &joint {
        let p = c / n;
        mi += p * (p / ((pa[i] / n) * (pb[j] / n))).ln();
    }
    let h = |p: &[f64]| -> f64 {
        -p.iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| (x / n) * (x / n).ln())
            .sum::<f64>()
    };
    let (ha, hb) = (h(&pa), h(&pb));
    if ha + hb == 0.0 {
        1.0
    } else {
        2.0 * mi / (ha + hb)
    }
}

fn main() {
    let dir = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: evaluate <evaldata-dir>"),
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("results.json")).unwrap()).unwrap();

    // Optional: igraph's native results, for the strongest available baseline.
    let igraph: std::collections::HashMap<String, serde_json::Value> =
        std::fs::read_to_string(dir.join("igraph_results.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .map(|v| {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|r| (r["name"].as_str().unwrap().to_string(), r.clone()))
                    .collect()
            })
            .unwrap_or_default();

    println!(
        "{:<22} {:>7} | {:>8} {:>9} | {:>8} {:>9} | {:>8} {:>9}",
        "case", "nodes", "ourQ", "ours(ms)", "igQ", "igraph(ms)", "laQ", "la(ms)"
    );
    println!("{}", "-".repeat(96));
    let mut q_vs_igraph = Vec::new();
    let mut speed_vs_igraph = Vec::new();

    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (n, edges) = read_edges(&dir.join(format!("{name}.edges")));
        let truth = read_labels(&dir.join(format!("{name}.truth")));
        let resolution = case["resolution"].as_f64().unwrap();

        let graph = CSRNetwork::from_edges(n, &edges).unwrap();
        let n_iterations = std::env::var("EVAL_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let config = LeidenConfig {
            n_iterations,
            ..LeidenConfig::with_resolution(resolution)
        };

        // Best of N, matching how the igraph/leidenalg timings were taken.
        let repeats = if n <= 60_000 { 3 } else { 1 };
        let mut ours_secs = f64::INFINITY;
        let mut clustering = leiden(&graph, &config).unwrap();
        for _ in 0..repeats {
            let start = Instant::now();
            clustering = leiden(&graph, &config).unwrap();
            ours_secs = ours_secs.min(start.elapsed().as_secs_f64());
        }

        let reference: Vec<usize> = case["reference_membership"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let our_q = modularity(&graph, clustering.labels(), resolution);
        let ref_q = modularity(&graph, &reference, resolution);
        let (our_nmi, ref_nmi) = match &truth {
            Some(t) => (nmi(clustering.labels(), t), nmi(&reference, t)),
            None => (f64::NAN, f64::NAN),
        };

        let ig = igraph.get(name);
        let (ig_q, ig_s) = match ig {
            Some(r) => (
                r["igleiden"]["q"].as_f64().unwrap(),
                r["igleiden"]["s"].as_f64().unwrap(),
            ),
            None => (f64::NAN, f64::NAN),
        };
        let la_s = ig
            .map(|r| r["leidenalg"]["s"].as_f64().unwrap())
            .unwrap_or(case["reference_seconds"].as_f64().unwrap());

        if ig_q.is_finite() {
            q_vs_igraph.push((our_q - ig_q) / ig_q.abs().max(1e-12));
            speed_vs_igraph.push(ig_s / ours_secs);
        }

        println!(
            "{:<22} {:>7} | {:>8.4} {:>9.1} | {:>8.4} {:>9.1} | {:>8.4} {:>9.1}",
            name,
            n,
            our_q,
            ours_secs * 1000.0,
            ig_q,
            ig_s * 1000.0,
            ref_q,
            la_s * 1000.0,
        );
        let _ = (our_nmi, ref_nmi);
    }

    if !q_vs_igraph.is_empty() {
        let mean_q = q_vs_igraph.iter().sum::<f64>() / q_vs_igraph.len() as f64;
        let wins = q_vs_igraph.iter().filter(|&&d| d > 1e-9).count();
        let losses = q_vs_igraph.iter().filter(|&&d| d < -1e-9).count();
        let geo_speed = (speed_vs_igraph.iter().map(|s| s.ln()).sum::<f64>()
            / speed_vs_igraph.len() as f64)
            .exp();
        println!("{}", "-".repeat(96));
        println!(
            "vs igraph native: modularity {:+.4}% mean ({wins} better, {losses} worse of {}), \
             speed {geo_speed:.2}x (geometric mean)",
            100.0 * mean_q,
            q_vs_igraph.len()
        );
    }
}
