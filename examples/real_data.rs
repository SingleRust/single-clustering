//! Validates against a real single-cell dataset exported by `tools/export_h5ad.py`.
//!
//! ```text
//! uv run tools/export_h5ad.py --dataset pbmc3k --out /tmp/pbmc3k
//! cargo run --release --no-default-features --example real_data -- /tmp/pbmc3k
//! ```
//!
//! Clusters the *same* connectivity matrix scanpy did, so any difference is the algorithm,
//! not graph construction. Reports:
//!
//! * cluster count vs resolution, ours against scanpy's
//! * agreement with scanpy's partition (ARI / NMI)
//! * agreement with the author cell-type annotation — the only number here that's about
//!   biology rather than matching another implementation
//! * modularity and wall time

use single_clustering::community_search::leiden::{LeidenConfig, leiden, modularity};
use single_clustering::network::CSRNetwork;
use std::path::Path;
use std::time::Instant;

fn read_u64(path: &Path) -> Vec<usize> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Adjusted Rand Index: agreement between two partitions, corrected for chance.
///
/// The standard measure for comparing clusterings against a reference labelling. 1.0 is exact
/// agreement, 0.0 is what random labelling would score.
fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len() as f64;
    if n == 0.0 {
        return 1.0;
    }
    let ka = a.iter().max().map_or(0, |m| m + 1);
    let kb = b.iter().max().map_or(0, |m| m + 1);
    let mut joint = std::collections::HashMap::new();
    let (mut sa, mut sb) = (vec![0.0f64; ka], vec![0.0f64; kb]);
    for i in 0..a.len() {
        *joint.entry((a[i], b[i])).or_insert(0.0f64) += 1.0;
        sa[a[i]] += 1.0;
        sb[b[i]] += 1.0;
    }
    let comb2 = |x: f64| x * (x - 1.0) / 2.0;
    let sum_ij: f64 = joint.values().map(|&c| comb2(c)).sum();
    let sum_a: f64 = sa.iter().map(|&c| comb2(c)).sum();
    let sum_b: f64 = sb.iter().map(|&c| comb2(c)).sum();
    let total = comb2(n);
    let expected = sum_a * sum_b / total;
    let max = 0.5 * (sum_a + sum_b);
    if (max - expected).abs() < 1e-12 {
        return 1.0;
    }
    (sum_ij - expected) / (max - expected)
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
        *joint.entry((a[i], b[i])).or_insert(0.0f64) += 1.0;
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
            .expect("usage: real_data <export-dir>"),
    );

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();

    let indptr = read_u64(&dir.join("indptr.bin"));
    let indices = read_u32(&dir.join("indices.bin"));
    let data = read_f32(&dir.join("data.bin"));

    println!(
        "dataset {} — {} cells, {} directed entries, {}-NN, scanpy {}",
        meta["dataset"].as_str().unwrap(),
        meta["n_cells"],
        meta["n_entries"],
        meta["n_neighbors"],
        meta["scanpy"].as_str().unwrap()
    );

    // The zero-copy path: exactly what a pipeline would hand over.
    let build = Instant::now();
    let graph = CSRNetwork::from_csr_parts(indptr, indices, data, None)
        .expect("scanpy connectivities should be a valid symmetric graph");
    let build_time = build.elapsed();

    println!(
        "graph built in {:.2?}: {} nodes, {} undirected edges, {:.1} MB, self-loops: {}",
        build_time,
        graph.node_count(),
        graph.edge_count(),
        graph.memory_bytes() as f64 / 1e6,
        graph.has_self_loops()
    );
    graph
        .validate_symmetry()
        .expect("exhaustive symmetry check should pass on scanpy output");
    println!("exhaustive symmetry check: OK");

    let annotation: Option<Vec<usize>> = reference.get("annotation").map(|a| {
        a["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap().max(0) as usize)
            .collect()
    });
    if let Some(a) = reference.get("annotation") {
        println!(
            "author annotation '{}': {} cell types",
            a["name"].as_str().unwrap(),
            a["categories"].as_array().unwrap().len()
        );
    }

    println!();
    println!(
        "{:>6} | {:>7} {:>7} | {:>7} {:>7} | {:>8} {:>8} | {:>9} {:>9}",
        "res", "ours_k", "scanpy_k", "ARI", "NMI", "ourQ", "scanpyQ", "ours(ms)", "scanpy(ms)"
    );
    println!("{}", "-".repeat(96));

    // Keep the original key strings: Rust renders 1.0f64 as "1" while Python wrote "1.0", so
    // reconstructing the key from the parsed float would miss.
    let mut resolutions: Vec<(f64, String)> = reference["resolutions"]
        .as_object()
        .unwrap()
        .keys()
        .map(|k| (k.parse().unwrap(), k.clone()))
        .collect();
    resolutions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut best_ari_to_truth = 0.0f64;
    let mut best_ari_res = 0.0f64;
    let mut scanpy_best_ari = 0.0f64;

    for (res, key) in resolutions {
        let entry = &reference["resolutions"][&key];
        let scanpy_labels: Vec<usize> = entry["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let config = LeidenConfig::with_resolution(res);
        let t = Instant::now();
        let clustering = leiden(&graph, &config).unwrap();
        let elapsed = t.elapsed();

        let ours = clustering.labels();
        println!(
            "{:>6} | {:>7} {:>7} | {:>7.4} {:>7.4} | {:>8.4} {:>8.4} | {:>9.1} {:>9.1}",
            res,
            clustering.n_clusters(),
            entry["n_clusters"].as_u64().unwrap(),
            adjusted_rand_index(ours, &scanpy_labels),
            nmi(ours, &scanpy_labels),
            modularity(&graph, ours, res),
            modularity(&graph, &scanpy_labels, res),
            elapsed.as_secs_f64() * 1000.0,
            entry["seconds"].as_f64().unwrap() * 1000.0,
        );

        if let Some(truth) = &annotation {
            let ari = adjusted_rand_index(ours, truth);
            if ari > best_ari_to_truth {
                best_ari_to_truth = ari;
                best_ari_res = res;
            }
            scanpy_best_ari = scanpy_best_ari.max(adjusted_rand_index(&scanpy_labels, truth));
        }
    }

    if annotation.is_some() {
        println!();
        println!(
            "best agreement with author cell types: ours ARI {best_ari_to_truth:.4} \
             (at resolution {best_ari_res}), scanpy ARI {scanpy_best_ari:.4}"
        );
    }
}
