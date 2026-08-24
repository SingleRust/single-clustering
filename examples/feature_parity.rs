//! Prints a fingerprint of everything the `parallel` feature touches.
//!
//! Run it both ways and diff. Same numbers means the feature changes speed and nothing else.
//!
//! ```text
//! cargo run -q --release --no-default-features --example feature_parity > seq.txt
//! cargo run -q --release --no-default-features --features parallel --example feature_parity > par.txt
//! diff seq.txt par.txt
//! ```

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use single_clustering::community_search::leiden::{
    LeidenConfig, ObjectiveKind, leiden, modularity,
};
use single_clustering::network::CSRNetwork;
use single_clustering::spatial::{SpatialWeight, Symmetry, knn_graph, radius_graph};

fn hash(vals: impl Iterator<Item = u64>) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for v in vals {
        h = (h ^ v).wrapping_mul(0x100000001b3);
    }
    h
}

fn main() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);

    // clustering: ring lattice + noise
    let n = 60_000usize;
    let mut edges = Vec::new();
    for v in 0..n {
        for d in 1..=5 {
            edges.push((v, (v + d) % n, 1.0));
        }
        for _ in 0..2 {
            edges.push((v, rng.random_range(0..n), 1.0));
        }
    }
    let g = CSRNetwork::from_edges(n, &edges).unwrap();
    for obj in [
        ObjectiveKind::Rb { resolution: 1.0 },
        ObjectiveKind::Cpm { resolution: 0.05 },
    ] {
        for seed in [1u64, 7] {
            let c = leiden(
                &g,
                &LeidenConfig {
                    objective: obj,
                    seed: Some(seed),
                    ..Default::default()
                },
            )
            .unwrap();
            println!(
                "leiden {obj:?} seed={seed}: k={} labels={:016x} q={:.12}",
                c.n_clusters(),
                hash(c.labels().iter().map(|&l| l as u64)),
                modularity(&g, c.labels(), 1.0)
            );
        }
    }

    // spatial builders
    let pts: Vec<[f64; 2]> = (0..40_000)
        .map(|_| [rng.random::<f64>() * 500.0, rng.random::<f64>() * 500.0])
        .collect();
    for (name, gr) in [
        (
            "knn",
            knn_graph(&pts, 8, Symmetry::Union, SpatialWeight::Uniform).unwrap(),
        ),
        (
            "radius",
            radius_graph(&pts, 6.0, SpatialWeight::Gaussian { sigma: 3.0 }, None).unwrap(),
        ),
    ] {
        let (ptrs, nb, w) = gr.to_csr_parts();
        println!(
            "{name}: e={} ptrs={:016x} nb={:016x} w={:016x}",
            gr.edge_count(),
            hash(ptrs.iter().map(|&p| p as u64)),
            hash(nb.iter().map(|&x| x as u64)),
            hash(w.iter().map(|&x| x.to_bits() as u64)),
        );
    }
}
