//! Results must not depend on how the work was divided.
//!
//! The one test the parallel path really needs. A staleness bug doesn't crash — it returns a
//! slightly different, entirely plausible clustering, and comparing across thread counts is
//! what makes that visible.
//!
//! Same inputs under 1, 2, 4, 8 and 16 rayon threads, demanding byte-identical labels. Only
//! meaningful with the `parallel` feature, which is on by default; without it the crate
//! ignores thread count entirely and this passes trivially.

mod common;

use common::{karate, sbm};
use single_clustering::community_search::leiden::{LeidenConfig, ObjectiveKind, leiden};
use single_clustering::network::CSRNetwork;

/// Clusters `graph` inside a rayon pool of exactly `threads` workers.
fn labels_with_threads(graph: &CSRNetwork, config: &LeidenConfig, threads: usize) -> Vec<usize> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool");
    pool.install(|| leiden(graph, config).unwrap().labels().to_vec())
}

fn graphs() -> Vec<(String, CSRNetwork)> {
    let mut out = vec![("karate".to_string(), karate())];
    for (name, n_per, blocks, p_in, p_out, seed) in [
        ("sbm_small", 40, 5, 0.30, 0.02, 4u64),
        ("sbm_mid", 80, 8, 0.25, 0.01, 11),
        ("sbm_sparse", 60, 6, 0.12, 0.02, 19),
    ] {
        let (edges, _) = sbm(n_per, blocks, p_in, p_out, seed);
        out.push((
            name.to_string(),
            CSRNetwork::from_edges(n_per * blocks, &edges).unwrap(),
        ));
    }
    out
}

#[test]
fn labels_are_identical_across_thread_counts() {
    let configs = [
        ("rb_1.0", ObjectiveKind::Rb { resolution: 1.0 }),
        ("rb_0.5", ObjectiveKind::Rb { resolution: 0.5 }),
        ("cpm_0.1", ObjectiveKind::Cpm { resolution: 0.1 }),
    ];

    for (gname, graph) in graphs() {
        for (cname, objective) in configs {
            for seed in [1u64, 42] {
                let config = LeidenConfig {
                    objective,
                    seed: Some(seed),
                    ..Default::default()
                };
                let reference = labels_with_threads(&graph, &config, 1);
                for threads in [2usize, 4, 8, 16] {
                    let got = labels_with_threads(&graph, &config, threads);
                    assert_eq!(
                        got, reference,
                        "{gname}/{cname}/seed {seed}: {threads} threads disagreed with 1"
                    );
                }
            }
        }
    }
}

/// Refinement randomness is the part most likely to depend on scheduling, so exercise it
/// specifically — including the greedy `0.0` case and a strongly exploratory one.
#[test]
fn randomised_refinement_is_thread_independent() {
    let (edges, _) = sbm(50, 6, 0.28, 0.02, 7);
    let graph = CSRNetwork::from_edges(300, &edges).unwrap();

    for randomness in [0.0, 0.01, 0.5] {
        let config = LeidenConfig {
            refine_randomness: randomness,
            seed: Some(5),
            ..Default::default()
        };
        let reference = labels_with_threads(&graph, &config, 1);
        for threads in [2usize, 8, 16] {
            assert_eq!(
                labels_with_threads(&graph, &config, threads),
                reference,
                "randomness {randomness}: {threads} threads disagreed with 1"
            );
        }
    }
}
