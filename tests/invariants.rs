//! Properties that must hold for any input, exercised through the public API.
//!
//! Each of the four below maps to a bug that shipped in 0.6.x. Cheap, no fixtures needed, and
//! any one failing means the algorithm is wrong rather than unlucky.

use single_clustering::community_search::leiden::{
    Cpm, LeidenConfig, Objective, ObjectiveKind, Partition, Rb, leiden,
};
use single_clustering::network::CSRNetwork;

mod common;
use common::{disconnected_community_count, sbm, test_graphs};

/// Invariant 1: collapsing communities must not change what a partition scores. Broken in
/// 0.6.x, where a self-loop added its weight once instead of twice to strength, switching the
/// null model off above level 0.
///
/// Holds to stored-weight precision. Adjacency is `f32`, so a collapsed self-loop carrying a
/// whole community's internal weight rounds once per level — `1e-6` relative, about 10x f32
/// epsilon. The bug this guards against was a factor of ~1.75, not 1e-8.
#[test]
fn quality_is_invariant_under_aggregation() {
    for (name, graph) in test_graphs() {
        if graph.node_count() == 0 {
            continue;
        }
        let clustering = leiden(&graph, &LeidenConfig::default()).unwrap();
        let mut partition = Partition::from_membership(&graph, clustering.labels());

        let grouping = partition.renumber_into_grouping();
        let coarse = graph.aggregate(&grouping);
        let coarse_labels: Vec<usize> = (0..coarse.node_count()).collect();
        let coarse_partition = Partition::from_membership(&coarse, &coarse_labels);

        for gamma in [0.25, 1.0, 4.0] {
            let fine = Rb::new(gamma).quality(&partition);
            let agg = Rb::new(gamma).quality(&coarse_partition);
            assert!(
                (fine - agg).abs() < 1e-6 * fine.abs().max(1.0),
                "{name} at gamma={gamma}: fine {fine} != aggregated {agg}"
            );
        }
        let fine = Cpm::new(0.05).quality(&partition);
        let agg = Cpm::new(0.05).quality(&coarse_partition);
        assert!(
            (fine - agg).abs() < 1e-6 * fine.abs().max(1.0),
            "{name} under CPM: fine {fine} != aggregated {agg}"
        );
    }
}

/// Invariant 2: the graph's degree sum must equal twice its total edge weight, before and
/// after aggregation. The cheapest possible guard on the self-loop convention.
#[test]
fn degree_sum_matches_total_weight() {
    for (name, graph) in test_graphs() {
        let check = |g: &CSRNetwork, label: &str| {
            let sum: f64 = (0..g.node_count()).map(|v| g.strength(v)).sum();
            let expected = 2.0 * g.total_weight();
            assert!(
                (sum - expected).abs() < 1e-9 * expected.abs().max(1.0),
                "{name} ({label}): degree sum {sum} != 2*total_weight {expected}"
            );
        };
        check(&graph, "original");

        if graph.node_count() > 0 {
            let clustering = leiden(&graph, &LeidenConfig::default()).unwrap();
            let mut partition = Partition::from_membership(&graph, clustering.labels());
            let coarse = graph.aggregate(&partition.renumber_into_grouping());
            check(&coarse, "aggregated");
            let (fine_w, coarse_w) = (graph.total_weight(), coarse.total_weight());
            assert!(
                (fine_w - coarse_w).abs() < 1e-6 * fine_w.abs().max(1.0),
                "{name}: aggregation changed total weight, {fine_w} -> {coarse_w}"
            );
        }
    }
}

/// Invariant 3: every community internally connected — the guarantee refinement exists for.
/// 0.6.x lost it by hanging the "only singletons may merge" guard off the wrong statement.
#[test]
fn communities_are_internally_connected() {
    for (name, graph) in test_graphs() {
        for seed in 0..5u64 {
            for objective in [
                ObjectiveKind::Rb { resolution: 0.5 },
                ObjectiveKind::Rb { resolution: 1.0 },
                ObjectiveKind::Rb { resolution: 3.0 },
                ObjectiveKind::Cpm { resolution: 0.05 },
            ] {
                let clustering = leiden(
                    &graph,
                    &LeidenConfig {
                        objective,
                        seed: Some(seed),
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_eq!(
                    disconnected_community_count(&graph, clustering.labels()),
                    0,
                    "{name} (seed {seed}, {objective:?}) produced a disconnected community"
                );
            }
        }
    }
}

/// Invariant 4: a fixed seed gives bit-identical results. 0.6.x iterated candidates out of a
/// `HashSet`, whose order varies per process, so `seed: Some(42)` still drifted run to run.
#[test]
fn a_fixed_seed_is_reproducible() {
    for (name, graph) in test_graphs() {
        let config = LeidenConfig {
            seed: Some(7),
            ..Default::default()
        };
        let first = leiden(&graph, &config).unwrap();
        for attempt in 0..10 {
            let again = leiden(&graph, &config).unwrap();
            assert_eq!(first, again, "{name} differed on attempt {attempt}");
        }
    }
}

/// The optimizer must never return a partition worse than the singleton partition it starts
/// from — a blunt check that the reported search actually improves the objective.
#[test]
fn result_beats_the_starting_point() {
    for (name, graph) in test_graphs() {
        if graph.node_count() == 0 {
            continue;
        }
        for gamma in [0.5, 1.0, 2.0] {
            let objective = Rb::new(gamma);
            let singletons: Vec<usize> = (0..graph.node_count()).collect();
            let start = objective.quality(&Partition::from_membership(&graph, &singletons));

            let clustering = leiden(&graph, &LeidenConfig::with_resolution(gamma)).unwrap();
            let end = objective.quality(&Partition::from_membership(&graph, clustering.labels()));
            assert!(
                end >= start - 1e-9,
                "{name} at gamma={gamma}: ended at {end}, worse than singletons {start}"
            );
        }
    }
}

/// More resolution must not mean fewer communities.
#[test]
fn cluster_count_is_monotone_in_resolution() {
    let (edges, _) = sbm(40, 5, 0.30, 0.02, 4);
    let graph = CSRNetwork::from_edges(200, &edges).unwrap();

    let counts: Vec<usize> = [0.1, 0.5, 1.0, 2.0, 8.0, 32.0]
        .iter()
        .map(|&r| {
            leiden(&graph, &LeidenConfig::with_resolution(r))
                .unwrap()
                .n_clusters()
        })
        .collect();

    assert!(
        counts.windows(2).all(|w| w[0] <= w[1]),
        "cluster counts should be non-decreasing in resolution: {counts:?}"
    );
    assert!(
        counts[0] < *counts.last().unwrap(),
        "resolution had no effect at all: {counts:?}"
    );
}

/// Labels must always be consecutive from 0 and cover every node.
#[test]
fn labels_are_well_formed() {
    for (name, graph) in test_graphs() {
        let clustering = leiden(&graph, &LeidenConfig::default()).unwrap();
        assert_eq!(clustering.len(), graph.node_count(), "{name}: wrong length");
        assert_eq!(
            clustering.n_clusters(),
            clustering
                .labels()
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            "{name}: n_clusters disagrees with distinct labels"
        );
        for &l in clustering.labels() {
            assert!(
                l < clustering.n_clusters(),
                "{name}: label {l} out of range"
            );
        }
        let sizes = clustering.cluster_sizes();
        assert_eq!(sizes.iter().sum::<usize>(), graph.node_count());
        assert!(
            sizes.iter().all(|&s| s > 0),
            "{name}: an empty cluster survived"
        );
    }
}

/// Nodes in different connected components must never share a community.
#[test]
fn connected_components_are_never_merged() {
    // three cliques with no edges between them
    let mut edges = Vec::new();
    for b in 0..3usize {
        for i in 0..6 {
            for j in (i + 1)..6 {
                edges.push((b * 6 + i, b * 6 + j, 1.0));
            }
        }
    }
    let graph = CSRNetwork::from_edges(18, &edges).unwrap();

    for seed in 0..10u64 {
        let clustering = leiden(
            &graph,
            &LeidenConfig {
                seed: Some(seed),
                ..Default::default()
            },
        )
        .unwrap();
        let labels = clustering.labels();
        for a in 0..18 {
            for b in 0..18 {
                if a / 6 != b / 6 {
                    assert_ne!(
                        labels[a], labels[b],
                        "seed {seed}: nodes {a} and {b} are in different components"
                    );
                }
            }
        }
    }
}
