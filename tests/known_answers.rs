//! Graphs whose right answer is known independently of any implementation.

use single_clustering::community_search::leiden::{
    LeidenConfig, ObjectiveKind, leiden, modularity,
};
use single_clustering::network::CSRNetwork;

mod common;
use common::{karate, nmi, sbm};

/// Zachary's karate club has a known modularity optimum of Q = 0.4198 with 4 communities.
#[test]
fn karate_club_reaches_the_known_optimum() {
    let graph = karate();
    let clustering = leiden(&graph, &LeidenConfig::default()).unwrap();
    let q = modularity(&graph, clustering.labels(), 1.0);

    assert_eq!(
        clustering.n_clusters(),
        4,
        "sizes: {:?}",
        clustering.cluster_sizes()
    );
    assert!(
        (q - 0.4198).abs() < 1e-3,
        "expected Q = 0.4198, got {q} with {} communities",
        clustering.n_clusters()
    );
}

/// The karate club's classic 2-way split (the factions the club actually broke into) should
/// appear at a lower resolution.
#[test]
fn karate_club_splits_in_two_at_low_resolution() {
    let graph = karate();
    let clustering = leiden(&graph, &LeidenConfig::with_resolution(0.5)).unwrap();
    assert!(
        (2..=3).contains(&clustering.n_clusters()),
        "expected the 2-faction split, got {} communities",
        clustering.n_clusters()
    );
}

/// A well-separated planted partition must be recovered essentially exactly.
#[test]
fn recovers_a_well_separated_planted_partition() {
    for (n_per, blocks, p_in, p_out, seed) in [
        (60, 6, 0.25, 0.01, 11),
        (40, 4, 0.35, 0.02, 5),
        (80, 3, 0.20, 0.01, 9),
    ] {
        let (edges, truth) = sbm(n_per, blocks, p_in, p_out, seed);
        let graph = CSRNetwork::from_edges(n_per * blocks, &edges).unwrap();
        let clustering = leiden(&graph, &LeidenConfig::default()).unwrap();

        let score = nmi(clustering.labels(), &truth);
        assert!(
            score > 0.95,
            "{blocks} blocks of {n_per}: NMI {score}, {} clusters",
            clustering.n_clusters()
        );
        assert_eq!(
            clustering.n_clusters(),
            blocks,
            "{blocks} blocks of {n_per}: found {} clusters",
            clustering.n_clusters()
        );
    }
}

/// A found partition should never score worse than the planted one it is meant to recover.
#[test]
fn found_partition_scores_at_least_the_planted_one() {
    let (edges, truth) = sbm(60, 6, 0.25, 0.01, 11);
    let graph = CSRNetwork::from_edges(360, &edges).unwrap();
    let clustering = leiden(&graph, &LeidenConfig::default()).unwrap();

    let found = modularity(&graph, clustering.labels(), 1.0);
    let planted = modularity(&graph, &truth, 1.0);
    assert!(
        found >= planted - 1e-9,
        "found Q={found} is below the planted Q={planted}"
    );
}

/// A ring of cliques is the textbook resolution-limit case: modularity merges adjacent
/// cliques, CPM does not.
#[test]
fn cpm_avoids_the_resolution_limit() {
    let n_cliques = 20;
    let size = 5;
    let mut edges = Vec::new();
    for c in 0..n_cliques {
        let base = c * size;
        for i in 0..size {
            for j in (i + 1)..size {
                edges.push((base + i, base + j, 1.0));
            }
        }
        edges.push((base, ((c + 1) % n_cliques) * size, 1.0));
    }
    let graph = CSRNetwork::from_edges(n_cliques * size, &edges).unwrap();

    let cpm = leiden(
        &graph,
        &LeidenConfig {
            objective: ObjectiveKind::Cpm { resolution: 0.1 },
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        cpm.n_clusters(),
        n_cliques,
        "CPM should recover every clique, got {:?}",
        cpm.cluster_sizes()
    );
}

/// Degenerate inputs must produce sensible answers rather than panicking.
#[test]
fn degenerate_graphs_behave() {
    let none: &[(usize, usize, f64)] = &[];

    let empty = leiden(
        &CSRNetwork::from_edges(0, none).unwrap(),
        &LeidenConfig::default(),
    )
    .unwrap();
    assert_eq!(empty.n_clusters(), 0);
    assert!(empty.is_empty());

    let single = leiden(
        &CSRNetwork::from_edges(1, none).unwrap(),
        &LeidenConfig::default(),
    )
    .unwrap();
    assert_eq!(single.n_clusters(), 1);

    // Isolated nodes each form their own community, since merging them cannot help.
    let isolated = leiden(
        &CSRNetwork::from_edges(6, none).unwrap(),
        &LeidenConfig::default(),
    )
    .unwrap();
    assert_eq!(isolated.n_clusters(), 6);

    // A graph of nothing but self-loops likewise stays fully split.
    let loops = CSRNetwork::from_edges(3, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]).unwrap();
    let clustered = leiden(&loops, &LeidenConfig::default()).unwrap();
    assert_eq!(clustered.n_clusters(), 3);
}

/// Scaling every edge weight by the same factor must not change the partition under RB, whose
/// null model is scale-free.
#[test]
fn rb_is_invariant_to_uniform_edge_scaling() {
    let (edges, _) = sbm(30, 4, 0.30, 0.03, 2);
    let graph = CSRNetwork::from_edges(120, &edges).unwrap();
    let scaled: Vec<(usize, usize, f64)> =
        edges.iter().map(|&(a, b, w)| (a, b, w * 17.5)).collect();
    let scaled_graph = CSRNetwork::from_edges(120, &scaled).unwrap();

    let a = leiden(&graph, &LeidenConfig::default()).unwrap();
    let b = leiden(&scaled_graph, &LeidenConfig::default()).unwrap();
    assert_eq!(
        a.labels(),
        b.labels(),
        "uniform scaling changed the partition"
    );
}

/// Relabelling nodes must not change the partition, only its labels.
#[test]
fn results_are_stable_under_node_relabelling() {
    let (edges, _) = sbm(25, 4, 0.35, 0.02, 6);
    let n = 100;
    let graph = CSRNetwork::from_edges(n, &edges).unwrap();

    // reverse the node numbering
    let permuted: Vec<(usize, usize, f64)> = edges
        .iter()
        .map(|&(a, b, w)| (n - 1 - a, n - 1 - b, w))
        .collect();
    let permuted_graph = CSRNetwork::from_edges(n, &permuted).unwrap();

    let a = leiden(&graph, &LeidenConfig::default()).unwrap();
    let b = leiden(&permuted_graph, &LeidenConfig::default()).unwrap();

    // The RNG sees a different node order, so labels differ; the grouping should not.
    let a_labels = a.labels().to_vec();
    let b_reversed: Vec<usize> = b.labels().iter().rev().copied().collect();
    assert!(
        nmi(&a_labels, &b_reversed) > 0.95,
        "relabelling changed the grouping: NMI {}",
        nmi(&a_labels, &b_reversed)
    );
}
