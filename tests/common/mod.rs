//! Test fixtures shared by the integration tests. Each binary uses a different subset.
#![allow(dead_code, unused_imports)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use single_clustering::network::CSRNetwork;

pub use single_clustering::testdata::{
    disconnected_community_count, karate, karate_edges, nmi, sbm,
};

/// A varied set of graphs, including the degenerate shapes that tend to break assumptions.
pub fn test_graphs() -> Vec<(String, CSRNetwork)> {
    let none: &[(usize, usize, f64)] = &[];
    let mut out: Vec<(String, CSRNetwork)> = vec![
        ("empty".into(), CSRNetwork::from_edges(0, none).unwrap()),
        (
            "single_node".into(),
            CSRNetwork::from_edges(1, none).unwrap(),
        ),
        ("no_edges".into(), CSRNetwork::from_edges(8, none).unwrap()),
        (
            "one_edge".into(),
            CSRNetwork::from_edges(2, &[(0, 1, 1.0)]).unwrap(),
        ),
        (
            "self_loops_only".into(),
            CSRNetwork::from_edges(4, &[(0, 0, 1.0), (1, 1, 2.0), (2, 2, 0.5)]).unwrap(),
        ),
        (
            "with_self_loops".into(),
            CSRNetwork::from_edges(
                5,
                &[
                    (0, 1, 1.0),
                    (1, 2, 1.0),
                    (0, 0, 3.0),
                    (3, 4, 1.0),
                    (3, 3, 2.0),
                ],
            )
            .unwrap(),
        ),
        (
            "star".into(),
            CSRNetwork::from_edges(9, &(1..9).map(|i| (0usize, i, 1.0)).collect::<Vec<_>>())
                .unwrap(),
        ),
        (
            "path".into(),
            CSRNetwork::from_edges(12, &(0..11).map(|i| (i, i + 1, 1.0)).collect::<Vec<_>>())
                .unwrap(),
        ),
        ("karate".into(), karate()),
    ];

    // weighted karate, so the weighted paths get the same coverage
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let weighted: Vec<(usize, usize, f64)> = karate_edges()
        .into_iter()
        .map(|(a, b, _)| (a, b, 0.25 + 3.5 * rng.random::<f64>()))
        .collect();
    out.push((
        "karate_weighted".into(),
        CSRNetwork::from_edges(34, &weighted).unwrap(),
    ));

    let (edges, _) = sbm(25, 4, 0.35, 0.03, 1);
    out.push(("sbm".into(), CSRNetwork::from_edges(100, &edges).unwrap()));

    // three cliques, fully disconnected from each other
    let mut edges = Vec::new();
    for b in 0..3usize {
        for i in 0..5 {
            for j in (i + 1)..5 {
                edges.push((b * 5 + i, b * 5 + j, 1.0));
            }
        }
    }
    out.push((
        "disconnected_cliques".into(),
        CSRNetwork::from_edges(15, &edges).unwrap(),
    ));

    out
}

/// Adjusted Rand Index: agreement between two labellings, corrected for chance.
///
/// 1.0 is exact agreement, 0.0 is what random labelling scores. Unlike NMI it does not
/// reward splitting, so a partition that shatters one of the two is penalised.
pub fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len() as f64;
    if a.len() < 2 {
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
    let expected = sum_a * sum_b / comb2(n);
    let max = 0.5 * (sum_a + sum_b);
    if (max - expected).abs() < 1e-12 {
        return 1.0;
    }
    (sum_ij - expected) / (max - expected)
}
