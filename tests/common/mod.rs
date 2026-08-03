//! Graph fixtures and metrics shared by the integration tests.
//!
//! Included by several test binaries, each of which uses a different subset.
#![allow(dead_code)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use single_clustering::network::CSRNetwork;

/// Edges of Zachary's karate club.
pub fn karate_edges() -> Vec<(usize, usize, f64)> {
    // dense on purpose — one tuple per line is unreadable
    #[rustfmt::skip]
    const E: &[(usize, usize)] = &[
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11),
        (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13),
        (1, 17), (1, 19), (1, 21), (1, 30), (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27),
        (2, 28), (2, 32), (3, 7), (3, 12), (3, 13), (4, 6), (4, 10), (5, 6), (5, 10), (5, 16),
        (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32), (14, 33), (15, 32),
        (15, 33), (18, 32), (18, 33), (19, 33), (20, 32), (20, 33), (22, 32), (22, 33),
        (23, 25), (23, 27), (23, 29), (23, 32), (23, 33), (24, 25), (24, 27), (24, 31),
        (25, 31), (26, 29), (26, 33), (27, 33), (28, 31), (28, 33), (29, 32), (29, 33),
        (30, 32), (30, 33), (31, 32), (31, 33), (32, 33),
    ];
    E.iter().map(|&(a, b)| (a, b, 1.0)).collect()
}

/// The karate club as a graph.
pub fn karate() -> CSRNetwork {
    CSRNetwork::from_edges(34, &karate_edges()).unwrap()
}

/// A stochastic block model with the given within/between edge probabilities.
pub fn sbm(
    n_per: usize,
    blocks: usize,
    p_in: f64,
    p_out: f64,
    seed: u64,
) -> (Vec<(usize, usize, f64)>, Vec<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = n_per * blocks;
    let truth: Vec<usize> = (0..n).map(|i| i / n_per).collect();
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let p = if truth[i] == truth[j] { p_in } else { p_out };
            if rng.random::<f64>() < p {
                edges.push((i, j, 1.0));
            }
        }
    }
    (edges, truth)
}

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

/// Normalized mutual information between two labellings, in `[0, 1]`.
pub fn nmi(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len() as f64;
    if n == 0.0 {
        return 1.0;
    }
    let ka = a.iter().max().map_or(0, |m| m + 1);
    let kb = b.iter().max().map_or(0, |m| m + 1);
    let mut joint = vec![0.0f64; ka * kb];
    let (mut pa, mut pb) = (vec![0.0f64; ka], vec![0.0f64; kb]);
    for i in 0..a.len() {
        joint[a[i] * kb + b[i]] += 1.0;
        pa[a[i]] += 1.0;
        pb[b[i]] += 1.0;
    }
    let mut mi = 0.0;
    for i in 0..ka {
        for j in 0..kb {
            let p = joint[i * kb + j] / n;
            if p > 0.0 {
                mi += p * (p / ((pa[i] / n) * (pb[j] / n))).ln();
            }
        }
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

/// Number of communities that are not internally connected.
pub fn disconnected_community_count(graph: &CSRNetwork, labels: &[usize]) -> usize {
    let n = graph.node_count();
    let n_comms = labels.iter().max().map_or(0, |m| m + 1);
    let mut seen = vec![false; n];
    let mut components = vec![0usize; n_comms];

    for v in 0..n {
        if seen[v] {
            continue;
        }
        components[labels[v]] += 1;
        let mut stack = vec![v];
        seen[v] = true;
        while let Some(u) = stack.pop() {
            for (w, _) in graph.neighbors(u) {
                if !seen[w] && labels[w] == labels[u] {
                    seen[w] = true;
                    stack.push(w);
                }
            }
        }
    }
    components.iter().filter(|&&c| c > 1).count()
}
