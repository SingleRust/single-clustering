//! Exhaustive optimality checks on small graphs.
//!
//! Small enough to enumerate every set partition, so the true optimum is computable. The only
//! test here that checks ground truth rather than another implementation — nothing in it can
//! be fooled by a shared misunderstanding of the objective.
//!
//! Kept small enough for the normal suite; `exhaustive_optimality_larger` does `n = 10..12`
//! and is `#[ignore]`d.

use single_clustering::community_search::leiden::{LeidenConfig, leiden, modularity};
use single_clustering::network::CSRNetwork;

mod common;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Visits every set partition of `n` items via restricted growth strings, calling `f` with
/// each labelling. There are `Bell(n)` of them: 4140 for n=8, 115975 for n=10, 4213597 for
/// n=12.
fn for_each_partition(n: usize, mut f: impl FnMut(&[usize])) {
    let mut a = vec![0usize; n];
    let mut max_seen = vec![0usize; n];

    loop {
        f(&a);

        // advance the restricted growth string, right to left
        let mut i = n - 1;
        loop {
            if i == 0 {
                return;
            }
            if a[i] <= max_seen[i - 1] {
                a[i] += 1;
                let new_max = max_seen[i - 1].max(a[i]);
                for j in (i + 1)..n {
                    a[j] = 0;
                    max_seen[j] = new_max;
                }
                if i < n {
                    max_seen[i] = new_max;
                }
                break;
            }
            a[i] = 0;
            i -= 1;
        }
    }
}

/// True modularity optimum of `graph` at the given resolution, by exhaustive enumeration.
fn brute_force_optimum(graph: &CSRNetwork, resolution: f64) -> f64 {
    let n = graph.node_count();
    let mut best = f64::NEG_INFINITY;
    for_each_partition(n, |labels| {
        let q = modularity(graph, labels, resolution);
        if q > best {
            best = q;
        }
    });
    best
}

/// Random graphs spanning a range of densities, plus a few structured ones.
fn small_graphs(n: usize, count: usize, seed: u64) -> Vec<(String, CSRNetwork)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::new();

    for i in 0..count {
        let p = 0.15 + 0.5 * (i as f64 / count as f64);
        let mut edges = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                if rng.random::<f64>() < p {
                    edges.push((a, b, 1.0));
                }
            }
        }
        if edges.is_empty() {
            continue;
        }
        out.push((
            format!("random_n{n}_p{p:.2}_{i}"),
            CSRNetwork::from_edges(n, &edges).unwrap(),
        ));
    }

    // two cliques joined by one edge: an unambiguous optimum
    if n >= 6 {
        let half = n / 2;
        let mut edges = Vec::new();
        for a in 0..half {
            for b in (a + 1)..half {
                edges.push((a, b, 1.0));
            }
        }
        for a in half..n {
            for b in (a + 1)..n {
                edges.push((a, b, 1.0));
            }
        }
        edges.push((0, half, 1.0));
        out.push((
            format!("barbell_n{n}"),
            CSRNetwork::from_edges(n, &edges).unwrap(),
        ));
    }

    out
}

fn check_optimality(sizes: &[usize], graphs_per_size: usize, resolutions: &[f64]) {
    let mut checked = 0;
    let mut optimal = 0;
    let mut worst_gap: f64 = 0.0;
    let mut worst_case = String::new();

    for &n in sizes {
        for (name, graph) in small_graphs(n, graphs_per_size, 17 + n as u64) {
            for &resolution in resolutions {
                let best = brute_force_optimum(&graph, resolution);

                // Give the optimizer the same budget a user would.
                let clustering = leiden(
                    &graph,
                    &LeidenConfig {
                        objective: single_clustering::community_search::leiden::ObjectiveKind::Rb {
                            resolution,
                        },
                        seed: Some(42),
                        n_iterations: std::env::var("OPT_ITERS")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(2),
                        ..Default::default()
                    },
                )
                .unwrap();
                let ours = modularity(&graph, clustering.labels(), resolution);

                checked += 1;
                if ours >= best - 1e-9 {
                    optimal += 1;
                } else {
                    // absolute, not relative: modularity is in [-1, 1] and these optima sit
                    // near zero, where relative error explodes meaninglessly — -0.034 vs
                    // -0.018 is "87% worse" relatively but 0.016 absolutely
                    let gap = best - ours;
                    if gap > worst_gap {
                        worst_gap = gap;
                        worst_case =
                            format!("{name} at resolution {resolution}: {ours:.6} vs {best:.6}");
                    }
                }
                assert!(
                    ours <= best + 1e-9,
                    "{name}: scored {ours} above the exhaustive optimum {best} - the \
                     objective and the reported modularity disagree"
                );
            }
        }
    }

    let rate = 100.0 * optimal as f64 / checked as f64;
    println!(
        "exhaustive: optimal on {optimal}/{checked} ({rate:.1}%), worst gap {worst_gap:.4} \
         modularity ({worst_case})"
    );

    // Measured on these fixtures: the true optimum is found on 93.5% of 108 cases, worst gap
    // 0.027 modularity (0.050 for the n=10..12 variant). `leidenalg` on the identical graphs gets 95.4% at its default budget
    // and 96.3% at 10x that, hitting the same hardest instance — it defeats both. These are
    // small dense random graphs with no community structure, i.e. a deliberately flat,
    // adversarial landscape; the bounds below sit just outside the measured values so a
    // genuine regression trips them.
    assert!(
        rate >= 90.0,
        "found the true optimum in only {rate:.1}% of {checked} cases (expected ~93.5%)"
    );
    assert!(
        worst_gap < 0.05,
        "worst gap {worst_gap:.4} modularity on {worst_case}"
    );
}

/// Every partition enumerated, for n = 6..9.
#[test]
fn finds_the_true_optimum_on_small_graphs() {
    check_optimality(&[6, 7, 8, 9], 8, &[0.5, 1.0, 2.0]);
}

/// Same, for n = 10..12. Slow (Bell(12) = 4.2M partitions per graph per resolution).
#[test]
#[ignore = "exhaustive over Bell(12); run with --ignored"]
fn exhaustive_optimality_larger() {
    check_optimality(&[10, 11, 12], 6, &[0.5, 1.0, 2.0]);
}

/// The enumerator itself must be right, or the whole test is worthless.
#[test]
fn partition_enumerator_is_correct() {
    // Bell numbers
    for (n, expected) in [
        (1usize, 1usize),
        (2, 2),
        (3, 5),
        (4, 15),
        (5, 52),
        (6, 203),
        (7, 877),
        (8, 4140),
    ] {
        let mut count = 0;
        let mut seen = std::collections::HashSet::new();
        for_each_partition(n, |labels| {
            count += 1;
            // canonical form: labels are already a restricted growth string, so they are a
            // unique representative of the set partition
            seen.insert(labels.to_vec());
            assert_eq!(labels.len(), n);
            assert_eq!(labels[0], 0, "restricted growth strings start at 0");
        });
        assert_eq!(count, expected, "Bell({n})");
        assert_eq!(seen.len(), expected, "Bell({n}): duplicates emitted");
    }
}
