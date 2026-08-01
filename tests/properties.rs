//! Randomised property tests.
//!
//! The rest of the suite uses hand-picked graphs, which only covers what somebody thought to
//! write down. This generates graphs and configs at random and checks the invariants hold —
//! that's what catches the combination nobody considered.
//!
//! Failures print the seed, so counterexamples are reproducible.

use single_clustering::community_search::leiden::{
    Cpm, LeidenConfig, Objective, ObjectiveKind, Partition, Rb, leiden, modularity,
};
use single_clustering::network::CSRNetwork;

mod common;
use common::disconnected_community_count;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// A randomly shaped graph: varying size, density, weight distribution and self-loops.
fn random_graph(seed: u64) -> (String, CSRNetwork) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = rng.random_range(1..=120usize);
    let density = rng.random::<f64>().powi(2) * 0.6 + 0.01;
    let weighted = rng.random::<bool>();
    let self_loops = rng.random::<f64>() < 0.3;
    // Sometimes plant block structure, sometimes leave it structureless.
    let blocks = if rng.random::<bool>() {
        rng.random_range(2..=6usize).min(n)
    } else {
        1
    };

    let mut edges = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            let same = blocks > 1 && (a * blocks / n) == (b * blocks / n);
            let p = if same { density * 4.0 } else { density };
            if rng.random::<f64>() < p.min(1.0) {
                let w = if weighted {
                    0.05 + 4.0 * rng.random::<f64>()
                } else {
                    1.0
                };
                edges.push((a, b, w));
            }
        }
    }
    if self_loops {
        for v in 0..n {
            if rng.random::<f64>() < 0.2 {
                edges.push((v, v, 0.5 + 2.0 * rng.random::<f64>()));
            }
        }
    }

    let name = format!(
        "seed={seed} n={n} density={density:.3} blocks={blocks} weighted={weighted} loops={self_loops}"
    );
    (name, CSRNetwork::from_edges(n, &edges).unwrap())
}

fn random_config(rng: &mut ChaCha8Rng, seed: u64) -> LeidenConfig {
    let objective = if rng.random::<bool>() {
        ObjectiveKind::Rb {
            resolution: [0.1, 0.5, 1.0, 2.0, 5.0][rng.random_range(0..5)],
        }
    } else {
        ObjectiveKind::Cpm {
            resolution: [0.01, 0.05, 0.2, 1.0][rng.random_range(0..4)],
        }
    };
    LeidenConfig {
        objective,
        seed: Some(seed),
        n_iterations: rng.random_range(1..=4),
        refine: rng.random::<bool>(),
        refine_randomness: [0.0, 0.01, 0.1][rng.random_range(0..3)],
        max_community_weight: if rng.random::<f64>() < 0.2 {
            Some(rng.random_range(2..=40) as f64)
        } else {
            None
        },
        ..Default::default()
    }
}

fn quality(graph: &CSRNetwork, objective: ObjectiveKind, labels: &[usize]) -> f64 {
    let p = Partition::from_membership(graph, labels);
    match objective {
        ObjectiveKind::Rb { resolution } => Rb::new(resolution).quality(&p),
        ObjectiveKind::Cpm { resolution } => Cpm::new(resolution).quality(&p),
    }
}

/// Every invariant, over 400 random graph/config combinations.
#[test]
fn invariants_hold_on_random_inputs() {
    for seed in 0..400u64 {
        let (name, graph) = random_graph(seed);
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x9e37_79b9);
        let config = random_config(&mut rng, seed);

        let clustering = leiden(&graph, &config).unwrap();
        let labels = clustering.labels();
        let ctx = format!("{name} {:?}", config.objective);

        // labels are well formed
        assert_eq!(clustering.len(), graph.node_count(), "{ctx}: label count");
        for &l in labels {
            assert!(l < clustering.n_clusters(), "{ctx}: label {l} out of range");
        }
        assert!(
            clustering.cluster_sizes().iter().all(|&s| s > 0),
            "{ctx}: empty cluster survived"
        );

        // never worse than the singleton partition it started from
        let singletons: Vec<usize> = (0..graph.node_count()).collect();
        let start = quality(&graph, config.objective, &singletons);
        let end = quality(&graph, config.objective, labels);
        assert!(
            end >= start - 1e-6 * start.abs().max(1.0),
            "{ctx}: ended at {end}, worse than singletons {start}"
        );

        // communities are internally connected (refinement's guarantee)
        if config.refine {
            assert_eq!(
                disconnected_community_count(&graph, labels),
                0,
                "{ctx}: disconnected community"
            );
        }

        // a max size cap is honoured
        if let Some(limit) = config.max_community_weight {
            for (id, size) in clustering.cluster_sizes().iter().enumerate() {
                assert!(
                    *size as f64 <= limit,
                    "{ctx}: cluster {id} has {size} nodes, limit {limit}"
                );
            }
        }

        // deterministic
        let again = leiden(&graph, &config).unwrap();
        assert_eq!(clustering, again, "{ctx}: not reproducible");
    }
}

/// Quality must survive aggregation, on random graphs rather than chosen ones.
#[test]
fn aggregation_preserves_quality_on_random_inputs() {
    for seed in 0..250u64 {
        let (name, graph) = random_graph(seed);
        if graph.node_count() == 0 {
            continue;
        }
        let clustering = leiden(&graph, &LeidenConfig::default()).unwrap();
        let mut partition = Partition::from_membership(&graph, clustering.labels());

        let grouping = partition.renumber_into_grouping();
        let coarse = graph.aggregate(&grouping);
        let coarse_labels: Vec<usize> = (0..coarse.node_count()).collect();
        let coarse_partition = Partition::from_membership(&coarse, &coarse_labels);

        // structural invariants of the collapsed graph
        let fine_deg: f64 = (0..graph.node_count()).map(|v| graph.strength(v)).sum();
        let coarse_deg: f64 = (0..coarse.node_count()).map(|v| coarse.strength(v)).sum();
        assert!(
            (fine_deg - coarse_deg).abs() < 1e-6 * fine_deg.abs().max(1.0),
            "{name}: degree sum {fine_deg} -> {coarse_deg}"
        );
        coarse.validate_symmetry().unwrap();

        for gamma in [0.25, 1.0, 3.0] {
            let a = Rb::new(gamma).quality(&partition);
            let b = Rb::new(gamma).quality(&coarse_partition);
            assert!(
                (a - b).abs() < 1e-6 * a.abs().max(1.0),
                "{name} gamma={gamma}: {a} != {b}"
            );
        }
        let a = Cpm::new(0.05).quality(&partition);
        let b = Cpm::new(0.05).quality(&coarse_partition);
        assert!(
            (a - b).abs() < 1e-6 * a.abs().max(1.0),
            "{name} cpm: {a} != {b}"
        );
    }
}

/// The delta a move reports must equal the change it causes, on random inputs.
#[test]
fn move_deltas_match_quality_changes_on_random_inputs() {
    use single_clustering::community_search::leiden::NeighborWeights;
    use single_clustering::community_search::leiden::objective::InsertContext;

    for seed in 0..150u64 {
        let (name, graph) = random_graph(seed);
        let n = graph.node_count();
        if n < 2 {
            continue;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xabcd);
        let labels: Vec<usize> = (0..n).map(|_| rng.random_range(0..4usize)).collect();

        for objective in [
            &Rb::new(1.0) as &dyn Objective,
            &Rb::new(0.3),
            &Cpm::new(0.05),
        ] {
            let mut p = Partition::from_membership(&graph, &labels);
            let mut acc = NeighborWeights::with_capacity(p.slots() + 1);

            for v in 0..n.min(25) {
                for target in 0..p.slots() {
                    let before = objective.quality(&p);
                    let ctx = InsertContext::for_node(&graph, v);

                    acc.collect(&graph, p.membership_raw(), v);
                    let own = p.membership(v);
                    p.remove_node(v, &graph, acc.weight_to(own));
                    let leave = objective.delta_insert(&p, &ctx, own, acc.weight_to(own));
                    let join = objective.delta_insert(&p, &ctx, target, acc.weight_to(target));
                    p.insert_node(v, target, &graph, acc.weight_to(target));

                    let actual = objective.quality(&p) - before;
                    let predicted = join - leave;
                    assert!(
                        (actual - predicted).abs() < 1e-6 * actual.abs().max(1.0),
                        "{name}: node {v} -> {target}: predicted {predicted}, actual {actual}"
                    );

                    // restore
                    acc.collect(&graph, p.membership_raw(), v);
                    p.remove_node(v, &graph, acc.weight_to(target));
                    p.insert_node(v, own, &graph, acc.weight_to(own));
                }
            }
            p.verify_against(&graph).unwrap();
        }
    }
}

/// The zero-copy constructor is the path used at scale, so it has to be indistinguishable
/// from the edge-list one, not just close.
#[test]
fn both_constructors_produce_identical_results() {
    for seed in 0..300u64 {
        let (name, via_edges) = random_graph(seed);

        let (node_ptrs, neighbors, weights) = via_edges.to_csr_parts();
        let node_weights = via_edges.node_weights().to_vec();
        let via_csr =
            CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, Some(node_weights)).unwrap();

        // identical structure
        assert_eq!(
            via_edges.node_count(),
            via_csr.node_count(),
            "{name}: nodes"
        );
        assert_eq!(
            via_edges.edge_count(),
            via_csr.edge_count(),
            "{name}: edges"
        );
        assert_eq!(
            via_edges.total_weight(),
            via_csr.total_weight(),
            "{name}: total weight"
        );
        for v in 0..via_edges.node_count() {
            assert_eq!(
                via_edges.strength(v),
                via_csr.strength(v),
                "{name}: strength({v})"
            );
            assert_eq!(
                via_edges.neighbors(v).collect::<Vec<_>>(),
                via_csr.neighbors(v).collect::<Vec<_>>(),
                "{name}: neighbors({v})"
            );
        }
        via_csr.validate_symmetry().unwrap();

        // and identical clustering, bit for bit
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5555);
        let config = random_config(&mut rng, seed);
        assert_eq!(
            leiden(&via_edges, &config).unwrap(),
            leiden(&via_csr, &config).unwrap(),
            "{name}: clusterings differ between constructors"
        );
    }
}

/// Raising the resolution must never reduce the number of communities.
#[test]
fn resolution_is_monotone_on_random_inputs() {
    for seed in 0..120u64 {
        let (name, graph) = random_graph(seed);
        if graph.node_count() < 10 || graph.edge_count() == 0 {
            continue;
        }
        let counts: Vec<usize> = [0.05, 0.25, 1.0, 4.0, 16.0, 64.0]
            .iter()
            .map(|&r| {
                leiden(&graph, &LeidenConfig::with_resolution(r))
                    .unwrap()
                    .n_clusters()
            })
            .collect();
        // Heuristics can wobble by a cluster at adjacent resolutions; the trend must hold.
        assert!(
            counts[0] <= counts[counts.len() - 1],
            "{name}: {counts:?} is not increasing overall"
        );
        for w in counts.windows(2) {
            assert!(
                w[1] + 2 >= w[0],
                "{name}: {counts:?} drops sharply as resolution rises"
            );
        }
    }
}

/// Uniformly scaling every edge weight must not change an RB partition.
#[test]
fn rb_is_scale_invariant_on_random_inputs() {
    for seed in 0..120u64 {
        let (name, graph) = random_graph(seed);
        if graph.edge_count() == 0 {
            continue;
        }
        let (ptrs, nbrs, w) = graph.to_csr_parts();
        // 2^6 is exact in binary, so scaling introduces no f32 rounding of its own.
        let scaled: Vec<f32> = w.iter().map(|x| x * 64.0).collect();
        let scaled_graph =
            CSRNetwork::from_csr_parts(ptrs, nbrs, scaled, Some(graph.node_weights().to_vec()))
                .unwrap();

        let a = leiden(&graph, &LeidenConfig::default()).unwrap();
        let b = leiden(&scaled_graph, &LeidenConfig::default()).unwrap();
        assert_eq!(
            a.labels(),
            b.labels(),
            "{name}: scaling changed the partition"
        );
        assert!(
            (modularity(&graph, a.labels(), 1.0) - modularity(&scaled_graph, b.labels(), 1.0))
                .abs()
                < 1e-6,
            "{name}: scaling changed modularity"
        );
    }
}
