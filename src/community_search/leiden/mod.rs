//! The Leiden algorithm for community detection.
//!
//! ```
//! use single_clustering::network::CSRNetwork;
//! use single_clustering::community_search::leiden::{leiden, LeidenConfig, ObjectiveKind};
//!
//! # fn main() -> single_clustering::Result<()> {
//! let graph = CSRNetwork::from_edges(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])?;
//! let config = LeidenConfig {
//!     objective: ObjectiveKind::Rb { resolution: 1.0 },
//!     seed: Some(42),
//!     ..Default::default()
//! };
//! let clustering = leiden(&graph, &config)?;
//! println!("{} communities", clustering.n_clusters());
//! # Ok(())
//! # }
//! ```
//!
//! # How it works
//!
//! Each level: [`local_move()`] settles nodes into communities, [`refine()`] re-derives a
//! finer partition inside each one, and [`CSRNetwork::aggregate`] collapses the **refined**
//! communities into super-nodes. The next level starts from the coarse partition the working
//! communities induce, so the same partition keeps improving at coarser granularity until
//! collapsing stops shrinking the graph.
//!
//! Collapsing the refined partition instead of the working one is the whole difference from
//! Louvain — it's why badly connected communities don't get locked in.

pub mod local_move;
pub mod objective;
pub mod partition;
pub mod refine;

pub use local_move::local_move;
pub use objective::{Cpm, Objective, ObjectiveKind, Rb, modularity};
pub use partition::{MoveScratch, NeighborWeights, Partition};
pub use refine::refine;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::clustering::Clustering;
use crate::error::{ClusteringError, Result};
use crate::network::CSRNetwork;

/// Configuration for [`leiden`].
#[derive(Debug, Clone, PartialEq)]
pub struct LeidenConfig {
    /// Quality function to optimize, carrying the resolution parameter.
    ///
    /// [`ObjectiveKind::Rb`] at `resolution = 1.0` is modularity, matching `scanpy`'s default.
    pub objective: ObjectiveKind,
    /// Seed for the random node orderings. `Some(_)` makes runs bit-for-bit reproducible;
    /// `None` draws from the OS.
    pub seed: Option<u64>,
    /// Number of passes, each continuing from the previous result.
    ///
    /// Matches `leidenalg` — passes keep improving the same partition rather than restarting
    /// from singletons, which is how it escapes local optima a single pass falls into. Stops
    /// early once a pass finds nothing.
    pub n_iterations: usize,
    /// Cap on aggregation levels within one pass. Reaching it is unusual — a pass normally
    /// stops earlier, when collapsing no longer shrinks the graph.
    pub max_levels: usize,
    /// Minimum improvement for a pass to count as progress, relative to total edge weight.
    ///
    /// Relative, so the stopping rule doesn't depend on what units the weights are in.
    ///
    /// Between passes only, never between levels — a level finding nothing says nothing about
    /// the coarser level above it, which is exactly where aggregation earns its keep.
    pub tolerance: f64,
    /// Whether to run the refinement phase. Disabling it turns the algorithm into Louvain.
    pub refine: bool,
    /// Randomness of community selection during refinement, as in the Leiden paper.
    ///
    /// Higher explores more. `0.0` is a deterministic argmax, which gets stuck in local optima
    /// no number of passes can escape. Deterministic for a fixed [`seed`](Self::seed) either
    /// way.
    pub refine_randomness: f64,
    /// How many consecutive passes may find nothing before stopping.
    ///
    /// Refinement is randomised, so one fruitless pass isn't proof of convergence. `1` gives
    /// `leidenalg`'s stop-on-first-stall behaviour.
    pub patience: usize,
    /// Optional cap on a community's total node weight. Node weights add up on aggregation, so
    /// this means the same thing at every level — a limit on original nodes.
    pub max_community_weight: Option<f64>,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            objective: ObjectiveKind::default(),
            seed: Some(42),
            n_iterations: 2,
            max_levels: 64,
            tolerance: 1e-9,
            refine: true,
            refine_randomness: 0.01,
            patience: 2,
            max_community_weight: None,
        }
    }
}

impl LeidenConfig {
    /// Convenience constructor for RB (modularity-style) clustering at a given resolution.
    pub fn with_resolution(resolution: f64) -> Self {
        Self {
            objective: ObjectiveKind::Rb { resolution },
            ..Default::default()
        }
    }

    fn validate(&self) -> Result<()> {
        let r = self.objective.resolution();
        if !r.is_finite() || r < 0.0 {
            return Err(ClusteringError::InvalidConfig(format!(
                "resolution must be finite and non-negative, got {r}"
            )));
        }
        if self.n_iterations == 0 {
            return Err(ClusteringError::InvalidConfig(
                "n_iterations must be at least 1".into(),
            ));
        }
        if self.max_levels == 0 {
            return Err(ClusteringError::InvalidConfig(
                "max_levels must be at least 1".into(),
            ));
        }
        if self.patience == 0 {
            return Err(ClusteringError::InvalidConfig(
                "patience must be at least 1".into(),
            ));
        }
        if !self.refine_randomness.is_finite() || self.refine_randomness < 0.0 {
            return Err(ClusteringError::InvalidConfig(format!(
                "refine_randomness must be finite and non-negative, got {}",
                self.refine_randomness
            )));
        }
        if let Some(w) = self.max_community_weight
            && (!w.is_finite() || w <= 0.0)
        {
            return Err(ClusteringError::InvalidConfig(format!(
                "max_community_weight must be finite and positive, got {w}"
            )));
        }
        Ok(())
    }
}

fn build_objective(kind: ObjectiveKind) -> Box<dyn Objective> {
    match kind {
        ObjectiveKind::Rb { resolution } => Box::new(Rb::new(resolution)),
        ObjectiveKind::Cpm { resolution } => Box::new(Cpm::new(resolution)),
    }
}

/// Detects communities in `graph`.
///
/// Runs the algorithm [`LeidenConfig::n_iterations`] times, each pass continuing from the
/// previous result, and stops early once a pass finds no improvement. With a fixed
/// [`seed`](LeidenConfig::seed) the result is deterministic.
pub fn leiden(graph: &CSRNetwork, config: &LeidenConfig) -> Result<Clustering> {
    config.validate()?;

    let objective = build_objective(config.objective);
    let mut rng = match config.seed {
        Some(seed) => ChaCha8Rng::seed_from_u64(seed),
        None => ChaCha8Rng::from_os_rng(),
    };

    let n = graph.node_count();
    if n == 0 {
        return Ok(Clustering::from_normalized(Vec::new(), 0));
    }

    // singletons; each pass picks up the last one's result
    let mut labels: Vec<usize> = (0..n).collect();
    let mut n_clusters = n;
    let mut stagnant = 0usize;
    for _ in 0..config.n_iterations {
        let (next, clusters, improvement) =
            run_once(graph, objective.as_ref(), &mut rng, config, &labels);
        labels = next;
        n_clusters = clusters;
        let scale = graph.total_weight().abs().max(1.0);
        if improvement <= config.tolerance * scale {
            stagnant += 1;
            // refinement is randomised, so the next pass may see moves this one couldn't
            if stagnant >= config.patience {
                break;
            }
        } else {
            stagnant = 0;
        }
    }

    Ok(Clustering::from_normalized(labels, n_clusters))
}

/// One pass: local move, refine, aggregate, repeat until the graph stops shrinking.
///
/// Starts from `initial`, not singletons — that's what makes successive passes build on each
/// other. Returns consecutive labels over the original nodes, the community count, and the
/// total improvement.
///
/// The improvement is just the sum of per-level local-moving gains, which is exact:
/// aggregation doesn't change quality, and the coarse partition is the same partition.
fn run_once(
    graph: &CSRNetwork,
    objective: &dyn Objective,
    rng: &mut impl RngCore,
    config: &LeidenConfig,
    initial: &[usize],
) -> (Vec<usize>, usize, f64) {
    let n = graph.node_count();
    let mut level_rng = ChaCha8Rng::seed_from_u64(rng.next_u64());

    let mut level_graph = graph.clone();
    let mut partition = Partition::from_membership(&level_graph, initial);
    // one set for the whole pass; per-level realloc is ~1.3 GB of churn at 75M nodes
    let mut scratch = MoveScratch::with_capacity(n, partition.slots().max(n) + 1);
    let mut total_improvement = 0.0;

    // where each original node lives now
    let mut node_at_level: Vec<usize> = (0..n).collect();

    for _ in 0..config.max_levels {
        let improvement = local_move(
            &level_graph,
            &mut partition,
            objective,
            &mut scratch,
            &mut level_rng,
            config.max_community_weight,
        );
        total_improvement += improvement;

        // refining before the collapse is what makes this Leiden
        let mut collapse_by = if config.refine {
            refine(
                &level_graph,
                partition.membership_raw(),
                objective,
                &mut scratch,
                &mut level_rng,
                config.max_community_weight,
                config.refine_randomness,
            )
        } else {
            partition.clone()
        };

        let grouping = collapse_by.renumber_into_grouping();
        let coarse_graph = level_graph.aggregate(&grouping);

        // no shrink, nothing left to gain
        if coarse_graph.node_count() >= level_graph.node_count() {
            break;
        }

        // several coarse nodes can share a working community — that's what lets the next
        // level pull them apart again
        let mut coarse_membership = vec![0usize; coarse_graph.node_count()];
        for v in 0..level_graph.node_count() {
            coarse_membership[collapse_by.membership(v)] = partition.membership(v);
        }

        for slot in node_at_level.iter_mut() {
            *slot = collapse_by.membership(*slot);
        }

        level_graph = coarse_graph;
        partition = Partition::from_membership(&level_graph, &coarse_membership);
    }

    partition.renumber();
    let labels: Vec<usize> = node_at_level
        .iter()
        .map(|&node| partition.membership(node))
        .collect();
    let n_clusters = partition.community_count();
    (labels, n_clusters, total_improvement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::{disconnected_community_count, karate, nmi, sbm};

    #[test]
    fn finds_the_karate_club_optimum() {
        let g = karate();
        let c = leiden(&g, &LeidenConfig::default()).unwrap();
        let q = modularity(&g, c.labels(), 1.0);
        assert_eq!(c.n_clusters(), 4, "sizes: {:?}", c.cluster_sizes());
        assert!(
            (q - 0.4198).abs() < 1e-3,
            "expected the known optimum 0.4198, got {q}"
        );
    }

    #[test]
    fn recovers_a_planted_partition() {
        let (edges, truth) = sbm(60, 6, 0.25, 0.01, 11);
        let g = CSRNetwork::from_edges(360, &edges).unwrap();
        let c = leiden(&g, &LeidenConfig::default()).unwrap();
        let score = nmi(c.labels(), &truth);
        assert!(score > 0.95, "NMI {score}, {} clusters", c.n_clusters());
        assert_eq!(c.n_clusters(), 6);
    }

    #[test]
    fn is_deterministic_for_a_fixed_seed() {
        let (edges, _) = sbm(40, 5, 0.3, 0.02, 4);
        let g = CSRNetwork::from_edges(200, &edges).unwrap();
        let first = leiden(&g, &LeidenConfig::default()).unwrap();
        for _ in 0..10 {
            let again = leiden(&g, &LeidenConfig::default()).unwrap();
            assert_eq!(first, again);
        }
    }

    #[test]
    fn different_seeds_are_allowed_to_differ_but_stay_good() {
        let (edges, truth) = sbm(40, 5, 0.3, 0.02, 4);
        let g = CSRNetwork::from_edges(200, &edges).unwrap();
        for seed in 0..8 {
            let c = leiden(
                &g,
                &LeidenConfig {
                    seed: Some(seed),
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(nmi(c.labels(), &truth) > 0.9, "seed {seed}");
        }
    }

    #[test]
    fn communities_are_internally_connected() {
        // what refinement is for
        let (edges, _) = sbm(40, 5, 0.25, 0.03, 8);
        let g = CSRNetwork::from_edges(200, &edges).unwrap();
        for seed in 0..8 {
            let c = leiden(
                &g,
                &LeidenConfig {
                    seed: Some(seed),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                disconnected_community_count(&g, c.labels()),
                0,
                "seed {seed} produced a disconnected community"
            );
        }
    }

    #[test]
    fn resolution_controls_granularity() {
        let (edges, _) = sbm(40, 5, 0.3, 0.02, 4);
        let g = CSRNetwork::from_edges(200, &edges).unwrap();
        let mut counts = Vec::new();
        for r in [0.25, 1.0, 4.0, 16.0] {
            let c = leiden(&g, &LeidenConfig::with_resolution(r)).unwrap();
            counts.push(c.n_clusters());
        }
        assert!(
            counts.windows(2).all(|w| w[0] <= w[1]),
            "cluster count should be non-decreasing in resolution, got {counts:?}"
        );
        assert!(
            counts[0] < counts[3],
            "resolution had no effect: {counts:?}"
        );
    }

    #[test]
    fn cpm_also_recovers_the_planted_partition() {
        let (edges, truth) = sbm(40, 5, 0.3, 0.02, 4);
        let g = CSRNetwork::from_edges(200, &edges).unwrap();
        let c = leiden(
            &g,
            &LeidenConfig {
                objective: ObjectiveKind::Cpm { resolution: 0.1 },
                ..Default::default()
            },
        )
        .unwrap();
        assert!(nmi(c.labels(), &truth) > 0.9, "{} clusters", c.n_clusters());
    }

    #[test]
    fn respects_max_community_weight() {
        let (edges, _) = sbm(40, 5, 0.3, 0.02, 4);
        let g = CSRNetwork::from_edges(200, &edges).unwrap();
        let c = leiden(
            &g,
            &LeidenConfig {
                max_community_weight: Some(25.0),
                ..Default::default()
            },
        )
        .unwrap();
        for (id, size) in c.cluster_sizes().iter().enumerate() {
            assert!(*size <= 25, "cluster {id} has {size} nodes");
        }
    }

    #[test]
    fn handles_degenerate_graphs() {
        let cases: Vec<(CSRNetwork, usize)> = vec![
            (
                CSRNetwork::from_edges(0, &[] as &[(usize, usize, f64)]).unwrap(),
                0,
            ),
            (
                CSRNetwork::from_edges(1, &[] as &[(usize, usize, f64)]).unwrap(),
                1,
            ),
            (
                CSRNetwork::from_edges(5, &[] as &[(usize, usize, f64)]).unwrap(),
                5,
            ),
            (CSRNetwork::from_edges(2, &[(0, 1, 1.0)]).unwrap(), 1),
        ];
        for (g, expected) in cases {
            let c = leiden(&g, &LeidenConfig::default()).unwrap();
            assert_eq!(c.len(), g.node_count());
            assert_eq!(c.n_clusters(), expected, "graph: {g}");
        }
    }

    #[test]
    fn disconnected_components_are_never_merged() {
        // two components with no edge between them must not share a community
        let mut edges = Vec::new();
        for b in 0..2usize {
            for i in 0..5 {
                for j in (i + 1)..5 {
                    edges.push((b * 5 + i, b * 5 + j, 1.0));
                }
            }
        }
        let g = CSRNetwork::from_edges(10, &edges).unwrap();
        let c = leiden(&g, &LeidenConfig::default()).unwrap();
        assert_eq!(c.n_clusters(), 2);
        assert_ne!(c.labels()[0], c.labels()[5]);
    }

    #[test]
    fn rejects_invalid_config() {
        let g = karate();
        for bad in [
            LeidenConfig {
                n_iterations: 0,
                ..Default::default()
            },
            LeidenConfig {
                max_levels: 0,
                ..Default::default()
            },
            LeidenConfig::with_resolution(-1.0),
            LeidenConfig {
                max_community_weight: Some(0.0),
                ..Default::default()
            },
        ] {
            assert!(leiden(&g, &bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn louvain_mode_still_works() {
        let g = karate();
        let c = leiden(
            &g,
            &LeidenConfig {
                refine: false,
                ..Default::default()
            },
        )
        .unwrap();
        let q = modularity(&g, c.labels(), 1.0);
        assert!(q > 0.41, "Louvain mode reached only {q}");
    }
}
