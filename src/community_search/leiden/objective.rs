//! Quality functions.
//!
//! An [`Objective`] scores a [`Partition`], and more usefully scores *inserting* a removed
//! node into a candidate community. Both read only maintained aggregates, never the graph.
//!
//! Everything is `Σ_c [ internal_c − null_c ]`, each internal edge once. [`Rb`]'s null term is
//! `γ·K_c²/(4m)` — that `4m` is load-bearing, since `2m` gives `Rb` at resolution `2γ`, which
//! is the 0.6.x bug. [`modularity`] reports a value directly comparable with `scanpy`.

use crate::community_search::leiden::partition::Partition;
use crate::network::CSRNetwork;

/// What an objective needs about the node being moved. Built once per node, reused across
/// every candidate.
#[derive(Debug, Clone, Copy)]
pub struct InsertContext {
    /// Incident edge weight, self-loop counted twice.
    pub node_strength: f64,
    /// The node's own weight, summed across members after aggregation.
    pub node_weight: f64,
    /// Weight of the node's self-loop, or `0.0`.
    pub self_loop: f64,
}

impl InsertContext {
    /// Reads them for `node` out of `graph`.
    #[inline]
    pub fn for_node(graph: &CSRNetwork, node: usize) -> Self {
        Self {
            node_strength: graph.strength(node),
            node_weight: graph.node_weight(node),
            self_loop: graph.self_loop_weight(node),
        }
    }
}

/// The community aggregates a move depends on.
///
/// Three numbers, which is what lets refinement work on local state instead of the whole
/// partition — and so run in parallel.
#[derive(Debug, Clone, Copy, Default)]
pub struct CommunityView {
    /// Total edge weight of the graph, each edge once. Global, not per community.
    pub total_weight: f64,
    /// Summed node strength of the community — `K_c` in the RB null model.
    pub strength: f64,
    /// Summed node weight of the community — `n_c` in CPM.
    pub weight: f64,
}

/// A quality function to maximize.
pub trait Objective: Send + Sync {
    /// Quality change from inserting a removed node into a community.
    ///
    /// The one place each formula lives — [`delta_insert`](Self::delta_insert) reads the view
    /// out of a [`Partition`] and calls this, so the two paths can't drift.
    ///
    /// `weight_to_comm` excludes the self-loop, which gets added here.
    fn delta_insert_view(
        &self,
        view: CommunityView,
        ctx: &InsertContext,
        weight_to_comm: f64,
    ) -> f64;

    /// Quality change from inserting a removed node into `community`, which may be an
    /// unallocated slot — that reads as empty.
    #[inline]
    fn delta_insert(
        &self,
        partition: &Partition,
        ctx: &InsertContext,
        community: usize,
        weight_to_comm: f64,
    ) -> f64 {
        self.delta_insert_view(
            CommunityView {
                total_weight: partition.total_weight(),
                strength: partition.strength(community),
                weight: partition.weight(community),
            },
            ctx,
            weight_to_comm,
        )
    }

    /// Total quality, in the same units as [`delta_insert`](Self::delta_insert).
    fn quality(&self, partition: &Partition) -> f64;

    /// Higher values favour smaller communities.
    fn resolution(&self) -> f64;
}

/// Reichardt–Bornholdt configuration model.
///
/// `Q = Σ_c [ internal_c − γ · K_c² / (4m) ]`. At `γ = 1` this is modularity, matching
/// `leidenalg`'s `RBConfigurationVertexPartition` and so scanpy's default.
#[derive(Debug, Clone, Copy)]
pub struct Rb {
    resolution: f64,
}

impl Rb {
    /// RB at the given resolution.
    pub fn new(resolution: f64) -> Self {
        Self { resolution }
    }
}

impl Default for Rb {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Objective for Rb {
    #[inline]
    fn delta_insert_view(
        &self,
        view: CommunityView,
        ctx: &InsertContext,
        weight_to_comm: f64,
    ) -> f64 {
        let four_m = 4.0 * view.total_weight;
        if four_m == 0.0 {
            return 0.0;
        }
        let k_c = view.strength;
        let k = ctx.node_strength;
        // d/dK of γ·K²/(4m) discretised: ((K+k)² − K²) = k·(2K + k)
        (weight_to_comm + ctx.self_loop) - self.resolution * k * (2.0 * k_c + k) / four_m
    }

    fn quality(&self, partition: &Partition) -> f64 {
        let four_m = 4.0 * partition.total_weight();
        if four_m == 0.0 {
            return 0.0;
        }
        (0..partition.slots())
            .map(|c| {
                let k_c = partition.strength(c);
                partition.internal(c) - self.resolution * k_c * k_c / four_m
            })
            .sum()
    }

    #[inline]
    fn resolution(&self) -> f64 {
        self.resolution
    }
}

/// Constant Potts Model.
///
/// `Q = Σ_c [ internal_c − γ · n_c(n_c − 1) / 2 ]` over summed node weight `n_c`. No
/// resolution limit: `γ` is directly the internal edge density below which a community won't
/// form, so it behaves predictably on large graphs.
///
/// The pair count matches `leidenalg`'s `CPMVertexPartition`, which reports exactly twice
/// this. Node weights sum on aggregation, so `n_c` counts original nodes at every level.
#[derive(Debug, Clone, Copy)]
pub struct Cpm {
    resolution: f64,
}

impl Cpm {
    /// CPM at the given resolution.
    pub fn new(resolution: f64) -> Self {
        Self { resolution }
    }
}

impl Default for Cpm {
    fn default() -> Self {
        Self::new(0.05)
    }
}

impl Objective for Cpm {
    #[inline]
    fn delta_insert_view(
        &self,
        view: CommunityView,
        ctx: &InsertContext,
        weight_to_comm: f64,
    ) -> f64 {
        let n_c = view.weight;
        let a = ctx.node_weight;
        // ((n+a)(n+a-1) - n(n-1)) / 2 = a(2n + a - 1) / 2
        (weight_to_comm + ctx.self_loop) - self.resolution * a * (2.0 * n_c + a - 1.0) / 2.0
    }

    fn quality(&self, partition: &Partition) -> f64 {
        (0..partition.slots())
            .map(|c| {
                let n_c = partition.weight(c);
                partition.internal(c) - self.resolution * n_c * (n_c - 1.0) / 2.0
            })
            .sum()
    }

    #[inline]
    fn resolution(&self) -> f64 {
        self.resolution
    }
}

/// Which quality function to optimize.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObjectiveKind {
    /// Reichardt–Bornholdt configuration model. `resolution = 1.0` is modularity.
    Rb {
        /// Resolution parameter; higher values yield smaller communities.
        resolution: f64,
    },
    /// Constant Potts Model, free of the resolution limit.
    Cpm {
        /// Resolution parameter; the internal density threshold for forming a community.
        resolution: f64,
    },
}

impl Default for ObjectiveKind {
    fn default() -> Self {
        Self::Rb { resolution: 1.0 }
    }
}

impl ObjectiveKind {
    /// The resolution this carries.
    pub fn resolution(&self) -> f64 {
        match *self {
            Self::Rb { resolution } | Self::Cpm { resolution } => resolution,
        }
    }
}

/// Newman–Girvan modularity of a labelling, straight from the graph.
///
/// Independent of [`Partition`] and [`Objective`] on purpose: it's what to report and compare
/// against scanpy, and it doubles as an outside check on the optimizer's arithmetic. At
/// `resolution = 1.0` it's textbook modularity in `[-1, 1]`.
pub fn modularity(graph: &CSRNetwork, labels: &[usize], resolution: f64) -> f64 {
    let m = graph.total_weight();
    if m == 0.0 {
        return 0.0;
    }
    let n_comms = labels.iter().copied().max().map_or(0, |c| c + 1);
    let mut internal = vec![0.0f64; n_comms];
    let mut strength = vec![0.0f64; n_comms];

    for v in 0..graph.node_count() {
        let c = labels[v];
        strength[c] += graph.strength(v);
        for (u, w) in graph.neighbors(v) {
            if labels[u] == c && (u == v || v < u) {
                internal[c] += w;
            }
        }
    }

    let two_m = 2.0 * m;
    (0..n_comms)
        .map(|c| 2.0 * internal[c] / two_m - resolution * (strength[c] / two_m).powi(2))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::{karate, karate_optimum};

    #[test]
    fn rb_at_resolution_one_is_modularity() {
        let g = karate();
        let labels = karate_optimum();
        let p = Partition::from_membership(&g, &labels);

        let q_scaled = Rb::new(1.0).quality(&p);
        let q_standard = modularity(&g, &labels, 1.0);

        // Rb::quality is m * Q, the scaling the old code got wrong by 2x
        assert!(
            (q_scaled / g.total_weight() - q_standard).abs() < 1e-12,
            "Rb::quality/m = {}, modularity = {q_standard}",
            q_scaled / g.total_weight()
        );
        assert!(
            (q_standard - 0.4198).abs() < 1e-3,
            "karate optimum should score 0.4198, got {q_standard}"
        );
    }

    #[test]
    fn rb_resolution_scales_the_null_term_not_the_objective() {
        let g = karate();
        let labels = karate_optimum();
        let p = Partition::from_membership(&g, &labels);
        // Rb(gamma) == textbook Q(gamma), not Q(2*gamma) as before
        for gamma in [0.5, 1.0, 2.0, 4.0] {
            let got = Rb::new(gamma).quality(&p) / g.total_weight();
            let want = modularity(&g, &labels, gamma);
            assert!((got - want).abs() < 1e-12, "gamma={gamma}: {got} vs {want}");
        }
    }

    /// Catches any drift between the delta and the objective.
    fn delta_matches_quality_change<O: Objective>(obj: &O, graph: &CSRNetwork, labels: &[usize]) {
        use crate::community_search::leiden::partition::NeighborWeights;
        let mut p = Partition::from_membership(graph, labels);
        let mut acc = NeighborWeights::with_capacity(p.slots());
        let n_comms = p.slots();

        for v in 0..graph.node_count() {
            for target in 0..n_comms {
                let before = obj.quality(&p);
                let ctx = InsertContext::for_node(graph, v);

                acc.collect(graph, p.membership_raw(), v);
                let own = p.membership(v);
                p.remove_node(v, graph, acc.weight_to(own));
                let predicted = obj.delta_insert(&p, &ctx, target, acc.weight_to(target));
                p.insert_node(v, target, graph, acc.weight_to(target));
                let after = obj.quality(&p);

                let removed_gain = {
                    // undo
                    acc.collect(graph, p.membership_raw(), v);
                    p.remove_node(v, graph, acc.weight_to(target));
                    let g = obj.delta_insert(&p, &ctx, own, acc.weight_to(own));
                    p.insert_node(v, own, graph, acc.weight_to(own));
                    g
                };

                let actual = after - before;
                let expected = predicted - removed_gain;
                assert!(
                    (expected - actual).abs() < 1e-9,
                    "node {v} -> comm {target}: delta {expected} != actual {actual}"
                );
            }
        }
    }

    #[test]
    fn rb_delta_equals_quality_change() {
        let g = karate();
        delta_matches_quality_change(&Rb::new(1.0), &g, &karate_optimum());
        delta_matches_quality_change(&Rb::new(0.37), &g, &karate_optimum());
    }

    #[test]
    fn cpm_delta_equals_quality_change() {
        let g = karate();
        delta_matches_quality_change(&Cpm::new(0.1), &g, &karate_optimum());
    }

    #[test]
    fn quality_is_invariant_under_aggregation() {
        // what the old aggregate() broke: collapsing must not change the score
        let g = karate();
        let labels = karate_optimum();
        let mut p = Partition::from_membership(&g, &labels);

        let grouping = p.renumber_into_grouping();
        let coarse_graph = g.aggregate(&grouping);
        let coarse_labels: Vec<usize> = (0..coarse_graph.node_count()).collect();
        let coarse_p = Partition::from_membership(&coarse_graph, &coarse_labels);

        for gamma in [0.5, 1.0, 2.0] {
            let fine = Rb::new(gamma).quality(&p);
            let coarse = Rb::new(gamma).quality(&coarse_p);
            assert!(
                (fine - coarse).abs() < 1e-9,
                "gamma={gamma}: fine {fine} != coarse {coarse}"
            );
        }
        let fine = Cpm::new(0.1).quality(&p);
        let coarse = Cpm::new(0.1).quality(&coarse_p);
        assert!((fine - coarse).abs() < 1e-9, "cpm: {fine} != {coarse}");
    }

    #[test]
    fn empty_graph_scores_zero() {
        let g = CSRNetwork::from_edges(5, &[] as &[(usize, usize, f64)]).unwrap();
        let p = Partition::singleton(&g);
        assert_eq!(Rb::new(1.0).quality(&p), 0.0);
        assert_eq!(modularity(&g, &[0, 1, 2, 3, 4], 1.0), 0.0);
    }
}
