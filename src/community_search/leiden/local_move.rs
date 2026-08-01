//! Local moving: the inner loop of Leiden.
//!
//! Take a node out of its community, put it wherever it scores best, repeat until nothing
//! wants to move. Nodes come off a queue seeded with a shuffled order; a node that moves
//! re-queues its neighbours, so only the affected part gets revisited.
//!
//! O(degree) per node, and nothing allocates per node or per candidate.

use rand_chacha::ChaCha8Rng;

use crate::community_search::leiden::objective::{InsertContext, Objective};
#[allow(unused_imports)]
use crate::community_search::leiden::partition::NeighborWeights;
use crate::community_search::leiden::partition::{MoveScratch, Partition};
use crate::network::CSRNetwork;

/// Improvement floor, as a fraction of total edge weight.
///
/// Relative, not absolute: otherwise scaling every weight by a constant could change the
/// clustering, even though the objective itself is scale-invariant. Low enough to sit under
/// any real gain, high enough that float dust doesn't cause no-op move cycles.
pub(crate) const MIN_GAIN_RELATIVE: f64 = 1e-12;

/// The absolute improvement threshold for a given graph.
#[inline]
pub(crate) fn min_gain(partition: &Partition) -> f64 {
    MIN_GAIN_RELATIVE * partition.total_weight().abs().max(1.0)
}

/// Move cap, as a multiple of node count. Only a safety valve — this terminates on its own,
/// since every accepted move strictly increases a bounded objective.
const MAX_MOVES_PER_NODE: usize = 100;

/// Moves nodes between communities until no single move improves the objective.
///
/// Returns the total quality improvement. `max_community_weight` caps a community's summed
/// node weight; node weights add up when communities collapse, so it means the same thing at
/// every level.
pub fn local_move(
    graph: &CSRNetwork,
    partition: &mut Partition,
    objective: &dyn Objective,
    scratch: &mut MoveScratch,
    rng: &mut ChaCha8Rng,
    max_community_weight: Option<f64>,
) -> f64 {
    let n = graph.node_count();
    if n == 0 {
        return 0.0;
    }

    scratch.seed_queue(n, rng);
    let threshold = min_gain(partition);
    let mut total_gain = 0.0;
    let mut moves = 0usize;
    let move_budget = n.saturating_mul(MAX_MOVES_PER_NODE);

    while let Some(v) = scratch.pop() {
        let ctx = InsertContext::for_node(graph, v);
        // hoisted out of the neighbour loop, ~15x fewer calls
        scratch.weights.ensure_capacity(partition.slots() + 1);
        scratch
            .weights
            .collect(graph, partition.membership_raw(), v);

        let old = partition.membership(v);
        let weight_to_old = scratch.weights.weight_to(old);
        partition.remove_node(v, graph, weight_to_old);

        // staying put is just another candidate — keeps the empty case uniform
        let base_gain = objective.delta_insert(partition, &ctx, old, weight_to_old);
        let mut best = old;
        let mut best_gain = base_gain;

        let fits = |partition: &Partition, c: usize| match max_community_weight {
            Some(limit) => partition.weight(c) + ctx.node_weight <= limit,
            None => true,
        };

        for i in 0..scratch.weights.touched().len() {
            let c = scratch.weights.touched()[i];
            if c == old || !fits(partition, c) {
                continue;
            }
            let gain = objective.delta_insert(partition, &ctx, c, scratch.weights.weight_to(c));
            if gain > best_gain || (gain == best_gain && c < best) {
                best = c;
                best_gain = gain;
            }
        }

        // one per node, claimed below, so no two get the same
        let empty = partition.empty_community();
        if empty != old && fits(partition, empty) {
            let gain = objective.delta_insert(partition, &ctx, empty, 0.0);
            if gain > best_gain || (gain == best_gain && empty < best) {
                best = empty;
                best_gain = gain;
            }
        }

        let accept = best != old && best_gain - base_gain > threshold && moves < move_budget;
        let target = if accept { best } else { old };
        partition.insert_node(v, target, graph, scratch.weights.weight_to(target));

        if !accept {
            continue;
        }

        total_gain += best_gain - base_gain;
        moves += 1;

        // only neighbours outside the new community can have gone unstable
        for (u, _) in graph.neighbors(v) {
            if u != v && !scratch.is_queued(u) && partition.membership(u) != target {
                scratch.push(u);
            }
        }
    }

    total_gain
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::community_search::leiden::objective::{Rb, modularity};
    use rand::SeedableRng;

    fn two_cliques() -> CSRNetwork {
        let mut edges = Vec::new();
        for b in 0..2usize {
            for i in 0..5 {
                for j in (i + 1)..5 {
                    edges.push((b * 5 + i, b * 5 + j, 1.0));
                }
            }
        }
        edges.push((0, 5, 1.0));
        CSRNetwork::from_edges(10, &edges).unwrap()
    }

    fn run(graph: &CSRNetwork, seed: u64) -> (Partition, f64) {
        let mut p = Partition::singleton(graph);
        let mut scratch = MoveScratch::with_capacity(graph.node_count(), p.slots());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let gain = local_move(graph, &mut p, &Rb::new(1.0), &mut scratch, &mut rng, None);
        (p, gain)
    }

    #[test]
    fn separates_two_cliques() {
        let g = two_cliques();
        let (mut p, _) = run(&g, 42);
        p.renumber();
        let labels = p.membership_vec();
        assert_eq!(p.community_count(), 2, "labels: {labels:?}");
        for i in 1..5 {
            assert_eq!(labels[i], labels[0]);
            assert_eq!(labels[5 + i], labels[5]);
        }
        assert_ne!(labels[0], labels[5]);
    }

    #[test]
    fn reported_gain_matches_the_objective() {
        let g = two_cliques();
        let obj = Rb::new(1.0);
        let mut p = Partition::singleton(&g);
        let before = obj.quality(&p);
        let mut scratch = MoveScratch::with_capacity(g.node_count(), p.slots());
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let gain = local_move(&g, &mut p, &obj, &mut scratch, &mut rng, None);
        let after = obj.quality(&p);
        assert!(
            (after - before - gain).abs() < 1e-9,
            "reported {gain}, actual {}",
            after - before
        );
    }

    #[test]
    fn aggregates_stay_exact_after_local_moving() {
        let g = two_cliques();
        let (p, _) = run(&g, 3);
        p.verify_against(&g).unwrap();
    }

    #[test]
    fn quality_never_decreases() {
        let g = two_cliques();
        let obj = Rb::new(1.0);
        for seed in 0..25 {
            let mut p = Partition::singleton(&g);
            let before = obj.quality(&p);
            let mut scratch = MoveScratch::with_capacity(g.node_count(), p.slots());
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            local_move(&g, &mut p, &obj, &mut scratch, &mut rng, None);
            assert!(obj.quality(&p) >= before - 1e-12, "seed {seed}");
        }
    }

    #[test]
    fn is_deterministic_for_a_fixed_seed() {
        let g = two_cliques();
        let (mut first, gain) = run(&g, 99);
        first.renumber();
        for _ in 0..10 {
            let (mut again, g2) = run(&g, 99);
            again.renumber();
            assert_eq!(first.membership_vec(), again.membership_vec());
            assert_eq!(gain, g2);
        }
    }

    #[test]
    fn respects_max_community_weight() {
        let g = two_cliques();
        let mut p = Partition::singleton(&g);
        let mut scratch = MoveScratch::with_capacity(g.node_count(), p.slots());
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        local_move(&g, &mut p, &Rb::new(1.0), &mut scratch, &mut rng, Some(3.0));
        for c in 0..p.slots() {
            assert!(
                p.weight(c) <= 3.0,
                "community {c} has weight {}",
                p.weight(c)
            );
        }
    }

    #[test]
    fn handles_degenerate_graphs() {
        for g in [
            CSRNetwork::from_edges(0, &[] as &[(usize, usize, f64)]).unwrap(),
            CSRNetwork::from_edges(1, &[] as &[(usize, usize, f64)]).unwrap(),
            CSRNetwork::from_edges(4, &[] as &[(usize, usize, f64)]).unwrap(),
            CSRNetwork::from_edges(2, &[(0, 0, 1.0), (1, 1, 1.0)]).unwrap(),
        ] {
            let (p, _) = run(&g, 5);
            p.verify_against(&g).unwrap();
        }
    }

    #[test]
    fn improves_karate_modularity_over_singletons() {
        let g = crate::testdata::karate();
        let (mut p, _) = run(&g, 42);
        p.renumber();
        let q = modularity(&g, &p.membership_vec(), 1.0);
        // one pass from singletons gives Q = 0.31..0.40; 0.4198 needs the aggregation levels
        assert!(q > 0.30, "karate modularity after one pass: {q}");
        assert!((3..=12).contains(&p.community_count()));
    }
}
