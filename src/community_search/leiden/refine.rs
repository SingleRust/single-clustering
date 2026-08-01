//! Refinement — what makes this Leiden and not Louvain.
//!
//! Re-derives a finer partition inside each community local moving found, from singletons,
//! merging only within a community. Aggregation then collapses the *refined* communities, so
//! Louvain's badly-connected (sometimes disconnected) communities don't get locked in.
//!
//! Two rules, both of which the previous implementation got wrong:
//!
//! 1. Only singletons may merge — once absorbed, a node is never moved again. Otherwise
//!    refined communities stop being bottom-up unions and the guarantee is gone.
//! 2. Moves stay inside the constraining community.
//!
//! Candidate selection is randomised (Leiden paper, Algorithm 3): draw with probability
//! proportional to `exp(gain / randomness)`. Greedy argmax gets stuck somewhere repeated
//! passes can't escape; this is how it explores past that. `randomness = 0.0` is greedy.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::community_search::leiden::objective::{InsertContext, Objective};
use crate::community_search::leiden::partition::{MoveScratch, Partition};
use crate::network::CSRNetwork;

/// Builds a refined partition inside the communities of `constraint`.
///
/// Every refined community sits inside exactly one community of `constraint`. One pass over a
/// shuffled order is enough — only singletons merge and each node is seen once, so it can't
/// cycle. That's why zero-gain merges are safe to accept here; they help escape plateaus.
pub fn refine(
    graph: &CSRNetwork,
    constraint: &[u32],
    objective: &dyn Objective,
    scratch: &mut MoveScratch,
    rng: &mut ChaCha8Rng,
    max_community_weight: Option<f64>,
    randomness: f64,
) -> Partition {
    let n = graph.node_count();
    let mut refined = Partition::singleton(graph);
    if n == 0 {
        return refined;
    }

    // relative to a typical edge, so rescaling weights doesn't change behaviour
    let mean_edge_weight = if graph.edge_count() > 0 {
        graph.total_weight() / graph.edge_count() as f64
    } else {
        1.0
    };
    let randomness = randomness * mean_edge_weight;

    // snapshot — the loop body needs `scratch` mutably
    let order: Vec<usize> = scratch.shuffled_order(n, rng).to_vec();
    let mut candidates: Vec<(usize, f64)> = Vec::new();
    let mut exp_buf: Vec<f64> = Vec::new();

    for v in order {
        let own = refined.membership(v);
        // rule 1: singletons only
        if refined.size(own) != 1 {
            continue;
        }

        let ctx = InsertContext::for_node(graph, v);
        // rule 2: stay inside the constraint
        scratch.weights.ensure_capacity(refined.slots() + 1);
        scratch
            .weights
            .collect_constrained(graph, refined.membership_raw(), v, constraint);

        let weight_to_own = scratch.weights.weight_to(own);
        refined.remove_node(v, graph, weight_to_own);

        let base_gain = objective.delta_insert(&refined, &ctx, own, weight_to_own);

        // candidates that don't lose quality
        candidates.clear();
        let mut best_rel = 0.0f64;
        for i in 0..scratch.weights.touched().len() {
            let c = scratch.weights.touched()[i];
            if c == own {
                continue;
            }
            if let Some(limit) = max_community_weight
                && refined.weight(c) + ctx.node_weight > limit
            {
                continue;
            }
            let rel =
                objective.delta_insert(&refined, &ctx, c, scratch.weights.weight_to(c)) - base_gain;
            // `>=` takes ties, like libleidenalg
            if rel >= 0.0 {
                candidates.push((c, rel));
                best_rel = best_rel.max(rel);
            }
        }

        let target = select(&candidates, best_rel, randomness, rng, &mut exp_buf).unwrap_or(own);
        refined.insert_node(v, target, graph, scratch.weights.weight_to(target));
    }

    refined
}

/// Draws a candidate with probability proportional to `exp(gain / randomness)`.
///
/// Subtracting `best_rel` before exponentiating keeps weights in `(0, 1]`, so a large
/// gain/randomness ratio can't overflow. `randomness <= 0` is a deterministic argmax, ties
/// going to the lowest id so traversal order never shows through.
fn select(
    candidates: &[(usize, f64)],
    best_rel: f64,
    randomness: f64,
    rng: &mut ChaCha8Rng,
    weights: &mut Vec<f64>,
) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    if randomness <= 0.0 {
        return candidates
            .iter()
            .filter(|&&(_, rel)| rel >= best_rel)
            .map(|&(c, _)| c)
            .min();
    }

    // one exp per candidate; computing it twice cost ~4%
    weights.clear();
    weights.extend(
        candidates
            .iter()
            .map(|&(_, rel)| ((rel - best_rel) / randomness).exp()),
    );
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return candidates
            .iter()
            .filter(|&&(_, rel)| rel >= best_rel)
            .map(|&(c, _)| c)
            .min();
    }

    let mut threshold = rng.random::<f64>() * total;
    for (&(c, _), &w) in candidates.iter().zip(weights.iter()) {
        threshold -= w;
        if threshold <= 0.0 {
            return Some(c);
        }
    }
    // float drift; take the last rather than lose the move
    candidates.last().map(|&(c, _)| c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::community_search::leiden::objective::Rb;
    use rand::SeedableRng;

    fn barbell() -> CSRNetwork {
        // two 5-cliques joined by a single edge
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

    #[test]
    fn refined_communities_never_cross_the_constraint() {
        let g = barbell();
        let constraint: Vec<u32> = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let mut scratch = MoveScratch::with_capacity(g.node_count(), g.node_count());
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let refined = refine(
            &g,
            &constraint,
            &Rb::new(1.0),
            &mut scratch,
            &mut rng,
            None,
            0.0,
        );

        // every refined community must sit inside exactly one constraint community
        let mut owner = std::collections::HashMap::new();
        for (v, &cv) in constraint.iter().enumerate() {
            let r = refined.membership(v);
            let entry = owner.entry(r).or_insert(cv);
            assert_eq!(*entry, cv, "refined community {r} straddles the constraint");
        }
        refined.verify_against(&g).unwrap();
    }

    #[test]
    fn a_single_constraint_community_can_still_split() {
        // one constraint community, should still split the barbell rather than merge it
        let g = barbell();
        let constraint: Vec<u32> = vec![0; 10];
        let mut scratch = MoveScratch::with_capacity(g.node_count(), g.node_count());
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut refined = refine(
            &g,
            &constraint,
            &Rb::new(1.0),
            &mut scratch,
            &mut rng,
            None,
            0.0,
        );
        refined.renumber();
        assert!(
            refined.community_count() >= 2,
            "expected the barbell to split, got {} communities",
            refined.community_count()
        );
    }

    #[test]
    fn refinement_is_deterministic() {
        let g = barbell();
        let constraint: Vec<u32> = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let mut first: Option<Vec<usize>> = None;
        for _ in 0..10 {
            let mut scratch = MoveScratch::with_capacity(g.node_count(), g.node_count());
            let mut rng = ChaCha8Rng::seed_from_u64(77);
            let mut r = refine(
                &g,
                &constraint,
                &Rb::new(1.0),
                &mut scratch,
                &mut rng,
                None,
                0.0,
            );
            r.renumber();
            let labels = r.membership_vec();
            match &first {
                None => first = Some(labels),
                Some(f) => assert_eq!(f, &labels),
            }
        }
    }

    #[test]
    fn singletons_only_rule_holds() {
        // aggregates stay exact and nothing straddles the constraint
        let (edges, truth) = crate::testdata::sbm(20, 4, 0.4, 0.02, 5);
        let g = CSRNetwork::from_edges(80, &edges).unwrap();
        let mut scratch = MoveScratch::with_capacity(g.node_count(), g.node_count());
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let truth32: Vec<u32> = truth.iter().map(|&c| c as u32).collect();
        let refined = refine(
            &g,
            &truth32,
            &Rb::new(1.0),
            &mut scratch,
            &mut rng,
            None,
            0.0,
        );
        refined.verify_against(&g).unwrap();
        for (v, &tv) in truth.iter().enumerate() {
            for (u, _) in g.neighbors(v) {
                if refined.membership(u) == refined.membership(v) {
                    assert_eq!(truth[u], tv, "refined community crosses the constraint");
                }
            }
        }
    }
}
