//! Refinement — what makes this Leiden and not Louvain.
//!
//! Re-derives a finer partition inside each community local moving found, starting from
//! singletons. Aggregation then collapses the *refined* communities, so Louvain's
//! badly-connected ones don't get locked in.
//!
//! Two rules, both of which the pre-0.7 implementation got wrong:
//!
//! 1. Only singletons may merge. Once absorbed a node never moves again, otherwise refined
//!    communities stop being bottom-up unions and the guarantee is gone.
//! 2. Moves stay inside the constraining community.
//!
//! Candidate selection is randomised (Leiden paper, Algorithm 3): draw with probability
//! proportional to `exp(gain / randomness)`. Greedy argmax gets stuck where repeated passes
//! can't help. `randomness = 0.0` is greedy.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::community_search::leiden::objective::{CommunityView, InsertContext, Objective};
use crate::community_search::leiden::partition::{MoveScratch, Partition};
use crate::network::CSRNetwork;

/// Builds a refined partition inside the communities of `constraint`.
///
/// One pass over a shuffled order is enough: only singletons merge and each node is seen once,
/// so it can't cycle. That's why zero-gain merges are safe here — they escape plateaus.
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
    if n == 0 {
        return Partition::singleton(graph);
    }

    // relative to a typical edge, so rescaling weights changes nothing
    let mean_edge_weight = if graph.edge_count() > 0 {
        graph.total_weight() / graph.edge_count() as f64
    } else {
        1.0
    };
    let randomness = randomness * mean_edge_weight;

    // one draw per node, up front. A shared stream would make a node's randomness depend on
    // how many before it needed one, coupling communities that are otherwise independent.
    let draws: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();

    // bucketing keeps each community's relative order, so it sees the same sequence either way
    let order = scratch.shuffled_order(n, rng);
    // sized by the largest constraint id, not the node count — these are inherited from the
    // previous level and routinely exceed it
    let slots = constraint
        .iter()
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); slots];
    for &v in order.iter() {
        members[constraint[v] as usize].push(v as u32);
    }

    let total_weight = graph.total_weight();

    // Each community writes only its own members' labels, so the slices are disjoint. Refine
    // one at a time and merge, which keeps that obvious whether or not it runs in parallel.
    let refine_one = |m: &Vec<u32>| {
        let mut sorted = m.clone();
        sorted.sort_unstable();
        let mut out = vec![0u32; sorted.len()];
        refine_into(
            graph,
            m,
            &sorted,
            objective,
            total_weight,
            &draws,
            max_community_weight,
            randomness,
            &mut out,
        );
        (sorted, out)
    };
    let nonempty: Vec<&Vec<u32>> = members.iter().filter(|m| !m.is_empty()).collect();
    let refined = crate::par::map_range(nonempty.len(), |i| refine_one(nonempty[i]));

    let mut labels = vec![0u32; n];
    for (sorted, out) in refined {
        for (&g, &label) in sorted.iter().zip(out.iter()) {
            labels[g as usize] = label;
        }
    }

    // rebuild from the labels, so aggregates are exact rather than maintained across a split
    let as_usize: Vec<usize> = labels.iter().map(|&c| c as usize).collect();
    Partition::from_membership(graph, &as_usize)
}

/// Draws a candidate with probability proportional to `exp(gain / randomness)`.
///
/// Subtracting `best_rel` first keeps weights in `(0, 1]` so a large ratio can't overflow.
/// `randomness <= 0` is argmax, ties to the lowest id.
fn select(
    candidates: &[(usize, f64)],
    best_rel: f64,
    randomness: f64,
    draw: f64,
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

    let mut threshold = draw * total;
    for (&(c, _), &w) in candidates.iter().zip(weights.iter()) {
        threshold -= w;
        if threshold <= 0.0 {
            return Some(c);
        }
    }
    // float drift; take the last rather than lose the move
    candidates.last().map(|&(c, _)| c)
}

/// Refines one constraining community, independently of every other.
///
/// A node can only merge inside its own constraining community, so two of them share no state.
/// Local ids rather than the global partition is what makes that disjointness visible to the
/// compiler instead of merely true.
///
/// `members` is the community's global node ids in visit order, `sorted` the same set
/// ascending for global-to-local lookup. Each member's label goes into `out` as the global id
/// its refined community formed around, so labels can't collide across communities.
#[allow(clippy::too_many_arguments)]
fn refine_into(
    graph: &CSRNetwork,
    members: &[u32],
    sorted: &[u32],
    objective: &dyn Objective,
    total_weight: f64,
    draws: &[f64],
    max_community_weight: Option<f64>,
    randomness: f64,
    out: &mut [u32],
) {
    let k = sorted.len();
    if k == 0 {
        return;
    }
    let local_of =
        |g: u32| -> usize { sorted.binary_search(&g).expect("member of this community") };

    // local singleton partition. Only the three aggregates `delta_insert_view` reads are
    // tracked; `internal` is never consulted here, and the caller rebuilds anyway.
    let mut membership: Vec<u32> = (0..k as u32).collect();
    let mut size = vec![1u32; k];
    let mut strength = vec![0.0f64; k];
    let mut weight = vec![0.0f64; k];
    for (local, &g) in sorted.iter().enumerate() {
        strength[local] = graph.strength(g as usize);
        weight[local] = graph.node_weight(g as usize);
    }

    // weight to each local community, cleared via `touched`
    let mut to_comm = vec![0.0f64; k];
    let mut touched: Vec<usize> = Vec::new();
    let mut candidates: Vec<(usize, f64)> = Vec::new();
    let mut exp_buf: Vec<f64> = Vec::new();

    for &g in members {
        let v = local_of(g);
        let own = membership[v] as usize;
        // rule 1: singletons only
        if size[own] != 1 {
            continue;
        }

        for &c in &touched {
            to_comm[c] = 0.0;
        }
        touched.clear();

        // rule 2: neighbours outside the constraint aren't in `sorted`, so the search
        // filters them out
        for (u, w) in graph.neighbors(g as usize) {
            if u == g as usize {
                continue;
            }
            if let Ok(local_u) = sorted.binary_search(&(u as u32)) {
                let c = membership[local_u] as usize;
                if to_comm[c] == 0.0 {
                    touched.push(c);
                }
                to_comm[c] += w;
            }
        }

        let ctx = InsertContext::for_node(graph, g as usize);
        let weight_to_own = to_comm[own];

        // take it out; its community is a singleton, so it empties
        size[own] -= 1;
        strength[own] -= ctx.node_strength;
        weight[own] -= ctx.node_weight;

        let view_of = |c: usize| CommunityView {
            total_weight,
            strength: strength[c],
            weight: weight[c],
        };
        let base_gain = objective.delta_insert_view(view_of(own), &ctx, weight_to_own);

        // candidates that don't lose quality
        candidates.clear();
        let mut best_rel = 0.0f64;
        for &c in &touched {
            if c == own {
                continue;
            }
            if let Some(limit) = max_community_weight
                && weight[c] + ctx.node_weight > limit
            {
                continue;
            }
            let rel = objective.delta_insert_view(view_of(c), &ctx, to_comm[c]) - base_gain;
            // `>=` takes ties, like libleidenalg
            if rel >= 0.0 {
                candidates.push((c, rel));
                best_rel = best_rel.max(rel);
            }
        }

        let target = select(
            &candidates,
            best_rel,
            randomness,
            draws[g as usize],
            &mut exp_buf,
        )
        .unwrap_or(own);

        size[target] += 1;
        strength[target] += ctx.node_strength;
        weight[target] += ctx.node_weight;
        membership[v] = target as u32;
    }

    // label by the global id the refined community formed around, so labels are unique
    // without coordination. `out` is local, matching `sorted`; the caller scatters it.
    for local in 0..k {
        out[local] = sorted[membership[local] as usize];
    }
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
        // one constraint community; must still split the barbell
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
