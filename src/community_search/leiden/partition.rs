//! Partition state.
//!
//! [`Partition`] holds every aggregate the quality functions need — sizes, node weights,
//! strengths, internal edge weights — updated in O(1) per move. Nothing here walks the graph
//! or allocates per call, which is what keeps local moving at O(degree) rather than O(n).
//!
//! A move is [`remove_node`](Partition::remove_node) then
//! [`insert_node`](Partition::insert_node), not one operation. The node sits in no community
//! in between, so scoring every candidate — its old community, an empty one, anything else —
//! is the same calculation.

use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use crate::network::CSRNetwork;
use crate::network::grouping::{NetworkGrouping, VectorGrouping};

/// Membership value for a node that has been removed from its community.
///
/// Visible in [`Partition::membership_raw`] while a move is in progress; never in a
/// partition handed back to a caller.
pub const UNASSIGNED: u32 = u32::MAX;

/// A partition of a graph's nodes into communities, with incrementally maintained aggregates.
#[derive(Debug, Clone)]
pub struct Partition {
    /// Community of each node, or [`UNASSIGNED`] while a node is removed.
    ///
    /// `u32`, not `usize`: this is the hottest random access in the algorithm — every
    /// neighbour of every visited node reads it — so halving the array halves that cache
    /// footprint. Ids are bounded by the node count, which `CSRNetwork` caps at `u32::MAX`.
    membership: Vec<u32>,
    /// Number of nodes in each community.
    size: Vec<u32>,
    /// Sum of node weights in each community. This is what CPM measures.
    weight: Vec<f64>,
    /// Sum of node strengths in each community. This is `K_c` in the RB null model.
    strength: Vec<f64>,
    /// Internal edge weight of each community: edges with both endpoints inside, each
    /// undirected edge counted once, self-loops counted once at full weight.
    internal: Vec<f64>,
    /// Community slots that currently hold no nodes, available for reuse.
    free: Vec<usize>,
    /// Total edge weight, each edge once. Kept here so quality functions don't need the graph,
    /// and so it stays consistent across levels.
    total_weight: f64,
    /// Sum of all node weights.
    total_node_weight: f64,
}

impl Partition {
    /// Creates a partition where every node is alone in its own community.
    pub fn singleton(graph: &CSRNetwork) -> Self {
        let n = graph.node_count();
        let mut p = Self {
            membership: (0..n as u32).collect(),
            size: vec![1; n],
            weight: (0..n).map(|v| graph.node_weight(v)).collect(),
            strength: (0..n).map(|v| graph.strength(v)).collect(),
            // only internal edge is its own self-loop
            internal: (0..n).map(|v| graph.self_loop_weight(v)).collect(),
            free: Vec::new(),
            total_weight: graph.total_weight(),
            total_node_weight: graph.total_node_weight(),
        };
        // no nodes, no communities
        p.free.clear();
        p
    }

    /// Creates a partition from an explicit membership vector.
    ///
    /// Ids need not be consecutive — empty slots below the maximum are recorded as free and
    /// reused before anything new is allocated.
    pub fn from_membership(graph: &CSRNetwork, membership: &[usize]) -> Self {
        debug_assert_eq!(membership.len(), graph.node_count());
        let slots = membership.iter().copied().max().map_or(0, |m| m + 1);

        let mut p = Self {
            membership: membership.iter().map(|&c| c as u32).collect(),
            size: vec![0; slots],
            weight: vec![0.0; slots],
            strength: vec![0.0; slots],
            internal: vec![0.0; slots],
            free: Vec::new(),
            total_weight: graph.total_weight(),
            total_node_weight: graph.total_node_weight(),
        };

        for (v, &c) in membership.iter().enumerate() {
            p.size[c] += 1;
            p.weight[c] += graph.node_weight(v);
            p.strength[c] += graph.strength(v);
        }
        for (v, &c) in membership.iter().enumerate() {
            for (u, w) in graph.neighbors(v) {
                if membership[u] != c {
                    continue;
                }
                // each edge once: loops here, regular ones from the lower id
                if u == v || v < u {
                    p.internal[c] += w;
                }
            }
        }
        for c in (0..slots).rev() {
            if p.size[c] == 0 {
                p.free.push(c);
            }
        }
        p
    }

    /// Number of community slots, including empty ones.
    #[inline]
    pub fn slots(&self) -> usize {
        self.size.len()
    }

    /// Number of non-empty communities.
    #[inline]
    pub fn community_count(&self) -> usize {
        self.size.len() - self.free.len()
    }

    /// Number of nodes in the partition.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.membership.len()
    }

    /// Community of a node. Panics in debug builds if the node is currently removed.
    #[inline]
    pub fn membership(&self, node: usize) -> usize {
        debug_assert_ne!(self.membership[node], UNASSIGNED, "node {node} is removed");
        self.membership[node] as usize
    }

    /// The raw membership slice, which may contain [`UNASSIGNED`] mid-move.
    #[inline]
    pub fn membership_raw(&self) -> &[u32] {
        &self.membership
    }

    /// The membership as `usize` values, allocated fresh.
    pub fn membership_vec(&self) -> Vec<usize> {
        self.membership.iter().map(|&c| c as usize).collect()
    }

    /// Number of nodes in a community.
    #[inline]
    pub fn size(&self, community: usize) -> usize {
        self.size.get(community).copied().unwrap_or(0) as usize
    }

    /// Sum of node weights in a community — `n_c` in CPM.
    #[inline]
    pub fn weight(&self, community: usize) -> f64 {
        self.weight.get(community).copied().unwrap_or(0.0)
    }

    /// Sum of node strengths in a community — `K_c` in the RB null model.
    #[inline]
    pub fn strength(&self, community: usize) -> f64 {
        self.strength.get(community).copied().unwrap_or(0.0)
    }

    /// Internal edge weight of a community, each edge counted once.
    #[inline]
    pub fn internal(&self, community: usize) -> f64 {
        self.internal.get(community).copied().unwrap_or(0.0)
    }

    /// Total edge weight of the underlying graph, each edge counted once.
    #[inline]
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Sum of all node weights.
    #[inline]
    pub fn total_node_weight(&self) -> f64 {
        self.total_node_weight
    }

    /// Returns a community id that is currently empty, without allocating.
    ///
    /// Reuses a free slot if there is one, otherwise the next index, which
    /// [`insert_node`](Self::insert_node) allocates on demand. Each caller takes exactly one
    /// and fills it immediately, so two nodes can't get the same empty community.
    #[inline]
    pub fn empty_community(&self) -> usize {
        self.free.last().copied().unwrap_or(self.size.len())
    }

    /// Removes a node from its community, updating all aggregates in O(1).
    ///
    /// `weight_to_own` is the total weight of edges from `node` to other members of its
    /// community, **excluding** the node's self-loop — exactly what
    /// [`NeighborWeights`] accumulates. Returns the community the node was in.
    pub fn remove_node(&mut self, node: usize, graph: &CSRNetwork, weight_to_own: f64) -> usize {
        let c = self.membership[node] as usize;
        debug_assert_ne!(
            self.membership[node], UNASSIGNED,
            "node {node} was already removed"
        );

        self.size[c] -= 1;
        self.weight[c] -= graph.node_weight(node);
        self.strength[c] -= graph.strength(node);
        // its edges in, plus its self-loop, stop being internal
        self.internal[c] -= weight_to_own + graph.self_loop_weight(node);

        if self.size[c] == 0 {
            // clamp drift so empty is exactly empty
            self.weight[c] = 0.0;
            self.strength[c] = 0.0;
            self.internal[c] = 0.0;
            self.free.push(c);
        }

        self.membership[node] = UNASSIGNED;
        c
    }

    /// Inserts a removed node into a community, updating all aggregates in O(1).
    ///
    /// `community` may be [`empty_community`](Self::empty_community), in which case a new
    /// slot is allocated. `weight_to_comm` excludes the node's self-loop.
    pub fn insert_node(
        &mut self,
        node: usize,
        community: usize,
        graph: &CSRNetwork,
        weight_to_comm: f64,
    ) {
        debug_assert_eq!(
            self.membership[node], UNASSIGNED,
            "node {node} is not removed"
        );
        debug_assert!(community <= self.size.len(), "community id skips a slot");

        if community == self.size.len() {
            self.size.push(0);
            self.weight.push(0.0);
            self.strength.push(0.0);
            self.internal.push(0.0);
        } else if self.size[community] == 0 {
            // reusing a free slot
            if let Some(pos) = self.free.iter().rposition(|&c| c == community) {
                self.free.swap_remove(pos);
            }
        }

        self.size[community] += 1;
        self.weight[community] += graph.node_weight(node);
        self.strength[community] += graph.strength(node);
        self.internal[community] += weight_to_comm + graph.self_loop_weight(node);
        self.membership[node] = community as u32;
    }

    /// Renumbers communities consecutively from 0, keeping their relative order.
    ///
    /// Ascending order, so the output never depends on iteration order upstream.
    pub fn renumber(&mut self) {
        let slots = self.size.len();
        let mut new_id = vec![usize::MAX; slots];
        let mut next = 0;
        for (id, size) in new_id.iter_mut().zip(self.size.iter()) {
            if *size > 0 {
                *id = next;
                next += 1;
            }
        }

        for (c, &target) in new_id.iter().enumerate() {
            if target == usize::MAX || target == c {
                continue;
            }
            self.size[target] = self.size[c];
            self.weight[target] = self.weight[c];
            self.strength[target] = self.strength[c];
            self.internal[target] = self.internal[c];
        }
        self.size.truncate(next);
        self.weight.truncate(next);
        self.strength.truncate(next);
        self.internal.truncate(next);
        self.free.clear();

        for m in self.membership.iter_mut() {
            if *m != UNASSIGNED {
                *m = new_id[*m as usize] as u32;
            }
        }
    }

    /// Renumbers this partition and returns its membership as a [`VectorGrouping`].
    ///
    /// `&mut self` on purpose: the grouping's ids and this partition's must be the same
    /// numbers, since callers use `membership(v)` to index the graph
    /// [`CSRNetwork::aggregate`] produces. Renumbering a copy would let them drift silently.
    pub fn renumber_into_grouping(&mut self) -> VectorGrouping {
        self.renumber();
        VectorGrouping::from_assignments(&self.membership_vec())
    }

    /// Recomputes every aggregate from scratch and checks it against the maintained value.
    ///
    /// Used by tests to prove the incremental updates never drift.
    pub fn verify_against(&self, graph: &CSRNetwork) -> std::result::Result<(), String> {
        let rebuilt = Self::from_membership(graph, &self.membership_vec());
        for c in 0..self.size.len() {
            if self.size[c] as usize != rebuilt.size(c) {
                return Err(format!(
                    "community {c}: size {} != {}",
                    self.size[c],
                    rebuilt.size(c)
                ));
            }
            for (name, got, want) in [
                ("weight", self.weight[c], rebuilt.weight(c)),
                ("strength", self.strength[c], rebuilt.strength(c)),
                ("internal", self.internal[c], rebuilt.internal(c)),
            ] {
                if (got - want).abs() > 1e-9 * want.abs().max(1.0) {
                    return Err(format!("community {c}: {name} {got} != {want}"));
                }
            }
        }
        Ok(())
    }
}

/// Accumulates edge weight from one node to each neighbouring community.
///
/// Replaces the per-call `HashSet` and `get_group_members()` allocation that made the old
/// implementation O(n) per candidate. One pass over a node's neighbours gives every candidate
/// and its weight, so scoring all of them is O(degree), not O(n · degree).
///
/// Candidates come back in first-touch order, i.e. ascending neighbour id — deterministic,
/// unlike hash-set iteration.
#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    /// Generation this slot was last written in, avoiding an O(slots) clear between nodes.
    stamp: u64,
    /// Accumulated edge weight to this community, valid only when `stamp == generation`.
    weight: f64,
}

#[derive(Debug, Clone, Default)]
pub struct NeighborWeights {
    /// Stamp and weight packed together — the inner loop touches both for the same community,
    /// and at scale this array is way past cache, so adjacency turns two random misses per
    /// neighbour into one.
    ///
    /// Tried and reverted: a linear-scan path for low-degree nodes. 19% worse at 100k (the
    /// indexed array still fits in cache there) and only break-even at 3M, since the scan
    /// trades a miss for a branch-mispredicting loop.
    slots: Vec<Slot>,
    generation: u64,
    touched: Vec<usize>,
}

impl NeighborWeights {
    /// Creates an accumulator sized for `slots` communities.
    pub fn with_capacity(slots: usize) -> Self {
        Self {
            slots: vec![Slot::default(); slots],
            generation: 0,
            touched: Vec::new(),
        }
    }

    /// Grows the accumulator so community ids up to `slots - 1` are addressable.
    ///
    /// Call this once per node, before [`collect`](Self::collect) - not once per neighbour.
    #[inline]
    pub fn ensure_capacity(&mut self, slots: usize) {
        if self.slots.len() < slots {
            self.slots.resize(slots, Slot::default());
        }
    }

    /// Starts a new accumulation round.
    ///
    /// A generation counter is what makes clearing free. `u64` won't wrap in any realistic
    /// run, but the reset keeps that from being a silent correctness cliff.
    #[inline]
    fn begin(&mut self) {
        self.touched.clear();
        match self.generation.checked_add(1) {
            Some(next) => self.generation = next,
            None => {
                for slot in &mut self.slots {
                    slot.stamp = 0;
                }
                self.generation = 1;
            }
        }
    }

    /// Accumulates `w` into `community`, recording it as touched the first time.
    #[inline]
    fn add(&mut self, community: usize, w: f64) {
        // callers use `ensure_capacity`; grow anyway so forgetting is slow, not wrong
        if community >= self.slots.len() {
            self.slots.resize(community + 1, Slot::default());
        }
        let generation = self.generation;
        let slot = &mut self.slots[community];
        if slot.stamp != generation {
            slot.stamp = generation;
            slot.weight = w;
            self.touched.push(community);
        } else {
            slot.weight += w;
        }
    }

    /// Accumulates the weight from `node` to each community containing one of its neighbours.
    ///
    /// Self-loops are excluded, matching the convention of
    /// [`Partition::remove_node`]/[`Partition::insert_node`], which add the self-loop
    /// themselves. Returns the touched community ids.
    pub fn collect(&mut self, graph: &CSRNetwork, membership: &[u32], node: usize) -> &[usize] {
        self.begin();

        for (u, w) in graph.neighbors(node) {
            if u == node {
                continue; // self-loop handled by the partition
            }
            let c = membership[u];
            if c == UNASSIGNED {
                continue; // neighbour is mid-move; treated as belonging to nothing
            }
            self.add(c as usize, w);
        }
        &self.touched
    }

    /// Like [`collect`](Self::collect), but only counts neighbours that share the node's
    /// group in `constraint`.
    ///
    /// What keeps refinement inside the communities local moving found — a node can only
    /// merge with refined communities inside its own working community.
    pub fn collect_constrained(
        &mut self,
        graph: &CSRNetwork,
        membership: &[u32],
        node: usize,
        constraint: &[u32],
    ) -> &[usize] {
        self.begin();
        let own = constraint[node];

        for (u, w) in graph.neighbors(node) {
            if u == node || constraint[u] != own {
                continue;
            }
            let c = membership[u];
            if c == UNASSIGNED {
                continue;
            }
            self.add(c as usize, w);
        }
        &self.touched
    }

    /// Weight from the last collected node to `community`, or `0.0` if untouched.
    #[inline]
    pub fn weight_to(&self, community: usize) -> f64 {
        match self.slots.get(community) {
            Some(slot) if slot.stamp == self.generation => slot.weight,
            _ => 0.0,
        }
    }

    /// The communities touched by the last [`collect`](Self::collect).
    #[inline]
    pub fn touched(&self) -> &[usize] {
        &self.touched
    }
}

/// Reusable working buffers for local moving and refinement.
///
/// Both phases want a shuffled order, a work queue, and a queued-flag array, all sized to the
/// node count. Per-call allocation is ~1.3 GB of churn per level at 75M nodes, so the driver
/// owns one and hands it down.
#[derive(Debug, Clone, Default)]
pub struct MoveScratch {
    /// Per-node accumulator of edge weight to each neighbouring community.
    pub weights: NeighborWeights,
    order: Vec<usize>,
    queue: std::collections::VecDeque<usize>,
    queued: Vec<bool>,
}

impl MoveScratch {
    /// Creates scratch sized for `n` nodes and `slots` communities.
    pub fn with_capacity(n: usize, slots: usize) -> Self {
        Self {
            weights: NeighborWeights::with_capacity(slots),
            order: Vec::with_capacity(n),
            queue: std::collections::VecDeque::with_capacity(n),
            queued: Vec::with_capacity(n),
        }
    }

    /// Fills the queue with all `n` nodes in a freshly shuffled order and marks them queued.
    pub(crate) fn seed_queue(&mut self, n: usize, rng: &mut ChaCha8Rng) {
        self.order.clear();
        self.order.extend(0..n);
        self.order.shuffle(rng);

        self.queue.clear();
        self.queue.extend(self.order.iter().copied());

        self.queued.clear();
        self.queued.resize(n, true);
    }

    /// Produces a freshly shuffled node order without touching the queue.
    pub(crate) fn shuffled_order(&mut self, n: usize, rng: &mut ChaCha8Rng) -> &[usize] {
        self.order.clear();
        self.order.extend(0..n);
        self.order.shuffle(rng);
        &self.order
    }

    #[inline]
    pub(crate) fn pop(&mut self) -> Option<usize> {
        let v = self.queue.pop_front()?;
        self.queued[v] = false;
        Some(v)
    }

    #[inline]
    pub(crate) fn push(&mut self, v: usize) {
        if !self.queued[v] {
            self.queued[v] = true;
            self.queue.push_back(v);
        }
    }

    #[inline]
    pub(crate) fn is_queued(&self, v: usize) -> bool {
        self.queued[v]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_triangles() -> CSRNetwork {
        // 0-1-2 triangle, 3-4-5 triangle, bridge 2-3
        CSRNetwork::from_edges(
            6,
            &[
                (0, 1, 1.0),
                (1, 2, 1.0),
                (0, 2, 1.0),
                (3, 4, 1.0),
                (4, 5, 1.0),
                (3, 5, 1.0),
                (2, 3, 1.0),
            ],
        )
        .unwrap()
    }

    #[test]
    fn from_membership_computes_aggregates() {
        let g = two_triangles();
        let p = Partition::from_membership(&g, &[0, 0, 0, 1, 1, 1]);
        assert_eq!(p.size(0), 3);
        assert_eq!(p.internal(0), 3.0, "three triangle edges");
        assert_eq!(p.internal(1), 3.0);
        assert_eq!(p.strength(0), 2.0 + 2.0 + 3.0, "node 2 has the bridge");
        assert_eq!(p.community_count(), 2);
    }

    #[test]
    fn remove_then_insert_round_trips() {
        let g = two_triangles();
        let mut p = Partition::from_membership(&g, &[0, 0, 0, 1, 1, 1]);
        let before = p.clone();

        let mut acc = NeighborWeights::with_capacity(p.slots());
        acc.collect(&g, p.membership_raw(), 2);
        let w_own = acc.weight_to(0);
        assert_eq!(w_own, 2.0, "node 2 links to 0 and 1");

        let old = p.remove_node(2, &g, w_own);
        assert_eq!(old, 0);
        assert_eq!(p.size(0), 2);
        assert_eq!(p.internal(0), 1.0, "only edge 0-1 remains internal");

        p.insert_node(2, old, &g, w_own);
        assert_eq!(p.membership(2), 0);
        for c in 0..p.slots() {
            assert_eq!(p.size(c), before.size(c));
            assert!((p.internal(c) - before.internal(c)).abs() < 1e-12);
            assert!((p.strength(c) - before.strength(c)).abs() < 1e-12);
        }
        p.verify_against(&g).unwrap();
    }

    #[test]
    fn moving_a_node_keeps_aggregates_exact() {
        let g = two_triangles();
        let mut p = Partition::from_membership(&g, &[0, 0, 0, 1, 1, 1]);
        let mut acc = NeighborWeights::with_capacity(p.slots());

        // walk node 2 across to the other community and back
        for target in [1usize, 0] {
            acc.collect(&g, p.membership_raw(), 2);
            let own = p.membership(2);
            let w_own = acc.weight_to(own);
            p.remove_node(2, &g, w_own);
            let w_new = acc.weight_to(target);
            p.insert_node(2, target, &g, w_new);
            p.verify_against(&g).unwrap();
        }
        assert_eq!(p.membership(2), 0);
    }

    #[test]
    fn emptied_community_is_reused_not_leaked() {
        let g = two_triangles();
        let mut p = Partition::from_membership(&g, &[0, 1, 1, 1, 1, 1]);
        let mut acc = NeighborWeights::with_capacity(p.slots());

        acc.collect(&g, p.membership_raw(), 0);
        let old = p.remove_node(0, &g, acc.weight_to(0));
        assert_eq!(old, 0);
        assert_eq!(p.community_count(), 1);
        // the emptied slot is the one offered back as "an empty community"
        assert_eq!(p.empty_community(), 0);

        p.insert_node(0, 1, &g, acc.weight_to(1));
        assert_eq!(p.community_count(), 1);
        assert_eq!(p.empty_community(), 0, "slot 0 stays free");
        p.verify_against(&g).unwrap();
    }

    #[test]
    fn self_loops_are_excluded_from_the_accumulator() {
        let g = CSRNetwork::from_edges(2, &[(0, 1, 1.0), (0, 0, 5.0)]).unwrap();
        let p = Partition::singleton(&g);
        let mut acc = NeighborWeights::with_capacity(p.slots());
        acc.collect(&g, p.membership_raw(), 0);
        assert_eq!(acc.weight_to(0), 0.0, "own self-loop must not appear");
        assert_eq!(acc.weight_to(1), 1.0);
        // but the partition does account for it
        assert_eq!(p.internal(0), 5.0);
    }

    #[test]
    fn candidate_order_is_deterministic() {
        let g = two_triangles();
        let p = Partition::from_membership(&g, &[3, 1, 2, 0, 1, 2]);
        let mut acc = NeighborWeights::with_capacity(p.slots());
        let first: Vec<usize> = acc.collect(&g, p.membership_raw(), 2).to_vec();
        for _ in 0..20 {
            let again: Vec<usize> = acc.collect(&g, p.membership_raw(), 2).to_vec();
            assert_eq!(first, again);
        }
    }

    #[test]
    fn renumber_compacts_ids_in_ascending_order() {
        let g = two_triangles();
        let mut p = Partition::from_membership(&g, &[5, 5, 5, 2, 2, 2]);
        assert_eq!(p.community_count(), 2);
        p.renumber();
        assert_eq!(p.slots(), 2);
        assert_eq!(p.membership_raw(), &[1, 1, 1, 0, 0, 0]);
        p.verify_against(&g).unwrap();
    }
}
