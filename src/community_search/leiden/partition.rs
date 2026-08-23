//! Partition state.
//!
//! [`Partition`] holds every aggregate the quality functions need, updated in O(1) per move.
//! Nothing here walks the graph or allocates per call, which is what keeps local moving at
//! O(degree).
//!
//! A move is [`remove_node`](Partition::remove_node) then
//! [`insert_node`](Partition::insert_node), never one operation. The node sits in no community
//! in between, so every candidate — old community, empty one, anything else — scores the same
//! way.

use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use crate::network::CSRNetwork;
use crate::network::grouping::VectorGrouping;

/// Membership value for a removed node. Visible via [`Partition::membership_raw`] mid-move,
/// never in a partition handed back.
pub const UNASSIGNED: u32 = u32::MAX;

/// A partition of a graph's nodes into communities, with incrementally maintained aggregates.
#[derive(Debug, Clone)]
pub struct Partition {
    /// Community of each node, or [`UNASSIGNED`] mid-move.
    ///
    /// `u32` because this is the hottest random access in the algorithm — every neighbour of
    /// every visited node reads it — and halving the array halves that cache footprint.
    membership: Vec<u32>,
    /// Number of nodes in each community.
    size: Vec<u32>,
    /// Sum of node weights in each community. This is what CPM measures.
    weight: Vec<f64>,
    /// Sum of node strengths in each community. This is `K_c` in the RB null model.
    strength: Vec<f64>,
    /// Internal edge weight per community, each undirected edge once, self-loops at full
    /// weight.
    internal: Vec<f64>,
    /// Community slots that currently hold no nodes, available for reuse.
    free: Vec<usize>,
    /// Total edge weight, each edge once. Here so quality functions don't need the graph.
    total_weight: f64,
    /// Sum of all node weights.
    total_node_weight: f64,
}

impl Partition {
    /// Creates a partition where every node is alone in its own community.
    pub fn singleton(graph: &CSRNetwork) -> Self {
        let n = graph.node_count();
        Self {
            membership: (0..n as u32).collect(),
            size: vec![1; n],
            weight: (0..n).map(|v| graph.node_weight(v)).collect(),
            strength: (0..n).map(|v| graph.strength(v)).collect(),
            // its only internal edge is its own self-loop
            internal: (0..n).map(|v| graph.self_loop_weight(v)).collect(),
            free: Vec::new(),
            total_weight: graph.total_weight(),
            total_node_weight: graph.total_node_weight(),
        }
    }

    /// Creates a partition from an explicit membership vector.
    ///
    /// Ids needn't be consecutive; gaps become free slots and get reused first.
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
                // each edge once: loops here, the rest from the lower id
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

    /// Community of a node. Debug-panics if it's currently removed.
    #[inline]
    pub fn membership(&self, node: usize) -> usize {
        debug_assert_ne!(self.membership[node], UNASSIGNED, "node {node} is removed");
        self.membership[node] as usize
    }

    /// Raw membership, which may hold [`UNASSIGNED`] mid-move.
    #[inline]
    pub fn membership_raw(&self) -> &[u32] {
        &self.membership
    }

    /// Membership as `usize`, allocated fresh.
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

    /// An empty community id, without allocating.
    ///
    /// A free slot if there is one, otherwise the next index, allocated on insert. Each caller
    /// takes one and fills it immediately, so two nodes can't get the same one.
    #[inline]
    pub fn empty_community(&self) -> usize {
        self.free.last().copied().unwrap_or(self.size.len())
    }

    /// Removes a node from its community, updating aggregates in O(1). Returns the community
    /// it was in.
    ///
    /// `weight_to_own` excludes the self-loop — exactly what [`NeighborWeights`] accumulates.
    pub fn remove_node(&mut self, node: usize, graph: &CSRNetwork, weight_to_own: f64) -> usize {
        let c = self.membership[node] as usize;
        debug_assert_ne!(
            self.membership[node], UNASSIGNED,
            "node {node} was already removed"
        );

        self.size[c] -= 1;
        self.weight[c] -= graph.node_weight(node);
        self.strength[c] -= graph.strength(node);
        // its edges in, plus its self-loop, stop counting as internal
        self.internal[c] -= weight_to_own + graph.self_loop_weight(node);

        if self.size[c] == 0 {
            // clamp float drift so empty is exactly empty
            self.weight[c] = 0.0;
            self.strength[c] = 0.0;
            self.internal[c] = 0.0;
            self.free.push(c);
        }

        self.membership[node] = UNASSIGNED;
        c
    }

    /// Inserts a removed node into a community, updating aggregates in O(1).
    ///
    /// `community` may be [`empty_community`](Self::empty_community), which allocates a slot.
    /// `weight_to_comm` excludes the self-loop.
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

    /// Renumbers communities consecutively from 0, ascending, so the output never depends on
    /// iteration order upstream.
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
    /// `&mut self` on purpose: callers use `membership(v)` to index the graph
    /// [`CSRNetwork::aggregate`] produces, so the two sets of ids have to stay the same.
    pub fn renumber_into_grouping(&mut self) -> VectorGrouping {
        self.renumber();
        VectorGrouping::from_assignments(&self.membership_vec())
    }

    /// Recomputes every aggregate and checks it against the maintained one. Tests only.
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

#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    /// Generation last written, which is what avoids an O(slots) clear between nodes.
    stamp: u64,
    /// Edge weight to this community, valid only when `stamp == generation`.
    weight: f64,
}

/// Edge weight from one node to each neighbouring community.
///
/// One pass over a node's neighbours gives every candidate and its weight, so scoring all of
/// them is O(degree). Candidates come back in first-touch order, i.e. ascending neighbour id.
#[derive(Debug, Clone, Default)]
pub struct NeighborWeights {
    /// Stamp and weight packed together: the inner loop touches both for the same community,
    /// and past cache that's one random miss per neighbour instead of two.
    ///
    /// Tried and reverted: a linear scan for low-degree nodes. 19% worse at 100k, break-even
    /// at 3M — it trades a miss for a branch-mispredicting loop.
    slots: Vec<Slot>,
    generation: u64,
    touched: Vec<usize>,
}

impl NeighborWeights {
    /// Sized for `slots` communities.
    pub fn with_capacity(slots: usize) -> Self {
        Self {
            slots: vec![Slot::default(); slots],
            generation: 0,
            touched: Vec::new(),
        }
    }

    /// Grows so ids up to `slots - 1` are addressable. Once per node, not per neighbour.
    #[inline]
    pub fn ensure_capacity(&mut self, slots: usize) {
        if self.slots.len() < slots {
            self.slots.resize(slots, Slot::default());
        }
    }

    /// Starts a new round. The generation counter makes clearing free; `u64` won't wrap in
    /// any realistic run, but the reset keeps that from being a silent correctness cliff.
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

    /// Weight from `node` to each community holding one of its neighbours, self-loop
    /// excluded — [`Partition::remove_node`]/[`insert_node`](Partition::insert_node) add that
    /// themselves. Returns the touched ids.
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

    /// Weight from the last collected node to `community`, or `0.0`.
    #[inline]
    pub fn weight_to(&self, community: usize) -> f64 {
        match self.slots.get(community) {
            Some(slot) if slot.stamp == self.generation => slot.weight,
            _ => 0.0,
        }
    }

    /// Communities touched by the last [`collect`](Self::collect).
    #[inline]
    pub fn touched(&self) -> &[usize] {
        &self.touched
    }
}

/// Reusable buffers for local moving and refinement.
///
/// Both want a shuffled order, a queue, and a queued flag per node. Allocating per call is
/// ~1.3 GB of churn per level at 75M nodes, so the driver owns one and hands it down.
#[derive(Debug, Clone, Default)]
pub struct MoveScratch {
    /// Per-node accumulator of edge weight to each neighbouring community.
    pub weights: NeighborWeights,
    order: Vec<usize>,
    queue: std::collections::VecDeque<usize>,
    queued: Vec<bool>,
}

impl MoveScratch {
    /// Sized for `n` nodes and `slots` communities.
    pub fn with_capacity(n: usize, slots: usize) -> Self {
        Self {
            weights: NeighborWeights::with_capacity(slots),
            order: Vec::with_capacity(n),
            queue: std::collections::VecDeque::with_capacity(n),
            queued: Vec::with_capacity(n),
        }
    }

    /// Queues all `n` nodes in a fresh shuffled order.
    pub(crate) fn seed_queue(&mut self, n: usize, rng: &mut ChaCha8Rng) {
        self.order.clear();
        self.order.extend(0..n);
        self.order.shuffle(rng);

        self.queue.clear();
        self.queue.extend(self.order.iter().copied());

        self.queued.clear();
        self.queued.resize(n, true);
    }

    /// A fresh shuffled node order, leaving the queue alone.
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
