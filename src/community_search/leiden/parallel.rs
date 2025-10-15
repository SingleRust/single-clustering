use std::{
    collections::{HashSet, VecDeque}, string::ParseError, thread::current
};

use num_traits::Float;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use single_utilities::traits::FloatOpsTS;

use crate::{
    community_search::leiden::{
        ConsiderComms,
        partition::{self, VertexPartition},
    },
    network::{CSRNetwork, grouping::NetworkGrouping},
};

#[derive(Debug, Clone, Copy)]
pub struct ProposedMove<N> {
    pub node: usize,
    pub from_comm: usize,
    pub to_comm: usize,
    pub improvement: N,
}

impl<N: FloatOpsTS> ProposedMove<N> {
    #[inline]
    pub fn new(node: usize, from_comm: usize, to_comm: usize, improvement: N) -> Self {
        Self {
            node,
            from_comm,
            to_comm,
            improvement,
        }
    }

    #[inline]
    pub fn is_beneficial(&self) -> bool {
        self.from_comm != self.to_comm && self.improvement > N::zero()
    }
}

#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pub max_batch_size: usize,
    pub max_iterations: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 10_000,
            max_iterations: 1_000,
        }
    }
}

pub struct ConflictFreeBatcher {
    max_batch_size: usize,
}

impl ConflictFreeBatcher {
    pub fn new(max_batch_size: usize) -> Self {
        Self { max_batch_size }
    }

    pub fn create_batches<N>(
        &self,
        nodes: &[usize],
        network: &CSRNetwork<N, N>,
        skip_stable: &[bool],
    ) -> Vec<Vec<usize>>
    where
        N: FloatOpsTS + 'static,
    {
        let mut batches = Vec::new();
        let mut remaining: Vec<usize> =
            nodes.iter().copied().filter(|&n| !skip_stable[n]).collect();

        while !remaining.is_empty() {
            let (batch, leftover) = self.extract_batch(&remaining, network);

            if batch.is_empty() {
                break;
            }

            batches.push(batch);
            remaining = leftover;
        }
        batches
    }

    fn extract_batch<N>(
        &self,
        candidates: &[usize],
        network: &CSRNetwork<N, N>,
    ) -> (Vec<usize>, Vec<usize>)
    where
        N: FloatOpsTS + 'static,
    {
        let mut batch = Vec::new();
        let mut leftover = Vec::new();
        let mut locked = vec![false; network.node_count()];

        for &node in candidates {
            if self.has_conflict(node, network, &locked) {
                leftover.push(node);
            } else {
                batch.push(node);
                self.mark_locked(node, network, &mut locked);
                if batch.len() >= self.max_batch_size {
                    leftover.extend_from_slice(
                        &candidates[candidates.iter().position(|&n| n == node).unwrap() + 1..],
                    );
                    break;
                }
            }
        }
        (batch, leftover)
    }

    #[inline]
    fn has_conflict<N>(&self, node: usize, network: &CSRNetwork<N, N>, locked: &[bool]) -> bool
    where
        N: FloatOpsTS + 'static,
    {
        if locked[node] {
            return true;
        }

        for (neighbor, _) in network.neighbors(node) {
            if locked[neighbor] {
                return true;
            }
        }
        false
    }

    #[inline]
    fn mark_locked<N>(&self, node: usize, network: &CSRNetwork<N, N>, locked: &mut [bool])
    where
        N: FloatOpsTS + 'static,
    {
        locked[node] = true;
        for (neighbor, _) in network.neighbors(node) {
            locked[neighbor] = true;
        }
    }
}

pub struct ParallelEvaluator;

impl ParallelEvaluator {
    pub fn evaluate_batch<N, G, P>(
        batch: &[usize],
        partitions: &[P],
        layer_weights: &[N],
        consider_comms: ConsiderComms,
        consider_empty: bool,
        max_comm_size: Option<usize>,
    ) -> Vec<ProposedMove<N>>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping,
        P: VertexPartition<N, G>,
    {
        batch
            .par_iter()
            .filter_map(|&node| {
                Self::evaluate_node(
                    node,
                    partitions,
                    layer_weights,
                    consider_comms,
                    consider_empty,
                    max_comm_size,
                )
            })
            .collect()
    }

    fn evaluate_node<N, G, P>(
        node: usize,
        partitions: &[P],
        layer_weights: &[N],
        consider_comms: ConsiderComms,
        consider_empty: bool,
        max_comm_size: Option<usize>,
    ) -> Option<ProposedMove<N>>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping,
        P: VertexPartition<N, G>,
    {
        let current_comm = partitions[0].membership(node);

        let candidates = Self::get_candidates(node, partitions, consider_comms);

        let (best_comm, best_improv) = Self::find_best(
            node,
            current_comm,
            &candidates,
            partitions,
            layer_weights,
            consider_empty,
            max_comm_size,
        )?;

        if best_comm != current_comm && best_improv > N::zero() {
            Some(ProposedMove {
                node,
                from_comm: current_comm,
                to_comm: best_comm,
                improvement: best_improv,
            })
        } else {
            None
        }
    }

    fn get_candidates<N, G, P>(node: usize, partitions: &[P], stategy: ConsiderComms) -> Vec<usize>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping,
        P: VertexPartition<N, G>,
    {
        let mut comms = HashSet::new();
        match stategy {
            ConsiderComms::AllComms => {
                for comm in 0..partitions[0].community_count() {
                    if partitions[0].cnodes(comm) > 0 {
                        comms.insert(comm);
                    }
                }
            }
            ConsiderComms::AllNeighComms => {
                for partition in partitions {
                    for (neighbor, _) in partition.network().neighbors(node) {
                        comms.insert(partition.membership(neighbor));
                    }
                }
            }
            ConsiderComms::RandComm | ConsiderComms::RandNeighComm => {
                for partition in partitions {
                    for (neighbor, _) in partition.network().neighbors(node) {
                        comms.insert(partition.membership(neighbor));
                    }
                }
            }
        }

        comms.into_iter().collect()
    }

    fn find_best<N, G, P>(
        node: usize,
        current_comm: usize,
        candidates: &[usize],
        partitions: &[P],
        layer_weights: &[N],
        consider_empty: bool,
        max_comm_size: Option<usize>,
    ) -> Option<(usize, N)>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping,
        P: VertexPartition<N, G>,
    {
        let epsilon = N::from(10.0).unwrap() * <N as num_traits::Float>::epsilon();
        let mut best_comm = current_comm;
        let mut best_improv = epsilon;

        let mut to_check = candidates.to_vec();
        if consider_empty && partitions[0].cnodes(current_comm) > 1 {
            to_check.push(partitions[0].community_count());
        }

        for &comm in &to_check {
            if let Some(max_size) = max_comm_size {
                if comm < partitions[0].community_count()
                    && partitions[0].csize(comm) + 1 > max_size
                {
                    continue;
                }
            }

            let mut total_improv = N::zero();
            for (idx, partitions) in partitions.iter().enumerate() {
                let layer_improv = partitions.diff_move_readonly(node, comm);
                total_improv += layer_weights[idx] * layer_improv;
            }

            if total_improv > best_improv {
                best_comm = comm;
                best_improv = total_improv;
            }
        }

        Some((best_comm, best_improv))
    }
}

pub struct MoveApplicator;

impl MoveApplicator {
    pub fn apply_moves<N, G, P>(
        moves: Vec<ProposedMove<N>>,
        partitions: &mut [P],
        is_stable: &mut [bool],
        is_fixed: &[bool],
        pending: &mut VecDeque<usize>,
    ) -> N
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping + Clone,
        P: VertexPartition<N, G>,
    {

        let mut total_improv = N::zero();

        if moves.is_empty() {
            return N::zero();
        }

        for m in moves {
            Self::ensure_community_exists(m.to_comm, partitions);

            for partition in partitions.iter_mut() {
                partition.move_node(m.node, m.to_comm);
            }

            total_improv += m.improvement;
            is_stable[m.node] = true;

            let network = partitions[0].network();
            Self::mark_neighbors_unstable(m.node, network, is_stable, is_fixed, pending);
        }
        total_improv
    }

    fn ensure_community_exists<N, G, P>(comm: usize, partitions: &mut [P]) 
    where 
        N: FloatOpsTS + 'static,
        G: NetworkGrouping + Clone,
        P: VertexPartition<N, G> {
            for partition in partitions.iter_mut() {
                while partition.community_count() <= comm {
                    partition.add_empty_community();
                }
            }
        }

    fn mark_neighbors_unstable<N>(
        node: usize,
        network: &CSRNetwork<N, N>,
        is_stable: &mut [bool],
        is_fixed: &[bool],
        pending: &mut VecDeque<usize>
    )
    where 
    N: FloatOpsTS + 'static {
        for (neighbor, _) in network.neighbors(node) {
            if !is_fixed[neighbor] && is_stable[neighbor] {
                is_stable[neighbor] = false;
                pending.push_back(neighbor);
            }
        }
    }
}
