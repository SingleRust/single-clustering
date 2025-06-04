use std::{collections::VecDeque, os::unix::process::parent_id};

use anyhow::Ok;
use num_traits::Float;
use rand::{Rng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use single_utilities::traits::FloatOpsTS;

use crate::{
    community_search::leiden::{
        ConsiderComms, LeidenConfig,
        partition::{self, VertexPartition},
    },
    network::grouping::NetworkGrouping,
};

pub struct LeidenOptimizer {
    config: LeidenConfig,
    rng: ChaCha8Rng,
}

impl LeidenOptimizer {
    fn merge_nodes<N, G, P>(
        &mut self,
        partitions: &mut [P],
        layer_weights: &[N],
        is_membership_fixed: &[bool],
        consider_comms: ConsiderComms,
        renumber_fixed_nodes: bool,
        max_comm_size: Option<usize>,
    ) -> anyhow::Result<N>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping + Clone + Default,
        P: VertexPartition<N, G>,
    {
        let nb_layers = partitions.len();
        if nb_layers == 0 {
            return Ok(N::from(-1.0).unwrap());
        }

        let n = partitions[0].node_count();

        for partition in partitions.iter() {
            if partition.node_count() != n {
                return Err(anyhow::anyhow!("Unequal node size in partitions detected!"));
            }
        }

        let mut fixed_nodes = Vec::new();
        let mut fixed_membership = vec![0; n];
        if renumber_fixed_nodes {
            for v in (0..n) {
                if is_membership_fixed[v] {
                    fixed_nodes.push(v);
                    fixed_membership[v] = partitions[0].membership(v);
                }
            }
        }

        let mut total_improvement = N::zero();

        let mut vertex_order: Vec<usize> = (0..n).filter(|&v| !is_membership_fixed[v]).collect();

        vertex_order.shuffle(&mut self.rng);

        let mut comm_added = vec![false; partitions[0].community_count()];
        let mut comms = Vec::new();

        for v in vertex_order {
            let v_comm = partitions[0].membership(v);

            for &comm in &comms {
                if comm < comm_added.len() {
                    comm_added[comm] = false;
                }
            }

            comms.clear();
            if partitions[0].cnodes(v_comm) == 1 {
                self.collect_candidate_communities(
                    v,
                    partitions,
                    consider_comms,
                    &mut comms,
                    &mut comm_added,
                );
            }

            let mut max_comm = v_comm;
            let mut max_improv = if let Some(max_size) = max_comm_size {
                if max_size < partitions[0].csize(v_comm) {
                    N::from(f64::NEG_INFINITY).unwrap()
                } else {
                    N::zero()
                }
            } else {
                N::zero()
            };

            let v_size = N::one();

            for &comm in &comms {
                if let Some(max_size) = max_comm_size {
                    let comm_size = N::from(partitions[0].csize(comm)).unwrap();
                    if N::from(max_size).unwrap() < comm_size + v_size {
                        continue;
                    }
                }

                let mut possible_improvement = N::zero();
                for layer in 0..nb_layers {
                    let diff = partitions[layer].diff_move(v, comm);
                    possible_improvement += layer_weights[layer] * diff;
                }

                if possible_improvement >= max_improv {
                    max_comm = comm;
                    max_improv = possible_improvement;
                }
            }

            if max_comm != v_comm {
                total_improvement += max_improv;

                for partition in partitions.iter_mut() {
                    // reflect changes to all partitions
                    partition.move_node(v, max_comm);
                }
            }
        }

        partitions[0].renumber_communities();
        if renumber_fixed_nodes {
            partitions[0].renumber_communities_fixed(&fixed_nodes, &fixed_membership);
        }

        let membership = partitions[0].membership_vector();
        for partition in partitions.iter_mut().skip(1) {
            partition.set_membership(&membership);
        }
        Ok(total_improvement)
    }

    fn find_best_community_move<N, G, P>(
        &self,
        v: usize,
        v_comm: usize,
        comms: &[usize],
        partitions: &[P],
        layer_weights: &[N],
        max_comm_size: Option<usize>,
    ) -> anyhow::Result<(usize, N)>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping,
        P: VertexPartition<N, G>,
    {
        let mut max_comm = v_comm;
        let mut max_improv = if let Some(max_size) = max_comm_size {
            if max_size < partitions[0].csize(v_comm) {
                N::from(f64::NEG_INFINITY).unwrap()
            } else {
                N::from(10.0 * f64::EPSILON).unwrap()
            }
        } else {
            N::from(10.0 * f64::EPSILON).unwrap()
        };

        let v_size = 1; // assuming unsit size

        for &comm in comms {
            if let Some(max_size) = max_comm_size {
                if max_size < partitions[0].csize(comm) + v_size {
                    continue;
                }
            }

            let mut possible_improv = N::zero();
            for (layer, partition) in partitions.iter().enumerate() {
                let layer_improv = partition.diff_move(v, comm);
                possible_improv += layer_weights[layer] * layer_improv;
            }

            if possible_improv > max_improv {
                max_comm = comm;
                max_improv = possible_improv;
            }
        }
        Ok((max_comm, max_improv))
    }

    fn collect_candidate_communities<N, G, P>(
        &mut self,
        v: usize,
        partitions: &[P],
        consider_comms: ConsiderComms,
        comms: &mut Vec<usize>,
        comm_added: &mut [bool],
    ) where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping,
        P: VertexPartition<N, G>,
    {
        match consider_comms {
            ConsiderComms::AllComms => {
                // Consider all non-empty communities across all layers
                for comm in 0..partitions[0].community_count() {
                    for partition in partitions {
                        if partition.cnodes(comm) > 0
                            && comm < comm_added.len()
                            && !comm_added[comm]
                        {
                            comms.push(comm);
                            comm_added[comm] = true;
                            break;
                        }
                    }
                }
            }
            ConsiderComms::AllNeighComms => {
                // Consider all neighbor communities across all layers
                for partition in partitions {
                    let neigh_comms = partition.get_neigh_comms(v, None);
                    for comm in neigh_comms {
                        if comm < comm_added.len() && !comm_added[comm] {
                            comms.push(comm);
                            comm_added[comm] = true;
                        }
                    }
                }
            }
            ConsiderComms::RandComm => {
                // Pick a random community from a random node
                if let Some(partition) = partitions.first() {
                    let random_node = self.rng.random_range(0..partition.node_count());
                    let rand_comm = partition.membership(random_node);
                    comms.push(rand_comm);
                    if rand_comm < comm_added.len() {
                        comm_added[rand_comm] = true;
                    }
                }
            }
            ConsiderComms::RandNeighComm => {
                // Pick a random neighbor's community from a random layer
                if !partitions.is_empty() {
                    let rand_layer = self.rng.random_range(0..partitions.len());
                    let neighbors: Vec<_> = partitions[rand_layer]
                        .network()
                        .neighbors(v)
                        .map(|(neighbor, _)| neighbor)
                        .collect();

                    if !neighbors.is_empty() {
                        let random_neighbor = neighbors[self.rng.random_range(0..neighbors.len())];
                        let rand_comm = partitions[0].membership(random_neighbor);
                        comms.push(rand_comm);
                        if rand_comm < comm_added.len() {
                            comm_added[rand_comm] = true;
                        }
                    }
                }
            }
        }
    }

    fn mark_neighbors_unstable<N, G, P>(
        &self,
        v: usize,
        new_comm: usize,
        partition: &P,
        is_node_stable: &mut [bool],
        is_membership_fixed: &[bool],
        vertex_order: &mut VecDeque<usize>,
    ) where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping,
        P: VertexPartition<N, G>,
    {
        for (neighbor, _) in partition.network().neighbors(v) {
            // If neighbor was stable, not in new community, and not fixed
            if is_node_stable[neighbor]
                && partition.membership(neighbor) != new_comm
                && !is_membership_fixed[neighbor]
            {
                vertex_order.push_back(neighbor);
                is_node_stable[neighbor] = false;
            }
        }
    }

    fn move_nodes<N, G, P>(
        &mut self,
        partitions: &mut [P],
        layer_weights: &[N],
        is_membership_fixed: &[bool],
        consider_comms: ConsiderComms,
        consider_empty_community: bool,
        renumber_fixed_nodes: bool,
        max_comm_size: Option<usize>,
    ) -> anyhow::Result<N>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping + Clone + Default,
        P: VertexPartition<N, G>,
    {
        if partitions.is_empty() {
            return Ok(N::from(-1.0).unwrap());
        }

        let nb_layers = partitions.len();
        let n = partitions[0].node_count();

        for partition in partitions.iter() {
            if partition.node_count() != n {
                panic!("Number of nodes are not equal for all graphs.");
            }
        }

        let mut fixed_nodes = Vec::new();
        let mut fixed_membership = vec![0; n];
        if renumber_fixed_nodes {
            for v in 0..n {
                if is_membership_fixed[v] {
                    fixed_nodes.push(v);
                    fixed_membership[v] = partitions[0].membership(v);
                }
            }
        }

        let mut total_improv = N::zero();

        let mut is_node_stable = is_membership_fixed.to_vec();

        let mut nodes: Vec<usize> = (0..n).filter(|&v| !is_membership_fixed[v]).collect();
        nodes.shuffle(&mut self.rng);
        let mut vertex_order: VecDeque<usize> = nodes.into();

        let mut comm_added = vec![false; partitions[0].community_count()];
        let mut comms = Vec::new();

        while let Some(v) = vertex_order.pop_front() {
            let v_comm = partitions[0].membership(v);
            for &comm in &comms {
                if comm < comm_added.len() {
                    comm_added[comm] = false;
                }
            }
            comms.clear();

            self.collect_candidate_communities(
                v,
                &partitions,
                consider_comms,
                &mut comms,
                &mut comm_added,
            );

            if consider_empty_community && partitions[0].cnodes(v_comm) > 1 {
                let n_comms_before = partitions[0].community_count();
                let empty_comm = partitions[0].get_empty_community();
                comms.push(empty_comm);

                if partitions[0].community_count() > n_comms_before {
                    for layer in 1..nb_layers {
                        partitions[layer].add_empty_community();
                    }
                    comm_added.resize(partitions[0].community_count(), false);
                    if empty_comm < comm_added.len() {
                        comm_added[empty_comm] = true;
                    }
                }
            }

            let (max_comm, max_improv) = self.find_best_community_move(
                v,
                v_comm,
                &comms,
                partitions,
                layer_weights,
                max_comm_size,
            )?;

            is_node_stable[v] = true;

            if max_comm != v_comm && max_improv > N::zero() {
                total_improv += max_improv;

                for partition in partitions.iter_mut() {
                    partition.move_node(v, max_comm);
                }

                self.mark_neighbors_unstable(
                    v,
                    max_comm,
                    &partitions[0],
                    &mut is_node_stable,
                    is_membership_fixed,
                    &mut vertex_order,
                );
            }
        }

        partitions[0].renumber_communities();
        if renumber_fixed_nodes {
            partitions[0].renumber_communities_fixed(&fixed_nodes, &fixed_membership);
        }

        let membership = partitions[0].membership_vector();
        for partition in partitions.iter_mut().skip(1) {
            partition.set_membership(&membership);
        }

        Ok(total_improv)
    }

    fn move_nodes_constrained<N, G, P>(
        &mut self,
        partitions: &mut [P],
        layer_weights: &[N],
        consider_comms: ConsiderComms,
        constrained_partition: &P,
        max_comm_size: Option<usize>,
    ) -> anyhow::Result<N>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping + Clone + Default,
        P: VertexPartition<N, G>,
    {
        if partitions.is_empty() {
            return Ok(N::from(-1.0).unwrap());
        }

        let nb_layers = partitions.len();
        let n = partitions[0].node_count();

        for partition in partitions.iter() {
            if partition.node_count() != n {
                return Err(anyhow::anyhow!(
                    "Number of nodes are not equal for all graphs."
                ));
            }
        }

        let mut total_improv = N::zero();
        let mut is_node_stable = vec![false; n];
        let mut nb_moves = 0;

        let mut nodes: Vec<usize> = (0..n).collect();
        nodes.shuffle(&mut self.rng);
        let mut vertex_order: VecDeque<usize> = nodes.into();

        // Get constrained communities structure
        let constrained_comms = constrained_partition.get_communities();

        let mut comm_added = vec![false; partitions[0].community_count()];
        let mut comms = Vec::new();

        while let Some(v) = vertex_order.pop_front() {
            for &comm in &comms {
                if comm < comm_added.len() {
                    comm_added[comm] = false;
                }
            }

            comms.clear();

            let v_comm = partitions[0].membership(v);

            self.collect_constrained_candidate_communities(
                v,
                partitions,
                constrained_partition,
                &constrained_comms,
                consider_comms,
                &mut comms,
                &mut comm_added,
            );

            let (max_comm, max_improv) = self.find_best_community_move_constrained(
                v,
                v_comm,
                &comms,
                partitions,
                layer_weights,
                max_comm_size,
            )?;

            is_node_stable[v] = true;

            if max_comm != v_comm {
                total_improv += max_improv;

                for partition in partitions.iter_mut() {
                    partition.move_node(v, max_comm);
                }

                self.mark_constrained_neighbors_unstable(
                    v,
                    max_comm,
                    &partitions[0],
                    constrained_partition,
                    &mut is_node_stable,
                    &mut vertex_order,
                );
                nb_moves += 1;
            }
        }

        partitions[0].renumber_communities();
        let membership = partitions[0].membership_vector();
        for partition in partitions.iter_mut().skip(1) {
            partition.set_membership(&membership);
        }

        Ok(total_improv)
    }

    fn collect_constrained_candidate_communities<N, G, P>(
        &mut self,
        v: usize,
        partitions: &[P],
        constrained_partition: &P,
        constrained_comms: &[Vec<usize>],
        consider_comms: ConsiderComms,
        comms: &mut Vec<usize>,
        comm_added: &mut [bool],
    ) where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping,
        P: VertexPartition<N, G>,
    {
        match consider_comms {
            ConsiderComms::AllComms => {
                let v_constrained_comm = constrained_partition.membership(v);
                if v_constrained_comm < constrained_comms.len() {
                    for &u in &constrained_comms[v_constrained_comm] {
                        let u_comm = partitions[0].membership(u);
                        if u_comm < comm_added.len() && !comm_added[u_comm] {
                            comms.push(u_comm);
                            comm_added[u_comm] = true;
                        }
                    }
                }
            }
            ConsiderComms::AllNeighComms => {
                for partition in partitions {
                    let constrained_membership = constrained_partition.membership_vector();
                    let neigh_comms = partition.get_neigh_comms(v, Some(&constrained_membership));
                    for comm in neigh_comms {
                        if comm < comm_added.len() && !comm_added[comm] {
                            comms.push(comm);
                            comm_added[comm] = true;
                        }
                    }
                }
            }
            ConsiderComms::RandComm => {
                let v_constrained_comm = constrained_partition.membership(v);
                if v_constrained_comm < constrained_comms.len()
                    && !constrained_comms[v_constrained_comm].is_empty()
                {
                    let random_idx = self
                        .rng
                        .random_range(0..constrained_comms[v_constrained_comm].len());
                    let random_node = constrained_comms[v_constrained_comm][random_idx];
                    let rand_comm = partitions[0].membership(random_node);
                    comms.push(rand_comm);
                    if rand_comm < comm_added.len() {
                        comm_added[rand_comm] = true;
                    }
                }
            }
            ConsiderComms::RandNeighComm => {
                let mut all_neigh_comms_incl_dupes = Vec::new();
                for partition in partitions {
                    let constrained_membership = constrained_partition.membership_vector();
                    let neigh_comms = partition.get_neigh_comms(v, Some(&constrained_membership));
                    all_neigh_comms_incl_dupes.extend(neigh_comms);
                }

                if !all_neigh_comms_incl_dupes.is_empty() {
                    let random_idx = self.rng.random_range(0..all_neigh_comms_incl_dupes.len());
                    let rand_comm = all_neigh_comms_incl_dupes[random_idx];
                    comms.push(rand_comm);
                    if rand_comm < comm_added.len() {
                        comm_added[rand_comm] = true;
                    }
                }
            }
        }
    }

    fn find_best_community_move_constrained<N, G, P>(
        &self,
        v: usize,
        v_comm: usize,
        comms: &[usize],
        partitions: &[P],
        layer_weights: &[N],
        max_comm_size: Option<usize>,
    ) -> anyhow::Result<(usize, N)>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping,
        P: VertexPartition<N, G>,
    {
        let mut max_comm = v_comm;
        let mut max_improv = if let Some(max_size) = max_comm_size {
            if max_size < partitions[0].csize(v_comm) {
                N::from(f64::NEG_INFINITY).unwrap()
            } else {
                N::from(10.0 * f64::EPSILON).unwrap()
            }
        } else {
            N::from(10.0 * f64::EPSILON).unwrap()
        };

        let v_size = 1;

        for &comm in comms {
            if let Some(max_size) = max_comm_size {
                if max_size < partitions[0].csize(comm) + v_size {
                    continue;
                }
            }

            let mut possible_improvement = N::zero();
            for (layer, partition) in partitions.iter().enumerate() {
                let layer_improv = partition.diff_move(v, comm);
                possible_improvement += layer_weights[layer] * layer_improv;
            }

            if possible_improvement > max_improv {
                max_comm = comm;
                max_improv = possible_improvement;
            }
        }

        Ok((max_comm, max_improv))
    }

    fn mark_constrained_neighbors_unstable<N, G, P>(
        &self,
        v: usize,
        new_comm: usize,
        partition: &P,
        constrained_partition: &P,
        is_node_stable: &mut [bool],
        vertex_order: &mut VecDeque<usize>,
    ) where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping,
        P: VertexPartition<N, G>,
    {
        let v_constrained_comm = constrained_partition.membership(v);
        for (neighbor, _) in partition.network().neighbors(v) {
            if is_node_stable[neighbor]
                && partition.membership(neighbor) != new_comm
                && constrained_partition.membership(neighbor) == v_constrained_comm
            {
                vertex_order.push_back(neighbor);
                is_node_stable[neighbor] = false;
            }
        }
    }

    fn merge_nodes_constrained<N, G, P>(
        &mut self,
        partitions: &mut [P],
        layer_weights: &[N],
        consider_comms: ConsiderComms,
        constrained_partition: &P,
        max_comm_size: Option<usize>,
    ) -> anyhow::Result<N>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping + Clone + Default,
        P: VertexPartition<N, G>,
    {
        let nb_layers = partitions.len();
        if nb_layers == 0 {
            return Ok(N::from(-1.0).unwrap());
        }

        let n = partitions[0].node_count();

        // Check all partitions have same number of nodes
        for partition in partitions.iter() {
            if partition.node_count() != n {
                return Err(anyhow::anyhow!(
                    "Number of nodes are not equal for all graphs."
                ));
            }
        }

        let mut total_improv = N::zero();

        // Establish vertex order and shuffle it
        let mut vertex_order: Vec<usize> = (0..n).collect();
        vertex_order.shuffle(&mut self.rng);

        // Get constrained communities structure
        let constrained_comms = constrained_partition.get_communities();

        let mut comm_added = vec![false; partitions[0].community_count()];
        let mut comms = Vec::new();

        for v in vertex_order {
            let v_comm = partitions[0].membership(v);

            if partitions[0].cnodes(v_comm) == 1 {
                for &comm in &comms {
                    if comm < comm_added.len() {
                        comm_added[comm] = false;
                    }
                }
            }

            comms.clear();

            self.collect_constrained_candidate_communities(
                v,
                partitions,
                constrained_partition,
                &constrained_comms,
                consider_comms,
                &mut comms,
                &mut comm_added,
            );

            let mut max_comm = v_comm;
            let mut max_improv = if let Some(max_size) = max_comm_size {
                if max_size < partitions[0].csize(v_comm) {
                    N::from(f64::NEG_INFINITY).unwrap()
                } else {
                    N::zero()
                }
            } else {
                N::zero()
            };

            let v_size = 1;

            for &comm in &comms {
                if let Some(max_size) = max_comm_size {
                    if max_size < partitions[0].csize(comm) + v_size {
                        continue;
                    }
                }

                let mut possible_improv = N::zero();

                for (layer, partition) in partitions.iter().enumerate() {
                    let layer_imrpov = partition.diff_move(v, comm);
                    possible_improv += layer_weights[layer] * layer_imrpov;
                }

                if possible_improv >= max_improv {
                    max_comm = comm;
                    max_improv = possible_improv;
                }
            }

            if max_comm != v_comm {
                total_improv += max_improv;

                for partition in partitions.iter_mut() {
                    partition.move_node(v, max_comm);
                }
            }
        }

        partitions[0].renumber_communities();
        let membership = partitions[0].membership_vector();
        for partition in partitions.iter_mut().skip(1) {
            partition.set_membership(&membership);
        }

        Ok(total_improv)
    }

    fn refine_and_collapse<N, G, P>(
        &mut self,
        collapsed_partitions: &[P],
        layer_weights: &[N],
        aggregate_node_per_individual_node: &[usize],
        original_n: usize,
    ) -> anyhow::Result<Vec<P>>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping + Clone + Default,
        P: VertexPartition<N, G>,
    {
        let nb_layers = collapsed_partitions.len();

        let mut sub_collapsed_partitions: Vec<P> = Vec::with_capacity(nb_layers);

        for partition in collapsed_partitions {
            let network = partition.network().clone();
            let sub_partition = partition.create_partition(network);
            sub_collapsed_partitions.push(sub_partition);
        }

        match self.config.refine_routine {
            super::OptimiseRoutine::MoveNodes => {
                self.move_nodes_constrained(
                    &mut sub_collapsed_partitions,
                    layer_weights,
                    self.config.refine_consider_comms,
                    &collapsed_partitions[0],
                    self.config.max_community_size,
                )?;
            }
            super::OptimiseRoutine::MergeNodes => {
                self.merge_nodes_constrained(
                    &mut sub_collapsed_partitions,
                    layer_weights,
                    self.config.refine_consider_comms,
                    &collapsed_partitions[0],
                    self.config.max_community_size,
                )?;
            }
        }

        let mut new_collapsed_partitions = Vec::with_capacity(nb_layers);

        for layer in 0..nb_layers {
            let collapsed_network = collapsed_partitions[layer]
                .network()
                .create_reduced_network(sub_collapsed_partitions[layer].grouping());
            let refined_membership = sub_collapsed_partitions[layer].membership_vector();
            let mut new_membership = vec![0; collapsed_network.nodes()];

            for v in 0..collapsed_partitions[layer].node_count() {
                let refined_comm = refined_membership[v];
                let original_comm = collapsed_partitions[layer].membership(v);
                if refined_comm < new_membership.len() {
                    new_membership[refined_comm] = original_comm;
                }
            }

            let new_partition = collapsed_partitions[layer]
                .create_with_membership(collapsed_network, &new_membership);
            new_collapsed_partitions.push(new_partition);
        }

        Ok(new_collapsed_partitions)
    }

    fn simple_collapse<N, G, P>(&self, collapsed_partitions: &[P]) -> anyhow::Result<Vec<P>>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping + Clone + Default,
        P: VertexPartition<N, G>,
    {
        let mut new_collapsed_partitions = Vec::new();

        for partition in collapsed_partitions {
            let collapsed_network = partition
                .network()
                .create_reduced_network(partition.grouping());
            let new_partition = partition.create_partition(collapsed_network);
            new_collapsed_partitions.push(new_partition);
        }

        Ok(new_collapsed_partitions)
    }

    fn from_coarse_partition<N, G, P>(
        &self,
        fine_partition: &mut P,
        coarse_partition: &P,
        aggregate_mapping: &[usize],
    ) where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping,
        P: VertexPartition<N, G>,
    {
        for node in 0..fine_partition.node_count() {
            if node < aggregate_mapping.len() {
                let aggregate_node = aggregate_mapping[node];
                if aggregate_node < coarse_partition.node_count() {
                    let new_comm = coarse_partition.membership(aggregate_node);
                    fine_partition.move_node(node, new_comm);
                }
            }
        }
    }

    fn should_aggregate_further<N, G, P>(
        &self,
        new_partition: &P,
        old_partition: &P,
        is_collapsed_membership_fixed: &[bool],
    ) -> bool
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping,
        P: VertexPartition<N, G>,
    {
        if is_collapsed_membership_fixed.iter().all(|&fixed| fixed) {
            return false;
        }

        new_partition.node_count() < old_partition.node_count()
            && old_partition.node_count() > old_partition.community_count()
    }

    pub fn optimize_partition<N, G, P>(
        &mut self,
        partitions: &mut [P],
        layer_weights: &[N],
        is_membership_fixed: &[bool],
    ) -> anyhow::Result<N>
    where
        N: FloatOpsTS + 'static,
        G: NetworkGrouping + Clone + Default,
        P: VertexPartition<N, G>,
    {
        if partitions.is_empty() {
            return Err(anyhow::anyhow!("No partitions provided"));
        }

        let nb_layers = partitions.len();
        let n = partitions[0].node_count();

        // Validate all partitions have same number of nodes
        for partition in partitions.iter() {
            if partition.node_count() != n {
                return Err(anyhow::anyhow!(
                    "Number of nodes are not equal for all graphs"
                ));
            }
        }

        // Store fixed node information
        let mut fixed_nodes = Vec::new();
        let mut fixed_membership = vec![0; n];
        for v in 0..n {
            if is_membership_fixed[v] {
                fixed_nodes.push(v);
                fixed_membership[v] = partitions[0].membership(v);
            }
        }

        // Initialize collapsed structures - start with original partitions
        let mut collapsed_partitions: Vec<P> = partitions.iter().cloned().collect();
        let mut is_collapsed_membership_fixed = is_membership_fixed.to_vec();
        let mut aggregate_node_per_individual_node: Vec<usize> = (0..n).collect();
        let mut is_first_iteration = true;

        let mut total_improvement = N::zero();
        let mut aggregate_further = true;

        while aggregate_further {
            let improvement = match self.config.optimise_routine {
                super::OptimiseRoutine::MoveNodes => self.move_nodes(
                    &mut collapsed_partitions,
                    layer_weights,
                    &is_collapsed_membership_fixed,
                    self.config.consider_comms,
                    self.config.consider_empty_community,
                    false,
                    self.config.max_community_size,
                )?,
                super::OptimiseRoutine::MergeNodes => self.merge_nodes(
                    &mut collapsed_partitions,
                    layer_weights,
                    &is_collapsed_membership_fixed,
                    self.config.consider_comms,
                    false, // renumber_fixed_nodes
                    self.config.max_community_size,
                )?,
            };

            total_improvement += improvement;

            if !is_first_iteration
                || aggregate_node_per_individual_node.len() != n
                || !aggregate_node_per_individual_node
                    .iter()
                    .enumerate()
                    .all(|(i, &v)| i == v)
            {
                for (layer, partition) in partitions.iter_mut().enumerate() {
                    self.from_coarse_partition(
                        partition,
                        &collapsed_partitions[layer],
                        &aggregate_node_per_individual_node,
                    );
                }
            } else {
                // if not first iteration copy back directly
                for (orig, collapsed) in partitions.iter_mut().zip(collapsed_partitions.iter()) {
                    let membership = collapsed.membership_vector();
                    orig.set_membership(&membership);
                }
            }

            let new_collapsed_partitions = if self.config.refine_partition {
                self.refine_and_collapse(
                    &collapsed_partitions,
                    layer_weights,
                    &aggregate_node_per_individual_node,
                    n,
                )?
            } else {
                self.simple_collapse(&collapsed_partitions)?
            };

            if self.config.refine_partition {
                for v in 0..n {
                    let aggregate_node = aggregate_node_per_individual_node[v];
                    if aggregate_node < new_collapsed_partitions[0].node_count() {
                        aggregate_node_per_individual_node[v] =
                            new_collapsed_partitions[0].membership(aggregate_node);
                    }
                }
            }

            is_collapsed_membership_fixed = vec![false; new_collapsed_partitions[0].node_count()];
            for v in 0..n {
                if is_membership_fixed[v]
                    && aggregate_node_per_individual_node[v] < is_collapsed_membership_fixed.len()
                {
                    is_collapsed_membership_fixed[aggregate_node_per_individual_node[v]] = true;
                }
            }

            aggregate_further = self.should_aggregate_further(
                &new_collapsed_partitions[0],
                &collapsed_partitions[0],
                &is_collapsed_membership_fixed,
            );

            collapsed_partitions = new_collapsed_partitions;
            is_first_iteration = false;
        }

        partitions[0].renumber_communities();
        if !fixed_nodes.is_empty() {
            partitions[0].renumber_communities_fixed(&fixed_nodes, &fixed_membership);
        }

        let membership = partitions[0].membership_vector();
        for partition in partitions.iter_mut().skip(1) {
            partition.set_membership(&membership);
        }

        Ok(total_improvement)
    }
}
