use std::collections::HashSet;

use single_utilities::traits::FloatOpsTS;

use crate::network::{grouping::NetworkGrouping, CSRNetwork};

mod modularity;
pub use modularity::ModularityPartition;
mod rb;
pub use rb::RBConfigurationPartition;

pub trait VertexPartition<N, G>: Send + Sync + Clone
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    fn create_partition(network: CSRNetwork<N, N>) -> Self;

    fn create_with_membership(network: CSRNetwork<N, N>, membership: &[usize]) -> Self;

    fn create_like(&self, network: CSRNetwork<N, N>) -> Self;

    fn create_like_with_membership(&self, network: CSRNetwork<N, N>, membership: &[usize]) -> Self;

    fn quality(&self) -> N;

    fn diff_move(&self, node: usize, new_community: usize) -> N;

    fn network(&self) -> &CSRNetwork<N, N>;

    fn grouping(&self) -> &G;

    fn grouping_mut(&mut self) -> &mut G;

    fn move_node(&mut self, node: usize, new_community: usize) {
        self.grouping_mut().set_group(node, new_community);
    }

    fn membership(&self, node: usize) -> usize {
        self.grouping().get_group(node)
    }

    fn membership_vector(&self) -> Vec<usize> {
        (0..self.node_count()).map(|i| self.membership(i)).collect()
    }

    fn set_membership(&mut self, membership: &[usize]) {
        for (node, &comm) in membership.iter().enumerate() {
            self.grouping_mut().set_group(node, comm);
        }
    }

    fn node_count(&self) -> usize {
        self.network().node_count()
    }

    fn community_count(&self) -> usize {
        self.grouping().group_count()
    }

    fn renumber_communities(&mut self) {
        self.grouping_mut().normalize_groups();
    }

    fn renumber_communities_fixed(&mut self, fixed_nodes: &[usize], fixed_membership: &[usize]) {
        // First renumber normally
        self.renumber_communities();
        
        // Then restore fixed nodes to their original communities
        for (i, &node) in fixed_nodes.iter().enumerate() {
            self.move_node(node, fixed_membership[i]);
        }
    }

    fn get_empty_community(&mut self) -> usize {
        
        self.community_count()
    }

    fn add_empty_community(&mut self) {
        
        // This is handled implicitly when we move a node to a new community number
    }

    fn from_coarse_partition<P: VertexPartition<N, G>>(&mut self, coarse_partition: &P, aggregate_mapping: &[usize]) {
        for node in 0..self.node_count() {
            let aggregate_node = aggregate_mapping[node];
            let new_comm = coarse_partition.membership(aggregate_node);
            self.move_node(node, new_comm);
        }
    }

    fn is_well_connected(&self, node: usize, community: usize) -> bool {
        let internal_connections = self.count_internal_connections(node, community);
        let total_connections = self.count_total_connections(node);

        if total_connections == 0 {
            return true;
        }

        N::from(internal_connections).unwrap() / N::from(total_connections).unwrap()
            > N::from(0.5).unwrap()
    }

    fn count_internal_connections(&self, node: usize, community: usize) -> usize {
        self.network()
            .neighbors(node)
            .filter(|(neighbor, _)| self.membership(*neighbor) == community)
            .count()
    }

    fn count_total_connections(&self, node: usize) -> usize {
        self.network().neighbors(node).count()
    }

    fn get_communities(&self) -> Vec<Vec<usize>> {
        self.grouping().get_group_members()
    }

    fn csize(&self, community: usize) -> usize {
        if community < self.community_count() {
            self.grouping().get_group_members()[community].len()
        } else {
            0
        }
    }

    fn cnodes(&self, community: usize) -> usize {
        self.csize(community)
    }

    fn get_neigh_comms(&self, node: usize, constrained_membership: Option<&[usize]>) -> Vec<usize> {
        let mut comms = HashSet::new();
        
        for (neighbor, _) in self.network().neighbors(node) {
            // If we have constraints, check them
            if let Some(constraints) = constrained_membership {
                if constraints[node] != constraints[neighbor] {
                    continue;
                }
            }
            comms.insert(self.membership(neighbor));
        }
        
        comms.into_iter().collect()
    }
}

