//! Vertex partition trait and implementations for community detection.
//!
//! Defines the core `VertexPartition` trait that abstracts different quality functions
//! (modularity, RB configuration) for community detection algorithms.

use std::collections::HashSet;

use single_utilities::traits::FloatOpsTS;

use crate::network::{grouping::NetworkGrouping, CSRNetwork};

mod modularity;
pub use modularity::ModularityPartition;
mod rb;
pub use rb::RBConfigurationPartition;

/// Core trait for vertex partitions in community detection algorithms.
///
/// Abstracts different quality functions (modularity, RB configuration, etc.) while
/// providing common operations for node movement, community management, and quality evaluation.
/// Implementations must be thread-safe and cloneable for parallel optimization.
pub trait VertexPartition<N, G>: Send + Sync + Clone
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    /// Creates a new partition with singleton communities (each node in its own community).
    fn create_partition(network: CSRNetwork<N, N>) -> Self;

    /// Creates a partition with the specified community membership.
    fn create_with_membership(network: CSRNetwork<N, N>, membership: &[usize]) -> Self;

    /// Creates a partition of the same type for a different network.
    fn create_like(&self, network: CSRNetwork<N, N>) -> Self;

    /// Creates a partition of the same type with specified membership.
    fn create_like_with_membership(&self, network: CSRNetwork<N, N>, membership: &[usize]) -> Self;

    /// Calculates the current partition quality (modularity, RB configuration, etc.).
    fn quality(&self) -> N;

    /// Calculates the quality change from moving a node to a new community.
    ///
    /// Returns the improvement (positive) or degradation (negative) in quality.
    fn diff_move(&mut self, node: usize, new_community: usize) -> N;

    /// Returns a reference to the underlying network.
    fn network(&self) -> &CSRNetwork<N, N>;

    /// Returns a reference to the community grouping structure.
    fn grouping(&self) -> &G;

    /// Returns a mutable reference to the community grouping structure.
    fn grouping_mut(&mut self) -> &mut G;

    /// Moves a node to a new community.
    fn move_node(&mut self, node: usize, new_community: usize) {
        self.grouping_mut().set_group(node, new_community);
    }

    /// Returns the community membership of a node.
    fn membership(&self, node: usize) -> usize {
        self.grouping().get_group(node)
    }

    /// Returns the membership vector for all nodes.
    fn membership_vector(&self) -> Vec<usize> {
        (0..self.node_count()).map(|i| self.membership(i)).collect()
    }

    /// Sets the community membership for all nodes.
    fn set_membership(&mut self, membership: &[usize]) {
        for (node, &comm) in membership.iter().enumerate() {
            self.grouping_mut().set_group(node, comm);
        }
    }

    /// Returns the total number of nodes in the network.
    fn node_count(&self) -> usize {
        self.network().node_count()
    }

    /// Returns the total number of communities in the partition.
    fn community_count(&self) -> usize {
        self.grouping().group_count()
    }

    /// Renumbers communities to be consecutive starting from 0.
    fn renumber_communities(&mut self) {
        self.grouping_mut().normalize_groups();
    }

    /// Renumbers communities while preserving fixed node memberships.
    fn renumber_communities_fixed(&mut self, fixed_nodes: &[usize], fixed_membership: &[usize]) {
        // First renumber normally
        self.renumber_communities();
        
        // Then restore fixed nodes to their original communities
        for (i, &node) in fixed_nodes.iter().enumerate() {
            self.move_node(node, fixed_membership[i]);
        }
    }

    /// Returns the next available empty community ID.
    fn get_empty_community(&mut self) -> usize {
        
        self.community_count()
    }

    /// Adds a new empty community (handled implicitly by moving nodes).
    fn add_empty_community(&mut self) {
        
        // This is handled implicitly when we move a node to a new community number
    }

    /// Updates partition from a coarser partition using node aggregation mapping.
    fn from_coarse_partition<P: VertexPartition<N, G>>(&mut self, coarse_partition: &P, aggregate_mapping: &[usize]) {
        for node in 0..self.node_count() {
            let aggregate_node = aggregate_mapping[node];
            let new_comm = coarse_partition.membership(aggregate_node);
            self.move_node(node, new_comm);
        }
    }

    /// Checks if a node is well-connected within its community.
    ///
    /// Returns true if more than half of the node's connections are internal to the community.
    fn is_well_connected(&self, node: usize, community: usize) -> bool {
        let internal_connections = self.count_internal_connections(node, community);
        let total_connections = self.count_total_connections(node);

        if total_connections == 0 {
            return true;
        }

        N::from(internal_connections).unwrap() / N::from(total_connections).unwrap()
            > N::from(0.5).unwrap()
    }

    /// Counts internal connections within a community for a node.
    fn count_internal_connections(&self, node: usize, community: usize) -> usize {
        self.network()
            .neighbors(node)
            .filter(|(neighbor, _)| self.membership(*neighbor) == community)
            .count()
    }

    /// Counts total connections for a node.
    fn count_total_connections(&self, node: usize) -> usize {
        self.network().neighbors(node).count()
    }

    /// Returns all communities as vectors of their member nodes.
    fn get_communities(&self) -> Vec<Vec<usize>> {
        self.grouping().get_group_members()
    }

    /// Returns the size (number of nodes) of a community.
    fn csize(&self, community: usize) -> usize {
        if community < self.community_count() {
            self.grouping().get_group_members()[community].len()
        } else {
            0
        }
    }

    /// Alias for `csize()` - returns the number of nodes in a community.
    fn cnodes(&self, community: usize) -> usize {
        self.csize(community)
    }

    /// Gets neighboring communities for a node, optionally constrained by membership.
    ///
    /// Returns communities that contain at least one neighbor of the given node.
    /// If constrained membership is provided, only considers neighbors in the same constraint group.
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

