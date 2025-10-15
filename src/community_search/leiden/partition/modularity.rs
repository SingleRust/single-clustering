//! Modularity-based vertex partition implementation.
//!
//! Implements the modularity quality function for community detection, which measures
//! the density of connections within communities compared to a random null model.

use single_utilities::traits::FloatOpsTS;

use crate::{
    community_search::leiden::partition::VertexPartition,
    network::{grouping::NetworkGrouping, CSRNetwork},
};

/// Modularity-based vertex partition for community detection.
///
/// Uses the modularity quality function to evaluate community structures by comparing
/// the density of internal connections against a null model based on node degrees.
#[derive(Clone)]
pub struct ModularityPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    network: CSRNetwork<N, N>,
    grouping: G,
    total_weight: N,
}

impl<N, G> ModularityPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    /// Creates a new modularity partition with the given network and grouping.
    pub fn new(network: CSRNetwork<N, N>, grouping: G) -> Self {
        let tot_weight = network.total_weight();
        Self {
            network,
            grouping,
            total_weight: tot_weight,
        }
    }

    /// Creates a new partition with singleton communities (each node in its own community).
    pub fn new_singleton(network: CSRNetwork<N, N>) -> Self {
        let grouping = G::create_isolated(network.node_count());
        Self::new(network, grouping)
    }

    /// Consumes the partition and returns the underlying grouping structure.
    pub fn into_grouping(self) -> G {
        self.grouping
    }

    /// Calculates the total weight of edges from a node to a specific community.
    fn weight_to_comm(&self, node: usize, community: usize) -> N {
        let mut weight = N::zero();
        for (neighbor, edge_weight) in self.network.neighbors(node) {
            if self.grouping.get_group(neighbor) == community {
                weight += edge_weight;
            }
        }
        weight
    }

    fn weight_from_comm(&self, node: usize, community: usize) -> N {
        // For undirected graphs, this is the same as weight_to_comm
        self.weight_to_comm(node, community)
    }

    /// Calculates the total degree (strength) of all nodes in a community.
    fn total_weight_from_comm(&self, community: usize) -> N {
        if community >= self.grouping.group_count() {
            return N::zero();
        }

        let members = &self.grouping.get_group_members()[community];
        let mut total_weight = N::zero();

        for &node in members {
            total_weight += self.network.strength(node);
        }
        total_weight
    }

    fn total_weight_to_comm(&self, community: usize) -> N {
        // For undirected graphs, this is the same as total_weight_from_comm
        self.total_weight_from_comm(community)
    }

    /// Calculates the total weight of edges within a community (internal edges).
    fn total_weight_in_comm(&self, community: usize) -> N {
        if community >= self.grouping.group_count() {
            return N::zero();
        }

        let members = &self.grouping.get_group_members()[community];
        let mut total_weight = N::zero();

        for &node in members {
            for (neighbor, weight) in self.network.neighbors(node) {
                if self.grouping.get_group(neighbor) == community {
                    if node == neighbor {
                        // Self-loop: count full weight
                        total_weight += weight;
                    } else if node < neighbor {
                        // Regular edge: count once to avoid double counting
                        total_weight += weight;
                    }
                }
            }
        }
        total_weight
    }

    /// Returns the weight of self-loops for a node (if any).
    fn node_self_weight(&self, node: usize) -> N {
        // Look for self-loop
        for (neighbor, weight) in self.network.neighbors(node) {
            if neighbor == node {
                return weight;
            }
        }
        N::zero()
    }

    /// Returns the strength (total degree) of a node.
    fn node_strength(&self, node: usize) -> N {
        self.network.strength(node)
    }

    pub fn diff_move_readonly(&self, node: usize, new_community: usize) -> N {
        let old_comm = self.grouping.get_group(node);
        if new_community == old_comm {
            return N::zero();
        }
        
        if self.total_weight == N::zero() {
            return N::zero();
        }

        let two_m = N::from(2.0).unwrap() * self.total_weight;
        
        let w_to_old = self.weight_to_comm(node, old_comm);
        let w_to_new = if new_community < self.grouping.group_count() {
            self.weight_to_comm(node, new_community)
        } else {
            N::zero() 
        };
        

        let k_i = self.network.strength(node);
        
        let k_old = self.total_weight_from_comm(old_comm);
        let k_new = if new_community < self.grouping.group_count() {
            self.total_weight_from_comm(new_community)
        } else {
            N::zero()
        };
        
        
        let delta_edges = w_to_new - w_to_old;
        let delta_expected = (k_new * k_i - (k_old - k_i) * k_i) / two_m;
        
        (delta_edges - delta_expected) / self.total_weight
    }
}

impl<N, G> VertexPartition<N, G> for ModularityPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping + Clone + Default,
{
    fn create_partition(network: CSRNetwork<N, N>) -> Self {
        let node_count = network.node_count();
        Self::new(network, G::create_isolated(node_count))
    }

    fn create_with_membership(network: CSRNetwork<N, N>, membership: &[usize]) -> Self {
        Self::new(network, G::from_assignments(membership))
    }

    fn quality(&self) -> N {
        let total_weight = self.total_weight; // changed here
        if total_weight == N::zero() {
            return N::zero();
        }

        // For undirected graphs: m = 2 * total_weight
        let m = N::from(2.0).unwrap() * total_weight;
        let mut modularity = N::zero();

        for community in 0..self.community_count() {
            let w = self.total_weight_in_comm(community);
            let w_out = self.total_weight_from_comm(community);
            let w_in = w_out;

            // Following C++ formula for undirected graphs:
            // mod += w - w_out*w_in/(4.0*total_weight)
            let null_model = (w_out * w_in) / (N::from(4.0).unwrap() * total_weight);
            modularity += w - null_model;
        }

        let q = N::from(2.0).unwrap() * modularity;
        q / m
    }

    fn diff_move(&mut self, node: usize, new_community: usize) -> N {
        let old_community = self.grouping.get_group(node);
        if new_community == old_community {
            return N::zero();
        }

        let total_weight = N::from(2.0).unwrap() * self.total_weight;
        if total_weight == N::zero() {
            return N::zero();
        }

        let w_to_old = self.weight_to_comm(node, old_community);
        let w_from_old = w_to_old;
        let w_to_new = self.weight_to_comm(node, new_community);
        let w_from_new = w_to_new;

        let k_out = self.node_strength(node);
        let k_in = k_out; 
        let self_weight = self.node_self_weight(node);

        let K_out_old = self.total_weight_from_comm(old_community);
        let K_in_old = K_out_old; 
        let k_new = self.total_weight_from_comm(new_community);
        let K_out_new = k_new + k_out;
        let K_in_new = k_new + k_in;

        let diff_old = (w_to_old - k_out*K_in_old/total_weight) + 
               (w_from_old - k_in*K_out_old/total_weight);

        let diff_new = (w_to_new + self_weight - k_out*K_in_new/total_weight) + 
               (w_from_new + self_weight - k_in*K_out_new/total_weight);

        let diff = diff_new - diff_old;

        let m = total_weight;
        diff / m
    }

    fn network(&self) -> &CSRNetwork<N, N> {
        &self.network
    }

    fn grouping(&self) -> &G {
        &self.grouping
    }

    fn grouping_mut(&mut self) -> &mut G {
        &mut self.grouping
    }
    
    fn create_like(&self, network: CSRNetwork<N, N>) -> Self {
        Self::create_partition(network)
    }
    
    fn create_like_with_membership(&self, network: CSRNetwork<N, N>, membership: &[usize]) -> Self {
        Self::create_with_membership(network, membership)
    }
    
    fn diff_move_readonly(&self, node: usize, new_community: usize) -> N {
        self.diff_move_readonly(node, new_community)
    }
}
