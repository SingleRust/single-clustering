use single_utilities::traits::FloatOpsTS;

use crate::{
    community_search::leiden::partition::VertexPartition,
    network::{Network, grouping::NetworkGrouping},
};

#[derive(Clone)]
pub struct ModularityPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    network: Network<N, N>,
    grouping: G,
    total_weight: N,
}

impl<N, G> ModularityPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    pub fn new(network: Network<N, N>, grouping: G) -> Self {
        let tot_weight = network.get_total_edge_weight_par();
        Self {
            network,
            grouping,
            total_weight: tot_weight,
        }
    }

    pub fn new_singleton(network: Network<N, N>) -> Self {
        let grouping = G::create_isolated(network.nodes());
        Self::new(network, grouping)
    }

    pub fn into_grouping(self) -> G {
        self.grouping
    }

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

    fn total_weight_from_comm(&self, community: usize) -> N {
        if community >= self.grouping.group_count() {
            return N::zero();
        }

        let members = &self.grouping.get_group_members()[community];
        let mut total_weight = N::zero();

        for &node in members {
            for (_, edge_weight) in self.network.neighbors(node) {
                total_weight += edge_weight;
            }
        }
        total_weight
    }

    fn total_weight_to_comm(&self, community: usize) -> N {
        // For undirected graphs, this is the same as total_weight_from_comm
        self.total_weight_from_comm(community)
    }

    fn total_weight_in_comm(&self, community: usize) -> N {
        if community >= self.grouping.group_count() {
            return N::zero();
        }

        let members = &self.grouping.get_group_members()[community];
        let mut total_weight = N::zero();

        for &node in members {
            for (neighbor, weight) in self.network.neighbors(node) {
                if self.grouping.get_group(neighbor) == community {
                    if node == neighbor || node < neighbor {
                        total_weight += weight;
                    }
                }
            }
        }
        total_weight
    }

    fn node_self_weight(&self, node: usize) -> N {
        // Look for self-loop
        for (neighbor, weight) in self.network.neighbors(node) {
            if neighbor == node {
                return weight;
            }
        }
        N::zero()
    }

    fn node_strength(&self, node: usize) -> N {
        self.network
            .neighbors(node)
            .map(|(_, weight)| weight)
            .fold(N::zero(), |acc, w| acc + w)
    }
}

impl<N, G> VertexPartition<N, G> for ModularityPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping + Clone + Default,
{
    fn create_partition(network: Network<N, N>) -> Self {
        let node_count = network.nodes();
        Self::new(network, G::create_isolated(node_count))
    }

    fn create_with_membership(network: Network<N, N>, membership: &[usize]) -> Self {
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

    fn diff_move(&self, node: usize, new_community: usize) -> N {
        let old_community = self.grouping.get_group(node);
        if new_community == old_community {
            return N::zero();
        }

        let total_weight = N::from(2.0).unwrap() * self.total_weight;
        if total_weight == N::zero() {
            return N::zero();
        }

        let w_to_old = self.weight_to_comm(node, old_community);
        let w_from_old = self.weight_from_comm(node, old_community);
        let w_to_new = self.weight_to_comm(node, new_community);
        let w_from_new = self.weight_from_comm(node, new_community);

        let k_out = self.node_strength(node);
        let k_in = k_out; 
        let self_weight = self.node_self_weight(node);

        let K_out_old = self.total_weight_from_comm(old_community);
        let K_in_old = self.total_weight_to_comm(old_community); 
        let K_out_new = self.total_weight_from_comm(new_community) + k_out;
        let K_in_new = self.total_weight_to_comm(new_community) + k_in;

        let diff_old = (w_to_old - k_out*K_in_old/total_weight) + 
               (w_from_old - k_in*K_out_old/total_weight);

        let diff_new = (w_to_new + self_weight - k_out*K_in_new/total_weight) + 
               (w_from_new + self_weight - k_in*K_out_new/total_weight);

        let diff = diff_new - diff_old;

        let m = total_weight;
        diff / m
    }

    fn network(&self) -> &Network<N, N> {
        &self.network
    }

    fn grouping(&self) -> &G {
        &self.grouping
    }

    fn grouping_mut(&mut self) -> &mut G {
        &mut self.grouping
    }
    
    fn create_like(&self, network: Network<N, N>) -> Self {
        Self::create_partition(network)
    }
    
    fn create_like_with_membership(&self, network: Network<N, N>, membership: &[usize]) -> Self {
        Self::create_with_membership(network, membership)
    }
}
