use single_utilities::traits::FloatOpsTS;

use crate::{community_search::leiden::partition::VertexPartition, network::{grouping::NetworkGrouping, Network}};

#[derive(Clone)]
pub struct RBConfigurationPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    network: Network<N, N>,
    grouping: G,
    resolution: N,
}

impl<N, G> RBConfigurationPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    pub fn new(network: Network<N, N>, grouping: G, resolution: N) -> Self {
        Self { network, grouping, resolution }
    }

    pub fn new_singleton(network: Network<N, N>, resolution: N) -> Self {
        let grouping = G::create_isolated(network.nodes());
        Self::new(network, grouping, resolution)
    }

    pub fn into_grouping(self) -> G {
        self.grouping
    }

    /// Calculate the weight of edges from a node to a specific community
    fn weight_to_comm(&self, node: usize, community: usize) -> N {
        let mut weight = N::zero();
        for (neighbor, edge_weight) in self.network.neighbors(node) {
            if self.grouping.get_group(neighbor) == community {
                weight += edge_weight;
            }
        }
        weight
    }

    /// For undirected graphs, weight_from_comm is the same as weight_to_comm
    fn weight_from_comm(&self, node: usize, community: usize) -> N {
        self.weight_to_comm(node, community)
    }

    /// Calculate total weight of edges going out from all nodes in a community
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

    /// For undirected graphs, total_weight_to_comm is the same as total_weight_from_comm
    fn total_weight_to_comm(&self, community: usize) -> N {
        self.total_weight_from_comm(community)
    }

    /// Calculate total weight of edges within a community
    fn total_weight_in_comm(&self, community: usize) -> N {
        if community >= self.grouping.group_count() {
            return N::zero();
        }

        let members = &self.grouping.get_group_members()[community];
        let mut total_weight = N::zero();

        for &node in members {
            for (neighbor, weight) in self.network.neighbors(node) {
                if self.grouping.get_group(neighbor) == community {
                    // For undirected graphs, count each edge only once
                    if node <= neighbor {
                        total_weight += weight;
                    }
                }
            }
        }
        total_weight
    }

    /// Get the self-loop weight of a node (if any)
    fn node_self_weight(&self, node: usize) -> N {
        for (neighbor, weight) in self.network.neighbors(node) {
            if neighbor == node {
                return weight;
            }
        }
        N::zero()
    }

    /// Calculate the strength (degree) of a node
    fn node_strength(&self, node: usize) -> N {
        self.network
            .neighbors(node)
            .map(|(_, weight)| weight)
            .fold(N::zero(), |acc, w| acc + w)
    }
}

impl<N, G> VertexPartition<N, G> for RBConfigurationPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping + Clone + Default,
{
    fn create_partition(network: Network<N, N>) -> Self {
        let node_count = network.nodes();
        Self::new(network, G::create_isolated(node_count), N::one())
    }

    fn create_with_membership(network: Network<N, N>, membership: &[usize]) -> Self {
        Self::new(network, G::from_assignments(membership), N::one())
    }

    /// Calculate the RB quality function
    /// Q = Σ_{ij} [A_{ij} - γ * k_i * k_j / (2m)] * δ(c_i, c_j)
    fn quality(&self) -> N {
        let total_edge_weight = self.network.get_total_edge_weight();
        if total_edge_weight == N::zero() {
            return N::zero();
        }

        let two_m = N::from(2.0).unwrap() * total_edge_weight;
        let mut quality = N::zero();

        // Calculate quality for each community
        for community in 0..self.community_count() {
            let w_in = self.total_weight_in_comm(community);
            let k_out = self.total_weight_from_comm(community);
            let k_in = self.total_weight_to_comm(community);

            // RB quality: w_in - γ * k_out * k_in / (2m)
            let null_model_term = self.resolution * k_out * k_in / two_m;
            quality += w_in - null_model_term;
        }

        quality
    }

    /// Calculate the difference in quality when moving a node to a new community
    fn diff_move(&self, node: usize, new_community: usize) -> N {
        let old_community = self.grouping.get_group(node);
        if new_community == old_community {
            return N::zero();
        }

        let total_edge_weight = self.network.get_total_edge_weight();
        if total_edge_weight == N::zero() {
            return N::zero();
        }

        let two_m = N::from(2.0).unwrap() * total_edge_weight;
        
        // Get node properties
        let k_i = self.node_strength(node);
        let w_to_old = self.weight_to_comm(node, old_community);
        let w_from_old = w_to_old;
        let w_to_new = self.weight_to_comm(node, new_community);
        let w_from_new = w_to_new;
        let self_weight = self.node_self_weight(node);

        // Get community strengths
        let k_out_old = self.total_weight_from_comm(old_community);
        let k_in_old = k_out_old;
        let k_out_new = self.total_weight_from_comm(new_community);
        let k_in_new = k_out_new;

        // Calculate change in quality
        // Loss from removing node from old community
        let diff_old = w_to_old + w_from_old - self_weight - 
                       self.resolution * k_i * (k_out_old + k_in_old - k_i) / two_m;

        // Gain from adding node to new community  
        let diff_new = w_to_new + w_from_new + self_weight - 
                       self.resolution * k_i * (k_out_new + k_in_new + k_i) / two_m;

        diff_new - diff_old
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
}

// Convenience constructor functions
impl<N, G> RBConfigurationPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping + Clone + Default,
{
    /// Create RB partition with specified resolution parameter
    pub fn with_resolution(network: Network<N, N>, resolution: N) -> Self {
        let node_count = network.nodes();
        Self::new(network, G::create_isolated(node_count), resolution)
    }

    /// Create RB partition with specified membership and resolution
    pub fn with_membership_and_resolution(
        network: Network<N, N>, 
        membership: &[usize], 
        resolution: N
    ) -> Self {
        Self::new(network, G::from_assignments(membership), resolution)
    }

    /// Get the current resolution parameter
    pub fn resolution(&self) -> N {
        self.resolution
    }

    /// Set a new resolution parameter
    pub fn set_resolution(&mut self, resolution: N) {
        self.resolution = resolution;
    }
}