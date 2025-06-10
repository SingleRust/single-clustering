use single_utilities::traits::FloatOpsTS;

use crate::{
    community_search::leiden::partition::VertexPartition,
    network::{Network, grouping::NetworkGrouping},
};

#[derive(Clone)]
pub struct RBConfigurationPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    network: Network<N, N>,
    grouping: G,
    resolution: N,
    total_weight: N,
    
    // Cache for node strengths (degrees)
    node_strengths: Vec<N>,
    
    // Cache for community total weights (out/in weights)
    community_weights: Vec<N>,
    community_weights_dirty: bool,
    
    // Cache for community internal weights
    community_internal_weights: Vec<N>,
    community_internal_weights_dirty: bool,
}

impl<N, G> RBConfigurationPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    pub fn new(network: Network<N, N>, grouping: G, resolution: N) -> Self {
        let tot_weight = network.get_total_edge_weight_par();
        let node_count = network.nodes();
        
        // Pre-compute node strengths since they never change
        let node_strengths: Vec<N> = (0..node_count)
            .map(|node| {
                network
                    .neighbors(node)
                    .map(|(_, weight)| weight)
                    .fold(N::zero(), |acc, w| acc + w)
            })
            .collect();
        
        let mut partition = Self {
            network,
            grouping,
            resolution,
            total_weight: tot_weight,
            node_strengths,
            community_weights: Vec::new(),
            community_weights_dirty: true,
            community_internal_weights: Vec::new(),
            community_internal_weights_dirty: true,
        };
        
        // Initialize caches
        partition.update_community_caches();
        partition
    }

    pub fn new_singleton(network: Network<N, N>, resolution: N) -> Self {
        let grouping = G::create_isolated(network.nodes());
        Self::new(network, grouping, resolution)
    }

    pub fn into_grouping(self) -> G {
        self.grouping
    }
    
    /// Update community weight caches when needed
    fn update_community_caches(&mut self) {
        if !self.community_weights_dirty && !self.community_internal_weights_dirty {
            return;
        }
        
        let community_count = self.grouping.group_count();
        
        if self.community_weights_dirty {
            self.community_weights = vec![N::zero(); community_count];
            
            // Calculate total weights for each community
            for node in 0..self.network.nodes() {
                let community = self.grouping.get_group(node);
                self.community_weights[community] += self.node_strengths[node];
            }
            self.community_weights_dirty = false;
        }
        
        if self.community_internal_weights_dirty {
            self.community_internal_weights = vec![N::zero(); community_count];
            
            // Calculate internal weights for each community
            for node in 0..self.network.nodes() {
                let node_community = self.grouping.get_group(node);
                for (neighbor, weight) in self.network.neighbors(node) {
                    if self.grouping.get_group(neighbor) == node_community {
                        if node == neighbor {
                            // Self-loop: count full weight
                            self.community_internal_weights[node_community] += weight;
                        } else if node < neighbor {
                            // Regular edge: count once to avoid double counting
                            self.community_internal_weights[node_community] += weight;
                        }
                    }
                }
            }
            self.community_internal_weights_dirty = false;
        }
    }
    
    /// Mark caches as dirty when community structure changes
    fn invalidate_caches(&mut self) {
        self.community_weights_dirty = true;
        self.community_internal_weights_dirty = true;
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

    /// Calculate total weight of edges going out from all nodes in a community (cached)
    fn total_weight_from_comm(&mut self, community: usize) -> N {
        if community >= self.grouping.group_count() {
            return N::zero();
        }
        
        self.update_community_caches();
        self.community_weights[community]
    }

    /// For undirected graphs, total_weight_to_comm is the same as total_weight_from_comm
    fn total_weight_to_comm(&mut self, community: usize) -> N {
        self.total_weight_from_comm(community)
    }

    /// Calculate total weight of edges within a community (cached)
    fn total_weight_in_comm(&mut self, community: usize) -> N {
        if community >= self.grouping.group_count() {
            return N::zero();
        }
        
        self.update_community_caches();
        self.community_internal_weights[community]
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

    /// Calculate the strength (degree) of a node (cached)
    fn node_strength(&self, node: usize) -> N {
        self.node_strengths[node]
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
    fn quality(&self) -> N {
        // We need a mutable reference for caching, but quality should be const
        // For now, we'll compute without caching in this method
        let mut modularity = N::zero();
        let m = N::from(4.0).unwrap() * self.total_weight;

        if m == N::zero() {
            return N::zero();
        }

        for c in 0..self.community_count() {
            // Compute without cache for const method
            let w = self.compute_total_weight_in_comm_uncached(c);
            let w_out = self.compute_total_weight_from_comm_uncached(c);
            let w_in = w_out; // For undirected graphs
            modularity += w - self.resolution * w_out * w_in / m;
        }
        N::from(2.0).unwrap() * modularity
    }

    /// Calculate the difference in quality when moving a node to a new community
    fn diff_move(&self, node: usize, new_community: usize) -> N {
        let old_comm = self.membership(node);
        if new_community == old_comm {
            return N::zero();
        }
        
        let total_weight = N::from(2.0).unwrap() * self.total_weight;
        if total_weight == N::zero() {
            return N::zero();
        }

        let w_to_old = self.weight_to_comm(node, old_comm);
        let w_from_old = self.weight_from_comm(node, old_comm);

        let w_to_new = self.weight_to_comm(node, new_community);
        let w_from_new = self.weight_from_comm(node, new_community);

        let k_out = self.node_strength(node);
        let k_in = k_out; // For undirected graphs

        let self_weight = self.node_self_weight(node);

        // Use cached values where possible
        let K_out_old = self.compute_total_weight_from_comm_uncached(old_comm);
        let K_in_old = K_out_old; // For undirected graphs

        let K_out_new = self.compute_total_weight_from_comm_uncached(new_community) + k_out;
        let K_in_new = K_out_new; // For undirected graphs

        let diff_old = (w_to_old - self.resolution * k_out * K_in_old / total_weight)
            + (w_from_old - self.resolution * k_in * K_out_old / total_weight);
        let diff_new = (w_to_new + self_weight
            - self.resolution * k_out * K_in_new / total_weight)
            + (w_from_new + self_weight - self.resolution * k_in * K_out_new / total_weight);

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
    
    fn move_node(&mut self, node: usize, new_community: usize) {
        if self.grouping.get_group(node) != new_community {
            self.grouping.set_group(node, new_community);
            self.invalidate_caches();
        }
    }
    
    fn create_like(&self, network: Network<N, N>) -> Self {
        Self::with_resolution(network, self.resolution)
    }
    
    fn create_like_with_membership(&self, network: Network<N, N>, membership: &[usize]) -> Self {
        Self::with_membership_and_resolution(network, membership, self.resolution)
    }
}

// Helper methods for computing without cache (for const methods)
impl<N, G> RBConfigurationPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    fn compute_total_weight_from_comm_uncached(&self, community: usize) -> N {
        if community >= self.grouping.group_count() {
            return N::zero();
        }

        let members = &self.grouping.get_group_members()[community];
        let mut total_weight = N::zero();

        for &node in members {
            total_weight += self.node_strengths[node];
        }
        total_weight
    }
    
    fn compute_total_weight_in_comm_uncached(&self, community: usize) -> N {
        if community >= self.grouping.group_count() {
            return N::zero();
        }

        let members = &self.grouping.get_group_members()[community];
        let mut total_weight = N::zero();

        for &node in members {
            for (neighbor, weight) in self.network.neighbors(node) {
                if self.grouping.get_group(neighbor) == community {
                    if node == neighbor {
                        total_weight += weight;
                    } else if node <= neighbor {
                        total_weight += weight;
                    }
                }
            }
        }
        total_weight
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
        resolution: N,
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