use single_utilities::traits::FloatOpsTS;

use crate::{
    community_search::leiden::partition::VertexPartition,
    network::{grouping::NetworkGrouping, CSRNetwork},
};

#[derive(Clone)]
pub struct RBConfigurationPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    network: CSRNetwork<N, N>,
    grouping: G,
    resolution: N,
    total_weight: N,
    two_m: N, // Pre-computed 2*total_weight
    
    // Cache for node strengths (degrees) - never changes
    node_strengths: Vec<N>,
    
    // Cache for community strengths - updated when communities change
    community_strengths: Vec<N>,
    community_strengths_dirty: bool,
    
    // Cache for community internal weights
    community_internal_weights: Vec<N>,
    community_internal_weights_dirty: bool,
}

impl<N, G> RBConfigurationPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    pub fn new(network: CSRNetwork<N, N>, grouping: G, resolution: N) -> Self {
        let tot_weight = network.total_weight();
        let two_m = N::from(2.0).unwrap() * tot_weight;
        let node_count = network.node_count();
        
        // Pre-compute node strengths since they never change
        let node_strengths: Vec<N> = (0..node_count)
            .map(|node| network.strength(node))  // Use the cached strength from CSRNetwork
            .collect();
        
        let mut partition = Self {
            network,
            grouping,
            resolution,
            total_weight: tot_weight,
            two_m,
            node_strengths,
            community_strengths: Vec::new(),
            community_strengths_dirty: true,
            community_internal_weights: Vec::new(),
            community_internal_weights_dirty: true,
        };
        
        // Initialize caches
        partition.update_community_caches();
        partition
    }

    pub fn new_singleton(network: CSRNetwork<N, N>, resolution: N) -> Self {
        let grouping = G::create_isolated(network.node_count());
        Self::new(network, grouping, resolution)
    }

    pub fn into_grouping(self) -> G {
        self.grouping
    }
    
    /// Update community caches when needed - optimized for performance
    #[inline]
    fn update_community_caches(&mut self) {
        let community_count = self.grouping.group_count();
        
        if self.community_strengths_dirty {
            // Resize and zero out the community strengths vector
            if self.community_strengths.len() != community_count {
                self.community_strengths.resize(community_count, N::zero());
            } else {
                // Fast zero-out for existing vector
                for strength in &mut self.community_strengths {
                    *strength = N::zero();
                }
            }
            
            // Calculate total strengths for each community
            for node in 0..self.network.node_count() {
                let community = self.grouping.get_group(node);
                self.community_strengths[community] += self.node_strengths[node];
            }
            self.community_strengths_dirty = false;
        }
        
        if self.community_internal_weights_dirty {
            // Resize and zero out
            if self.community_internal_weights.len() != community_count {
                self.community_internal_weights.resize(community_count, N::zero());
            } else {
                for weight in &mut self.community_internal_weights {
                    *weight = N::zero();
                }
            }
            
            // Calculate internal weights for each community
            for node in 0..self.network.node_count() {
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
    #[inline]
    fn invalidate_caches(&mut self) {
        self.community_strengths_dirty = true;
        self.community_internal_weights_dirty = true;
    }

    /// Fast weight calculation from node to community
    #[inline]
    fn weight_to_comm(&self, node: usize, community: usize) -> N {
        let mut weight = N::zero();
        // Use iterator directly to avoid function call overhead
        for (neighbor, edge_weight) in self.network.neighbors(node) {
            if self.grouping.get_group(neighbor) == community {
                weight += edge_weight;
            }
        }
        weight
    }

    /// Get cached community strength (fast lookup)
    #[inline]
    fn get_community_strength(&self, community: usize) -> N {
        if community < self.community_strengths.len() {
            self.community_strengths[community]
        } else {
            N::zero()
        }
    }

    /// Get the self-loop weight of a node (optimized)
    #[inline]
    fn node_self_weight(&self, node: usize) -> N {
        // Use the optimized method from CSRNetwork
        self.network.self_loop_weight(node)
    }

    /// Calculate the strength (degree) of a node (cached lookup)
    #[inline]
    fn node_strength(&self, node: usize) -> N {
        self.node_strengths[node]
    }
}

impl<N, G> VertexPartition<N, G> for RBConfigurationPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping + Clone + Default,
{
    fn create_partition(network: CSRNetwork<N, N>) -> Self {
        let node_count = network.node_count();
        Self::new(network, G::create_isolated(node_count), N::one())
    }

    fn create_with_membership(network: CSRNetwork<N, N>, membership: &[usize]) -> Self {
        Self::new(network, G::from_assignments(membership), N::one())
    }

    /// Calculate the RB quality function
    fn quality(&self) -> N {
        let mut quality_sum = N::zero();

        if self.two_m == N::zero() {
            return N::zero();
        }

        for c in 0..self.community_count() {
            let w_in = self.compute_total_weight_in_comm_uncached(c);
            let k_c = self.compute_total_weight_from_comm_uncached(c);
            
            // RB quality: w_in - γ * k_c² / (2m)
            quality_sum += w_in - self.resolution * k_c * k_c / self.two_m;
        }
        
        quality_sum
    }

    /// Fixed diff_move calculation - this was the main issue
    fn diff_move(&mut self, node: usize, new_community: usize) -> N {
        self.diff_move_cached(node, new_community)
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
    
    #[inline]
    fn move_node(&mut self, node: usize, new_community: usize) {
        if self.grouping.get_group(node) != new_community {
            self.grouping.set_group(node, new_community);
            self.invalidate_caches();
        }
    }
    
    fn create_like(&self, network: CSRNetwork<N, N>) -> Self {
        Self::with_resolution(network, self.resolution)
    }
    
    fn create_like_with_membership(&self, network: CSRNetwork<N, N>, membership: &[usize]) -> Self {
        Self::with_membership_and_resolution(network, membership, self.resolution)
    }
}

// Helper methods for computing without cache (for const methods)
impl<N, G> RBConfigurationPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    /// Optimized diff_move that uses cached values when mutable reference is available
    pub fn diff_move_cached(&mut self, node: usize, new_community: usize) -> N {
        let old_comm = self.grouping.get_group(node);
        if new_community == old_comm {
            return N::zero();
        }
        
        if self.two_m == N::zero() {
            return N::zero();
        }

        // Ensure community strengths are up to date
        self.update_community_caches();

        let k_i = self.node_strengths[node];
        let self_weight = self.node_self_weight(node);
        
        // Calculate weights to old and new communities
        let w_to_old = self.weight_to_comm(node, old_comm);
        let w_to_new = self.weight_to_comm(node, new_community);

        // Use cached community strengths
        let k_old = self.get_community_strength(old_comm);
        let k_new = self.get_community_strength(new_community);

        let delta_w_in = (w_to_new + self_weight) - w_to_old;
        
        // Optimized delta k² calculation
        let delta_k_squared = N::from(2.0).unwrap() * k_i * (k_new - k_old + k_i);
        let delta_null_model = self.resolution * delta_k_squared / self.two_m;

        delta_w_in - delta_null_model
    }

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
                    } else if node < neighbor {
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
    pub fn with_resolution(network: CSRNetwork<N, N>, resolution: N) -> Self {
        let node_count = network.node_count();
        Self::new(network, G::create_isolated(node_count), resolution)
    }

    /// Create RB partition with specified membership and resolution
    pub fn with_membership_and_resolution(
        network: CSRNetwork<N, N>,
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

    /// Get the membership of a node
    #[inline]
    pub fn membership(&self, node: usize) -> usize {
        self.grouping.get_group(node)
    }

    /// Get the community count
    #[inline]
    pub fn community_count(&self) -> usize {
        self.grouping.group_count()
    }

    /// Get the node count
    #[inline]
    pub fn node_count(&self) -> usize {
        self.network.node_count()
    }
}