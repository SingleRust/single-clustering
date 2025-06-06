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
}

impl<N, G> RBConfigurationPartition<N, G>
where
    N: FloatOpsTS + 'static,
    G: NetworkGrouping,
{
    pub fn new(network: Network<N, N>, grouping: G, resolution: N) -> Self {
        Self {
            network,
            grouping,
            resolution,
        }
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
            total_weight += self.node_strength(node);
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
        let mut modularity = N::zero();

        let m = N::from(4.0).unwrap() * self.network.get_total_edge_weight();

        if m == N::zero() {
            return N::zero();
        }

        for c in 0..self.community_count() {
            let w = self.total_weight_in_comm(c);
            let w_out = self.total_weight_from_comm(c);
            let w_in = self.total_weight_to_comm(c);
            modularity += w - self.resolution * w_out * w_in / m;
        }
        N::from(2.0).unwrap() * modularity
    }

    /// Calculate the difference in quality when moving a node to a new community
    fn diff_move(&self, node: usize, new_community: usize) -> N {
        let mut diff: N = N::zero();
        let old_comm = self.membership(node);
        let total_weight = N::from(2.0).unwrap() * self.network.get_total_edge_weight();
        if total_weight == N::zero() {
            return N::zero();
        }
        if new_community != old_comm {
            let w_to_old = self.weight_to_comm(node, old_comm);
            let w_from_old = self.weight_from_comm(node, old_comm);

            let w_to_new = self.weight_to_comm(node, new_community);
            let w_from_new = self.weight_from_comm(node, new_community);

            let k_out = self.node_strength(node);
            let k_in = self.node_strength(node);

            let self_weight = self.node_self_weight(node);

            let K_out_old = self.total_weight_from_comm(old_comm);
            let K_in_old = self.total_weight_to_comm(old_comm);

            let K_out_new = self.total_weight_from_comm(new_community) + k_out;
            let K_in_new = self.total_weight_to_comm(new_community) + k_in;

            let diff_old = (w_to_old - self.resolution * k_out * K_in_old / total_weight)
                + (w_from_old - self.resolution * k_in * K_out_old / total_weight);
            let diff_new = (w_to_new + self_weight
                - self.resolution * k_out * K_in_new / total_weight)
                + (w_from_new + self_weight - self.resolution * k_in * K_out_new / total_weight);

            

            diff = diff_new - diff_old;

            if node < 5 {
                // Only log first few nodes to avoid spam
                println!(
                    "Node {}: resolution={:?}, diff_old={:?}, diff_new={:?}, final_diff={:?}",
                    node,
                    self.resolution,
                    diff_old,
                    diff_new,
                    diff_new - diff_old
                );
            }
        }
        diff
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
