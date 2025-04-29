use crate::network::Network;
use crate::network::grouping::NetworkGrouping;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::RngCore;
use rand::prelude::SliceRandom;
use single_utilities::traits::{FloatOpsTS, ZeroVec};
use std::iter::Sum;
use std::ops::MulAssign;

#[derive(Default)]
pub(crate) struct FastLocalMoving<T>
where
    T: FloatOpsTS,
{
    resolution: T,
    cluster_weights: Vec<T>,
    nodes_per_cluster: Vec<usize>,
    unused_clusters: Vec<usize>,
    node_order: Vec<usize>,
    edge_weight_per_cluster: Vec<T>,
    neighboring_clusters: Vec<usize>,
    stable_nodes: Vec<bool>,
}

impl<T> FastLocalMoving<T>
where
    T: FloatOpsTS,
{
    pub fn new(resolution: T) -> Self {
        FastLocalMoving {
            resolution,
            ..FastLocalMoving::default()
        }
    }

    pub fn iterate<C, R>(
        &mut self,
        network: &Network<T, T>,
        clustering: &mut C,
        rng: &mut R,
    ) -> bool
    where
        C: NetworkGrouping,
        R: RngCore,
    {
        let mut update = false;

        // Initialize vectors
        self.cluster_weights.zero_len(network.nodes());
        self.nodes_per_cluster.zero_len(network.nodes());
        self.edge_weight_per_cluster.zero_len(network.nodes());
        self.neighboring_clusters.zero_len(network.nodes());
        self.stable_nodes.zero_len(network.nodes());

        // Calculate initial cluster weights and sizes
        for i in 0..network.nodes() {
            let cluster = clustering.get_group(i);
            self.cluster_weights[cluster] += network.weight(i);
            self.nodes_per_cluster[cluster] += 1;
        }

        // Find unused clusters
        let mut num_unused_clusters = 0;
        self.unused_clusters.zero_len(network.nodes());
        for i in (0..network.nodes()).rev() {
            if self.nodes_per_cluster[i] == 0 {
                self.unused_clusters[num_unused_clusters] = i;
                num_unused_clusters += 1;
            }
        }

        // Create random order
        self.node_order.clear();
        self.node_order.extend(0..network.nodes());
        self.node_order.shuffle(rng);

        let mut num_unstable_nodes = network.nodes();
        let mut i = 0;

        while num_unstable_nodes > 0 {
            let j = self.node_order[i];
            let current_cluster = clustering.get_group(j);

            // Remove node from current cluster
            self.cluster_weights[current_cluster] =
                self.cluster_weights[current_cluster] - network.weight(j);
            self.nodes_per_cluster[current_cluster] -= 1;

            if self.nodes_per_cluster[current_cluster] == 0 {
                self.unused_clusters[num_unused_clusters] = current_cluster;
                num_unused_clusters += 1;
            }

            // Check available clusters
            self.neighboring_clusters[0] = self.unused_clusters[num_unused_clusters - 1];
            let mut num_neighboring_clusters = 1;

            // Get edge weights to neighbor clusters
            for (target, weight) in network.neighbors(j) {
                let neighbor_cluster = clustering.get_group(target);

                if self.edge_weight_per_cluster[neighbor_cluster] == T::zero() {
                    self.neighboring_clusters[num_neighboring_clusters] = neighbor_cluster;
                    num_neighboring_clusters += 1;
                }

                self.edge_weight_per_cluster[neighbor_cluster] += weight;
            }

            // Find best cluster
            let mut best_cluster = current_cluster;
            let mut max_qv_increment = self.edge_weight_per_cluster[current_cluster]
                - network.weight(j) * self.cluster_weights[current_cluster] * self.resolution;

            for k in 0..num_neighboring_clusters {
                let l = self.neighboring_clusters[k];

                let qv_increment = self.edge_weight_per_cluster[l]
                    - network.weight(j) * self.cluster_weights[l] * self.resolution;

                if qv_increment > max_qv_increment {
                    best_cluster = l;
                    max_qv_increment = qv_increment;
                }

                // Reset edge weights as we go
                self.edge_weight_per_cluster[l] = T::zero();
            }

            // Update cluster membership
            self.cluster_weights[best_cluster] += network.weight(j);
            self.nodes_per_cluster[best_cluster] += 1;

            if best_cluster == self.unused_clusters[num_unused_clusters - 1] {
                num_unused_clusters -= 1;
            }

            // Mark node as stable
            self.stable_nodes[j] = true;
            num_unstable_nodes -= 1;

            // Update if moved
            if best_cluster != current_cluster {
                clustering.set_group(j, best_cluster);

                // Check neighbors
                for (neighbor, _) in network.neighbors(j) {
                    if self.stable_nodes[neighbor] && clustering.get_group(neighbor) != best_cluster
                    {
                        self.stable_nodes[neighbor] = false;
                        num_unstable_nodes += 1;
                        self.node_order[(i + num_unstable_nodes) % network.nodes()] = neighbor;
                    }
                }

                update = true;
            }

            i = (i + 1) % network.nodes();
        }

        if update {
            clustering.normalize_groups();
        }

        update
    }
}
