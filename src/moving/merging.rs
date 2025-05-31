use crate::network::Network;
use rand::prelude::SliceRandom;
use rand::Rng;
use single_utilities::traits::{FloatOpsTS, ZeroVec};
use crate::network::grouping::{NetworkGrouping, VectorGrouping};

pub struct LocalMerging<T>
where
    T: FloatOpsTS,
{
    resolution: T,
    randomness: T,
    cluster_weights: Vec<T>,
    non_singleton_clusters: Vec<bool>,
    external_edge_weight_per_cluster: Vec<T>,
    edge_weight_per_cluster: Vec<T>,
    neighboring_clusters: Vec<usize>,
    cum_transformed_qv_incr_per_cluster: Vec<T>,
    node_order: Vec<usize>,
}

impl<T> LocalMerging<T>
where
    T: FloatOpsTS + 'static,
{
    pub fn new(resolution: T, randomness: T) -> Self {
        LocalMerging {
            resolution,
            randomness,
            cluster_weights: Vec::new(),
            non_singleton_clusters: Vec::new(),
            external_edge_weight_per_cluster: Vec::new(),
            edge_weight_per_cluster: Vec::new(),
            neighboring_clusters: Vec::new(),
            cum_transformed_qv_incr_per_cluster: Vec::new(),
            node_order: Vec::new(),
        }
    }

    pub fn run<R>(&mut self, network: &Network<T, T>, rng: &mut R) -> VectorGrouping
    where
        R: Rng,
    {
        let mut clustering = VectorGrouping::create_isolated(network.nodes());

        if network.nodes() == 1 {
            return clustering;
        }

        let mut update = false;

        let total_node_weight = network.get_total_node_weight();

        // Initialize cluster weights
        self.cluster_weights.clear();
        for i in 0..network.nodes() {
            self.cluster_weights
                .push(network.weight(i));
        }

        // Get total edge weight per node
        self.external_edge_weight_per_cluster.clear();
        network.get_total_edge_weight_per_node(&mut self.external_edge_weight_per_cluster);

        // Generate random permutation of nodes
        self.node_order.clear();
        self.node_order.extend(0..network.nodes());
        self.node_order.shuffle(rng);

        // Initialize arrays
        self.non_singleton_clusters.zero_len(network.nodes());
        self.edge_weight_per_cluster.zero_len(network.nodes());
        self.neighboring_clusters.zero_len(network.nodes());

        for i in 0..network.nodes() {
            let j = self.node_order[i];

            // Only nodes belonging to singleton clusters that are well connected
            // can be moved to a different cluster
            let thresh = self.cluster_weights[j]
                * (total_node_weight - self.cluster_weights[j])
                * self.resolution;
            let thresh = num_traits::Float::max((self.cluster_weights[j] * (total_node_weight - self.cluster_weights[j]) * self.resolution), T::from_f64(1e-10).unwrap());
            if !self.non_singleton_clusters[j] && self.external_edge_weight_per_cluster[j] >= thresh
            {
                // Remove current node from its cluster
                self.cluster_weights[j] = T::zero();
                self.external_edge_weight_per_cluster[j] = T::zero();

                // Identify neighboring clusters
                self.neighboring_clusters[0] = j;
                let mut num_neighboring_clusters = 1;

                for (neighbor, weight) in network.neighbors(j) {
                    let neighbor_cluster = clustering.get_group(neighbor);
                    if self.edge_weight_per_cluster[neighbor_cluster] == T::zero() {
                        self.neighboring_clusters[num_neighboring_clusters] = neighbor_cluster;
                        num_neighboring_clusters += 1;
                    }
                    self.edge_weight_per_cluster[neighbor_cluster] +=
                        T::from_f64(weight.to_f64().unwrap()).unwrap();
                }

                // Calculate quality increments for each neighboring cluster
                let mut best_cluster = j;
                let mut max_qv_increment = T::zero();
                let mut total_transformed_qv_increment = T::zero();

                self.cum_transformed_qv_incr_per_cluster.clear();

                for k in 0..num_neighboring_clusters {
                    let l = self.neighboring_clusters[k];

                    let thresh = num_traits::Float::max((self.cluster_weights[l] * (total_node_weight - self.cluster_weights[l]) * self.resolution), T::from_f64(1e-10).unwrap());
                    if self.external_edge_weight_per_cluster[l] >= thresh {
                        let node_weight = T::from_f64(network.weight(j).to_f64().unwrap()).unwrap();
                        let qv_increment = self.edge_weight_per_cluster[l]
                            - node_weight * self.cluster_weights[l] * self.resolution;

                        if qv_increment > max_qv_increment {
                            best_cluster = l;
                            max_qv_increment = qv_increment;
                        }

                        if qv_increment >= T::zero() {
                            total_transformed_qv_increment +=
                                (qv_increment / self.randomness).exp();
                        }
                    }

                    // Store cumulative value for this cluster
                    self.cum_transformed_qv_incr_per_cluster
                        .push(total_transformed_qv_increment);

                    // Reset edge weight for this cluster
                    self.edge_weight_per_cluster[l] = T::zero();
                }

                // Determine which cluster to move to
                let mut chosen_cluster = best_cluster;

                if total_transformed_qv_increment < <T as num_traits::Float>::infinity() {
                    let r = total_transformed_qv_increment * T::from_f64(rng.random::<f64>()).unwrap();
                    let mut min_idx = -1isize;
                    let mut max_idx = num_neighboring_clusters as isize;

                    while min_idx < max_idx - 1 {
                        let mid_idx = (min_idx + max_idx) / 2;
                        if mid_idx < 0 {
                            min_idx = mid_idx;
                        } else if (mid_idx as usize)
                            >= self.cum_transformed_qv_incr_per_cluster.len()
                        {
                            max_idx = mid_idx;
                        } else if self.cum_transformed_qv_incr_per_cluster[mid_idx as usize] >= r {
                            max_idx = mid_idx;
                        } else {
                            min_idx = mid_idx;
                        }
                    }

                    if max_idx >= 0 && (max_idx as usize) < num_neighboring_clusters {
                        chosen_cluster = self.neighboring_clusters[max_idx as usize];
                    }
                }

                // Move node to its new cluster and update statistics
                self.cluster_weights[chosen_cluster] +=
                    network.weight(j);

                for (neighbor, weight) in network.neighbors(j) {
                    let edge_weight = weight;
                    if clustering.get_group(neighbor) == chosen_cluster {
                        self.external_edge_weight_per_cluster[chosen_cluster] = self.external_edge_weight_per_cluster[chosen_cluster] - edge_weight;
                    } else {
                        self.external_edge_weight_per_cluster[chosen_cluster] += edge_weight;
                    }
                }

                if chosen_cluster != j {
                    clustering.set_group(j, chosen_cluster);
                    self.non_singleton_clusters[chosen_cluster] = true;
                    update = true;
                }
            }
        }

        if update {
            clustering.normalize_groups();
        }

        clustering
    }
}
