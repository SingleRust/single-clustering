use crate::network::Network;
use crate::network::grouping::NetworkGrouping;
use rand::RngCore;
use rand::prelude::SliceRandom;
use single_utilities::traits::FloatOpsTS;

use super::QualityFunction;

#[derive(Debug)]
pub struct StandardLocalMoving<T>
where
    T: FloatOpsTS,
{
    resolution: T,
    quality_function: QualityFunction,
    cluster_weights: Vec<T>,
    nodes_per_cluster: Vec<usize>,
    unused_clusters: Vec<usize>,
    node_order: Vec<usize>,
    edge_weight_per_cluster: Vec<T>,
    neighboring_clusters: Vec<usize>,
}

impl<T> StandardLocalMoving<T>
where
    T: FloatOpsTS + 'static,
{
    pub fn new(resolution: T) -> Self {
        StandardLocalMoving {
            resolution,
            quality_function: QualityFunction::CPM,
            cluster_weights: Vec::new(),
            nodes_per_cluster: Vec::new(),
            unused_clusters: Vec::new(),
            node_order: Vec::new(),
            edge_weight_per_cluster: Vec::new(),
            neighboring_clusters: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, size: usize) {
        self.cluster_weights.resize(size, T::zero());
        self.nodes_per_cluster.resize(size, 0);
        self.unused_clusters.resize(size, 0);
        self.node_order.resize(size, 0);
        self.edge_weight_per_cluster.resize(size, T::zero());
        self.neighboring_clusters.resize(size, 0);
    }

    pub fn new_with_quality_function(resolution: T, quality_function: QualityFunction) -> Self {
        StandardLocalMoving {
            resolution,
            quality_function,
            cluster_weights: Vec::new(),
            nodes_per_cluster: Vec::new(),
            unused_clusters: Vec::new(),
            node_order: Vec::new(),
            edge_weight_per_cluster: Vec::new(),
            neighboring_clusters: Vec::new(),
        }
    }

    pub fn new_rb_configuration(resolution: T) -> Self {
        Self::new_with_quality_function(resolution, QualityFunction::RBConfiguration)
    }

    fn calculate_quality_gain(
        &self,
        node_weight: T,
        k_i_in: T,
        cluster_tot: T,
        total_edge_weight_2m: T,
    ) -> T {
        match self.quality_function {
            QualityFunction::CPM => k_i_in - self.resolution * node_weight,
            QualityFunction::RBConfiguration => {
                let two_m = total_edge_weight_2m;
                let term1 = k_i_in;
                let term2 = (self.resolution * node_weight * cluster_tot) / two_m;
                term1 - term2
            }
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
        let node_count = network.nodes();
        if node_count == 0 {
            return false;
        }

        self.ensure_capacity(node_count);

        // Initialize cluster weights and counts
        self.cluster_weights[..node_count].fill(T::zero());
        self.nodes_per_cluster[..node_count].fill(0);
        self.edge_weight_per_cluster[..node_count].fill(T::zero());

        // Calculate initial cluster weights
        for i in 0..node_count {
            let cluster = clustering.get_group(i);
            self.cluster_weights[cluster] += network.weight(i);
            self.nodes_per_cluster[cluster] += 1;
        }

        // Find unused clusters
        let mut num_unused_clusters = 0;
        for i in (0..node_count).rev() {
            if self.nodes_per_cluster[i] == 0 {
                self.unused_clusters[num_unused_clusters] = i;
                num_unused_clusters += 1;
            }
        }

        // Shuffle node order once per iteration
        self.node_order.clear();
        self.node_order.extend(0..node_count);
        self.node_order.shuffle(rng);

        let total_edge_weight_2m = network.get_total_edge_weight() * T::from(2).unwrap();
        let mut global_update = false;

        let mut local_improvement = true;
        while local_improvement {
            local_improvement = false;

            for &node in &self.node_order {
                let current_cluster = clustering.get_group(node);
                let node_weight = network.weight(node);

                // Remove node from current cluster
                self.cluster_weights[current_cluster] -= node_weight;
                self.nodes_per_cluster[current_cluster] -= 1;

                if self.nodes_per_cluster[current_cluster] == 0 {
                    self.unused_clusters[num_unused_clusters] = current_cluster;
                    num_unused_clusters += 1;
                }

                // Clear edge weights from previous iteration
                self.edge_weight_per_cluster[..node_count].fill(T::zero());

                // Find neighboring clusters and calculate edge weights
                self.neighboring_clusters[0] = if num_unused_clusters > 0 {
                    self.unused_clusters[num_unused_clusters - 1]
                } else {
                    current_cluster // fallback
                };
                let mut num_neighboring_clusters = 1;

                for (target, weight) in network.neighbors(node) {
                    let neighbor_cluster = clustering.get_group(target);

                    if self.edge_weight_per_cluster[neighbor_cluster] == T::zero()
                        && neighbor_cluster != self.neighboring_clusters[0]
                    {
                        self.neighboring_clusters[num_neighboring_clusters] = neighbor_cluster;
                        num_neighboring_clusters += 1;
                    }
                    self.edge_weight_per_cluster[neighbor_cluster] += weight;
                }

                // Find best cluster
                let mut best_cluster = current_cluster;
                let mut max_quality_increment = self.calculate_quality_gain(
                    node_weight,
                    self.edge_weight_per_cluster[current_cluster],
                    self.cluster_weights[current_cluster],
                    total_edge_weight_2m,
                );

                for &cluster in &self.neighboring_clusters[..num_neighboring_clusters] {
                    let quality_increment = self.calculate_quality_gain(
                        node_weight,
                        self.edge_weight_per_cluster[cluster],
                        self.cluster_weights[cluster],
                        total_edge_weight_2m,
                    );

                    if quality_increment > max_quality_increment
                        || (quality_increment == max_quality_increment && cluster < best_cluster)
                    {
                        best_cluster = cluster;
                        max_quality_increment = quality_increment;
                    }
                }

                // Update cluster assignment
                self.cluster_weights[best_cluster] += node_weight;
                self.nodes_per_cluster[best_cluster] += 1;

                if best_cluster == self.unused_clusters[num_unused_clusters - 1] {
                    num_unused_clusters -= 1;
                }

                if best_cluster != current_cluster {
                    clustering.set_group(node, best_cluster);
                    local_improvement = true;
                    global_update = true;
                }
            }
        }

        if global_update {
            clustering.normalize_groups();
        }

        global_update
    }

    pub fn calculate_quality<C>(&self, network: &Network<T, T>, clustering: &C) -> T
    where
        C: NetworkGrouping,
    {
        match self.quality_function {
            QualityFunction::CPM => self.calculate_cpm_quality(network, clustering),
            QualityFunction::RBConfiguration => self.calculate_rb_configuration_quality(network, clustering),
        }
    }

    fn calculate_cpm_quality<C>(&self, network: &Network<T, T>, clustering: &C) -> T
    where
        C: NetworkGrouping,
    {
        let mut total_quality = T::zero();
        let mut cluster_internal_edges = vec![T::zero(); clustering.group_count()];
        let mut cluster_sizes = vec![0; clustering.group_count()];

        // Count internal edges and cluster sizes
        for node in 0..network.nodes() {
            let cluster = clustering.get_group(node);
            cluster_sizes[cluster] += 1;

            for (neighbor, weight) in network.neighbors(node) {
                if clustering.get_group(neighbor) == cluster && node <= neighbor {
                    cluster_internal_edges[cluster] += weight;
                }
            }
        }

        // Calculate CPM quality: Σᶜ (eᶜ - γ * nᶜ²)
        for (cluster_idx, &internal_edges) in cluster_internal_edges.iter().enumerate() {
            let cluster_size = T::from(cluster_sizes[cluster_idx]).unwrap();
            total_quality += internal_edges - self.resolution * cluster_size * cluster_size;
        }

        total_quality
    }

    fn calculate_rb_configuration_quality<C>(&self, network: &Network<T, T>, clustering: &C) -> T
    where
        C: NetworkGrouping,
    {
        let mut total_quality = T::zero();
        let mut cluster_internal_edges = vec![T::zero(); clustering.group_count()];
        let mut cluster_total_weight = vec![T::zero(); clustering.group_count()];

        // Count internal edges and total weights per cluster
        for node in 0..network.nodes() {
            let cluster = clustering.get_group(node);
            cluster_total_weight[cluster] += network.weight(node);

            for (neighbor, weight) in network.neighbors(node) {
                if clustering.get_group(neighbor) == cluster && node <= neighbor {
                    cluster_internal_edges[cluster] += weight;
                }
            }
        }

        let total_edge_weight_2m = network.get_total_edge_weight() * T::from(2).unwrap();

        // Calculate RBConfiguration quality: Σᶜ (eᶜ - γ * Kᶜ²/(4m))
        for (cluster_idx, &internal_edges) in cluster_internal_edges.iter().enumerate() {
            let cluster_weight = cluster_total_weight[cluster_idx];
            let expected_internal = (self.resolution * cluster_weight * cluster_weight) / total_edge_weight_2m;
            total_quality += internal_edges - expected_internal;
        }

        total_quality
    }

}
