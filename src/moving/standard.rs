use rand::prelude::SliceRandom;
use rand::RngCore;
use single_utilities::traits::FloatOpsTS;
use crate::network::grouping::NetworkGrouping;
use crate::network::Network;

#[derive(Debug)]
pub struct StandardLocalMoving<T>
where
    T: FloatOpsTS, {
    resolution: T,
    cluster_weights: Vec<T>,
    nodes_per_cluster: Vec<usize>,
    unused_clusters: Vec<usize>,
    node_order: Vec<usize>,
    edge_weight_per_cluster: Vec<T>,
    neighboring_clusters: Vec<usize>,
}

impl<T> StandardLocalMoving<T>
where
    T: FloatOpsTS {

    pub fn new(resolution: T) -> Self {
        StandardLocalMoving {
            resolution,
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
        let node_count = network.nodes();
        self.ensure_capacity(node_count);

        self.cluster_weights.fill(T::zero());
        self.nodes_per_cluster.fill(0);
        self.edge_weight_per_cluster.fill(T::zero());

        for i in 0..node_count {
            let cluster = clustering.get_group(i);
            self.cluster_weights[cluster] = self.cluster_weights[cluster]
                + T::from_f64(network.weight(i).to_f64().unwrap()).unwrap();
            self.nodes_per_cluster[cluster] += 1;
        }

        let mut num_unused_clusters = 0;
        for i in (0..node_count).rev() {
            if self.nodes_per_cluster[i] == 0 {
                self.unused_clusters[num_unused_clusters] = i;
                num_unused_clusters += 1;
            }
        }
        //println!("Number of unused clusters: {:?}", num_unused_clusters);

        self.node_order.clear();
        self.node_order.extend(0..node_count);
        self.node_order.shuffle(rng);

        let total_edge_weight =
            T::from_f64(network.get_total_edge_weight().to_f64().unwrap()).unwrap();
        let mut num_unstable_nodes = node_count;
        let mut i = 0;

        //println!("Total edge weight: {:?}, num unstable nodes: {:?}, i = {:?}", total_edge_weight.to_f64().unwrap(), num_unstable_nodes, i);

        while num_unstable_nodes > 0 {
            //println!("ITERATION | Total edge weight: {:?}, num unstable nodes: {:?}, i = {:?}", total_edge_weight.to_f64().unwrap(), num_unstable_nodes, i);
            let node = self.node_order[i];
            let current_cluster = clustering.get_group(node);
            //println!("ITERATION | Node: {:?}, Current Cluster: {:?}", node, current_cluster);

            // Remove node from current cluster
            let node_weight = T::from_f64(network.weight(node).to_f64().unwrap()).unwrap();
            self.cluster_weights[current_cluster] =
                self.cluster_weights[current_cluster] - node_weight;
            self.nodes_per_cluster[current_cluster] -= 1;

            if self.nodes_per_cluster[current_cluster] == 0 {
                self.unused_clusters[num_unused_clusters] = current_cluster;
                num_unused_clusters += 1;
                //println!("ITERATION | Nodes per cluster == 0, num unused clusters {:?}", num_unused_clusters);
            }

            // Find neighboring clusters
            self.neighboring_clusters[0] = self.unused_clusters[num_unused_clusters - 1];
            let mut num_neighboring_clusters = 1;

            for (target, weight) in network.neighbors(node) {
                let neighbor_cluster = clustering.get_group(target);
                let edge_weight = T::from_f64(weight.to_f64().unwrap()).unwrap();
                //println!("ITERATION | FORLOOP | target: {:?}, neighbor cluster: {:?}, edge_weight: {:?}", target, neighbor_cluster, edge_weight.to_f64().unwrap());
                if self.edge_weight_per_cluster[neighbor_cluster] == T::zero() {
                    //println!("ITERATION | FORLOOP | Is T::zero, edge_weight_per_cluster");
                    self.neighboring_clusters[num_neighboring_clusters] = neighbor_cluster;
                    num_neighboring_clusters += 1;
                }
                self.edge_weight_per_cluster[neighbor_cluster] =
                    self.edge_weight_per_cluster[neighbor_cluster] + edge_weight;
            }

            // Find best cluster
            let mut best_cluster = current_cluster;
            let mut max_quality_increment = self.edge_weight_per_cluster[current_cluster]
                - (node_weight * self.cluster_weights[current_cluster] * self.resolution)
                / (T::from_f64(2.0).unwrap() * total_edge_weight);

            //println!("ITERATION | Best Cluster {:?} Max Quality Increment {:?}", best_cluster, max_quality_increment.to_f64().unwrap());

            for &cluster in &self.neighboring_clusters[..num_neighboring_clusters] {
                let quality_increment = self.edge_weight_per_cluster[cluster]
                    - (node_weight * self.cluster_weights[cluster] * self.resolution)
                    / (T::from_f64(2.0).unwrap() * total_edge_weight);
                //println!("ITERATION | Cluster {:?} Quality Increment {:?}", cluster, quality_increment.to_f64().unwrap());
                if quality_increment > max_quality_increment
                    || (quality_increment == max_quality_increment && cluster < best_cluster)
                {
                    best_cluster = cluster;
                    max_quality_increment = quality_increment;
                    //println!("ITERATION | Passes if for quality improvement")
                }
                self.edge_weight_per_cluster[cluster] = T::zero();
            }

            // Update cluster assignment
            self.cluster_weights[best_cluster] = self.cluster_weights[best_cluster] + node_weight;
            self.nodes_per_cluster[best_cluster] += 1;

            if best_cluster == self.unused_clusters[num_unused_clusters - 1] {
                num_unused_clusters -= 1;
            }

            num_unstable_nodes -= 1;

            if best_cluster != current_cluster {
                clustering.set_group(node, best_cluster);
                update = true;
            }

            i = (i + 1) % node_count;
            //println!("END: {:?} best cluster {:?} num unstable nodes {:?} num unused clusters {:?}", i, best_cluster, num_unstable_nodes, num_unused_clusters)
        }

        if update {
            clustering.normalize_groups();
        }

        update
    }

}