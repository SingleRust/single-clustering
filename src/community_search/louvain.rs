use crate::moving::standard::StandardLocalMoving;
use crate::moving::QualityFunction;
use crate::network::grouping::{NetworkGrouping, VectorGrouping};
use crate::network::{Graph, Network};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use single_utilities::traits::FloatOpsTS;
use std::collections::HashSet;

pub struct Louvain<T>
where
    T: FloatOpsTS,
{
    rng: ChaCha20Rng,
    local_moving: StandardLocalMoving<T>,
    iterno: u64,
}

impl<T> Louvain<T>
where
    T: FloatOpsTS + 'static,
{
    pub fn new(resolution: T, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_default();
        println!(
            "WARNING!!!!! This implementation extremely highly unfinished and will be moved to a separate package in the future!"
        );
        Louvain {
            rng: ChaCha20Rng::seed_from_u64(seed),
            local_moving: StandardLocalMoving::new(resolution),
            iterno: 0,
        }
    }

    pub fn new_with_quality_function(
        resolution: T, 
        quality_function: QualityFunction, 
        seed: Option<u64>
    ) -> Self {
        let seed = seed.unwrap_or_default();
        println!(
            "WARNING!!!!! This implementation extremely highly unfinished and will be moved to a separate package in the future!"
        );
        Louvain {
            rng: ChaCha20Rng::seed_from_u64(seed),
            local_moving: StandardLocalMoving::new_with_quality_function(resolution, quality_function),
            iterno: 0,
        }
    }

    pub fn new_cpm(resolution: T, seed: Option<u64>) -> Self {
        Self::new_with_quality_function(resolution, QualityFunction::CPM, seed)
    }

    /// Create a new Louvain instance with RBConfiguration quality function (like Modularity)
    pub fn new_rb_configuration(resolution: T, seed: Option<u64>) -> Self {
        Self::new_with_quality_function(resolution, QualityFunction::RBConfiguration, seed)
    }

    /// Create a new Louvain instance with standard Modularity (resolution = 1.0)
    pub fn new_modularity(seed: Option<u64>) -> Self {
        Self::new_rb_configuration(T::one(), seed)
    }

    pub fn iterate_one_level(
        &mut self,
        network: &Network<T, T>,
        clustering: &mut VectorGrouping,
    ) -> bool {
        self.local_moving
            .iterate(network, clustering, &mut self.rng)
    }

    pub fn iterate(&mut self, network: &Network<T, T>, clustering: &mut VectorGrouping) -> bool {
        let num_nodes = network.graph.node_count();
        println!(
            "Running iteration, iteration: {:?}, num nodes: {:?}",
            self.iterno, num_nodes
        );
        self.iterno += 1;
        let mut update = self
            .local_moving
            .iterate(network, clustering, &mut self.rng);

        if clustering.group_count() == network.nodes() {
            return update;
        }

        let reduced_network = network.create_reduced_network(clustering);
        let mut reduced_clustering = VectorGrouping::create_isolated(reduced_network.nodes());
        update |= self.iterate(&reduced_network, &mut reduced_clustering);
        clustering.merge(&reduced_clustering);
        update
    }

    pub fn quality(&self, network: &Network<T, T>, clustering: &VectorGrouping) -> T {
        self.local_moving.calculate_quality(network, clustering)
    }

    pub fn build_network<I>(n_nodes: usize, n_edges: usize, adjacency: I) -> Network<f64, f64>
    where
        I: Iterator<Item = (u32, u32)>,
    {
        let mut graph = Graph::with_capacity(n_nodes, n_edges);
        let mut node_indices = Vec::with_capacity(n_nodes);

        for _ in 0..n_nodes {
            node_indices.push(graph.add_node(1.0));
        }

        let mut seen = vec![HashSet::<u32>::new(); n_nodes];
        let mut node_weights = vec![0.0; n_nodes];

        for (i, j) in adjacency {
            let (i, j) = if i < j { (i, j) } else { (j, i) };
            let i_ = i as usize;
            let j_ = j as usize;

            if seen[i_].insert(j) {
                graph.add_edge(node_indices[i_], node_indices[j_], 1.0);
                node_weights[j_] += 1.0;
                node_weights[i_] += 1.0;
            }
        }

        for &i in &node_indices {
            *graph.node_weight_mut(i).unwrap() = node_weights[i.index()];
        }

        Network::new_from_graph(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::Network;
    use crate::network::grouping::{NetworkGrouping, VectorGrouping};
    use petgraph::graph::NodeIndex;

    fn create_test_network() -> Network<f64, f64> {
        let mut graph = Graph::new_undirected();

        // Add 5 nodes
        for _ in 0..5 {
            graph.add_node(1.0);
        }

        // Add edges to create two communities
        graph.add_edge(NodeIndex::new(0), NodeIndex::new(1), 1.0);
        graph.add_edge(NodeIndex::new(1), NodeIndex::new(2), 1.0);
        graph.add_edge(NodeIndex::new(0), NodeIndex::new(2), 1.0);
        graph.add_edge(NodeIndex::new(3), NodeIndex::new(4), 1.0);

        Network::new_from_graph(graph)
    }

    #[test]
    fn test_louvain_clustering() {
        let network = create_test_network();
        let mut clustering = VectorGrouping::create_isolated(network.nodes());
        let mut louvain: Louvain<f64> = Louvain::new(1.0, Some(42));

        assert!(louvain.iterate(&network, &mut clustering));

        // Should identify two communities
        assert!(clustering.group_count() == 2);

        // Nodes 0,1,2 should be in same cluster
        let cluster1 = clustering.get_group(0);
        assert_eq!(clustering.get_group(1), cluster1);
        assert_eq!(clustering.get_group(2), cluster1);

        // Nodes 3,4 should be in different cluster
        let cluster2 = clustering.get_group(3);
        assert_eq!(clustering.get_group(4), cluster2);
        assert_ne!(cluster1, cluster2);
    }

    #[test]
    fn test_build_network() {
        let edges = vec![(0, 1), (1, 2), (2, 0), (3, 4)];
        let network = Louvain::<f64>::build_network(5, edges.len(), edges.into_iter());

        assert_eq!(network.nodes(), 5);
        assert_eq!(network.graph.edge_count(), 4);

        // Check node weights (should equal degree)
        for i in 0..5 {
            let weight = network.weight(i);
            let expected = match i {
                0..=2 => 2.0, // Nodes in triangle
                3..=4 => 1.0, // Nodes in single edge
                _ => unreachable!(),
            };
            assert_eq!(weight, expected);
        }
    }
}
