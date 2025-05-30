use crate::neighborhood::connectivity::GaussianConnectivity;
use crate::network::grouping::NetworkGrouping;
use nalgebra_sparse::CsrMatrix;
use petgraph::data::DataMap;
use petgraph::graph::{Edges, UnGraph};
use petgraph::prelude::{EdgeRef, NodeIndex};
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSlice;
use single_utilities::traits::FloatOpsTS;
use std::collections::HashMap;

pub mod grouping;

pub type Graph<N, E> = UnGraph<N, E>;

pub struct Network<N, E> {
    pub graph: Graph<N, E>,
}

pub struct NeighborAndWeightIterator<'a, N: 'a, E: 'a> {
    edge_iter: Edges<'a, E, petgraph::Undirected>,
    home_node: usize,
    _phantom: std::marker::PhantomData<&'a N>,
}

impl<'a, N, E> Iterator for NeighborAndWeightIterator<'a, N, E>
where
    E: Copy,
{
    type Item = (usize, E);

    fn next(&mut self) -> Option<Self::Item> {
        self.edge_iter.next().map(|edge_ref| {
            let neighbor = if edge_ref.source().index() == self.home_node {
                edge_ref.target().index()
            } else {
                edge_ref.source().index()
            };
            (neighbor, *edge_ref.weight())
        })
    }
}

impl<N, E> Default for Network<N, E>
where
    N: FloatOpsTS,
    E: FloatOpsTS,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> Network<N, E>
where
    N: FloatOpsTS,
    E: FloatOpsTS,
{
    pub fn new() -> Self {
        Network {
            graph: Graph::new_undirected(),
        }
    }

    pub fn new_from_graph(graph: Graph<N, E>) -> Self {
        Network { graph }
    }

    pub fn nodes(&self) -> usize {
        self.graph.node_count()
    }

    pub fn weight(&self, node: usize) -> N {
        *self
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(node))
            .unwrap()
    }

    pub fn neighbors(&self, node: usize) -> NeighborAndWeightIterator<'_, N, E> {
        NeighborAndWeightIterator {
            edge_iter: self.graph.edges(petgraph::graph::NodeIndex::new(node)),
            home_node: node,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn get_total_node_weight(&self) -> N {
        self.graph
            .node_weights()
            .fold(N::zero(), |sum, node| sum + *node)
    }

    pub fn get_total_edge_weight(&self) -> E {
        self.graph
            .edge_weights()
            .fold(E::zero(), |sum, edge| sum + *edge)
    }

    pub fn get_total_edge_weight_par(&self) -> E {
        let weights: Vec<_> = self.graph.edge_weights().collect();
        weights
            .par_chunks(256)
            .map(|chunk| chunk.iter().fold(E::zero(), |acc, &weight| acc + *weight))
            .sum()
    }

    pub fn get_total_edge_weight_per_node(&self, result: &mut Vec<E>) {
        result.clear();
        result.extend((0..self.nodes()).map(|i| {
            self.graph
                .edges(petgraph::graph::NodeIndex::new(i))
                .fold(E::zero(), |acc, edge| acc + *edge.weight())
        }));
    }

    pub fn create_reduced_network<T: NetworkGrouping>(&self, grouping: &T) -> Self {
        let mut cluster_g =
            Graph::with_capacity(grouping.group_count(), grouping.group_count() * 2);
        for _ in 0..grouping.group_count() {
            cluster_g.add_node(N::zero());
        }

        for node_idx in self.graph.node_indices() {
            let group = grouping.get_group(node_idx.index());
            let group_node = NodeIndex::new(group);
            let current_weight = self.graph.node_weight(node_idx).unwrap();
            let group_weight = cluster_g.node_weight_mut(group_node).unwrap();
            *group_weight += *current_weight;
        }

        let mut edge_memo = HashMap::new();

        // transferring all edges between nodes of previous graph
        // see if they are still there after reduction of nodes
        // if they are still kept, they will be reinserted (same weight or add if multiple edges in that direction!!!!!!!!)

        for edge in self.graph.edge_references() {
            let g1 = grouping.get_group(edge.source().index());
            let g2 = grouping.get_group(edge.target().index());
            if g1 == g2 {
                continue; // self loops are ignored !!
            }

            let (min_g, max_g) = if g1 < g2 { (g1, g2) } else { (g2, g1) };
            *edge_memo.entry((min_g, max_g)).or_insert(E::zero()) += *edge.weight();
        }

        for (&(g1, g2), &weight) in edge_memo.iter() {
            cluster_g.add_edge(NodeIndex::new(g1), NodeIndex::new(g2), weight);
        }

        Network { graph: cluster_g }
    }

    pub fn create_subnetworks<T: NetworkGrouping>(&self, grouping: &T) -> Vec<Self> {
        let mut graphs = vec![Graph::new_undirected(); grouping.group_count()];
        let mut new_id_map = vec![0; self.nodes()];
        let mut counts = vec![0; grouping.group_count()];

        for node_idx in self.graph.node_indices() {
            let node = node_idx.index();
            let group = grouping.get_group(node);
            let new_id = counts[group];
            new_id_map[node] = new_id;
            counts[group] += 1;
            graphs[group].add_node(*self.graph.node_weight(node_idx).unwrap());
        }

        for edge in self.graph.edge_references() {
            let n1 = edge.source().index();
            let g1 = grouping.get_group(n1);

            let n2 = edge.target().index();
            let g2 = grouping.get_group(n2);

            if g1 == g2 {
                graphs[g1].add_edge(
                    NodeIndex::new(new_id_map[n1]),
                    NodeIndex::new(new_id_map[n2]),
                    *edge.weight(),
                );
            }
        }

        graphs
            .into_iter()
            .map(Network::new_from_graph)
            .collect::<Vec<_>>()
    }

    pub fn create_subnetwork_from_group<T: NetworkGrouping>(
        &self,
        grouping: &T,
        group: usize,
    ) -> Self {
        let mut subgraph = Graph::new_undirected();
        let mut old_to_new = HashMap::new();

        for node_idx in self.graph.node_indices() {
            if grouping.get_group(node_idx.index()) == group {
                let new_idx = subgraph.add_node(*self.graph.node_weight(node_idx).unwrap());
                old_to_new.insert(node_idx, new_idx);
            }
        }

        for edge in self.graph.edge_references() {
            let source = edge.source();
            let target = edge.target();

            if let (Some(&new_source), Some(&new_target)) =
                (old_to_new.get(&source), old_to_new.get(&target))
            {
                subgraph.add_edge(new_source, new_target, *edge.weight());
            }
        }

        Network { graph: subgraph }
    }
}

pub fn network_from_csr_matrix<T>(csr_matrix: CsrMatrix<T>) -> Network<T, T>
where
    T: FloatOpsTS,
{
    let n_nodes = csr_matrix.ncols();
    let mut graph = Graph::with_capacity(n_nodes, csr_matrix.nnz());

    let mut node_indices = Vec::with_capacity(n_nodes);

    let mut node_weights = vec![T::zero(); n_nodes];

    for (row, row_vec) in csr_matrix.row_iter().enumerate() {
        let weight = row_vec.values().iter().fold(T::zero(), |acc, &x| acc + x);
        node_weights[row] = weight;
        node_indices.push(graph.add_node(weight));
    }

    for (row, col, &weight) in csr_matrix.triplet_iter() {
        if row <= col {
            graph.add_edge(node_indices[row], node_indices[col], weight);
        }
    }

    Network::new_from_graph(graph)
}

fn csr_to_petgraph<T>(connectivity: CsrMatrix<T>, node_weights: Vec<T>) -> Graph<T, T>
where
    T: FloatOpsTS,
{
    let mut graph = Graph::with_capacity(node_weights.len(), connectivity.nnz());

    let mut node_indices = Vec::with_capacity(node_weights.len());
    for weight in node_weights {
        node_indices.push(graph.add_node(weight));
    }

    for (row, col, &weight) in connectivity.triplet_iter() {
        if row <= col {
            graph.add_edge(node_indices[row], node_indices[col], weight);
        }
    }

    graph
}

pub fn network_from_gaussian_connectivity<T>(
    distances: &CsrMatrix<T>,
    node_weights: Vec<T>,
    n_neighbors: usize,
    knn: bool,
) -> Network<T, T>
where
    T: FloatOpsTS,
{
    let gauss_conn = GaussianConnectivity::new(knn);
    let connectivity_matrix = gauss_conn.compute_connectivities(distances, n_neighbors);
    let graph = csr_to_petgraph(connectivity_matrix, node_weights);
    Network::new_from_graph(graph)
}


