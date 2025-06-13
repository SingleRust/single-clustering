use core::num;
use std::collections::HashMap;

use nalgebra_sparse::CsrMatrix;
use rand::random;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use single_utilities::traits::FloatOpsTS;

use crate::network::grouping::{self, NetworkGrouping};

#[derive(Debug, Clone)]
pub struct CSRNetwork<N, E> {
    node_ptrs: Vec<usize>,
    neighbors: Vec<usize>,
    weights: Vec<E>,

    node_weights: Vec<N>,

    degrees: Vec<usize>,
    strengths: Vec<E>,
    total_weight: E,
    edge_count: usize,
}

impl<N, E> CSRNetwork<N, E>
where
    N: FloatOpsTS + 'static,
    E: FloatOpsTS + 'static,
{
    pub fn from_edges(edges: &[(usize, usize, E)], node_weights: Vec<N>) -> Self {
        let num_nodes = node_weights.len();

        let mut degrees = vec![0; num_nodes];
        for &(from, to, _) in edges {
            degrees[from] += 1;
            if from != to {
                degrees[to] += 1;
            }
        }

        let total_degree: usize = degrees.iter().sum();
        let mut node_ptrs = Vec::with_capacity(num_nodes + 1);
        let mut neighbors = Vec::with_capacity(total_degree);
        let mut weights = Vec::with_capacity(total_degree);
        let mut strengths = vec![E::zero(); num_nodes];

        let mut adjacency_lists: Vec<Vec<(usize, E)>> = vec![Vec::new(); num_nodes];
        for &(from, to, weight) in edges {
            adjacency_lists[from].push((to, weight));
            if from != to {
                adjacency_lists[to].push((from, weight));
            }
        }

        node_ptrs.push(0);
        let mut total_weight = E::zero();

        for (node, adj_list) in adjacency_lists.into_iter().enumerate() {
            let mut sorted_adj = adj_list;
            sorted_adj.sort_by_key(|&(neighbor, _)| neighbor);

            for (neighbor, weight) in sorted_adj {
                neighbors.push(neighbor);
                weights.push(weight);
                strengths[node] += weight;

                if node <= neighbor {
                    total_weight += weight;
                }
            }
            node_ptrs.push(neighbors.len());
        }

        Self {
            node_ptrs,
            neighbors,
            weights,
            node_weights,
            degrees,
            strengths,
            total_weight,
            edge_count: edges.len(),
        }
    }

    pub fn from_csr_matrix(matrix: CsrMatrix<E>, node_weights: Vec<N>) -> Self {
        let mut edges = Vec::new();

        for (row, col, &weight) in matrix.triplet_iter() {
            if weight != E::zero() {
                // Only add upper triangle for undirected graphs to avoid duplicates
                if row == col {
                    let tot_w = E::from(2.0).unwrap() * weight;
                    edges.push((row, col, tot_w));
                } else if row < col {
                    edges.push((row, col, weight));
                }
            }
        }

        Self::from_edges(&edges, node_weights)
    }

    #[inline]
    pub fn neighbors(&self, node: usize) -> CSRNeighborIterator<E> {
        debug_assert!(node < self.node_count());

        let start = self.node_ptrs[node];
        let end = self.node_ptrs[node + 1];

        CSRNeighborIterator {
            neighbor_ptr: unsafe { self.neighbors.as_ptr().add(start) },
            weight_ptr: unsafe { self.weights.as_ptr().add(start) },
            remaining: end - start,
        }
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_weights.len()
    }
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }
    #[inline]
    pub fn degree(&self, node: usize) -> usize {
        self.degrees[node]
    }
    #[inline]
    pub fn strength(&self, node: usize) -> E {
        self.strengths[node]
    }
    #[inline]
    pub fn node_weight(&self, node: usize) -> N {
        self.node_weights[node]
    }
    #[inline]
    pub fn total_weight(&self) -> E {
        self.total_weight
    }

    pub fn random_neighbor(&self, node: usize, rng: &mut impl rand::Rng) -> Option<usize> {
        let degree = self.degree(node);
        if degree == 0 {
            return None;
        }

        let random_idx = rng.random_range(0..degree);
        let neighbor_idx = self.node_ptrs[node] + random_idx;
        Some(self.neighbors[neighbor_idx])
    }

    pub fn edge_weight(&self, from: usize, to: usize) -> Option<E> {
        let (search_node, target) = if self.degree(from) <= self.degree(to) {
            (from, to)
        } else {
            (to, from)
        };

        let start = self.node_ptrs[search_node];
        let end = self.node_ptrs[search_node + 1];

        match self.neighbors[start..end].binary_search(&target) {
            Ok(pos) => Some(self.weights[start + pos]),
            Err(_) => None,
        }
    }

    pub fn aggregate<G: NetworkGrouping>(&self, grouping: &G) -> Self {
        let new_node_count = grouping.group_count();
        
        let mut new_node_weights = vec![N::zero(); new_node_count];
        
        for node in 0..self.node_count() {
            let group = grouping.get_group(node);
            new_node_weights[group] += self.node_weights[node];
        }
        
        let mut edge_memo = HashMap::new();
        let mut self_loop_weights = HashMap::new();
        
        for node in 0..self.node_count() {
            let start = self.node_ptrs[node];
            let end = self.node_ptrs[node + 1];
            
            for i in start..end {
                let neighbor = self.neighbors[i];
                let weight = self.weights[i];
                
                if node <= neighbor {
                    let g1 = grouping.get_group(node);
                    let g2 = grouping.get_group(neighbor);
                    
                    if g1 == g2 {
                        *self_loop_weights.entry(g1).or_insert(E::zero()) += weight;
                    } else {
                        let (min_g, max_g) = if g1 < g2 { (g1, g2) } else { (g2, g1) };
                        *edge_memo.entry((min_g, max_g)).or_insert(E::zero()) += weight;
                    }
                }
            }
        }
        
        let mut edges = Vec::new();
        
        for (&group, &weight) in self_loop_weights.iter() {
            if weight > E::zero() {
                edges.push((group, group, weight));
            }
        }
        
        for (&(g1, g2), &weight) in edge_memo.iter() {
            edges.push((g1, g2, weight));
        }
        
        Self::from_edges(&edges, new_node_weights)
    }

    pub fn subgraph<G: NetworkGrouping>(&self, grouping: &G, group: usize) -> Self {
        let group_members = &grouping.get_group_members()[group];
        let subgraph_size = group_members.len();

        let mut node_map = HashMap::with_capacity(subgraph_size);
        let mut new_node_weights = Vec::with_capacity(subgraph_size);

        for (new_id, &old_id) in group_members.iter().enumerate() {
            node_map.insert(old_id, new_id);
            new_node_weights.push(self.node_weights[old_id]);
        }

        let mut edges = Vec::new();
        for &node in group_members {
            let from_new = node_map[&node];

            for (neighbor, weight) in self.neighbors(node) {
                if let Some(&to_new) = node_map.get(&neighbor) {
                    if from_new <= to_new {
                        edges.push((from_new, to_new, weight));
                    }
                }
            }
        }
        Self::from_edges(&edges, new_node_weights)
    }

    pub fn to_csr_matrix(&self) -> CsrMatrix<E> {
        let n = self.node_count();
        let mut row_ptrs = vec![0; n + 1];
        let mut col_indices = Vec::with_capacity(self.neighbors.len());
        let mut values = Vec::with_capacity(self.weights.len());

        for node in 0..n {
            for (neighbor, weight) in self.neighbors(node) {
                col_indices.push(neighbor);
                values.push(weight);
            }
            row_ptrs[node + 1] = col_indices.len();
        }

        CsrMatrix::try_from_csr_data(n, n, row_ptrs, col_indices, values).unwrap()
    }

    pub fn has_self_loops(&self) -> bool {
        for node in 0..self.node_count() {
            for (neighbor, _) in self.neighbors(node) {
                if neighbor == node {
                    return true;
                }
            }
        }
        false
    }

    pub fn density(&self) -> f64 {
        let n = self.node_count() as f64;
        let m = self.edge_count as f64;
        let max_edges = n * (n - 1.0) / 2.0;

        if max_edges > 0.0 { m / max_edges } else { 0.0 }
    }

    #[inline]
    pub fn weight_to_comm(&self, node: usize, community: usize, grouping: &impl NetworkGrouping) -> E {
        let start = self.node_ptrs[node];
        let end = self.node_ptrs[node + 1];
        
        if start == end {
            return E::zero();
        }
        
        let mut weight = E::zero();
        
        // Direct unsafe pointer access for maximum performance
        unsafe {
            let mut neighbor_ptr = self.neighbors.as_ptr().add(start);
            let mut weight_ptr = self.weights.as_ptr().add(start);
            let mut remaining = end - start;
            
            while remaining > 0 {
                let neighbor = *neighbor_ptr;
                if grouping.get_group(neighbor) == community {
                    weight += *weight_ptr;
                }
                neighbor_ptr = neighbor_ptr.add(1);
                weight_ptr = weight_ptr.add(1);
                remaining -= 1;
            }
        }
        
        weight
    }

    #[inline]
    pub fn weight_to_two_comms(&self, node: usize, comm1: usize, comm2: usize, grouping: &impl NetworkGrouping) -> (E, E) {
        let start = self.node_ptrs[node];
        let end = self.node_ptrs[node + 1];
        
        if start == end {
            return (E::zero(), E::zero());
        }
        
        let mut w1 = E::zero();
        let mut w2 = E::zero();
        
        unsafe {
            let mut neighbor_ptr = self.neighbors.as_ptr().add(start);
            let mut weight_ptr = self.weights.as_ptr().add(start);
            let mut remaining = end - start;
            
            while remaining > 0 {
                let neighbor = *neighbor_ptr;
                let neighbor_comm = grouping.get_group(neighbor);
                let weight = *weight_ptr;
                
                if neighbor_comm == comm1 {
                    w1 += weight;
                } else if neighbor_comm == comm2 {
                    w2 += weight;
                }
                
                neighbor_ptr = neighbor_ptr.add(1);
                weight_ptr = weight_ptr.add(1);
                remaining -= 1;
            }
        }
        
        (w1, w2)
    }

    pub fn weight_to_comms_batch(&self, node: usize, communities: &[usize], grouping: &impl NetworkGrouping) -> Vec<E> {
        let mut weights = vec![E::zero(); communities.len()];
        
        // Create lookup map for community index
        let community_to_idx: HashMap<usize, usize> = communities
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        
        let start = self.node_ptrs[node];
        let end = self.node_ptrs[node + 1];
        
        unsafe {
            let mut neighbor_ptr = self.neighbors.as_ptr().add(start);
            let mut weight_ptr = self.weights.as_ptr().add(start);
            let mut remaining = end - start;
            
            while remaining > 0 {
                let neighbor = *neighbor_ptr;
                let neighbor_comm = grouping.get_group(neighbor);
                
                if let Some(&idx) = community_to_idx.get(&neighbor_comm) {
                    weights[idx] += *weight_ptr;
                }
                
                neighbor_ptr = neighbor_ptr.add(1);
                weight_ptr = weight_ptr.add(1);
                remaining -= 1;
            }
        }
        
        weights
    }

    #[inline]
    pub fn self_loop_weight(&self, node: usize) -> E {
        let start = self.node_ptrs[node];
        let end = self.node_ptrs[node + 1];
        
        if start >= end {
            return E::zero();
        }
        
        // Check if first neighbor is self (common optimization)
        if self.neighbors[start] == node {
            return self.weights[start];
        }
        
        // Binary search since neighbors are sorted
        match self.neighbors[start..end].binary_search(&node) {
            Ok(pos) => self.weights[start + pos],
            Err(_) => E::zero(),
        }
    }

    pub fn community_internal_weight(&self, community: usize, grouping: &impl NetworkGrouping) -> E {
        let members = &grouping.get_group_members()[community];
        let mut total_weight = E::zero();
        
        // Use parallel processing for large communities
        if members.len() > 100 {
            total_weight = members.par_iter()
                .map(|&node| {
                    let mut internal_weight = E::zero();
                    for (neighbor, weight) in self.neighbors(node) {
                        if grouping.get_group(neighbor) == community {
                            if node == neighbor {
                                internal_weight += weight; // Self-loop: full weight
                            } else if node < neighbor {
                                internal_weight += weight; // Edge: count once
                            }
                        }
                    }
                    internal_weight
                })
                .sum();
        } else {
            for &node in members {
                for (neighbor, weight) in self.neighbors(node) {
                    if grouping.get_group(neighbor) == community {
                        if node == neighbor {
                            total_weight += weight;
                        } else if node < neighbor {
                            total_weight += weight;
                        }
                    }
                }
            }
        }
        
        total_weight
    }

    pub fn community_total_strength(&self, community: usize, grouping: &impl NetworkGrouping) -> E {
        let members = &grouping.get_group_members()[community];
        
        if members.len() > 50 {
            // Parallel for large communities
            members.par_iter()
                .map(|&node| self.strength(node))
                .sum()
        } else {
            members.iter()
                .map(|&node| self.strength(node))
                .fold(E::zero(), |acc, x| acc + x)
        }
    }

    
}

pub struct CSRNeighborIterator<E> {
    neighbor_ptr: *const usize,
    weight_ptr: *const E,
    remaining: usize,
}

impl<E: Copy> Iterator for CSRNeighborIterator<E> {
    type Item = (usize, E);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        unsafe {
            let neighbor = *self.neighbor_ptr;
            let weight = *self.weight_ptr;

            self.neighbor_ptr = self.neighbor_ptr.add(1);
            self.weight_ptr = self.weight_ptr.add(1);
            self.remaining -= 1;
            Some((neighbor, weight))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<E> ExactSizeIterator for CSRNeighborIterator<E> where E: Copy {}

unsafe impl<E: Send> Send for CSRNeighborIterator<E> {}
unsafe impl<E: Sync> Sync for CSRNeighborIterator<E> {}

impl<N, E> std::fmt::Display for CSRNetwork<N, E>
where
    N: FloatOpsTS + std::fmt::Display + 'static,
    E: FloatOpsTS + std::fmt::Display + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CSRNetwork({} nodes, {} edges, total_weight: {}, density: {:.4})",
            self.node_count(),
            self.edge_count(),
            self.total_weight(),
            self.density()
        )
    }
}
