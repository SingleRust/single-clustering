//! CSR storage for a weighted, undirected graph.
//!
//! igraph/leidenalg weight conventions, which every quality function here depends on:
//!
//! - edge `{u, v}` of weight `w` → `w` to each endpoint's strength, `w` to `total_weight`
//! - **self-loop** on `v` of weight `w` → **`2w`** to `strength(v)`, `w` to `total_weight`
//!
//! So `Σ strength(v) == 2 · total_weight`. [`aggregate`](CSRNetwork::aggregate) packs a
//! community's internal weight into one self-loop, so getting that wrong kills the null model
//! above level 0.
//!
//! `u32` ids and `f32` weights: 16 bytes per undirected edge, ~12 GB of adjacency at 75M
//! nodes rather than ~24. Per-node data and all arithmetic stay `f64` — node weights count
//! original nodes after aggregation and would lose integer precision past 2^24.

use std::sync::Arc;

use nalgebra_sparse::CsrMatrix;
use single_utilities::traits::FloatOpsTS;

use crate::error::{ClusteringError, Result};
use crate::network::grouping::VectorGrouping;

/// Largest addressable node id. `u32` adjacency caps the graph at ~4.29 billion nodes.
pub const MAX_NODES: usize = u32::MAX as usize;

#[derive(Debug, Clone)]
struct CSRNetworkData {
    /// Offsets into `neighbors`/`weights`, length `n_nodes + 1`. `usize`, since it indexes
    /// up to `2m` entries.
    node_ptrs: Vec<usize>,
    /// Neighbour ids, strictly increasing within each node's slice.
    neighbors: Vec<u32>,
    /// Edge weights, parallel to `neighbors`.
    weights: Vec<f32>,
    /// Per-node weights (aggregate node size). Summed by [`CSRNetwork::aggregate`].
    node_weights: Vec<f64>,
    /// Per-node sum of incident edge weights, self-loops counted twice.
    strengths: Vec<f64>,
    /// Sum of edge weights, each undirected edge counted once.
    total_weight: f64,
    /// Number of distinct undirected edges, self-loops included.
    edge_count: usize,
    /// Lets `self_loop_weight` skip its binary search, which the level-0 graph normally can.
    /// That search was ~4% of runtime.
    any_self_loops: bool,
}

/// A weighted, undirected graph in compressed-sparse-row form.
///
/// Cloning is cheap — the data is shared behind an [`Arc`].
#[derive(Debug, Clone)]
pub struct CSRNetwork {
    data: Arc<CSRNetworkData>,
}

/// Iterator over one node's `(neighbor, weight)` pairs.
///
/// Yields `usize`/`f64` whatever the storage types, so callers can index and do arithmetic
/// directly.
pub struct CSRNeighborIterator<'a> {
    neighbors: std::slice::Iter<'a, u32>,
    weights: std::slice::Iter<'a, f32>,
}

impl Iterator for CSRNeighborIterator<'_> {
    type Item = (usize, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let n = *self.neighbors.next()?;
        let w = *self.weights.next()?;
        Some((n as usize, w as f64))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.neighbors.size_hint()
    }
}

impl ExactSizeIterator for CSRNeighborIterator<'_> {}

impl CSRNetwork {
    /// Builds a graph from an edge list, every node at unit weight.
    ///
    /// Give each undirected edge **once** — `(u, v, w)` and `(v, u, w)` are two parallel
    /// edges, not one. Parallel edges are merged by summing. `(v, v, w)` is a self-loop.
    pub fn from_edges<W>(n_nodes: usize, edges: &[(usize, usize, W)]) -> Result<Self>
    where
        W: FloatOpsTS + 'static,
    {
        Self::from_edges_with_node_weights(edges, vec![1.0; n_nodes])
    }

    /// Builds a graph from an edge list with explicit node weights.
    ///
    /// Node weight is a node's "size" and what CPM measures. Usually all `1.0`; it starts
    /// mattering after [`aggregate`](Self::aggregate), where a super-node's weight is the sum
    /// of its members'.
    ///
    /// Allocates once and sorts in place, so peak memory is the graph plus the caller's edge
    /// list, not double. [`from_csr_parts`](Self::from_csr_parts) needs no edge list at all.
    pub fn from_edges_with_node_weights<W>(
        edges: &[(usize, usize, W)],
        node_weights: Vec<f64>,
    ) -> Result<Self>
    where
        W: FloatOpsTS + 'static,
    {
        let n_nodes = node_weights.len();
        validate_node_weights(&node_weights)?;
        if n_nodes > MAX_NODES {
            return Err(ClusteringError::TooManyNodes { n_nodes });
        }

        // pass 1: validate and count slots per node
        let mut counts = vec![0usize; n_nodes];
        for &(from, to, w) in edges {
            check_edge(from, to, w.to_f64().unwrap_or(f64::NAN), n_nodes)?;
            counts[from] += 1;
            if from != to {
                counts[to] += 1;
            }
        }

        let mut node_ptrs = vec![0usize; n_nodes + 1];
        for v in 0..n_nodes {
            node_ptrs[v + 1] = node_ptrs[v] + counts[v];
        }
        let slots = node_ptrs[n_nodes];

        // pass 2: scatter into place; `cursor` is the write position in each slice
        let mut neighbors = vec![0u32; slots];
        let mut weights = vec![0.0f32; slots];
        let mut cursor = node_ptrs.clone();
        for &(from, to, w) in edges {
            let w = w.to_f64().unwrap() as f32;
            neighbors[cursor[from]] = to as u32;
            weights[cursor[from]] = w;
            cursor[from] += 1;
            if from != to {
                neighbors[cursor[to]] = from as u32;
                weights[cursor[to]] = w;
                cursor[to] += 1;
            }
        }
        drop(cursor);

        Self::finish(node_ptrs, neighbors, weights, node_weights, true)
    }

    /// Builds a graph directly from CSR arrays, taking ownership.
    ///
    /// The low-memory entry point: nothing proportional to the edge count is allocated, so a
    /// connectivity matrix you already hold — scanpy's `connectivities`, say — costs no copy.
    ///
    /// Must be the **full symmetric** adjacency: if `j` is in row `i`, `i` must be in row `j`
    /// at the same weight. Rows are sorted in place, duplicates summed, zero sums dropped.
    pub fn from_csr_parts(
        node_ptrs: Vec<usize>,
        neighbors: Vec<u32>,
        weights: Vec<f32>,
        node_weights: Option<Vec<f64>>,
    ) -> Result<Self> {
        if node_ptrs.is_empty() {
            return Err(ClusteringError::InvalidCsr(
                "node_ptrs must have at least one element".into(),
            ));
        }
        let n_nodes = node_ptrs.len() - 1;
        if n_nodes > MAX_NODES {
            return Err(ClusteringError::TooManyNodes { n_nodes });
        }
        if neighbors.len() != weights.len() {
            return Err(ClusteringError::InvalidCsr(format!(
                "neighbors has {} entries but weights has {}",
                neighbors.len(),
                weights.len()
            )));
        }
        if node_ptrs[n_nodes] != neighbors.len() {
            return Err(ClusteringError::InvalidCsr(format!(
                "node_ptrs ends at {} but there are {} entries",
                node_ptrs[n_nodes],
                neighbors.len()
            )));
        }
        for v in 0..n_nodes {
            if node_ptrs[v] > node_ptrs[v + 1] {
                return Err(ClusteringError::InvalidCsr(format!(
                    "node_ptrs is not non-decreasing at {v}"
                )));
            }
        }
        for (&u, &w) in neighbors.iter().zip(weights.iter()) {
            if u as usize >= n_nodes {
                return Err(ClusteringError::NodeIndexOutOfRange {
                    node: u as usize,
                    n_nodes,
                });
            }
            if !w.is_finite() {
                return Err(ClusteringError::NonFiniteWeight {
                    edge: (0, u as usize),
                });
            }
            if w < 0.0 {
                return Err(ClusteringError::NegativeWeight {
                    edge: (0, u as usize),
                    weight: w as f64,
                });
            }
        }

        let node_weights = match node_weights {
            Some(w) => {
                if w.len() != n_nodes {
                    return Err(ClusteringError::NodeWeightLengthMismatch {
                        got: w.len(),
                        expected: n_nodes,
                    });
                }
                validate_node_weights(&w)?;
                w
            }
            None => vec![1.0; n_nodes],
        };

        Self::finish(node_ptrs, neighbors, weights, node_weights, false)
    }

    /// Sorts each row, merges duplicates, compacts, derives the cached aggregates.
    ///
    /// One scratch buffer sized to the largest row, so the temporary is O(max degree).
    fn finish(
        mut node_ptrs: Vec<usize>,
        mut neighbors: Vec<u32>,
        mut weights: Vec<f32>,
        node_weights: Vec<f64>,
        may_have_gaps: bool,
    ) -> Result<Self> {
        let n_nodes = node_ptrs.len() - 1;
        let max_degree = (0..n_nodes)
            .map(|v| node_ptrs[v + 1] - node_ptrs[v])
            .max()
            .unwrap_or(0);
        let mut scratch: Vec<(u32, f32)> = Vec::with_capacity(max_degree);

        let mut strengths = vec![0.0f64; n_nodes];
        let mut total_weight = 0.0f64;
        let mut edge_count = 0usize;
        let mut write = 0usize;
        let mut any_self_loops = false;

        for v in 0..n_nodes {
            let (lo, hi) = (node_ptrs[v], node_ptrs[v + 1]);
            node_ptrs[v] = write;

            scratch.clear();
            scratch.extend(
                neighbors[lo..hi]
                    .iter()
                    .copied()
                    .zip(weights[lo..hi].iter().copied()),
            );
            scratch.sort_unstable_by_key(|&(u, _)| u);

            let mut i = 0;
            while i < scratch.len() {
                let u = scratch[i].0;
                let mut w = 0.0f64;
                while i < scratch.len() && scratch[i].0 == u {
                    w += scratch[i].1 as f64;
                    i += 1;
                }
                // an explicit zero isn't an edge — keeps degree/edge_count honest
                if w == 0.0 {
                    continue;
                }
                // `write <= lo` always, so this never clobbers unread input
                debug_assert!(write <= lo + (i - 1));
                neighbors[write] = u;
                weights[write] = w as f32;
                write += 1;

                // self-loops: twice toward strength, once toward total weight
                let stored = w as f32 as f64;
                if u as usize == v {
                    any_self_loops = true;
                    strengths[v] += 2.0 * stored;
                } else {
                    strengths[v] += stored;
                }
                if v <= u as usize {
                    total_weight += stored;
                    edge_count += 1;
                }
            }
        }
        node_ptrs[n_nodes] = write;

        if may_have_gaps || write < neighbors.len() {
            neighbors.truncate(write);
            weights.truncate(write);
            neighbors.shrink_to_fit();
            weights.shrink_to_fit();
        }

        let data = CSRNetworkData {
            node_ptrs,
            neighbors,
            weights,
            node_weights,
            strengths,
            total_weight,
            edge_count,
            any_self_loops,
        };
        check_degree_sum(&data)?;

        Ok(Self {
            data: Arc::new(data),
        })
    }

    /// Builds a graph from a sparse adjacency matrix, every node at unit weight.
    ///
    /// Treated as symmetric: only `row <= col` is read, and a diagonal entry is a self-loop.
    /// Allocates an intermediate edge list — prefer
    /// [`from_csr_parts`](Self::from_csr_parts) for large inputs.
    pub fn from_csr_matrix<W>(matrix: &CsrMatrix<W>) -> Result<Self>
    where
        W: FloatOpsTS + 'static,
    {
        Self::from_csr_matrix_with_node_weights(matrix, vec![1.0; matrix.nrows()])
    }

    /// Builds a graph from a sparse adjacency matrix with explicit node weights.
    pub fn from_csr_matrix_with_node_weights<W>(
        matrix: &CsrMatrix<W>,
        node_weights: Vec<f64>,
    ) -> Result<Self>
    where
        W: FloatOpsTS + 'static,
    {
        if node_weights.len() != matrix.nrows() {
            return Err(ClusteringError::NodeWeightLengthMismatch {
                got: node_weights.len(),
                expected: matrix.nrows(),
            });
        }

        let mut edges = Vec::with_capacity(matrix.nnz() / 2 + 1);
        for (row, col, &weight) in matrix.triplet_iter() {
            // upper triangle only; the diagonal picks up the self-loop convention later
            if row <= col && weight != W::zero() {
                edges.push((row, col, weight));
            }
        }

        Self::from_edges_with_node_weights(&edges, node_weights)
    }

    /// A node's `(neighbor, weight)` pairs, in ascending id order — which is what makes
    /// candidate enumeration deterministic.
    #[inline]
    pub fn neighbors(&self, node: usize) -> CSRNeighborIterator<'_> {
        let start = self.data.node_ptrs[node];
        let end = self.data.node_ptrs[node + 1];
        CSRNeighborIterator {
            neighbors: self.data.neighbors[start..end].iter(),
            weights: self.data.weights[start..end].iter(),
        }
    }

    /// Number of nodes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.data.node_weights.len()
    }

    /// Distinct undirected edges, self-loops included.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.data.edge_count
    }

    /// Number of distinct neighbours.
    #[inline]
    pub fn degree(&self, node: usize) -> usize {
        self.data.node_ptrs[node + 1] - self.data.node_ptrs[node]
    }

    /// Sum of incident edge weights, self-loops counted twice.
    #[inline]
    pub fn strength(&self, node: usize) -> f64 {
        self.data.strengths[node]
    }

    /// A node's weight, i.e. its "size".
    #[inline]
    pub fn node_weight(&self, node: usize) -> f64 {
        self.data.node_weights[node]
    }

    /// Total edge weight, each undirected edge once.
    #[inline]
    pub fn total_weight(&self) -> f64 {
        self.data.total_weight
    }

    /// Sum of all node weights.
    #[inline]
    pub fn total_node_weight(&self) -> f64 {
        self.data.node_weights.iter().sum()
    }

    /// A node's self-loop weight, or `0.0`.
    #[inline]
    pub fn self_loop_weight(&self, node: usize) -> f64 {
        if !self.data.any_self_loops {
            return 0.0;
        }
        let start = self.data.node_ptrs[node];
        let end = self.data.node_ptrs[node + 1];
        match self.data.neighbors[start..end].binary_search(&(node as u32)) {
            Ok(pos) => self.data.weights[start + pos] as f64,
            Err(_) => 0.0,
        }
    }

    /// Weight of the edge between two nodes, if there is one.
    pub fn edge_weight(&self, from: usize, to: usize) -> Option<f64> {
        // search the shorter row
        let (search_node, target) = if self.degree(from) <= self.degree(to) {
            (from, to)
        } else {
            (to, from)
        };
        let start = self.data.node_ptrs[search_node];
        let end = self.data.node_ptrs[search_node + 1];
        match self.data.neighbors[start..end].binary_search(&(target as u32)) {
            Ok(pos) => Some(self.data.weights[start + pos] as f64),
            Err(_) => None,
        }
    }

    /// Copies of the CSR arrays, as [`from_csr_parts`](Self::from_csr_parts) takes them.
    /// Round-tripping is lossless.
    pub fn to_csr_parts(&self) -> (Vec<usize>, Vec<u32>, Vec<f32>) {
        (
            self.data.node_ptrs.clone(),
            self.data.neighbors.clone(),
            self.data.weights.clone(),
        )
    }

    /// Per-node weights, as [`from_csr_parts`](Self::from_csr_parts) takes them.
    pub fn node_weights(&self) -> &[f64] {
        &self.data.node_weights
    }

    /// Rough resident size in bytes — ~14 GB at 75M nodes and 750M edges.
    pub fn memory_bytes(&self) -> usize {
        let d = &self.data;
        d.node_ptrs.len() * size_of::<usize>()
            + d.neighbors.len() * size_of::<u32>()
            + d.weights.len() * size_of::<f32>()
            + d.node_weights.len() * size_of::<f64>()
            + d.strengths.len() * size_of::<f64>()
    }

    /// Collapses each group into a super-node.
    ///
    /// Node weights sum, a group's internal weight becomes a self-loop, weight between groups
    /// becomes an edge. `total_weight` and `Σ strength` are preserved exactly, so quality
    /// doesn't change across levels.
    pub fn aggregate(&self, grouping: &VectorGrouping) -> Self {
        let n = self.node_count();
        let n_groups = grouping.group_count();

        let mut new_node_weights = vec![0.0f64; n_groups];
        for v in 0..n {
            new_node_weights[grouping.get_group(v)] += self.data.node_weights[v];
        }

        // counting sort into groups, so we can walk one group at a time
        let mut starts = vec![0usize; n_groups + 1];
        for v in 0..n {
            starts[grouping.get_group(v) + 1] += 1;
        }
        for g in 0..n_groups {
            starts[g + 1] += starts[g];
        }
        let mut members = vec![0u32; n];
        let mut cursor = starts.clone();
        for v in 0..n {
            let g = grouping.get_group(v);
            members[cursor[g]] = v as u32;
            cursor[g] += 1;
        }
        drop(cursor);

        // per group, accumulate to groups with id >= its own so each pair is seen once
        let mut acc = vec![0.0f64; n_groups];
        let mut touched: Vec<usize> = Vec::new();
        let mut counts = vec![0usize; n_groups];
        let mut pairs: Vec<(u32, u32, f32)> = Vec::new();

        for g in 0..n_groups {
            let mut self_loop_weight = 0.0f64;
            for &v in &members[starts[g]..starts[g + 1]] {
                for (u, w) in self.neighbors(v as usize) {
                    if u == v as usize {
                        self_loop_weight += w;
                    }
                    let h = grouping.get_group(u);
                    if h < g {
                        continue;
                    }
                    if acc[h] == 0.0 {
                        touched.push(h);
                    }
                    acc[h] += w;
                }
            }

            for &h in &touched {
                let w = acc[h];
                acc[h] = 0.0;
                if w <= 0.0 {
                    continue;
                }
                if h == g {
                    // regular edges seen from both ends, loops once: acc = 2·regular + loops
                    pairs.push((g as u32, g as u32, (0.5 * (w + self_loop_weight)) as f32));
                    counts[g] += 1;
                } else {
                    pairs.push((g as u32, h as u32, w as f32));
                    counts[g] += 1;
                    counts[h] += 1;
                }
            }
            touched.clear();
        }

        let mut node_ptrs = vec![0usize; n_groups + 1];
        for g in 0..n_groups {
            node_ptrs[g + 1] = node_ptrs[g] + counts[g];
        }
        let slots = node_ptrs[n_groups];
        let mut neighbors = vec![0u32; slots];
        let mut weights = vec![0.0f32; slots];
        let mut cursor = node_ptrs.clone();
        for &(a, b, w) in &pairs {
            neighbors[cursor[a as usize]] = b;
            weights[cursor[a as usize]] = w;
            cursor[a as usize] += 1;
            if a != b {
                neighbors[cursor[b as usize]] = a;
                weights[cursor[b as usize]] = w;
                cursor[b as usize] += 1;
            }
        }
        drop(cursor);

        // can't fail: group ids, already-validated weights
        Self::finish(node_ptrs, neighbors, weights, new_node_weights, false)
            .expect("aggregate produces a structurally valid graph")
    }

    /// Back to a symmetric sparse matrix.
    pub fn to_csr_matrix(&self) -> CsrMatrix<f64> {
        let n = self.node_count();
        let row_ptrs = self.data.node_ptrs.clone();
        let col_indices: Vec<usize> = self.data.neighbors.iter().map(|&u| u as usize).collect();
        let values: Vec<f64> = self.data.weights.iter().map(|&w| w as f64).collect();
        CsrMatrix::try_from_csr_data(n, n, row_ptrs, col_indices, values)
            .expect("CSR invariants are maintained by construction")
    }

    /// Checks every stored `(v, u, w)` has its `(u, v, w)`.
    ///
    /// Constructors already enforce `Σ strength == 2 · total_weight`, which catches the
    /// realistic mistakes. This is the exhaustive version, O(m log d) — worth running once
    /// when wiring up a new input source, not every call.
    pub fn validate_symmetry(&self) -> Result<()> {
        for v in 0..self.node_count() {
            for (u, w) in self.neighbors(v) {
                match self.edge_weight(u, v) {
                    Some(back) if (back - w).abs() <= 1e-6 * w.abs().max(1.0) => {}
                    Some(back) => {
                        return Err(ClusteringError::InvalidCsr(format!(
                            "edge ({v}, {u}) has weight {w} but ({u}, {v}) has {back}"
                        )));
                    }
                    None => {
                        return Err(ClusteringError::InvalidCsr(format!(
                            "edge ({v}, {u}) is present but ({u}, {v}) is missing"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether any node carries a self-loop.
    #[inline]
    pub fn has_self_loops(&self) -> bool {
        self.data.any_self_loops
    }

    /// Fraction of possible edges present.
    pub fn density(&self) -> f64 {
        let n = self.node_count() as f64;
        let max_edges = n * (n - 1.0) / 2.0;
        if max_edges > 0.0 {
            self.data.edge_count as f64 / max_edges
        } else {
            0.0
        }
    }
}

fn validate_node_weights(node_weights: &[f64]) -> Result<()> {
    for (node, &nw) in node_weights.iter().enumerate() {
        if !nw.is_finite() || nw < 0.0 {
            return Err(ClusteringError::InvalidNodeWeight { node, weight: nw });
        }
    }
    Ok(())
}

#[inline]
fn check_edge(from: usize, to: usize, w: f64, n_nodes: usize) -> Result<()> {
    if from >= n_nodes {
        return Err(ClusteringError::NodeIndexOutOfRange {
            node: from,
            n_nodes,
        });
    }
    if to >= n_nodes {
        return Err(ClusteringError::NodeIndexOutOfRange { node: to, n_nodes });
    }
    if !w.is_finite() {
        return Err(ClusteringError::NonFiniteWeight { edge: (from, to) });
    }
    if w < 0.0 {
        return Err(ClusteringError::NegativeWeight {
            edge: (from, to),
            weight: w,
        });
    }
    Ok(())
}

/// Checks `Σ strength == 2 · total_weight`.
///
/// Runs in release too: one O(n) pass against minutes of clustering, and it's the only thing
/// between an asymmetric input to [`CSRNetwork::from_csr_parts`] and quietly wrong clusters.
/// Necessary but not sufficient — [`CSRNetwork::validate_symmetry`] is exhaustive.
///
/// The `1e-6` relative tolerance is loose because both sides are naive f64 sums in different
/// orders. Real asymmetry is off by whole edge weights.
fn check_degree_sum(data: &CSRNetworkData) -> Result<()> {
    let sum: f64 = data.strengths.iter().sum();
    let expected = 2.0 * data.total_weight;
    let tol = 1e-6 * expected.abs().max(1.0);
    if (sum - expected).abs() > tol {
        return Err(ClusteringError::AsymmetricGraph {
            degree_sum: sum,
            expected,
        });
    }
    Ok(())
}

impl std::fmt::Display for CSRNetwork {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::grouping::VectorGrouping;

    fn triangle_with_loop() -> CSRNetwork {
        // triangle 0-1-2 (unit weights) plus a self-loop of weight 3 on node 0
        CSRNetwork::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (0, 0, 3.0)]).unwrap()
    }

    #[test]
    fn self_loop_counts_twice_toward_strength_once_toward_total() {
        let g = triangle_with_loop();
        assert_eq!(g.strength(0), 8.0, "2*3 self-loop + 1 + 1");
        assert_eq!(g.strength(1), 2.0);
        assert_eq!(g.strength(2), 2.0);
        assert_eq!(g.total_weight(), 6.0, "1 + 1 + 1 + 3");
        assert_eq!(g.self_loop_weight(0), 3.0);
        assert_eq!(g.self_loop_weight(1), 0.0);
    }

    #[test]
    fn degree_sum_invariant_holds() {
        for g in [
            triangle_with_loop(),
            CSRNetwork::from_edges(4, &[(0, 1, 2.5), (2, 3, 0.5)]).unwrap(),
            CSRNetwork::from_edges(3, &[] as &[(usize, usize, f64)]).unwrap(),
        ] {
            let sum: f64 = (0..g.node_count()).map(|v| g.strength(v)).sum();
            assert!((sum - 2.0 * g.total_weight()).abs() < 1e-12);
        }
    }

    #[test]
    fn parallel_edges_are_merged() {
        let g = CSRNetwork::from_edges(2, &[(0, 1, 1.0), (0, 1, 2.0)]).unwrap();
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.edge_weight(0, 1), Some(3.0));
        assert_eq!(g.total_weight(), 3.0);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn aggregate_preserves_total_weight_and_degree_sum() {
        // three 6-cliques joined by two bridges
        let mut edges = Vec::new();
        for b in 0..3usize {
            for i in 0..6 {
                for j in (i + 1)..6 {
                    edges.push((b * 6 + i, b * 6 + j, 1.0));
                }
            }
        }
        edges.push((0, 6, 1.0));
        edges.push((6, 12, 1.0));

        let g = CSRNetwork::from_edges(18, &edges).unwrap();
        let memb: Vec<usize> = (0..18).map(|i| i / 6).collect();
        let agg = g.aggregate(&VectorGrouping::from_assignments(&memb));

        let fine: f64 = (0..18).map(|v| g.strength(v)).sum();
        let coarse: f64 = (0..agg.node_count()).map(|v| agg.strength(v)).sum();
        assert_eq!(agg.node_count(), 3);
        assert!(
            (fine - coarse).abs() < 1e-12,
            "Σ strength {fine} vs {coarse}"
        );
        assert!((g.total_weight() - agg.total_weight()).abs() < 1e-12);

        // each super-node's strength equals the sum of its members' strengths
        for c in 0..3 {
            let members: f64 = (0..18)
                .filter(|&v| memb[v] == c)
                .map(|v| g.strength(v))
                .sum();
            assert!((members - agg.strength(c)).abs() < 1e-12, "community {c}");
        }
        // and node weights are summed
        assert_eq!(agg.node_weight(0), 6.0);
    }

    #[test]
    fn aggregate_is_idempotent_on_singletons() {
        let g = triangle_with_loop();
        let identity = VectorGrouping::from_assignments(&[0, 1, 2]);
        let agg = g.aggregate(&identity);
        for v in 0..3 {
            assert!((g.strength(v) - agg.strength(v)).abs() < 1e-12);
        }
        assert!((g.total_weight() - agg.total_weight()).abs() < 1e-12);
        assert_eq!(agg.self_loop_weight(0), 3.0);
    }

    #[test]
    fn rejects_invalid_input() {
        use crate::error::ClusteringError;
        assert_eq!(
            CSRNetwork::from_edges(2, &[(0, 5, 1.0)]).unwrap_err(),
            ClusteringError::NodeIndexOutOfRange {
                node: 5,
                n_nodes: 2
            }
        );
        assert_eq!(
            CSRNetwork::from_edges(2, &[(0, 1, -1.0)]).unwrap_err(),
            ClusteringError::NegativeWeight {
                edge: (0, 1),
                weight: -1.0
            }
        );
        assert!(matches!(
            CSRNetwork::from_edges(2, &[(0, 1, f64::NAN)]).unwrap_err(),
            ClusteringError::NonFiniteWeight { .. }
        ));
    }

    /// Zero-copy and edge-list paths must give exactly the same graph.
    #[test]
    fn from_csr_parts_matches_from_edges() {
        let edges = [
            (0usize, 1usize, 1.5f64),
            (1, 2, 2.0),
            (0, 2, 0.5),
            (0, 0, 3.0),
            (3, 3, 1.0),
        ];
        let via_edges = CSRNetwork::from_edges(4, &edges).unwrap();

        // the same graph, expressed as full symmetric CSR
        let node_ptrs = vec![0usize, 3, 5, 7, 8];
        let neighbors = vec![0u32, 1, 2, 0, 2, 0, 1, 3];
        let weights = vec![3.0f32, 1.5, 0.5, 1.5, 2.0, 0.5, 2.0, 1.0];
        let via_csr = CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None).unwrap();

        assert_eq!(via_edges.node_count(), via_csr.node_count());
        assert_eq!(via_edges.edge_count(), via_csr.edge_count());
        assert!((via_edges.total_weight() - via_csr.total_weight()).abs() < 1e-9);
        for v in 0..4 {
            assert!(
                (via_edges.strength(v) - via_csr.strength(v)).abs() < 1e-9,
                "strength({v})"
            );
            let a: Vec<_> = via_edges.neighbors(v).collect();
            let b: Vec<_> = via_csr.neighbors(v).collect();
            assert_eq!(a, b, "neighbors({v})");
        }
    }

    #[test]
    fn from_csr_parts_sorts_and_merges_rows() {
        // row 0 lists its neighbours out of order and twice
        let node_ptrs = vec![0usize, 3, 4];
        let neighbors = vec![1u32, 1, 0, 0];
        let weights = vec![1.0f32, 2.0, 0.0, 3.0];
        let g = CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None).unwrap();
        assert_eq!(g.edge_weight(0, 1), Some(3.0), "duplicates summed");
        assert_eq!(g.degree(0), 1, "the zero-weight self-loop is dropped");
    }

    #[test]
    fn from_csr_parts_rejects_malformed_input() {
        use crate::error::ClusteringError;
        // node_ptrs not matching the entry count
        assert!(matches!(
            CSRNetwork::from_csr_parts(vec![0, 5], vec![0u32], vec![1.0f32], None).unwrap_err(),
            ClusteringError::InvalidCsr(_)
        ));
        // mismatched parallel arrays
        assert!(matches!(
            CSRNetwork::from_csr_parts(vec![0, 1], vec![0u32], vec![1.0f32, 2.0], None)
                .unwrap_err(),
            ClusteringError::InvalidCsr(_)
        ));
        // out-of-range column index
        assert!(matches!(
            CSRNetwork::from_csr_parts(vec![0, 1], vec![9u32], vec![1.0f32], None).unwrap_err(),
            ClusteringError::NodeIndexOutOfRange { .. }
        ));
        // non-decreasing violation
        assert!(matches!(
            CSRNetwork::from_csr_parts(vec![0, 2, 1], vec![0u32, 1], vec![1.0f32, 1.0], None)
                .unwrap_err(),
            ClusteringError::InvalidCsr(_)
        ));
    }

    /// The realistic mistake: a kNN graph nobody symmetrised.
    #[test]
    fn asymmetric_input_is_rejected() {
        use crate::error::ClusteringError;
        let node_ptrs = vec![0usize, 1, 1];
        let neighbors = vec![1u32];
        let weights = vec![1.0f32];
        let err = CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None).unwrap_err();
        assert!(
            matches!(err, ClusteringError::AsymmetricGraph { .. }),
            "expected AsymmetricGraph, got {err}"
        );
        assert!(err.to_string().contains("symmetrised"));
    }

    #[test]
    fn asymmetric_weights_are_rejected() {
        use crate::error::ClusteringError;
        // both directions present, but with different weights
        let node_ptrs = vec![0usize, 1, 2];
        let neighbors = vec![1u32, 0];
        let weights = vec![1.0f32, 5.0];
        let err = CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None).unwrap_err();
        assert!(
            matches!(err, ClusteringError::AsymmetricGraph { .. }),
            "{err}"
        );
    }

    #[test]
    fn validate_symmetry_accepts_well_formed_graphs() {
        for g in [
            triangle_with_loop(),
            CSRNetwork::from_edges(6, &[(0, 1, 2.5), (2, 3, 0.5), (4, 4, 1.0)]).unwrap(),
            CSRNetwork::from_edges(3, &[] as &[(usize, usize, f64)]).unwrap(),
        ] {
            g.validate_symmetry().unwrap();
        }
    }

    /// Asymmetry that cancels in the totals, so only `validate_symmetry` sees it.
    #[test]
    fn validate_symmetry_catches_what_the_degree_sum_cannot() {
        // strengths and total both balance, but neither edge agrees with its reverse
        let node_ptrs = vec![0usize, 2, 3, 4];
        let neighbors = vec![1u32, 2, 0, 0];
        let weights = vec![1.0f32, 3.0, 3.0, 1.0];
        let g = CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None)
            .expect("degree sum balances, so construction succeeds");
        assert!(
            g.validate_symmetry().is_err(),
            "validate_symmetry should catch mismatched reverse weights"
        );
    }

    #[test]
    fn memory_is_eight_bytes_per_stored_entry() {
        let mut edges = Vec::new();
        for i in 0..1000usize {
            for d in 1..=5 {
                if i + d < 1000 {
                    edges.push((i, i + d, 1.0));
                }
            }
        }
        let g = CSRNetwork::from_edges(1000, &edges).unwrap();
        let entries = 2 * g.edge_count(); // both directions stored
        let node_side = 1000 * (8 + 8) + 1001 * 8;
        assert_eq!(g.memory_bytes(), entries * 8 + node_side);
    }
}
