//! Error types for the clustering crate.

use std::fmt;

/// Errors produced when building networks or running clustering algorithms.
#[derive(Debug, Clone, PartialEq)]
pub enum ClusteringError {
    /// An edge referenced a node index that does not exist in the graph.
    NodeIndexOutOfRange {
        /// The offending node index.
        node: usize,
        /// Number of nodes in the graph.
        n_nodes: usize,
    },
    /// An edge weight was NaN or infinite.
    NonFiniteWeight {
        /// Endpoints of the offending edge.
        edge: (usize, usize),
    },
    /// An edge weight was negative. The quality functions assume non-negative weights.
    NegativeWeight {
        /// Endpoints of the offending edge.
        edge: (usize, usize),
        /// The offending weight.
        weight: f64,
    },
    /// A node weight was NaN, infinite, or negative.
    InvalidNodeWeight {
        /// The offending node index.
        node: usize,
        /// The offending weight.
        weight: f64,
    },
    /// The supplied node-weight vector did not match the node count.
    NodeWeightLengthMismatch {
        /// Length of the supplied vector.
        got: usize,
        /// Expected length.
        expected: usize,
    },
    /// A membership vector did not match the node count.
    MembershipLengthMismatch {
        /// Length of the supplied vector.
        got: usize,
        /// Expected length.
        expected: usize,
    },
    /// A configuration value was outside its valid range.
    InvalidConfig(String),
    /// The graph has more nodes than the `u32` adjacency representation can address.
    TooManyNodes {
        /// The requested node count.
        n_nodes: usize,
    },
    /// The supplied CSR arrays were not a well-formed sparse matrix.
    InvalidCsr(String),
    /// Two observations claimed the same lattice position.
    DuplicateLatticePosition {
        /// Lattice row.
        row: u32,
        /// Lattice column.
        col: u32,
        /// The two node indices sharing it.
        nodes: (usize, usize),
    },
    /// Visium hex positions did not all share the same `(row + col)` parity.
    ///
    /// In doubled coordinates every spot has the same parity, so mixed parity means the
    /// coordinates are not doubled — most often offset coordinates (`col / 2`), which would
    /// silently produce a wrong graph rather than an empty one.
    InconsistentHexParity {
        /// A node with the majority parity.
        expected_node: usize,
        /// The first node found disagreeing with it.
        node: usize,
        /// The offending position.
        position: (u32, u32),
    },
    /// Parallel coordinate arrays had different lengths.
    CoordinateLengthMismatch {
        /// Length of the first array.
        got: usize,
        /// Expected length, from the first array supplied.
        expected: usize,
    },
    /// The adjacency is not symmetric, so `Σ strength != 2 · total_weight`.
    ///
    /// Most often means an un-symmetrised k-NN graph was passed to
    /// `CSRNetwork::from_csr_parts`: if `j` is among `i`'s nearest neighbours but `i` is not
    /// among `j`'s, the matrix has an entry in one direction only.
    AsymmetricGraph {
        /// The observed sum of node strengths.
        degree_sum: f64,
        /// Twice the total edge weight, which it should equal.
        expected: f64,
    },
}

impl fmt::Display for ClusteringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeIndexOutOfRange { node, n_nodes } => write!(
                f,
                "node index {node} is out of range for a graph with {n_nodes} nodes"
            ),
            Self::NonFiniteWeight { edge } => {
                write!(f, "edge ({}, {}) has a non-finite weight", edge.0, edge.1)
            }
            Self::NegativeWeight { edge, weight } => write!(
                f,
                "edge ({}, {}) has negative weight {weight}; weights must be non-negative",
                edge.0, edge.1
            ),
            Self::InvalidNodeWeight { node, weight } => write!(
                f,
                "node {node} has invalid weight {weight}; must be finite and non-negative"
            ),
            Self::NodeWeightLengthMismatch { got, expected } => write!(
                f,
                "node weight vector has length {got}, expected {expected}"
            ),
            Self::MembershipLengthMismatch { got, expected } => {
                write!(f, "membership vector has length {got}, expected {expected}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            Self::TooManyNodes { n_nodes } => write!(
                f,
                "graph has {n_nodes} nodes, exceeding the {} addressable by u32 adjacency",
                crate::network::MAX_NODES
            ),
            Self::InvalidCsr(msg) => write!(f, "malformed CSR input: {msg}"),
            Self::DuplicateLatticePosition { row, col, nodes } => write!(
                f,
                "nodes {} and {} both sit at lattice position ({row}, {col})",
                nodes.0, nodes.1
            ),
            Self::CoordinateLengthMismatch { got, expected } => {
                write!(f, "coordinate array has length {got}, expected {expected}")
            }
            Self::InconsistentHexParity {
                expected_node,
                node,
                position,
            } => write!(
                f,
                "node {node} at ({}, {}) disagrees in (row + col) parity with node \
                 {expected_node}. Visium doubled coordinates give every spot the same parity; \
                 mixed parity usually means offset coordinates were passed instead",
                position.0, position.1
            ),
            Self::AsymmetricGraph {
                degree_sum,
                expected,
            } => write!(
                f,
                "adjacency is not symmetric: Σ strength = {degree_sum}, expected \
                 2 · total_weight = {expected}. A k-NN graph must be symmetrised before use."
            ),
        }
    }
}

impl std::error::Error for ClusteringError {}

/// Convenience alias for results produced by this crate.
pub type Result<T> = std::result::Result<T, ClusteringError>;
