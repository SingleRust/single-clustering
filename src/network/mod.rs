//! # Network Module
//!
//! This module provides the network/graph representation used by the clustering
//! algorithms, plus the grouping abstraction used to express partitions.
//!
//! ## Key Components
//! - [`CSRNetwork`] - compressed-sparse-row weighted undirected graph
//! - [`grouping`] - node-to-group assignments (partitions/clusterings)

pub mod grouping;

mod csr_network;

pub use csr_network::CSRNeighborIterator;
pub use csr_network::{CSRNetwork, MAX_NODES};
