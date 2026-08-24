//! Graph representation: [`CSRNetwork`] plus the [`grouping`] type partitions come back as.

pub mod grouping;

mod csr_network;

pub use csr_network::CSRNeighborIterator;
pub use csr_network::{CSRNetwork, MAX_NODES};
