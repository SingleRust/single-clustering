#![doc = include_str!("../README.md")]

pub mod clustering;
pub mod error;
#[cfg(feature = "knn")]
pub mod neighborhood;
pub mod network;
#[cfg(test)]
pub(crate) mod testdata;

pub mod community_search;

pub use clustering::{Clustering, NOISE};
pub use error::{ClusteringError, Result};
