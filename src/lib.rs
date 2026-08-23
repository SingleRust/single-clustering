#![doc = include_str!("../README.md")]

pub mod clustering;
pub mod error;
#[cfg(feature = "knn")]
pub mod neighborhood;
pub mod network;
pub mod spatial;
#[cfg(any(test, feature = "testdata"))]
#[doc(hidden)]
pub mod testdata;

pub mod community_search;

pub use clustering::{Clustering, NOISE};
pub use error::{ClusteringError, Result};
