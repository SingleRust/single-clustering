//! Spatial neighbour graphs.
//!
//! Spatial domain detection is mostly a graph-construction problem: build the right graph and
//! the clustering is the same Leiden used everywhere else. Because refinement guarantees
//! internally-connected communities, running it on a spatial graph gives spatially contiguous
//! domains for free.
//!
//! Use [`ObjectiveKind::Cpm`](crate::community_search::leiden::ObjectiveKind::Cpm) rather than
//! RB for spatial domains — modularity's resolution limit bites harder here, where domains
//! tend to be numerous and small.

pub mod lattice;

pub use lattice::{Lattice, lattice_graph, visium_isometric_coords};
