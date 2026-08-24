//! Spatial neighbour graphs.
//!
//! Domain detection is mostly a graph-construction problem — build the right graph and the
//! clustering is the same Leiden as everywhere else. Refinement guarantees internally
//! connected communities, so domains come out spatially contiguous for free.
//!
//! Prefer [`ObjectiveKind::Cpm`](crate::community_search::leiden::ObjectiveKind::Cpm) over RB
//! here: modularity's resolution limit bites harder when domains are many and small.
//!
//! Gotchas around reading Visium data are in `docs/spatial-graphs.md`.

pub mod banksy;
pub mod combine;
pub mod delaunay;
pub(crate) mod grid;
pub mod lattice;
pub mod points;

pub use combine::{Normalization, fuse, per_sample};
pub use delaunay::{HullPruning, delaunay_graph};
pub use lattice::{Lattice, lattice_graph, visium_isometric_coords};
pub use points::{SpatialWeight, Symmetry, knn_graph, radius_graph};
