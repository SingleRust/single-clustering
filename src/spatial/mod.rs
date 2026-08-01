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

//! # Reading Visium data
//!
//! Two things to know before wiring up a loader, both verified against the upstream source:
//!
//! * Space Ranger ≥ 2.0 renamed `tissue_positions_list.csv` to `tissue_positions.csv` **and
//!   added a header row**. Parse the header rather than assuming column positions — scanpy
//!   branches on the filename to decide.
//! * `scanpy.read_visium` labels the two pixel columns the wrong way round, then selects them
//!   in the wrong order. The errors cancel, so `obsm["spatial"]` really is `(x, y)`, but the
//!   names in `obs` are swapped. Don't trust `pxl_row_in_fullres` / `pxl_col_in_fullres` if
//!   you read those columns directly.
//!
//! Neither affects `array_row` / `array_col`, which is what [`lattice_graph`] wants.

pub(crate) mod grid;
pub mod lattice;
pub mod points;

pub use lattice::{Lattice, lattice_graph, visium_isometric_coords};
pub use points::{SpatialWeight, Symmetry, knn_graph, radius_graph};
