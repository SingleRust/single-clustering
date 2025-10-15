//! Leiden algorithm implementation for community detection in networks.
//!
//! The Leiden algorithm is an improvement over the Louvain algorithm that guarantees
//! well-connected communities. It uses a refinement phase to ensure high-quality
//! partitions by preventing poorly connected communities. IMPORTANT: This code is currently work-in-progress and neither production ready nor optimize to the fullest!

pub mod partition;
mod optimizer;
pub use optimizer::LeidenOptimizer;
mod parallel;

/// Strategy for selecting communities to consider during optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsiderComms {
    /// Consider all non-empty communities.
    AllComms = 1,
    /// Consider communities of neighboring nodes only.
    AllNeighComms = 2,
    /// Randomly select one community.
    RandComm = 3,
    /// Randomly select a neighbor's community.
    RandNeighComm = 4,
}

/// Optimization routine for node movement during community detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimiseRoutine {
    /// Move individual nodes to different communities.
    MoveNodes = 10,
    /// Merge entire nodes/communities together.
    MergeNodes = 11,
}

/// Configuration parameters for the Leiden algorithm.
///
/// Controls the behavior of community detection including iteration limits,
/// convergence criteria, randomization, and optimization strategies.
#[derive(Debug, Clone)]
pub struct LeidenConfig {
    /// Maximum number of iterations before termination.
    pub max_iterations: usize,
    /// Convergence tolerance for quality improvement.
    pub tolerance: f64,
    /// Random seed for reproducible results. None for random seed.
    pub seed: Option<u64>,
    /// Maximum allowed community size. None for unlimited.
    pub max_community_size: Option<usize>,
    /// Whether to perform partition refinement for better connectivity.
    pub refine_partition: bool,
    /// Whether to consider moving nodes to empty communities.
    pub consider_empty_community: bool,
    /// Strategy for selecting communities during main optimization.
    pub consider_comms: ConsiderComms,
    /// Strategy for selecting communities during refinement.
    pub refine_consider_comms: ConsiderComms,
    /// Optimization routine for main phase.
    pub optimise_routine: OptimiseRoutine,
    /// Optimization routine for refinement phase.
    pub refine_routine: OptimiseRoutine,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            seed: Some(42),
            max_community_size: None,
            refine_partition: true,
            consider_empty_community: true,
            consider_comms: ConsiderComms::AllNeighComms,
            refine_consider_comms: ConsiderComms::AllNeighComms,
            optimise_routine: OptimiseRoutine::MoveNodes,
            refine_routine: OptimiseRoutine::MergeNodes,
        }
    }
}

