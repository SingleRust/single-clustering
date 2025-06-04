use std::collections::{HashSet, VecDeque};

use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use single_utilities::traits::FloatOpsTS;

use crate::network::{Network, grouping::NetworkGrouping};
pub mod partition;
mod optimizer;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsiderComms {
    AllComms = 1,
    AllNeighComms = 2,
    RandComm = 3,
    RandNeighComm = 4,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimiseRoutine {
    MoveNodes = 10,
    MergeNodes = 11,
}

#[derive(Debug, Clone)]
pub struct LeidenConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub seed: Option<u64>,
    pub max_community_size: Option<usize>,
    pub refine_partition: bool,
    pub consider_empty_community: bool,
    pub consider_comms: ConsiderComms,
    pub refine_consider_comms: ConsiderComms,
    pub optimise_routine: OptimiseRoutine,
    pub refine_routine: OptimiseRoutine,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            seed: None,
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

