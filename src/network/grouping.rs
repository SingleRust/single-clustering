//! Node-to-group assignments, as produced by clustering and consumed by
//! [`CSRNetwork::aggregate`](crate::network::CSRNetwork::aggregate).

/// A labelling of nodes into groups, with ids renumbered consecutively from 0.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VectorGrouping {
    assignments: Vec<usize>,
    group_count: usize,
}

impl VectorGrouping {
    /// Builds a grouping from raw assignments, renumbering to close any gaps.
    pub fn from_assignments(assignments: &[usize]) -> Self {
        let slots = assignments.iter().copied().max().map_or(0, |g| g + 1);
        let mut seen = vec![false; slots];
        for &g in assignments {
            seen[g] = true;
        }

        let mut new_id = vec![usize::MAX; slots];
        let mut next = 0;
        for (id, &used) in new_id.iter_mut().zip(&seen) {
            if used {
                *id = next;
                next += 1;
            }
        }

        Self {
            assignments: assignments.iter().map(|&g| new_id[g]).collect(),
            group_count: next,
        }
    }

    /// The group a node belongs to.
    #[inline]
    pub fn get_group(&self, node: usize) -> usize {
        self.assignments[node]
    }

    /// Number of groups, all of them non-empty.
    #[inline]
    pub fn group_count(&self) -> usize {
        self.group_count
    }

    /// Number of nodes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.assignments.len()
    }

    /// The assignments, consecutive from 0.
    #[inline]
    pub fn assignments(&self) -> &[usize] {
        &self.assignments
    }
}
