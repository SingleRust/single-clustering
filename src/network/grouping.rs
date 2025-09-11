//! # Network Grouping Module
//!
//! This module provides abstractions for representing and manipulating node groupings
//! in networks. It's essential for clustering algorithms where nodes need to be
//! assigned to communities or clusters.
//!
//! ## Key Components
//! - `NetworkGrouping` trait - Interface for node-to-group assignments
//! - `VectorGrouping` - Efficient vector-based implementation
//! - Group operations: merge, normalize, bulk updates

use std::fmt::Debug;

/// Trait for representing node-to-group assignments in networks.
///
/// This trait provides a common interface for different grouping implementations
/// used in clustering algorithms. It supports operations like group assignment,
/// merging, and normalization essential for community detection.
pub trait NetworkGrouping: Debug + Send + Sync {
    /// Creates a grouping where each node is in its own isolated group.
    fn create_isolated(node_count: usize) -> Self;

    /// Creates a grouping where all nodes are in a single unified group.
    fn create_unified(node_count: usize) -> Self;

    /// Creates a grouping from an array of group assignments.
    fn from_assignments(assignments: &[usize]) -> Self;

    /// Returns a vector of vectors, where each inner vector contains the nodes in that group.
    fn get_group_members(&self) -> Vec<Vec<usize>>;

    /// Gets the group ID for a given node.
    fn get_group(&self, node: usize) -> usize;

    /// Gets the group IDs for a range of nodes.
    fn get_groups_range(&self, range: std::ops::Range<usize>) -> &[usize];

    /// Sets the group for a given node.
    fn set_group(&mut self, node: usize, group: usize);

    /// Sets groups for multiple nodes at once.
    fn set_groups_bulk(&mut self, nodes: &[usize], group: usize);

    /// Gets the total number of nodes.
    fn node_count(&self) -> usize;

    /// Gets the total number of groups.
    fn group_count(&self) -> usize;

    /// Renumbers groups to eliminate gaps in group IDs.
    fn normalize_groups(&mut self);

    /// Merges groups based on a higher-level grouping scheme.
    ///
    /// This is used in multilevel clustering where groups themselves are
    /// further grouped into larger communities.
    fn merge<G: NetworkGrouping>(&mut self, arrangement: &G) {
        for node in 0..self.node_count() {
            let current_group = self.get_group(node);
            let new_group = arrangement.get_group(current_group);
            self.set_group(node, new_group);
        }
        self.normalize_groups();
    }

    /// Resets all nodes to group 0, effectively creating a single unified group.
    fn clear(&mut self) {
        for i in 0..self.node_count() {
            self.set_group(i, 0);
        }

        self.normalize_groups();
    }
}

/// Vector-based implementation of NetworkGrouping.
///
/// This implementation uses a simple vector to store node-to-group assignments,
/// providing efficient access and updates. It includes caching for group sizes
/// to optimize frequent size queries in clustering algorithms.
///
/// # Performance Features
/// - O(1) group assignment and lookup
/// - Cached group sizes with lazy updates
/// - Bulk operations for efficiency
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct VectorGrouping {
    /// Vector storing the group assignment for each node
    assignments: Vec<usize>,
    /// Total number of distinct groups
    group_count: usize,
    /// Cache for frequently accessed group sizes
    group_sizes: Vec<usize>,
    /// Flag indicating if group_sizes cache needs updating
    needs_size_update: bool,
}

impl VectorGrouping {
    /// Updates the cached group sizes if needed.
    ///
    /// This method is called automatically when group sizes are accessed
    /// to ensure the cache is current.
    #[inline]
    fn update_group_sizes(&mut self) {
        if !self.needs_size_update {
            return;
        }

        self.group_sizes = vec![0; self.group_count];
        for &group in &self.assignments {
            self.group_sizes[group] += 1;
        }
        self.needs_size_update = false;
    }

    /// Gets the size of a specific group.
    ///
    /// Updates the size cache if necessary before returning the result.
    #[inline]
    pub fn get_group_size(&mut self, group: usize) -> usize {
        self.update_group_sizes();
        self.group_sizes[group]
    }

    /// Returns an iterator over groups with their sizes.
    ///
    /// Yields (group_id, group_size) pairs for all groups.
    pub fn iter_group_sizes(&mut self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.update_group_sizes();
        self.group_sizes.iter().copied().enumerate()
    }
}

impl NetworkGrouping for VectorGrouping {
    fn create_isolated(node_count: usize) -> Self {
        let assignments = (0..node_count).collect();
        Self {
            assignments,
            group_count: node_count,
            group_sizes: vec![1; node_count],
            needs_size_update: false,
        }
    }

    fn create_unified(node_count: usize) -> Self {
        Self {
            assignments: vec![0; node_count],
            group_count: usize::from(node_count > 0),
            group_sizes: vec![node_count],
            needs_size_update: false,
        }
    }

    fn from_assignments(input: &[usize]) -> Self {
        let mut max_group = 0;
        let assignments = input
            .iter()
            .map(|&group| {
                max_group = max_group.max(group);
                group
            })
            .collect();

        let mut grouping = Self {
            assignments,
            group_count: max_group + 1,
            group_sizes: Vec::new(),
            needs_size_update: true,
        };
        grouping.normalize_groups();
        grouping
    }

    fn get_group_members(&self) -> Vec<Vec<usize>> {
        let mut groups = vec![Vec::new(); self.group_count];

        // Pre-allocate space based on average group size
        let avg_size = self.assignments.len() / self.group_count;
        for group in groups.iter_mut() {
            group.reserve(avg_size);
        }

        for (node, &group) in self.assignments.iter().enumerate() {
            groups[group].push(node);
        }
        groups
    }

    #[inline]
    fn get_group(&self, node: usize) -> usize {
        self.assignments[node]
    }

    #[inline]
    fn get_groups_range(&self, range: std::ops::Range<usize>) -> &[usize] {
        &self.assignments[range]
    }

    #[inline]
    fn set_group(&mut self, node: usize, group: usize) {
        if self.assignments[node] != group {
            self.assignments[node] = group;
            self.group_count = self.group_count.max(group + 1);
            self.needs_size_update = true;
        }
    }

    fn set_groups_bulk(&mut self, nodes: &[usize], group: usize) {
        for &node in nodes {
            self.assignments[node] = group;
        }
        self.group_count = self.group_count.max(group + 1);
        self.needs_size_update = true;
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.assignments.len()
    }

    #[inline]
    fn group_count(&self) -> usize {
        self.group_count
    }

    fn normalize_groups(&mut self) {
        // Use original sequential approach for small datasets
        let mut sizes = vec![0; self.group_count];
        for &group in &self.assignments {
            sizes[group] += 1;
        }

        let mut new_ids = Vec::with_capacity(self.group_count);
        let mut next_id = 0;

        for size in sizes {
            if size == 0 {
                new_ids.push(usize::MAX);
            } else {
                new_ids.push(next_id);
                next_id += 1;
            }
        }

        for group in self.assignments.iter_mut() {
            let new_id = new_ids[*group];
            debug_assert!(new_id != usize::MAX, "Invalid group assignment");
            *group = new_id;
        }

        self.group_count = next_id;

        self.needs_size_update = true;
    }
}
