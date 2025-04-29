use std::fmt::Debug;

pub trait NetworkGrouping: Debug + Send + Sync {
    fn create_isolated(node_count: usize) -> Self;

    fn create_unified(node_count: usize) -> Self;

    fn from_assignments(assignments: &[usize]) -> Self;

    fn get_group_members(&self) -> Vec<Vec<usize>>;

    /// Gets the group ID for a given node
    fn get_group(&self, node: usize) -> usize;

    /// Gets the group IDs for a range of nodes
    fn get_groups_range(&self, range: std::ops::Range<usize>) -> &[usize];

    /// Sets the group for a given node
    fn set_group(&mut self, node: usize, group: usize);

    /// Sets groups for multiple nodes at once
    fn set_groups_bulk(&mut self, nodes: &[usize], group: usize);

    /// Gets the total number of nodes
    fn node_count(&self) -> usize;

    /// Gets the total number of groups
    fn group_count(&self) -> usize;

    /// Renumbers groups to eliminate gaps in group IDs
    fn normalize_groups(&mut self);

    /// Merges groups based on a higher-level grouping scheme
    fn merge<G: NetworkGrouping>(&mut self, arrangement: &G) {
        for node in 0..self.node_count() {
            let current_group = self.get_group(node);
            let new_group = arrangement.get_group(current_group);
            self.set_group(node, new_group);
        }
        self.normalize_groups();
    }

    fn clear(&mut self) {
        for i in 0..self.node_count() {
            self.set_group(i, 0);
        }

        self.normalize_groups();
    }
}

#[derive(Debug, Clone)]
pub struct VectorGrouping {
    assignments: Vec<usize>,
    group_count: usize,
    // Cache for frequently accessed group sizes
    group_sizes: Vec<usize>,
    needs_size_update: bool,
}

impl Default for VectorGrouping {
    fn default() -> Self {
        Self {
            assignments: Vec::new(),
            group_count: 0,
            group_sizes: Vec::new(),
            needs_size_update: false,
        }
    }
}

impl VectorGrouping {
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

    /// Gets the size of a specific group
    #[inline]
    pub fn get_group_size(&mut self, group: usize) -> usize {
        self.update_group_sizes();
        self.group_sizes[group]
    }

    /// Returns an iterator over groups with their sizes
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
