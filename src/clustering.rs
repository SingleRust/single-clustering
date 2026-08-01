//! The result type shared by every clustering algorithm in this crate.

/// Label assigned to points that an algorithm considers noise.
///
/// Leiden never produces it — every node lands somewhere — but density-based methods will, so
/// it's here from the start rather than as a breaking change later.
pub const NOISE: usize = usize::MAX;

/// An assignment of items to clusters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clustering {
    labels: Vec<usize>,
    n_clusters: usize,
}

impl Clustering {
    /// Builds a clustering from labels, renumbering them to be consecutive from 0.
    ///
    /// [`NOISE`] labels are preserved as-is and excluded from the cluster count.
    pub fn from_labels(labels: Vec<usize>) -> Self {
        let mut remap = std::collections::HashMap::new();
        let mut next = 0usize;
        let mut out = Vec::with_capacity(labels.len());
        // node order, so numbering is deterministic
        for &l in &labels {
            if l == NOISE {
                out.push(NOISE);
                continue;
            }
            let id = *remap.entry(l).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            });
            out.push(id);
        }
        Self {
            labels: out,
            n_clusters: next,
        }
    }

    /// Builds a clustering from labels already known to be consecutive from 0.
    pub(crate) fn from_normalized(labels: Vec<usize>, n_clusters: usize) -> Self {
        Self { labels, n_clusters }
    }

    /// Cluster label of each item, in input order.
    #[inline]
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Number of clusters, excluding noise.
    #[inline]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Number of items.
    #[inline]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether there are no items.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Number of items in each cluster, indexed by cluster id.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.n_clusters];
        for &l in &self.labels {
            if l != NOISE {
                sizes[l] += 1;
            }
        }
        sizes
    }

    /// The items in each cluster, indexed by cluster id.
    pub fn clusters(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.n_clusters];
        for (item, &l) in self.labels.iter().enumerate() {
            if l != NOISE {
                out[l].push(item);
            }
        }
        out
    }

    /// Items labelled as noise.
    pub fn noise(&self) -> Vec<usize> {
        self.labels
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l == NOISE)
            .map(|(i, _)| i)
            .collect()
    }

    /// Consumes the clustering and returns the raw labels.
    pub fn into_labels(self) -> Vec<usize> {
        self.labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_labels_renumbers_in_first_seen_order() {
        let c = Clustering::from_labels(vec![7, 7, 3, 9, 3]);
        assert_eq!(c.labels(), &[0, 0, 1, 2, 1]);
        assert_eq!(c.n_clusters(), 3);
        assert_eq!(c.cluster_sizes(), vec![2, 2, 1]);
    }

    #[test]
    fn noise_is_preserved_and_excluded() {
        let c = Clustering::from_labels(vec![5, NOISE, 5, NOISE]);
        assert_eq!(c.n_clusters(), 1);
        assert_eq!(c.labels(), &[0, NOISE, 0, NOISE]);
        assert_eq!(c.noise(), vec![1, 3]);
        assert_eq!(c.cluster_sizes(), vec![2]);
    }

    #[test]
    fn empty_is_empty() {
        let c = Clustering::from_labels(vec![]);
        assert!(c.is_empty());
        assert_eq!(c.n_clusters(), 0);
    }
}
