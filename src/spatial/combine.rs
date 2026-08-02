//! Combining graphs, and building them per sample.
//!
//! Two operations that both exist because real experiments are not one clean tissue section:
//! you usually have an expression graph *and* a spatial graph over the same cells, and you
//! usually have several slices rather than one.

use crate::error::{ClusteringError, Result};
use crate::network::CSRNetwork;

/// How to put two graphs on comparable footing before mixing them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Normalization {
    /// Mix the weights as they are.
    ///
    /// Almost never what you want across an expression and a spatial graph: their weights
    /// live on unrelated scales, so whichever is larger dominates at every `alpha` and the
    /// parameter stops meaning anything.
    None,
    /// Scale each graph so its total edge weight is 1, so `alpha` divides a fixed budget.
    #[default]
    TotalWeight,
}

/// Blends two graphs over the same nodes: `alpha · a + (1 − alpha) · b`.
///
/// An edge present in only one contributes only its own side, so the result is the union of
/// the two edge sets. Both graphs must cover the same nodes.
///
/// With [`Normalization::TotalWeight`] each graph is first scaled to unit total weight, which
/// is what makes `alpha` behave like a proportion rather than an arbitrary knob. It is the
/// same problem the BANKSY blocks solve by z-scoring, and the same fix.
///
/// # Errors
///
/// If the graphs have different node counts, or `alpha` is outside `[0, 1]`.
pub fn fuse(
    a: &CSRNetwork,
    b: &CSRNetwork,
    alpha: f64,
    normalization: Normalization,
) -> Result<CSRNetwork> {
    if !(0.0..=1.0).contains(&alpha) || !alpha.is_finite() {
        return Err(ClusteringError::InvalidConfig(format!(
            "alpha must be in [0, 1], got {alpha}"
        )));
    }
    if a.node_count() != b.node_count() {
        return Err(ClusteringError::InvalidConfig(format!(
            "graphs cover different node sets: {} vs {} nodes",
            a.node_count(),
            b.node_count()
        )));
    }
    let n = a.node_count();
    if n == 0 {
        return CSRNetwork::from_csr_parts(vec![0], Vec::new(), Vec::new(), None);
    }

    let (sa, sb) = match normalization {
        Normalization::None => (1.0, 1.0),
        Normalization::TotalWeight => (
            scale_to_unit(a.total_weight()),
            scale_to_unit(b.total_weight()),
        ),
    };
    let (wa, wb) = (alpha * sa, (1.0 - alpha) * sb);

    // Merge each node's two neighbour lists. Both are sorted ascending, so this is a linear
    // two-pointer walk and the output stays sorted.
    let mut node_ptrs = Vec::with_capacity(n + 1);
    node_ptrs.push(0usize);
    let mut neighbors: Vec<u32> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();

    let (mut ra, mut rb) = (Vec::new(), Vec::new());
    for v in 0..n {
        ra.clear();
        rb.clear();
        ra.extend(a.neighbors(v));
        rb.extend(b.neighbors(v));

        let (mut i, mut j) = (0usize, 0usize);
        while i < ra.len() || j < rb.len() {
            let take_a = j >= rb.len() || (i < ra.len() && ra[i].0 <= rb[j].0);
            let take_b = i >= ra.len() || (j < rb.len() && rb[j].0 <= ra[i].0);
            let u = if take_a { ra[i].0 } else { rb[j].0 };

            let mut w = 0.0;
            if take_a {
                w += wa * ra[i].1;
                i += 1;
            }
            if take_b {
                w += wb * rb[j].1;
                j += 1;
            }
            if w > 0.0 {
                neighbors.push(u as u32);
                weights.push(w as f32);
            }
        }
        node_ptrs.push(neighbors.len());
    }

    CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None)
}

/// Reciprocal of a total weight, or 1.0 when there is nothing to scale.
fn scale_to_unit(total: f64) -> f64 {
    if total > 0.0 && total.is_finite() {
        1.0 / total
    } else {
        1.0
    }
}

/// Runs a graph builder separately within each sample and reassembles one graph.
///
/// Spatial neighbours are only meaningful within a slice: two cells on different sections are
/// not adjacent however close their stored coordinates happen to be, and sections are often
/// laid out overlapping in the same coordinate space. Building globally and deleting
/// cross-sample edges afterwards is **not** equivalent for k-nearest builders — a cell at a
/// section's edge would lose neighbours rather than take k from its own section. So the
/// builder runs per sample.
///
/// `samples` labels each point. Node indices in the result are the caller's original ones.
///
/// # Errors
///
/// If `samples` does not match the point count, or the builder fails on a sample.
pub fn per_sample<F>(points: &[[f64; 2]], samples: &[u32], build: F) -> Result<CSRNetwork>
where
    F: Fn(&[[f64; 2]]) -> Result<CSRNetwork>,
{
    if samples.len() != points.len() {
        return Err(ClusteringError::CoordinateLengthMismatch {
            got: samples.len(),
            expected: points.len(),
        });
    }
    let n = points.len();
    if n == 0 {
        return CSRNetwork::from_csr_parts(vec![0], Vec::new(), Vec::new(), None);
    }

    // Group point indices by sample, keeping each group in ascending original order so the
    // result never depends on how the samples were labelled.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&v| (samples[v as usize], v));

    let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    let mut start = 0usize;
    while start < n {
        let sample = samples[order[start] as usize];
        let mut end = start;
        while end < n && samples[order[end] as usize] == sample {
            end += 1;
        }

        let members = &order[start..end];
        let subset: Vec<[f64; 2]> = members.iter().map(|&v| points[v as usize]).collect();
        let sub = build(&subset)?;
        if sub.node_count() != subset.len() {
            return Err(ClusteringError::InvalidConfig(format!(
                "builder returned {} nodes for a sample of {}",
                sub.node_count(),
                subset.len()
            )));
        }

        for local in 0..sub.node_count() {
            let global = members[local] as usize;
            for (u, w) in sub.neighbors(local) {
                rows[global].push((members[u], w as f32));
            }
        }
        start = end;
    }

    let mut node_ptrs = Vec::with_capacity(n + 1);
    node_ptrs.push(0usize);
    let mut neighbors = Vec::new();
    let mut weights = Vec::new();
    for row in &mut rows {
        row.sort_unstable_by_key(|&(u, _)| u);
        for &(u, w) in row.iter() {
            neighbors.push(u);
            weights.push(w);
        }
        node_ptrs.push(neighbors.len());
    }

    CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{
        HullPruning, SpatialWeight, Symmetry, delaunay_graph, knn_graph, radius_graph,
    };
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    fn edge_set(g: &CSRNetwork) -> HashSet<(usize, usize)> {
        let mut out = HashSet::new();
        for v in 0..g.node_count() {
            for (u, _) in g.neighbors(v) {
                out.insert((v.min(u), v.max(u)));
            }
        }
        out
    }

    fn two_slices(seed: u64) -> (Vec<[f64; 2]>, Vec<u32>) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut pts = Vec::new();
        let mut samples = Vec::new();
        // Deliberately overlapping coordinates: two sections stored in the same frame.
        for s in 0..2u32 {
            for _ in 0..300 {
                pts.push([rng.random::<f64>() * 50.0, rng.random::<f64>() * 50.0]);
                samples.push(s);
            }
        }
        (pts, samples)
    }

    #[test]
    fn fusion_endpoints_recover_each_input() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let pts: Vec<[f64; 2]> = (0..200)
            .map(|_| [rng.random::<f64>() * 30.0, rng.random::<f64>() * 30.0])
            .collect();
        let a = radius_graph(&pts, 4.0, SpatialWeight::Uniform, None).unwrap();
        let b = knn_graph(&pts, 5, Symmetry::Union, SpatialWeight::Uniform).unwrap();

        let at_one = fuse(&a, &b, 1.0, Normalization::TotalWeight).unwrap();
        assert_eq!(edge_set(&at_one), edge_set(&a));
        let at_zero = fuse(&a, &b, 0.0, Normalization::TotalWeight).unwrap();
        assert_eq!(edge_set(&at_zero), edge_set(&b));
    }

    #[test]
    fn fusion_is_the_union_of_both_edge_sets() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(2);
        let pts: Vec<[f64; 2]> = (0..150)
            .map(|_| [rng.random::<f64>() * 30.0, rng.random::<f64>() * 30.0])
            .collect();
        let a = radius_graph(&pts, 3.0, SpatialWeight::Uniform, None).unwrap();
        let b = delaunay_graph(&pts, HullPruning::default(), SpatialWeight::Uniform).unwrap();

        let f = fuse(&a, &b, 0.5, Normalization::TotalWeight).unwrap();
        let union: HashSet<_> = edge_set(&a).union(&edge_set(&b)).copied().collect();
        assert_eq!(edge_set(&f), union);
        f.validate_symmetry().unwrap();
    }

    /// The point of normalising: without it, the graph with heavier weights wins whatever
    /// alpha says.
    #[test]
    fn normalisation_stops_one_graph_dominating() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let pts: Vec<[f64; 2]> = (0..200)
            .map(|_| [rng.random::<f64>() * 30.0, rng.random::<f64>() * 30.0])
            .collect();
        let light = radius_graph(&pts, 3.0, SpatialWeight::Uniform, None).unwrap();
        // Same shape, weights a thousand times larger.
        let heavy = {
            let (ptrs, nbrs, w) = light.to_csr_parts();
            CSRNetwork::from_csr_parts(ptrs, nbrs, w.iter().map(|x| x * 1000.0).collect(), None)
                .unwrap()
        };

        // Normalised, an even blend of a graph with itself is that graph, whatever the scale.
        let fused = fuse(&light, &heavy, 0.5, Normalization::TotalWeight).unwrap();
        let first = fused.neighbors(0).next().unwrap().1;
        for v in 0..fused.node_count() {
            for (_, w) in fused.neighbors(v) {
                assert!(
                    (w / first - 1.0).abs() < 1e-4,
                    "normalised weights should be uniform here, saw {w} vs {first}"
                );
            }
        }

        // Unnormalised, the heavy graph contributes ~1000x as much.
        let raw = fuse(&light, &heavy, 0.5, Normalization::None).unwrap();
        let w = raw.neighbors(0).next().unwrap().1;
        assert!(w > 400.0, "expected the heavy graph to dominate, got {w}");
    }

    #[test]
    fn fusion_rejects_mismatched_inputs() {
        let a = radius_graph(&[[0.0, 0.0], [1.0, 0.0]], 2.0, SpatialWeight::Uniform, None).unwrap();
        let b = radius_graph(
            &[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            2.0,
            SpatialWeight::Uniform,
            None,
        )
        .unwrap();
        assert!(fuse(&a, &b, 0.5, Normalization::TotalWeight).is_err());
        assert!(fuse(&a, &a, 1.5, Normalization::TotalWeight).is_err());
        assert!(fuse(&a, &a, f64::NAN, Normalization::TotalWeight).is_err());
    }

    /// Cells in different sections must never be joined, even where their coordinates
    /// coincide — which they do here by construction.
    #[test]
    fn per_sample_never_joins_two_slices() {
        let (pts, samples) = two_slices(5);

        for name in ["knn", "radius", "delaunay"] {
            let g = per_sample(&pts, &samples, |sub| match name {
                "knn" => knn_graph(sub, 6, Symmetry::Union, SpatialWeight::Uniform),
                "radius" => radius_graph(sub, 4.0, SpatialWeight::Uniform, None),
                _ => delaunay_graph(sub, HullPruning::default(), SpatialWeight::Uniform),
            })
            .unwrap();

            g.validate_symmetry()
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            for (a, b) in edge_set(&g) {
                assert_eq!(
                    samples[a], samples[b],
                    "{name}: joined sample {} to {}",
                    samples[a], samples[b]
                );
            }
            assert!(g.edge_count() > 100, "{name} produced almost nothing");
        }
    }

    /// The reason it builds per sample rather than filtering afterwards: with kNN, a cell at
    /// a section's edge must take k neighbours from its own section, not lose them.
    #[test]
    fn per_sample_knn_keeps_every_cell_at_full_degree() {
        let (pts, samples) = two_slices(6);
        let k = 6;
        let g = per_sample(&pts, &samples, |sub| {
            knn_graph(sub, k, Symmetry::Union, SpatialWeight::Uniform)
        })
        .unwrap();
        for v in 0..g.node_count() {
            assert!(g.degree(v) >= k, "node {v} has degree {}", g.degree(v));
        }

        // Build globally and delete cross-sample edges, and cells lose neighbours instead.
        let global = knn_graph(&pts, k, Symmetry::Union, SpatialWeight::Uniform).unwrap();
        let starved = (0..global.node_count())
            .filter(|&v| {
                global
                    .neighbors(v)
                    .filter(|&(u, _)| samples[u] == samples[v])
                    .count()
                    < k
            })
            .count();
        assert!(
            starved > 0,
            "the filter-afterwards approach should starve some cells; if it no longer does, \
             this test's premise needs revisiting"
        );
    }

    #[test]
    fn per_sample_handles_edge_cases() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [50.0, 50.0]];
        // A sample of one has no neighbours, and must not break the reassembly.
        let samples = vec![0u32, 0, 1];
        let g = per_sample(&pts, &samples, |sub| {
            radius_graph(sub, 5.0, SpatialWeight::Uniform, None)
        })
        .unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.degree(2), 0);

        let none: Vec<[f64; 2]> = Vec::new();
        assert_eq!(
            per_sample(&none, &[], |sub| radius_graph(
                sub,
                1.0,
                SpatialWeight::Uniform,
                None
            ))
            .unwrap()
            .node_count(),
            0
        );
        assert!(
            per_sample(&pts, &[0, 1], |sub| radius_graph(
                sub,
                1.0,
                SpatialWeight::Uniform,
                None
            ))
            .is_err()
        );
    }

    /// Sample labels are arbitrary; only the grouping matters.
    #[test]
    fn per_sample_is_independent_of_label_values() {
        let (pts, samples) = two_slices(7);
        let build = |sub: &[[f64; 2]]| radius_graph(sub, 4.0, SpatialWeight::Uniform, None);

        let a = per_sample(&pts, &samples, build).unwrap();
        let relabelled: Vec<u32> = samples
            .iter()
            .map(|&s| if s == 0 { 77 } else { 3 })
            .collect();
        let b = per_sample(&pts, &relabelled, build).unwrap();
        assert_eq!(edge_set(&a), edge_set(&b));
    }
}
