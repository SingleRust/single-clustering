//! Delaunay neighbour graphs.
//!
//! Parameter-free: no radius, no `k`. Each cell is joined to the ones it actually borders,
//! and the triangulation adapts to local density on its own — dense regions get short edges,
//! sparse regions long ones, with no threshold to tune. Average degree is just under 6, since
//! a triangulation of `n` points with `h` on the convex hull has exactly `3n - 3 - h` edges.
//!
//! # Why pruning is still needed
//!
//! A triangulation covers the *convex hull* of the points, so it invents edges wherever the
//! tissue is not convex: across ventricles and vessels, around concave boundaries, and
//! between physically separate fragments on the same section. Those edges are long, and
//! joining two sides of a hole is exactly the kind of error that produces a plausible-looking
//! but meaningless domain.
//!
//! Note that the parameter-free subgraphs — Gabriel, relative-neighbourhood — do **not** fix
//! this. Their tests ask whether other points lie near the edge, and the whole point of a
//! hole is that nothing does, so a hole-spanning edge passes. Measured on two detached
//! fragments, Gabriel leaves 19 of 42 bridges standing and drops 21% of genuine edges
//! elsewhere; the relative-neighbourhood graph drops 52%. Removing an artefact edge needs a
//! rule about length. See [`HullPruning`].
//!
//! # What other tools do
//!
//! Worth knowing, because the defaults differ sharply:
//!
//! * **squidpy** applies no pruning at all — `delaunay=True` returns the raw triangulation,
//!   convex hull included. Its `radius` and `percentile` filters both default to off, and in
//!   1.6.x a scalar `radius` is *silently ignored* on the Delaunay path.
//! * **Giotto** is the only one that derives a cutoff automatically, using a global Tukey
//!   whisker over all edge lengths. Being global, it fails exactly where tissue density
//!   varies: on a dense block beside 24x sparser stroma it shatters the sparse side into
//!   ~570 components while barely touching the dense one.
//!
//! That is the case for a local rule, and why [`HullPruning::Adaptive`] is the default here.

use crate::error::{ClusteringError, Result};
use crate::network::{CSRNetwork, MAX_NODES};
use crate::spatial::points::SpatialWeight;

/// How to drop the edges a triangulation invents across empty space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HullPruning {
    /// Keep the whole triangulation. Correct only when the tissue really is convex and
    /// gap-free, which is rare.
    None,
    /// Drop edges longer than this, in coordinate units. Use when the tissue has a known
    /// scale and roughly uniform density.
    MaxDistance(f64),
    /// Drop the longest `fraction` of all edges.
    ///
    /// One global cutoff, so it over-prunes sparse regions and under-prunes dense ones
    /// whenever density varies — which in tissue it always does.
    Percentile {
        /// Share of edges to drop, in `[0, 1)`.
        fraction: f64,
    },
    /// Drop an edge longer than `factor` times the local edge scale at **both** its
    /// endpoints, where local scale is a low quantile of a node's incident edge lengths.
    ///
    /// Compares against `max` of the two scales rather than `min`, so an edge that is long
    /// only because one end sits in sparse tissue survives — a real boundary between a dense
    /// nest and loose stroma should stay connected. A hole-spanning edge is long relative to
    /// both ends and goes.
    ///
    /// The default of 2.0 is measured, by `calibrate_adaptive_factor`: it is the largest
    /// factor that fully separates two detached tissue fragments, which is the one
    /// unambiguous requirement — separate pieces must never be joined. It costs about 5% of
    /// edges on uniform-random points, and less on real tissue, whose cells pack more
    /// regularly than a Poisson process.
    ///
    /// What it will *not* do is clear a hole only two or three cells across. Removing those
    /// needs a factor around 1.5, which drops ~16% of genuine edges everywhere. That is a
    /// resolution limit rather than a tuning failure: at two cells wide it is genuinely
    /// arguable whether the cells either side are neighbours.
    Adaptive {
        /// Multiple of the local scale above which an edge is dropped.
        factor: f64,
    },
}

impl Default for HullPruning {
    fn default() -> Self {
        Self::Adaptive { factor: 2.0 }
    }
}

impl HullPruning {
    fn validate(self) -> Result<()> {
        match self {
            Self::MaxDistance(d) if !d.is_finite() || d <= 0.0 => {
                Err(ClusteringError::InvalidConfig(format!(
                    "max distance must be finite and positive, got {d}"
                )))
            }
            Self::Percentile { fraction } if !(0.0..1.0).contains(&fraction) => {
                Err(ClusteringError::InvalidConfig(format!(
                    "fraction must be in [0, 1), got {fraction}"
                )))
            }
            Self::Adaptive { factor } if !factor.is_finite() || factor <= 0.0 => {
                Err(ClusteringError::InvalidConfig(format!(
                    "factor must be finite and positive, got {factor}"
                )))
            }
            _ => Ok(()),
        }
    }
}

/// Builds a Delaunay neighbour graph from 2D coordinates.
///
/// 2D only — the 3D equivalent is a tetrahedralisation, which is a different algorithm and a
/// much heavier dependency. For 3D use [`radius_graph`](super::radius_graph) or
/// [`knn_graph`](super::knn_graph).
///
/// # Errors
///
/// If the points are degenerate (all collinear, or fewer than three distinct positions), so
/// no triangulation exists. That is reported rather than returning an empty graph, since an
/// edgeless graph would cluster into singletons and look like a result.
pub fn delaunay_graph(
    points: &[[f64; 2]],
    pruning: HullPruning,
    weight: SpatialWeight,
) -> Result<CSRNetwork> {
    pruning.validate()?;
    if points.len() > MAX_NODES {
        return Err(ClusteringError::TooManyNodes {
            n_nodes: points.len(),
        });
    }
    for (i, p) in points.iter().enumerate() {
        if !p[0].is_finite() || !p[1].is_finite() {
            return Err(ClusteringError::InvalidConfig(format!(
                "point {i} has a non-finite coordinate"
            )));
        }
    }

    let n = points.len();
    if n < 2 {
        return CSRNetwork::from_csr_parts(vec![0; n + 1], Vec::new(), Vec::new(), None);
    }
    // Two points have no triangle but do border each other.
    if n == 2 {
        let d2 = distance_sq(points[0], points[1]);
        return build(n, &[(0, 1, d2)], weight);
    }

    let coords: Vec<delaunator::Point> = points
        .iter()
        .map(|p| delaunator::Point { x: p[0], y: p[1] })
        .collect();
    let tri = delaunator::triangulate(&coords);
    if tri.triangles.is_empty() {
        return Err(ClusteringError::DegenerateTriangulation { n_points: n });
    }

    // Each triangle contributes three edges; neighbouring triangles share them, so collect
    // into a set keyed on the ordered pair.
    let mut edges: Vec<(u32, u32, f64)> = Vec::with_capacity(tri.triangles.len());
    for t in tri.triangles.chunks_exact(3) {
        for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            if a < b {
                edges.push((a as u32, b as u32, distance_sq(points[a], points[b])));
            } else {
                edges.push((b as u32, a as u32, distance_sq(points[a], points[b])));
            }
        }
    }
    edges.sort_unstable_by_key(|e| (e.0, e.1));
    edges.dedup_by_key(|e| (e.0, e.1));

    let kept = prune(&edges, n, pruning);
    build(n, &kept, weight)
}

#[inline]
fn distance_sq(a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
    dx * dx + dy * dy
}

fn prune(edges: &[(u32, u32, f64)], n: usize, pruning: HullPruning) -> Vec<(u32, u32, f64)> {
    match pruning {
        HullPruning::None => edges.to_vec(),
        HullPruning::MaxDistance(d) => {
            let limit = d * d;
            edges.iter().copied().filter(|e| e.2 <= limit).collect()
        }
        HullPruning::Percentile { fraction } => {
            let mut lengths: Vec<f64> = edges.iter().map(|e| e.2).collect();
            lengths.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let keep = ((edges.len() as f64) * (1.0 - fraction)).ceil() as usize;
            if keep >= edges.len() {
                return edges.to_vec();
            }
            let limit = lengths[keep.saturating_sub(1).min(lengths.len() - 1)];
            edges.iter().copied().filter(|e| e.2 <= limit).collect()
        }
        HullPruning::Adaptive { factor } => {
            let scale = local_scale(edges, n);
            let f2 = factor * factor;
            edges
                .iter()
                .copied()
                .filter(|&(a, b, d2)| {
                    // Squared throughout, so the factor is squared too.
                    let s = scale[a as usize].max(scale[b as usize]);
                    d2 <= f2 * s
                })
                .collect()
        }
    }
}

/// Quantile of incident edge length used as a node's local spacing.
///
/// Deliberately below the median. A node on the rim of a hole has several long edges reaching
/// across it, and those drag its median up until the spurious edges look normal — the median
/// hides exactly what we are trying to find. A lower quantile reads the genuine local
/// spacing, since at least a quarter of any node's edges go to real neighbours.
const SCALE_QUANTILE: f64 = 0.25;

/// Squared local length scale per node. One small sort each, since Delaunay degree averages
/// just under 6.
fn local_scale(edges: &[(u32, u32, f64)], n: usize) -> Vec<f64> {
    let mut incident: Vec<Vec<f64>> = vec![Vec::new(); n];
    for &(a, b, d2) in edges {
        incident[a as usize].push(d2);
        incident[b as usize].push(d2);
    }
    incident
        .iter_mut()
        .map(|lengths| {
            if lengths.is_empty() {
                return f64::INFINITY;
            }
            lengths.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((lengths.len() - 1) as f64 * SCALE_QUANTILE).round() as usize;
            lengths[idx]
        })
        .collect()
}

/// Turns an undirected edge list into a symmetric graph.
fn build(n: usize, edges: &[(u32, u32, f64)], weight: SpatialWeight) -> Result<CSRNetwork> {
    let mut degree = vec![0u32; n];
    for &(a, b, _) in edges {
        degree[a as usize] += 1;
        degree[b as usize] += 1;
    }
    let mut node_ptrs = Vec::with_capacity(n + 1);
    node_ptrs.push(0usize);
    let mut total = 0usize;
    for &d in &degree {
        total += d as usize;
        node_ptrs.push(total);
    }

    let mut neighbors = vec![0u32; total];
    let mut weights = vec![0.0f32; total];
    let mut cursor: Vec<usize> = node_ptrs[..n].to_vec();
    for &(a, b, d2) in edges {
        let w = weight.of(d2) as f32;
        neighbors[cursor[a as usize]] = b;
        weights[cursor[a as usize]] = w;
        cursor[a as usize] += 1;
        neighbors[cursor[b as usize]] = a;
        weights[cursor[b as usize]] = w;
        cursor[b as usize] += 1;
    }

    CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    fn uniform(n: usize, span: f64, seed: u64) -> Vec<[f64; 2]> {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| [rng.random::<f64>() * span, rng.random::<f64>() * span])
            .collect()
    }

    /// A square of tissue with a circular hole punched out of the middle.
    fn tissue_with_hole(seed: u64) -> (Vec<[f64; 2]>, [f64; 2], f64) {
        tissue_with_hole_radius(seed, 15.0)
    }

    fn tissue_with_hole_radius(seed: u64, hole: f64) -> (Vec<[f64; 2]>, [f64; 2], f64) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let centre = [50.0, 50.0];
        let pts = (0..3000)
            .map(|_| [rng.random::<f64>() * 100.0, rng.random::<f64>() * 100.0])
            .filter(|p: &[f64; 2]| {
                (p[0] - centre[0]).powi(2) + (p[1] - centre[1]).powi(2) > hole * hole
            })
            .collect();
        (pts, centre, hole)
    }

    /// Tissue in two pieces with a gap between, as a torn or folded section gives.
    fn two_fragments(seed: u64) -> Vec<[f64; 2]> {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        (0..2000)
            .map(|i| {
                let off = if i % 2 == 0 { 0.0 } else { 75.0 };
                [off + rng.random::<f64>() * 50.0, rng.random::<f64>() * 50.0]
            })
            .collect()
    }

    /// Does the segment a-b pass through the disc?
    fn crosses(a: [f64; 2], b: [f64; 2], centre: [f64; 2], r: f64) -> bool {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = dx * dx + dy * dy;
        if len2 == 0.0 {
            return false;
        }
        let t = (((centre[0] - a[0]) * dx + (centre[1] - a[1]) * dy) / len2).clamp(0.0, 1.0);
        let (px, py) = (a[0] + t * dx, a[1] + t * dy);
        (px - centre[0]).powi(2) + (py - centre[1]).powi(2) < r * r
    }

    fn edge_set(g: &CSRNetwork) -> HashSet<(usize, usize)> {
        let mut out = HashSet::new();
        for v in 0..g.node_count() {
            for (u, _) in g.neighbors(v) {
                out.insert((v.min(u), v.max(u)));
            }
        }
        out
    }

    /// Euler's formula: a Delaunay triangulation of `n` points with `h` on the convex hull
    /// has exactly `3n - 3 - h` edges. Nothing approximate about it.
    #[test]
    fn unpruned_edge_count_matches_eulers_formula() {
        for (n, seed) in [(50usize, 1u64), (500, 2), (2000, 3)] {
            let pts = uniform(n, 100.0, seed);
            let g = delaunay_graph(&pts, HullPruning::None, SpatialWeight::Uniform).unwrap();

            let coords: Vec<delaunator::Point> = pts
                .iter()
                .map(|p| delaunator::Point { x: p[0], y: p[1] })
                .collect();
            let h = delaunator::triangulate(&coords).hull.len();

            assert_eq!(g.edge_count(), 3 * n - 3 - h, "n={n}, hull={h}");
            // Which puts average degree just under 6.
            let avg = 2.0 * g.edge_count() as f64 / n as f64;
            assert!(avg < 6.0 && avg > 5.0, "average degree {avg} at n={n}");
        }
    }

    /// The reason pruning exists: a triangulation spans holes, and the default must not.
    #[test]
    fn adaptive_pruning_clears_a_tissue_hole() {
        let (pts, centre, hole) = tissue_with_hole(7);
        let count_crossing = |g: &CSRNetwork| {
            edge_set(g)
                .iter()
                .filter(|&&(a, b)| crosses(pts[a], pts[b], centre, hole * 0.9))
                .count()
        };

        let raw = delaunay_graph(&pts, HullPruning::None, SpatialWeight::Uniform).unwrap();
        let pruned = delaunay_graph(&pts, HullPruning::default(), SpatialWeight::Uniform).unwrap();

        let before = count_crossing(&raw);
        let after = count_crossing(&pruned);
        println!("hole-crossing edges: {before} raw -> {after} pruned");
        assert!(
            before >= 5,
            "the hole should be bridged before pruning: {before}"
        );
        assert_eq!(after, 0, "{after} edges still cross the hole");
        // And it must not gut the rest of the graph. The default costs ~5% on uniform-random
        // points, which are less regularly packed than real cells, so this is the pessimistic
        // end of the range.
        assert!(
            pruned.edge_count() as f64 > 0.92 * raw.edge_count() as f64,
            "pruning removed {} of {} edges",
            raw.edge_count() - pruned.edge_count(),
            raw.edge_count()
        );
    }

    /// Why the rule compares against `max` of the two local scales, not `min`: a genuine
    /// boundary between dense and sparse tissue has to survive.
    #[test]
    fn adaptive_pruning_keeps_a_dense_to_sparse_boundary() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let mut pts: Vec<[f64; 2]> = Vec::new();
        // Dense nest on the left, loose stroma on the right, touching at x = 50.
        for _ in 0..1500 {
            pts.push([rng.random::<f64>() * 50.0, rng.random::<f64>() * 50.0]);
        }
        for _ in 0..150 {
            pts.push([
                50.0 + rng.random::<f64>() * 50.0,
                rng.random::<f64>() * 50.0,
            ]);
        }

        let g = delaunay_graph(&pts, HullPruning::default(), SpatialWeight::Uniform).unwrap();
        let spanning = edge_set(&g)
            .iter()
            .filter(|&&(a, b)| (pts[a][0] - 50.0).signum() != (pts[b][0] - 50.0).signum())
            .count();
        assert!(
            spanning > 10,
            "only {spanning} edges cross the density boundary; the two regions are severed"
        );
    }

    /// Sweeps the one tuned constant across every artefact it has to remove and everything it
    /// must not, over several seeds, so the default is a measurement rather than a guess.
    ///
    /// A large hole is easy and tells you little. The discriminating cases are a
    /// capillary-scale hole a few cell spacings across, and two separated fragments.
    ///
    /// ```text
    /// cargo test --release --no-default-features calibrate_adaptive -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "diagnostic sweep, not an assertion"]
    fn calibrate_adaptive_factor() {
        println!(
            "{:>6} | {:>9} | {:>8} | {:>9} | {:>6} | {:>8}",
            "factor", "hole r=15", "hole r=4", "fragments", "kept", "boundary"
        );

        for factor in [1.5, 2.0, 2.5, 3.0, 4.0, 6.0] {
            let p = HullPruning::Adaptive { factor };
            let (mut big, mut small, mut bridges, mut kept, mut boundary) =
                (0usize, 0usize, 0usize, 0.0f64, 0usize);

            for seed in 0..4u64 {
                for (radius, counter) in [(15.0, &mut big), (4.0, &mut small)] {
                    let (pts, centre, hole) = tissue_with_hole_radius(seed, radius);
                    let g = delaunay_graph(&pts, p, SpatialWeight::Uniform).unwrap();
                    *counter += edge_set(&g)
                        .iter()
                        .filter(|&&(a, b)| crosses(pts[a], pts[b], centre, hole * 0.9))
                        .count();
                    if radius == 15.0 {
                        let raw = delaunay_graph(&pts, HullPruning::None, SpatialWeight::Uniform)
                            .unwrap();
                        kept += 100.0 * g.edge_count() as f64 / raw.edge_count() as f64;
                    }
                }

                let frag = two_fragments(seed);
                let g = delaunay_graph(&frag, p, SpatialWeight::Uniform).unwrap();
                bridges += edge_set(&g)
                    .iter()
                    .filter(|&&(a, b)| (frag[a][0] < 60.0) != (frag[b][0] < 60.0))
                    .count();

                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(100 + seed);
                let mut mixed: Vec<[f64; 2]> = Vec::new();
                for _ in 0..1500 {
                    mixed.push([rng.random::<f64>() * 50.0, rng.random::<f64>() * 50.0]);
                }
                for _ in 0..150 {
                    mixed.push([
                        50.0 + rng.random::<f64>() * 50.0,
                        rng.random::<f64>() * 50.0,
                    ]);
                }
                let m = delaunay_graph(&mixed, p, SpatialWeight::Uniform).unwrap();
                boundary += edge_set(&m)
                    .iter()
                    .filter(|&&(a, b)| {
                        (mixed[a][0] - 50.0).signum() != (mixed[b][0] - 50.0).signum()
                    })
                    .count();
            }

            println!(
                "{factor:>6.1} | {big:>9} | {small:>8} | {bridges:>9} | {:>5.1}% | {boundary:>8}",
                kept / 4.0
            );
        }
    }

    /// Detached tissue pieces must never be joined — the criterion the default is set by.
    #[test]
    fn detached_fragments_are_never_bridged() {
        for seed in 0..4 {
            let pts = two_fragments(seed);
            let g = delaunay_graph(&pts, HullPruning::default(), SpatialWeight::Uniform).unwrap();
            let bridges = edge_set(&g)
                .iter()
                .filter(|&&(a, b)| (pts[a][0] < 60.0) != (pts[b][0] < 60.0))
                .count();
            assert_eq!(bridges, 0, "seed {seed} left {bridges} bridging edges");

            // And each fragment must survive intact rather than being shattered.
            let c = crate::testdata::disconnected_community_count(
                &g,
                &pts.iter()
                    .map(|p| usize::from(p[0] >= 60.0))
                    .collect::<Vec<_>>(),
            );
            assert_eq!(c, 0, "seed {seed}: a fragment came apart");
        }
    }

    #[test]
    fn pruning_modes_are_ordered_by_aggressiveness() {
        let pts = uniform(1000, 100.0, 11);
        let raw = delaunay_graph(&pts, HullPruning::None, SpatialWeight::Uniform).unwrap();

        let mut previous = raw.edge_count();
        for fraction in [0.01, 0.05, 0.2] {
            let g = delaunay_graph(
                &pts,
                HullPruning::Percentile { fraction },
                SpatialWeight::Uniform,
            )
            .unwrap();
            assert!(g.edge_count() < previous, "fraction {fraction} kept more");
            previous = g.edge_count();
        }

        let mut previous = raw.edge_count();
        for factor in [4.0, 2.0, 1.2] {
            let g = delaunay_graph(
                &pts,
                HullPruning::Adaptive { factor },
                SpatialWeight::Uniform,
            )
            .unwrap();
            assert!(g.edge_count() <= previous, "factor {factor} kept more");
            previous = g.edge_count();
        }
    }

    #[test]
    fn is_symmetric_and_order_independent() {
        use rand::seq::SliceRandom;
        let pts = uniform(600, 80.0, 13);
        let g = delaunay_graph(&pts, HullPruning::default(), SpatialWeight::Uniform).unwrap();
        g.validate_symmetry().unwrap();
        assert!(!g.has_self_loops());

        let reference: HashSet<(usize, usize)> = edge_set(&g);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        for _ in 0..4 {
            let mut perm: Vec<usize> = (0..pts.len()).collect();
            perm.shuffle(&mut rng);
            let shuffled: Vec<[f64; 2]> = perm.iter().map(|&i| pts[i]).collect();
            let h =
                delaunay_graph(&shuffled, HullPruning::default(), SpatialWeight::Uniform).unwrap();
            let mapped: HashSet<(usize, usize)> = edge_set(&h)
                .into_iter()
                .map(|(a, b)| (perm[a].min(perm[b]), perm[a].max(perm[b])))
                .collect();
            assert_eq!(mapped, reference);
        }
    }

    #[test]
    fn max_distance_is_exact() {
        let pts = uniform(500, 50.0, 17);
        let limit = 3.0;
        let g = delaunay_graph(
            &pts,
            HullPruning::MaxDistance(limit),
            SpatialWeight::Uniform,
        )
        .unwrap();
        for (a, b) in edge_set(&g) {
            let d = distance_sq(pts[a], pts[b]).sqrt();
            assert!(d <= limit + 1e-9, "kept an edge of length {d}");
        }
    }

    #[test]
    fn degenerate_input_is_reported_not_silently_empty() {
        // Collinear points have no triangulation.
        let line: Vec<[f64; 2]> = (0..50).map(|i| [i as f64, 0.0]).collect();
        assert!(matches!(
            delaunay_graph(&line, HullPruning::None, SpatialWeight::Uniform),
            Err(ClusteringError::DegenerateTriangulation { .. })
        ));

        // As do coincident ones.
        let same = vec![[1.0, 1.0]; 40];
        assert!(matches!(
            delaunay_graph(&same, HullPruning::None, SpatialWeight::Uniform),
            Err(ClusteringError::DegenerateTriangulation { .. })
        ));
    }

    #[test]
    fn handles_small_inputs() {
        let none: Vec<[f64; 2]> = Vec::new();
        assert_eq!(
            delaunay_graph(&none, HullPruning::None, SpatialWeight::Uniform)
                .unwrap()
                .node_count(),
            0
        );
        let one = vec![[0.0, 0.0]];
        let g = delaunay_graph(&one, HullPruning::None, SpatialWeight::Uniform).unwrap();
        assert_eq!((g.node_count(), g.edge_count()), (1, 0));

        // Two points border each other even though there is no triangle.
        let two = vec![[0.0, 0.0], [1.0, 0.0]];
        let g = delaunay_graph(&two, HullPruning::None, SpatialWeight::Uniform).unwrap();
        assert_eq!((g.node_count(), g.edge_count()), (2, 1));

        let three = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let g = delaunay_graph(&three, HullPruning::None, SpatialWeight::Uniform).unwrap();
        assert_eq!(g.edge_count(), 3, "a single triangle");
    }

    #[test]
    fn rejects_bad_config() {
        let pts = uniform(50, 10.0, 1);
        for bad in [
            HullPruning::MaxDistance(0.0),
            HullPruning::MaxDistance(f64::NAN),
            HullPruning::Percentile { fraction: 1.0 },
            HullPruning::Percentile { fraction: -0.1 },
            HullPruning::Adaptive { factor: 0.0 },
        ] {
            assert!(
                delaunay_graph(&pts, bad, SpatialWeight::Uniform).is_err(),
                "{bad:?} should be rejected"
            );
        }
        let bad = vec![[0.0, 0.0], [f64::NAN, 1.0], [1.0, 1.0]];
        assert!(delaunay_graph(&bad, HullPruning::None, SpatialWeight::Uniform).is_err());
    }

    /// Leiden on a Delaunay graph must still give spatially contiguous domains.
    #[test]
    fn leiden_domains_stay_contiguous() {
        use crate::community_search::leiden::{LeidenConfig, ObjectiveKind, leiden};
        let pts = uniform(2000, 100.0, 23);
        let g = delaunay_graph(&pts, HullPruning::default(), SpatialWeight::Uniform).unwrap();
        let c = leiden(
            &g,
            &LeidenConfig {
                objective: ObjectiveKind::Cpm { resolution: 0.2 },
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            crate::testdata::disconnected_community_count(&g, c.labels()),
            0
        );
        assert!(c.n_clusters() > 1);
    }
}
