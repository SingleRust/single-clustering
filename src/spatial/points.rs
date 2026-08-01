//! Neighbour graphs from point coordinates, for assays that are not on a lattice.
//!
//! Xenium, MERFISH, CosMx and Slide-seq give each cell a position rather than a grid slot, so
//! neighbours have to be searched for. Both builders here are exact, over the uniform grid in
//! [`grid`](super::grid) — no approximation, so no recall gaps to make a graph
//! irreproducible. Both are checked against brute force on degenerate inputs.
//!
//! On Visium data, pass [`visium_isometric_coords`](super::visium_isometric_coords) rather
//! than raw lattice coordinates; see the module docs for why.

use rayon::prelude::*;

use crate::error::{ClusteringError, Result};
use crate::network::{CSRNetwork, MAX_NODES};
use crate::spatial::grid::{Grid, distance_sq};

/// How to reconcile the two directions of a k-nearest-neighbour relation.
///
/// kNN is directed — `j` being among `i`'s nearest does not make `i` among `j`'s — but the
/// quality functions need a symmetric graph, so one of these has to be chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Symmetry {
    /// Keep an edge if either endpoint chose the other. Degrees come out at or above `k`.
    /// This is what scanpy and squidpy do.
    #[default]
    Union,
    /// Keep an edge only if both endpoints chose each other. Degrees come out at or below
    /// `k`, and the graph is sparser and more conservative — hub cells stop dragging in
    /// distant neighbours.
    Intersection,
}

/// Weight to give an edge, as a function of its length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpatialWeight {
    /// Every edge weighs 1.0. Adjacency alone carries the signal.
    Uniform,
    /// `exp(-d² / (2σ²))`, so nearby cells count for more. `sigma` is in the same units as
    /// the coordinates.
    Gaussian {
        /// Kernel bandwidth.
        sigma: f64,
    },
}

impl SpatialWeight {
    #[inline]
    fn apply(self, distance_sq: f64) -> f64 {
        match self {
            Self::Uniform => 1.0,
            Self::Gaussian { sigma } => (-distance_sq / (2.0 * sigma * sigma)).exp(),
        }
    }

    fn validate(self) -> Result<()> {
        match self {
            Self::Gaussian { sigma } if !sigma.is_finite() || sigma <= 0.0 => {
                Err(ClusteringError::InvalidConfig(format!(
                    "gaussian sigma must be finite and positive, got {sigma}"
                )))
            }
            _ => Ok(()),
        }
    }
}

/// A neighbour and the square of its distance, kept squared to avoid a pointless `sqrt`.
type Candidate = (u32, f64);

fn validate_points<const D: usize>(points: &[[f64; D]]) -> Result<()> {
    if points.len() > MAX_NODES {
        return Err(ClusteringError::TooManyNodes {
            n_nodes: points.len(),
        });
    }
    for (i, p) in points.iter().enumerate() {
        if let Some(bad) = p.iter().find(|c| !c.is_finite()) {
            return Err(ClusteringError::InvalidConfig(format!(
                "point {i} has a non-finite coordinate {bad}"
            )));
        }
    }
    Ok(())
}

/// Joins every pair of points closer than `radius`.
///
/// The relation is symmetric by construction, so no reconciliation step is needed and the
/// result does not depend on any tie-breaking. `radius` is in coordinate units — for spatial
/// assays that means it is directly interpretable, e.g. 30 µm.
///
/// Degree is unbounded: in dense tissue a generous radius can give a cell thousands of
/// neighbours, and cost downstream is linear in degree. `max_degree` caps it by keeping only
/// the nearest, but note the cap is applied **before** symmetrisation, so a node can still
/// finish slightly above it when a neighbour kept an edge it dropped.
///
/// # Errors
///
/// If `radius` is not finite and positive, a coordinate is not finite, or there are more
/// points than `u32` adjacency can address.
pub fn radius_graph<const D: usize>(
    points: &[[f64; D]],
    radius: f64,
    weight: SpatialWeight,
    max_degree: Option<usize>,
) -> Result<CSRNetwork> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(ClusteringError::InvalidConfig(format!(
            "radius must be finite and positive, got {radius}"
        )));
    }
    validate_points(points)?;
    weight.validate()?;
    if points.is_empty() {
        return CSRNetwork::from_csr_parts(vec![0], Vec::new(), Vec::new(), None);
    }

    // Cells the size of the radius, so everything in range sits in the 3^D block around a
    // point and one ring is all that ever needs visiting.
    let grid = Grid::new(points, radius);
    let r2 = radius * radius;

    let mut adjacency: Vec<Vec<Candidate>> = points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let centre = grid.cell_of(p);
            let reach = (radius / grid.cell_size()).ceil() as usize;
            let mut found: Vec<Candidate> = Vec::new();
            for ring in 0..=reach.min(grid.max_ring(&centre)) {
                grid.for_each_in_ring(&centre, ring, |j| {
                    if j as usize != i {
                        let d2 = distance_sq(p, &points[j as usize]);
                        if d2 <= r2 {
                            found.push((j, d2));
                        }
                    }
                });
            }
            // Sorted by (distance, id) so a capped result never depends on visit order —
            // points at equal distance are the norm on regular tissue.
            found.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
            if let Some(cap) = max_degree {
                found.truncate(cap);
            }
            found
        })
        .collect();

    // Capping breaks symmetry, so restore it. Without a cap this is already a no-op.
    if max_degree.is_some() {
        symmetrise(&mut adjacency, Symmetry::Union);
    }
    assemble(adjacency, weight)
}

/// Joins every point to its `k` nearest neighbours, then makes the relation symmetric.
///
/// Ties are broken on the lower node index, so a fixed input gives a fixed graph even where
/// many points sit at identical coordinates — which happens whenever coordinates have been
/// rounded to a pixel grid.
///
/// # Errors
///
/// If `k` is zero, a coordinate is not finite, or there are more points than `u32` adjacency
/// can address.
pub fn knn_graph<const D: usize>(
    points: &[[f64; D]],
    k: usize,
    symmetry: Symmetry,
    weight: SpatialWeight,
) -> Result<CSRNetwork> {
    if k == 0 {
        return Err(ClusteringError::InvalidConfig(
            "k must be at least 1".into(),
        ));
    }
    validate_points(points)?;
    weight.validate()?;
    if points.is_empty() {
        return CSRNetwork::from_csr_parts(vec![0], Vec::new(), Vec::new(), None);
    }

    // Aim for a handful of points per cell, so the first ring or two usually suffices.
    let grid = Grid::new(points, typical_spacing(points, k));

    let mut adjacency: Vec<Vec<Candidate>> = points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let centre = grid.cell_of(p);
            let limit = grid.max_ring(&centre);
            let mut found: Vec<Candidate> = Vec::with_capacity(k * 2);

            for ring in 0..=limit {
                grid.for_each_in_ring(&centre, ring, |j| {
                    if j as usize != i {
                        found.push((j, distance_sq(p, &points[j as usize])));
                    }
                });
                if found.len() < k {
                    continue;
                }
                // Anything still unvisited sits at Chebyshev distance >= ring + 1, so it is
                // at least `ring * cell` away — a point can be anywhere inside its own cell,
                // which costs one ring of slack. Once the k-th best beats that, stop.
                found.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
                found.truncate(k.max(1).min(found.len()));
                let safe = ring as f64 * grid.cell_size();
                if found[found.len() - 1].1 <= safe * safe {
                    break;
                }
            }

            found.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
            found.truncate(k);
            found
        })
        .collect();

    symmetrise(&mut adjacency, symmetry);
    assemble(adjacency, weight)
}

/// A cell size that puts roughly `k` points in each cell.
fn typical_spacing<const D: usize>(points: &[[f64; D]], k: usize) -> f64 {
    let mut min = [f64::INFINITY; D];
    let mut max = [f64::NEG_INFINITY; D];
    for p in points {
        for d in 0..D {
            min[d] = min[d].min(p[d]);
            max[d] = max[d].max(p[d]);
        }
    }
    let mut volume = 1.0f64;
    for d in 0..D {
        volume *= (max[d] - min[d]).max(f64::MIN_POSITIVE);
    }
    let per_cell = (k.max(1) as f64).max(1.0);
    let cells = (points.len() as f64 / per_cell).max(1.0);
    (volume / cells).powf(1.0 / D as f64).max(f64::MIN_POSITIVE)
}

/// Rewrites `adjacency` in place so `j ∈ adj[i]` exactly when `i ∈ adj[j]`.
///
/// Done explicitly rather than by emitting both directions and letting construction merge
/// them: duplicates are *summed* there, which would quietly give mutual pairs twice the
/// weight of one-sided ones.
fn symmetrise(adjacency: &mut [Vec<Candidate>], symmetry: Symmetry) {
    let n = adjacency.len();

    // Reverse lists via counting sort — O(total edges), and no per-node allocation.
    let mut counts = vec![0u32; n + 1];
    for list in adjacency.iter() {
        for &(u, _) in list {
            counts[u as usize + 1] += 1;
        }
    }
    for i in 0..n {
        counts[i + 1] += counts[i];
    }
    let mut reverse = vec![(0u32, 0.0f64); counts[n] as usize];
    let mut cursor: Vec<u32> = counts[..n].to_vec();
    for (v, list) in adjacency.iter().enumerate() {
        for &(u, d) in list {
            reverse[cursor[u as usize] as usize] = (v as u32, d);
            cursor[u as usize] += 1;
        }
    }

    for (v, list) in adjacency.iter_mut().enumerate() {
        let incoming = &reverse[counts[v] as usize..counts[v + 1] as usize];
        match symmetry {
            Symmetry::Union => {
                list.extend_from_slice(incoming);
                list.sort_unstable_by_key(|&(u, _)| u);
                list.dedup_by_key(|&mut (u, _)| u);
            }
            Symmetry::Intersection => {
                let mut mutual: Vec<u32> = incoming.iter().map(|&(u, _)| u).collect();
                mutual.sort_unstable();
                list.retain(|&(u, _)| mutual.binary_search(&u).is_ok());
                list.sort_unstable_by_key(|&(u, _)| u);
            }
        }
    }
}

/// Flattens symmetric adjacency lists into a graph.
fn assemble(adjacency: Vec<Vec<Candidate>>, weight: SpatialWeight) -> Result<CSRNetwork> {
    let n = adjacency.len();
    let mut node_ptrs = Vec::with_capacity(n + 1);
    node_ptrs.push(0usize);
    let mut total = 0usize;
    for list in &adjacency {
        total += list.len();
        node_ptrs.push(total);
    }

    let mut neighbors = Vec::with_capacity(total);
    let mut weights = Vec::with_capacity(total);
    for list in &adjacency {
        for &(u, d2) in list {
            neighbors.push(u);
            weights.push(weight.apply(d2) as f32);
        }
    }

    CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    fn grid(side: usize) -> Vec<[f64; 2]> {
        (0..side)
            .flat_map(|r| (0..side).map(move |c| [c as f64, r as f64]))
            .collect()
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

    /// On a unit grid the radius picks out a known connectivity, so this is exact.
    #[test]
    fn radius_on_a_unit_grid_is_exactly_the_expected_connectivity() {
        let pts = grid(6);
        let four = radius_graph(&pts, 1.01, SpatialWeight::Uniform, None).unwrap();
        let eight = radius_graph(&pts, 1.5, SpatialWeight::Uniform, None).unwrap();

        // 6x6 grid: 4-connected has 2*6*5 = 60 edges, 8-connected adds 2*5*5 = 50 diagonals.
        assert_eq!(four.edge_count(), 60);
        assert_eq!(eight.edge_count(), 110);
        assert!(edge_set(&four).is_subset(&edge_set(&eight)));

        for (v, p) in pts.iter().enumerate() {
            if p[0] > 0.0 && p[0] < 5.0 && p[1] > 0.0 && p[1] < 5.0 {
                assert_eq!(four.degree(v), 4);
                assert_eq!(eight.degree(v), 8);
            }
        }
    }

    /// A radius relation is symmetric before anything is done to it, so this must hold with
    /// no reconciliation step at all.
    #[test]
    fn radius_is_symmetric_without_reconciliation() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(4);
        let pts: Vec<[f64; 2]> = (0..400)
            .map(|_| [rng.random::<f64>() * 20.0, rng.random::<f64>() * 20.0])
            .collect();
        let g = radius_graph(&pts, 2.0, SpatialWeight::Uniform, None).unwrap();
        g.validate_symmetry().unwrap();
        assert!(g.edge_count() > 0);
    }

    /// The cross-check that matters: the two builders must agree on Visium.
    #[test]
    fn isometric_radius_reproduces_the_exact_hex_lattice() {
        use crate::spatial::lattice::{Lattice, lattice_graph, visium_isometric_coords};

        let (mut rows, mut cols) = (Vec::new(), Vec::new());
        for r in 0..20u32 {
            for i in 0..16u32 {
                rows.push(r);
                cols.push(2 * i + (r % 2));
            }
        }
        let exact = lattice_graph(&rows, &cols, Lattice::VisiumHex).unwrap();

        let (x, y) = visium_isometric_coords(&rows, &cols, 100.0).unwrap();
        let pts: Vec<[f64; 2]> = x.iter().zip(&y).map(|(&a, &b)| [a, b]).collect();
        // Anything past the pitch but short of the next shell at pitch*sqrt(3).
        let searched = radius_graph(&pts, 110.0, SpatialWeight::Uniform, None).unwrap();

        assert_eq!(edge_set(&exact), edge_set(&searched));
    }

    #[test]
    fn knn_union_contains_intersection_and_both_are_symmetric() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
        let pts: Vec<[f64; 2]> = (0..300)
            .map(|_| [rng.random::<f64>() * 50.0, rng.random::<f64>() * 50.0])
            .collect();

        let u = knn_graph(&pts, 6, Symmetry::Union, SpatialWeight::Uniform).unwrap();
        let i = knn_graph(&pts, 6, Symmetry::Intersection, SpatialWeight::Uniform).unwrap();
        u.validate_symmetry().unwrap();
        i.validate_symmetry().unwrap();

        assert!(edge_set(&i).is_subset(&edge_set(&u)));
        assert!(
            i.edge_count() < u.edge_count(),
            "random points give some one-sided pairs"
        );

        for v in 0..pts.len() {
            assert!(u.degree(v) >= 6, "union degree is at least k");
            assert!(i.degree(v) <= 6, "intersection degree is at most k");
        }
    }

    /// Coordinates rounded to a pixel grid produce many exact ties; those must not make the
    /// result depend on kd-tree traversal order.
    #[test]
    fn duplicate_points_still_give_a_deterministic_graph() {
        let mut pts = Vec::new();
        for i in 0..40 {
            // Four coincident points at each of ten sites.
            for _ in 0..4 {
                pts.push([(i % 10) as f64, 0.0]);
            }
        }
        let first = knn_graph(&pts, 5, Symmetry::Union, SpatialWeight::Uniform).unwrap();
        for _ in 0..8 {
            let again = knn_graph(&pts, 5, Symmetry::Union, SpatialWeight::Uniform).unwrap();
            assert_eq!(edge_set(&first), edge_set(&again));
        }
        assert!(
            !first.has_self_loops(),
            "a point must not be its own neighbour"
        );
    }

    #[test]
    fn gaussian_weights_fall_off_with_distance() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let g = radius_graph(&pts, 2.5, SpatialWeight::Gaussian { sigma: 1.0 }, None).unwrap();

        let w = |a: usize, b: usize| g.neighbors(a).find(|&(u, _)| u == b).unwrap().1;
        // exp(-1/2) and exp(-4/2), to f32 precision.
        assert!((w(0, 1) - (-0.5f64).exp()).abs() < 1e-6, "got {}", w(0, 1));
        assert!((w(0, 2) - (-2.0f64).exp()).abs() < 1e-6, "got {}", w(0, 2));
        assert!(w(0, 1) > w(0, 2));
        assert_eq!(w(0, 1), w(1, 0), "the kernel must be symmetric");
    }

    #[test]
    fn max_degree_bounds_the_hubs() {
        // One dense clump: without a cap, every point sees every other.
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(2);
        let pts: Vec<[f64; 2]> = (0..200)
            .map(|_| [rng.random::<f64>(), rng.random::<f64>()])
            .collect();

        let uncapped = radius_graph(&pts, 5.0, SpatialWeight::Uniform, None).unwrap();
        let capped = radius_graph(&pts, 5.0, SpatialWeight::Uniform, Some(10)).unwrap();
        assert_eq!(uncapped.degree(0), 199, "everything is within the radius");
        capped.validate_symmetry().unwrap();
        assert!(capped.edge_count() < uncapped.edge_count() / 5);
        // Documented as a soft cap: symmetrisation can push a node back over it.
        for v in 0..pts.len() {
            assert!(capped.degree(v) >= 10, "each node keeps its own 10");
        }
    }

    #[test]
    fn works_in_three_dimensions() {
        let pts: Vec<[f64; 3]> = (0..4)
            .flat_map(|z| {
                (0..4).flat_map(move |y| (0..4).map(move |x| [x as f64, y as f64, z as f64]))
            })
            .collect();
        let g = radius_graph(&pts, 1.01, SpatialWeight::Uniform, None).unwrap();
        g.validate_symmetry().unwrap();
        // 4x4x4 grid, 6-connected: 3 * 4 * 4 * 3 = 144 edges.
        assert_eq!(g.edge_count(), 144);
        let interior = (0..pts.len())
            .find(|&v| pts[v].iter().all(|&c| c > 0.0 && c < 3.0))
            .unwrap();
        assert_eq!(g.degree(interior), 6);
    }

    #[test]
    fn node_order_does_not_change_the_graph() {
        use rand::seq::SliceRandom;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
        let pts: Vec<[f64; 2]> = (0..250)
            .map(|_| [rng.random::<f64>() * 30.0, rng.random::<f64>() * 30.0])
            .collect();
        let reference: HashSet<(u64, u64)> = {
            let g = knn_graph(&pts, 5, Symmetry::Union, SpatialWeight::Uniform).unwrap();
            edge_set(&g)
                .into_iter()
                .map(|(a, b)| (a as u64, b as u64))
                .collect()
        };

        for _ in 0..5 {
            let mut perm: Vec<usize> = (0..pts.len()).collect();
            perm.shuffle(&mut rng);
            let shuffled: Vec<[f64; 2]> = perm.iter().map(|&i| pts[i]).collect();
            let g = knn_graph(&shuffled, 5, Symmetry::Union, SpatialWeight::Uniform).unwrap();
            // Map back through the permutation and compare as sets.
            let mapped: HashSet<(u64, u64)> = edge_set(&g)
                .into_iter()
                .map(|(a, b)| {
                    let (a, b) = (perm[a] as u64, perm[b] as u64);
                    (a.min(b), a.max(b))
                })
                .collect();
            assert_eq!(mapped, reference);
        }
    }

    /// `KdTree` panics past 32 items sharing a value on one axis. Spatial data does that as a
    /// matter of course, so this covers the shapes that actually occur.
    #[test]
    fn survives_heavy_axis_alignment() {
        // A full Visium row is 64 spots at one y; a capture area is 78 of them.
        let mut visium = Vec::new();
        for r in 0..78 {
            for i in 0..64 {
                visium.push([(2 * i + (r % 2)) as f64 * 50.0, r as f64 * 86.6]);
            }
        }
        let g = knn_graph(&visium, 6, Symmetry::Union, SpatialWeight::Uniform).unwrap();
        g.validate_symmetry().unwrap();
        assert_eq!(g.node_count(), 4992);

        // A whole HD-sized row at a single y.
        let row: Vec<[f64; 2]> = (0..3250).map(|i| [i as f64, 0.0]).collect();
        let g = radius_graph(&row, 1.5, SpatialWeight::Uniform, None).unwrap();
        assert_eq!(g.edge_count(), 3249, "a path along the row");

        // Every point identical — the worst case for any tree.
        let same = vec![[7.0, 7.0]; 200];
        for sym in [Symmetry::Union, Symmetry::Intersection] {
            let g = knn_graph(&same, 5, sym, SpatialWeight::Uniform).unwrap();
            g.validate_symmetry()
                .unwrap_or_else(|e| panic!("{sym:?}: {e}"));
            assert!(!g.has_self_loops());
        }
    }

    /// Point sets covering the shapes that break spatial indices: duplicates, collinear runs,
    /// tight clumps, near-empty space.
    fn awkward_point_sets(seed: u64) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut out: Vec<(String, Vec<[f64; 2]>)> = Vec::new();

        out.push((
            "uniform".into(),
            (0..300)
                .map(|_| [rng.random::<f64>() * 40.0, rng.random::<f64>() * 40.0])
                .collect(),
        ));
        out.push((
            "collinear".into(),
            (0..200).map(|i| [i as f64, 0.0]).collect(),
        ));
        out.push((
            "grid_with_ties".into(),
            (0..20)
                .flat_map(|r| (0..20).map(move |c| [c as f64, r as f64]))
                .collect(),
        ));
        out.push(("all_identical".into(), vec![[3.0, 4.0]; 150]));
        out.push((
            "duplicated_pairs".into(),
            (0..150).map(|i| [(i / 2) as f64, 0.0]).collect(),
        ));
        out.push((
            "two_far_clumps".into(),
            (0..200)
                .map(|i| {
                    let off = if i < 100 { 0.0 } else { 1000.0 };
                    [off + rng.random::<f64>(), rng.random::<f64>()]
                })
                .collect(),
        ));
        out.push((
            "one_dense_spike".into(),
            (0..250)
                .map(|i| {
                    if i < 200 {
                        [rng.random::<f64>() * 0.001, rng.random::<f64>() * 0.001]
                    } else {
                        [rng.random::<f64>() * 100.0, rng.random::<f64>() * 100.0]
                    }
                })
                .collect(),
        ));
        out
    }

    /// The radius search must agree with O(n²) brute force exactly, on every shape.
    #[test]
    fn radius_matches_brute_force() {
        for (name, pts) in awkward_point_sets(21) {
            for &radius in &[0.5f64, 1.5, 7.0] {
                let g = radius_graph(&pts, radius, SpatialWeight::Uniform, None).unwrap();
                let got = edge_set(&g);

                let r2 = radius * radius;
                let mut want = HashSet::new();
                for i in 0..pts.len() {
                    for j in (i + 1)..pts.len() {
                        let d2 = (pts[i][0] - pts[j][0]).powi(2) + (pts[i][1] - pts[j][1]).powi(2);
                        if d2 <= r2 {
                            want.insert((i, j));
                        }
                    }
                }
                assert_eq!(got, want, "{name} at radius {radius}");
            }
        }
    }

    /// The k-nearest search must pick exactly the same neighbours brute force would,
    /// including how it breaks ties.
    #[test]
    fn knn_matches_brute_force() {
        for (name, pts) in awkward_point_sets(22) {
            for &k in &[1usize, 3, 8] {
                // Compare pre-symmetrisation, which is where the search itself is decided.
                let mut want: Vec<Vec<u32>> = Vec::with_capacity(pts.len());
                for i in 0..pts.len() {
                    let mut all: Vec<(f64, u32)> = (0..pts.len())
                        .filter(|&j| j != i)
                        .map(|j| {
                            let d2 =
                                (pts[i][0] - pts[j][0]).powi(2) + (pts[i][1] - pts[j][1]).powi(2);
                            (d2, j as u32)
                        })
                        .collect();
                    all.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
                    all.truncate(k);
                    let mut ids: Vec<u32> = all.into_iter().map(|(_, j)| j).collect();
                    ids.sort_unstable();
                    want.push(ids);
                }

                // Intersection keeps only mutual pairs, so recover the directed lists by
                // asking for the union and checking containment both ways instead.
                let g = knn_graph(&pts, k, Symmetry::Union, SpatialWeight::Uniform).unwrap();
                for (i, truth) in want.iter().enumerate() {
                    let neigh: HashSet<usize> = g.neighbors(i).map(|(u, _)| u).collect();
                    for &j in truth {
                        assert!(
                            neigh.contains(&(j as usize)),
                            "{name} k={k}: node {i} lost its true neighbour {j}"
                        );
                    }
                }
                // And every union edge must be justified by one side's true k-nearest.
                for (i, truth) in want.iter().enumerate() {
                    for (u, _) in g.neighbors(i) {
                        assert!(
                            truth.contains(&(u as u32)) || want[u].contains(&(i as u32)),
                            "{name} k={k}: edge {i}-{u} is in neither node's k-nearest"
                        );
                    }
                }
            }
        }
    }

    /// Xenium-scale: a few hundred thousand cells over a tissue section.
    ///
    /// ```text
    /// cargo test --release --no-default-features xenium_scale -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "takes a few seconds"]
    fn xenium_scale() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let n = 500_000;
        // ~12 mm square at roughly Xenium cell density.
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|_| {
                [
                    rng.random::<f64>() * 12_000.0,
                    rng.random::<f64>() * 12_000.0,
                ]
            })
            .collect();

        let t = std::time::Instant::now();
        let g = knn_graph(&pts, 6, Symmetry::Union, SpatialWeight::Uniform).unwrap();
        let knn = t.elapsed();

        let t = std::time::Instant::now();
        let r = radius_graph(&pts, 30.0, SpatialWeight::Uniform, None).unwrap();
        let rad = t.elapsed();

        println!(
            "{n} cells\n  knn(6)      {:.2?}  {} edges\n  radius(30)  {:.2?}  {} edges",
            knn,
            g.edge_count(),
            rad,
            r.edge_count()
        );
        g.validate_symmetry().unwrap();
        r.validate_symmetry().unwrap();
    }

    #[test]
    fn rejects_bad_input() {
        let pts = grid(3);
        assert!(radius_graph(&pts, 0.0, SpatialWeight::Uniform, None).is_err());
        assert!(radius_graph(&pts, f64::NAN, SpatialWeight::Uniform, None).is_err());
        assert!(knn_graph(&pts, 0, Symmetry::Union, SpatialWeight::Uniform).is_err());
        assert!(radius_graph(&pts, 1.0, SpatialWeight::Gaussian { sigma: 0.0 }, None).is_err());
        let bad = vec![[0.0, 0.0], [f64::NAN, 1.0]];
        assert!(radius_graph(&bad, 1.0, SpatialWeight::Uniform, None).is_err());
    }

    #[test]
    fn handles_degenerate_input() {
        let none: Vec<[f64; 2]> = Vec::new();
        assert_eq!(
            radius_graph(&none, 1.0, SpatialWeight::Uniform, None)
                .unwrap()
                .node_count(),
            0
        );
        let one = vec![[1.0, 2.0]];
        let g = knn_graph(&one, 5, Symmetry::Union, SpatialWeight::Uniform).unwrap();
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);

        // Nothing within the radius.
        let far = vec![[0.0, 0.0], [1000.0, 0.0]];
        let g = radius_graph(&far, 1.0, SpatialWeight::Uniform, None).unwrap();
        assert_eq!(g.edge_count(), 0);
    }
}
