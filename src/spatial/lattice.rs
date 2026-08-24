//! Neighbour graphs for observations on a regular lattice.
//!
//! Visium spots and HD bins sit on a known grid, so neighbours are integer arithmetic — no
//! spatial index, nothing approximate. At HD scale that's seconds instead of minutes.
//!
//! `squidpy.gr.spatial_neighbors` instead runs Euclidean kNN on pixel coordinates with a
//! global cutoff, which bridges gaps on a sparse lattice: at 50% occupancy about half its
//! edges aren't lattice-adjacent, at 20% three quarters.
//!
//! Never run Euclidean kNN on raw doubled coordinates. In-row neighbours are 2 units apart
//! and diagonal ones only √2, so a metric method drops the in-row pair and hands back a
//! diagonal-only graph that still clusters plausibly. [`visium_isometric_coords`] rescales so
//! all six are equidistant.

use crate::error::{ClusteringError, Result};
use crate::network::{CSRNetwork, MAX_NODES};

/// The lattice the coordinates describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lattice {
    /// Visium's hex grid in doubled coordinates: `col` steps by 2 within a row, and odd rows
    /// are offset by 1, so a spot exists only where `(row + col)` is even. Six neighbours:
    /// `(row, col ± 2)` and `(row ± 1, col ± 1)`.
    VisiumHex,
    /// Square grid, edge-adjacent only. Four neighbours: `(row ± 1, col)`, `(row, col ± 1)`.
    Square4,
    /// Square grid including diagonals. Eight neighbours.
    Square8,
}

impl Lattice {
    /// Half the offsets, so each undirected edge is emitted once by its lower endpoint.
    const fn forward_offsets(self) -> &'static [(i64, i64)] {
        match self {
            Self::VisiumHex => &[(0, 2), (1, -1), (1, 1)],
            Self::Square4 => &[(0, 1), (1, 0)],
            Self::Square8 => &[(0, 1), (1, 0), (1, 1), (1, -1)],
        }
    }

    /// Neighbours an interior position has. Docs and tests only.
    pub const fn degree(self) -> usize {
        match self {
            Self::VisiumHex => 6,
            Self::Square4 => 4,
            Self::Square8 => 8,
        }
    }
}

/// Where each distinct row sits in the sorted node order.
struct RowIndex {
    /// Distinct row values, ascending.
    rows: Vec<u32>,
    /// `starts[i]..starts[i + 1]` is row `rows[i]`'s slice of the sorted order.
    starts: Vec<u32>,
}

impl RowIndex {
    /// Slice of the sorted order holding `row`, or `None` if the row is empty.
    #[inline]
    fn range(&self, row: u32) -> Option<std::ops::Range<usize>> {
        let i = self.rows.binary_search(&row).ok()?;
        Some(self.starts[i] as usize..self.starts[i + 1] as usize)
    }
}

/// Builds a neighbour graph from integer lattice coordinates.
///
/// Every edge has weight 1.0. Absent positions — masked spots, trimmed bins — just have no
/// edge, so a partial capture area needs no special handling.
///
/// The graph is exactly the lattice adjacency, so Leiden's internal-connectivity guarantee
/// becomes spatial contiguity.
///
/// # Errors
///
/// If the arrays differ in length, two observations share a position, or there are more
/// nodes than `u32` adjacency can address.
pub fn lattice_graph(rows: &[u32], cols: &[u32], lattice: Lattice) -> Result<CSRNetwork> {
    let n = rows.len();
    if cols.len() != n {
        return Err(ClusteringError::CoordinateLengthMismatch {
            got: cols.len(),
            expected: n,
        });
    }
    if n > MAX_NODES {
        return Err(ClusteringError::TooManyNodes { n_nodes: n });
    }
    if n == 0 {
        return CSRNetwork::from_csr_parts(vec![0], Vec::new(), Vec::new(), None);
    }

    // sorted by (row, col), so a lookup is a binary search inside one row
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&v| key(rows[v as usize], cols[v as usize]));

    // held separately so the search reads contiguous memory. `cols[order[i]]` gathers
    // randomly instead: 1.21s vs 954ms on 10.5M bins.
    let sorted_cols: Vec<u32> = order.iter().map(|&v| cols[v as usize]).collect();

    for (i, w) in order.windows(2).enumerate() {
        let (a, b) = (w[0] as usize, w[1] as usize);
        if rows[a] == rows[b] && sorted_cols[i] == sorted_cols[i + 1] {
            return Err(ClusteringError::DuplicateLatticePosition {
                row: rows[a],
                col: cols[a],
                nodes: (a.min(b), a.max(b)),
            });
        }
    }

    if lattice == Lattice::VisiumHex {
        check_hex_parity(rows, cols)?;
    }

    let index = build_row_index(&order, rows);
    let offsets = lattice.forward_offsets();

    // two passes so the CSR arrays are allocated once. An edge list first would cost
    // ~700 MB at HD scale.
    let mut degree = vec![0u32; n];
    visit_edges(&order, rows, cols, &sorted_cols, &index, offsets, |v, u| {
        degree[v] += 1;
        degree[u] += 1;
    });

    let mut node_ptrs = Vec::with_capacity(n + 1);
    node_ptrs.push(0usize);
    let mut total = 0usize;
    for &d in &degree {
        total += d as usize;
        node_ptrs.push(total);
    }

    let mut neighbors = vec![0u32; total];
    let mut weights = vec![1.0f32; total];
    let mut cursor: Vec<usize> = node_ptrs[..n].to_vec();
    visit_edges(&order, rows, cols, &sorted_cols, &index, offsets, |v, u| {
        neighbors[cursor[v]] = u as u32;
        cursor[v] += 1;
        neighbors[cursor[u]] = v as u32;
        cursor[u] += 1;
    });
    weights.truncate(total);

    CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None)
}

/// Packs a position into one sortable integer. `i64` so negative offsets don't wrap.
#[inline]
const fn key(row: u32, col: u32) -> u64 {
    ((row as u64) << 32) | col as u64
}

/// Rejects coordinates that aren't doubled.
///
/// Only consistency is required, so a shifted origin is fine. Without this, passing `col / 2`
/// gives a wrong graph rather than an empty one — the offsets still find something.
fn check_hex_parity(rows: &[u32], cols: &[u32]) -> Result<()> {
    let parity = (rows[0] + cols[0]) % 2;
    for v in 1..rows.len() {
        if (rows[v] + cols[v]) % 2 != parity {
            return Err(ClusteringError::InconsistentHexParity {
                expected_node: 0,
                node: v,
                position: (rows[v], cols[v]),
            });
        }
    }
    Ok(())
}

/// Rescales Visium doubled coordinates so all six neighbours sit at the same distance.
///
/// Hand these to anything distance-based. `pitch` is centre-to-centre spot spacing, 100.0 µm
/// on standard Visium; output is in the same units and every neighbour lands exactly `pitch`
/// away.
///
/// # Errors
///
/// If the arrays differ in length.
pub fn visium_isometric_coords(
    rows: &[u32],
    cols: &[u32],
    pitch: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if cols.len() != rows.len() {
        return Err(ClusteringError::CoordinateLengthMismatch {
            got: cols.len(),
            expected: rows.len(),
        });
    }
    // col carries the doubling, so pitch/2; row spacing is the hex height
    let (sx, sy) = (pitch / 2.0, pitch * f64::sqrt(3.0) / 2.0);
    Ok((
        cols.iter().map(|&c| c as f64 * sx).collect(),
        rows.iter().map(|&r| r as f64 * sy).collect(),
    ))
}

fn build_row_index(order: &[u32], rows: &[u32]) -> RowIndex {
    let mut index = RowIndex {
        rows: Vec::new(),
        starts: Vec::new(),
    };
    for (pos, &v) in order.iter().enumerate() {
        let r = rows[v as usize];
        if index.rows.last() != Some(&r) {
            index.rows.push(r);
            index.starts.push(pos as u32);
        }
    }
    index.starts.push(order.len() as u32);
    index
}

/// Calls `f(v, u)` once per undirected edge, in the caller's numbering. Identical in both
/// passes, so degrees and fill can't disagree.
#[inline]
fn visit_edges(
    order: &[u32],
    rows: &[u32],
    cols: &[u32],
    sorted_cols: &[u32],
    index: &RowIndex,
    offsets: &[(i64, i64)],
    mut f: impl FnMut(usize, usize),
) {
    for &v in order {
        let v = v as usize;
        let (r, c) = (rows[v] as i64, cols[v] as i64);
        for &(dr, dc) in offsets {
            let (tr, tc) = (r + dr, c + dc);
            if tr < 0 || tc < 0 || tr > u32::MAX as i64 || tc > u32::MAX as i64 {
                continue;
            }
            if let Some(u) = lookup(order, sorted_cols, index, tr as u32, tc as u32) {
                f(v, u);
            }
        }
    }
}

/// Node at `(row, col)`, or `None` if nothing sits there.
#[inline]
fn lookup(
    order: &[u32],
    sorted_cols: &[u32],
    index: &RowIndex,
    row: u32,
    col: u32,
) -> Option<usize> {
    let range = index.range(row)?;
    // rows are short (128 on Visium, ~3250 on HD), so this stays in L1
    let i = sorted_cols[range.clone()].binary_search(&col).ok()?;
    Some(order[range.start + i] as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::seq::SliceRandom;
    use std::collections::HashSet;

    /// Every position in an R x C block.
    fn full_square(r: u32, c: u32) -> (Vec<u32>, Vec<u32>) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..r {
            for j in 0..c {
                rows.push(i);
                cols.push(j);
            }
        }
        (rows, cols)
    }

    /// Visium's layout: `(row + col)` even, `col` stepping by 2 within a row.
    fn full_hex(r: u32, c: u32) -> (Vec<u32>, Vec<u32>) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..r {
            for j in 0..c {
                if (i + j) % 2 == 0 {
                    rows.push(i);
                    cols.push(j);
                }
            }
        }
        (rows, cols)
    }

    /// Edges as coordinate pairs, so orderings can be compared.
    fn edge_set(g: &CSRNetwork, rows: &[u32], cols: &[u32]) -> HashSet<((u32, u32), (u32, u32))> {
        let mut out = HashSet::new();
        for v in 0..g.node_count() {
            for (u, _) in g.neighbors(v) {
                out.insert(canon((rows[v], cols[v]), (rows[u], cols[u])));
            }
        }
        out
    }

    /// Orders endpoints so both directions collapse to one entry.
    fn canon(a: (u32, u32), b: (u32, u32)) -> ((u32, u32), (u32, u32)) {
        if a <= b { (a, b) } else { (b, a) }
    }

    #[test]
    fn square4_3x3_has_exactly_the_lattice_edges() {
        let (rows, cols) = full_square(3, 3);
        let g = lattice_graph(&rows, &cols, Lattice::Square4).unwrap();

        let mut expected = HashSet::new();
        for r in 0..3u32 {
            for c in 0..3u32 {
                if c + 1 < 3 {
                    expected.insert(canon((r, c), (r, c + 1)));
                }
                if r + 1 < 3 {
                    expected.insert(canon((r, c), (r + 1, c)));
                }
            }
        }
        assert_eq!(edge_set(&g, &rows, &cols), expected);
        assert_eq!(g.edge_count(), 12, "3x3 grid: 6 horizontal + 6 vertical");
    }

    #[test]
    fn square8_3x3_adds_exactly_the_diagonals() {
        let (rows, cols) = full_square(3, 3);
        let four = lattice_graph(&rows, &cols, Lattice::Square4).unwrap();
        let eight = lattice_graph(&rows, &cols, Lattice::Square8).unwrap();
        let (a, b) = (
            edge_set(&four, &rows, &cols),
            edge_set(&eight, &rows, &cols),
        );
        assert!(a.is_subset(&b));
        assert_eq!(b.len() - a.len(), 8, "2x2 blocks x 2 diagonals each");
    }

    /// Closed form for a full grid, over a range of shapes.
    #[test]
    fn square_edge_counts_match_the_closed_form() {
        for (r, c) in [(1, 1), (1, 7), (7, 1), (2, 2), (5, 9), (16, 16)] {
            let (rows, cols) = full_square(r, c);
            let g4 = lattice_graph(&rows, &cols, Lattice::Square4).unwrap();
            assert_eq!(
                g4.edge_count() as u32,
                r * (c - 1) + (r - 1) * c,
                "Square4 {r}x{c}"
            );
            let g8 = lattice_graph(&rows, &cols, Lattice::Square8).unwrap();
            assert_eq!(
                g8.edge_count() as u32,
                r * (c - 1) + (r - 1) * c + 2 * (r - 1) * (c - 1),
                "Square8 {r}x{c}"
            );
        }
    }

    /// Duplicates are summed on construction, so a double emission shows up as weight 2.0.
    #[test]
    fn every_edge_has_weight_exactly_one() {
        for lattice in [Lattice::Square4, Lattice::Square8, Lattice::VisiumHex] {
            let (rows, cols) = match lattice {
                Lattice::VisiumHex => full_hex(9, 13),
                _ => full_square(9, 13),
            };
            let g = lattice_graph(&rows, &cols, lattice).unwrap();
            for v in 0..g.node_count() {
                for (u, w) in g.neighbors(v) {
                    assert_eq!(w, 1.0, "{lattice:?}: edge {v}-{u} has weight {w}");
                }
            }
            assert!(!g.has_self_loops(), "{lattice:?} produced a self-loop");
        }
    }

    #[test]
    fn interior_degree_matches_the_lattice() {
        for lattice in [Lattice::Square4, Lattice::Square8, Lattice::VisiumHex] {
            let (r, c) = (11u32, 15u32);
            let (rows, cols) = match lattice {
                Lattice::VisiumHex => full_hex(r, c),
                _ => full_square(r, c),
            };
            let g = lattice_graph(&rows, &cols, lattice).unwrap();
            // hex reaches two columns sideways, so interior starts one further in
            let margin = if lattice == Lattice::VisiumHex { 2 } else { 1 };
            let mut checked = 0;
            for v in 0..g.node_count() {
                let (vr, vc) = (rows[v], cols[v]);
                if vr >= margin && vr < r - margin && vc >= margin && vc < c - margin {
                    assert_eq!(
                        g.degree(v),
                        lattice.degree(),
                        "{lattice:?}: interior spot ({vr}, {vc})"
                    );
                    checked += 1;
                }
            }
            assert!(checked > 10, "{lattice:?}: only {checked} interior spots");
        }
    }

    /// Neighbours of an even-parity spot are also even-parity.
    #[test]
    fn hex_neighbours_preserve_parity_and_are_the_six_offsets() {
        let (rows, cols) = full_hex(9, 13);
        let g = lattice_graph(&rows, &cols, Lattice::VisiumHex).unwrap();
        let expected: HashSet<(i64, i64)> =
            [(0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1)].into();

        for v in 0..g.node_count() {
            assert_eq!((rows[v] + cols[v]) % 2, 0, "spot {v} has odd parity");
            for (u, _) in g.neighbors(v) {
                let d = (
                    rows[u] as i64 - rows[v] as i64,
                    cols[u] as i64 - cols[v] as i64,
                );
                assert!(expected.contains(&d), "unexpected hex offset {d:?}");
                assert_eq!((rows[u] + cols[u]) % 2, 0);
            }
        }
    }

    /// Boundary spots, worked out by hand.
    #[test]
    fn hex_boundary_degrees_are_exact() {
        let (rows, cols) = full_hex(5, 9);
        let g = lattice_graph(&rows, &cols, Lattice::VisiumHex).unwrap();
        let at = |r: u32, c: u32| {
            (0..g.node_count())
                .find(|&v| rows[v] == r && cols[v] == c)
                .unwrap_or_else(|| panic!("no spot at ({r}, {c})"))
        };
        assert_eq!(g.degree(at(2, 4)), 6, "interior");
        assert_eq!(g.degree(at(0, 4)), 4, "top edge: no row above");
        assert_eq!(g.degree(at(2, 0)), 3, "left edge: nothing to the left");
        assert_eq!(g.degree(at(0, 0)), 2, "corner");
        assert_eq!(g.degree(at(4, 8)), 2, "opposite corner");
    }

    /// Masking removes spots; the rest just lose those edges.
    #[test]
    fn missing_positions_leave_holes_rather_than_shifting_neighbours() {
        let (rows, cols) = full_square(5, 5);
        let keep: Vec<usize> = (0..rows.len())
            .filter(|&i| !(rows[i] == 2 && cols[i] == 2))
            .collect();
        let r: Vec<u32> = keep.iter().map(|&i| rows[i]).collect();
        let c: Vec<u32> = keep.iter().map(|&i| cols[i]).collect();

        let g = lattice_graph(&r, &c, Lattice::Square4).unwrap();
        let full = lattice_graph(&rows, &cols, Lattice::Square4).unwrap();
        assert_eq!(g.node_count(), 24);
        assert_eq!(g.edge_count(), full.edge_count() - 4, "the hole's 4 edges");

        for v in 0..g.node_count() {
            for (u, _) in g.neighbors(v) {
                let d = (r[u] as i64 - r[v] as i64).abs() + (c[u] as i64 - c[v] as i64).abs();
                assert_eq!(d, 1, "a hole must not join spots two apart");
            }
        }
    }

    /// Node order is the caller's; the graph must not depend on it.
    #[test]
    fn result_is_invariant_under_input_permutation() {
        let (rows, cols) = full_hex(9, 13);
        let reference = {
            let g = lattice_graph(&rows, &cols, Lattice::VisiumHex).unwrap();
            edge_set(&g, &rows, &cols)
        };

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        for _ in 0..8 {
            let mut perm: Vec<usize> = (0..rows.len()).collect();
            perm.shuffle(&mut rng);
            let r: Vec<u32> = perm.iter().map(|&i| rows[i]).collect();
            let c: Vec<u32> = perm.iter().map(|&i| cols[i]).collect();
            let g = lattice_graph(&r, &c, Lattice::VisiumHex).unwrap();
            assert_eq!(edge_set(&g, &r, &c), reference);
        }
    }

    #[test]
    fn adjacency_is_symmetric() {
        for lattice in [Lattice::Square4, Lattice::Square8, Lattice::VisiumHex] {
            let (rows, cols) = match lattice {
                Lattice::VisiumHex => full_hex(12, 17),
                _ => full_square(12, 17),
            };
            let g = lattice_graph(&rows, &cols, lattice).unwrap();
            g.validate_symmetry()
                .unwrap_or_else(|e| panic!("{lattice:?}: {e}"));
        }
    }

    /// Real Visium geometry per Space Ranger docs: 78 rows, 64 spots each.
    #[test]
    fn matches_the_documented_visium_capture_area() {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for r in 0..78u32 {
            for i in 0..64u32 {
                rows.push(r);
                cols.push(2 * i + (r % 2));
            }
        }
        assert_eq!(rows.len(), 4992, "10x documents 4992 spots");

        let g = lattice_graph(&rows, &cols, Lattice::VisiumHex).unwrap();
        assert_eq!(g.node_count(), 4992);
        // interior spots: rows 1..77 with both in-row neighbours
        let interior = (0..g.node_count())
            .filter(|&v| rows[v] > 0 && rows[v] < 77 && cols[v] >= 2 && cols[v] <= 125)
            .count();
        assert!(interior > 4000);
        for v in 0..g.node_count() {
            if rows[v] > 0 && rows[v] < 77 && cols[v] >= 2 && cols[v] <= 125 {
                assert_eq!(g.degree(v), 6, "spot ({}, {})", rows[v], cols[v]);
            }
        }
    }

    /// Offset coordinates look valid but describe a different lattice.
    #[test]
    fn rejects_offset_coordinates_masquerading_as_doubled() {
        let (rows, doubled) = full_hex(9, 13);
        let offset: Vec<u32> = doubled.iter().map(|&c| c / 2).collect();
        assert!(
            matches!(
                lattice_graph(&rows, &offset, Lattice::VisiumHex),
                Err(ClusteringError::InconsistentHexParity { .. })
            ),
            "offset coordinates must not be silently accepted"
        );
        // square lattices have no parity constraint
        assert!(lattice_graph(&rows, &offset, Lattice::Square4).is_ok());
    }

    /// A shifted origin is still doubled.
    #[test]
    fn accepts_a_shifted_hex_origin() {
        let (rows, cols) = full_hex(9, 13);
        let shifted: Vec<u32> = cols.iter().map(|&c| c + 1).collect();
        let g = lattice_graph(&rows, &shifted, Lattice::VisiumHex).unwrap();
        let reference = lattice_graph(&rows, &cols, Lattice::VisiumHex).unwrap();
        assert_eq!(g.edge_count(), reference.edge_count());
    }

    /// All six neighbours land at exactly `pitch` once rescaled.
    #[test]
    fn isometric_coords_equalise_all_six_neighbours() {
        let (rows, cols) = full_hex(9, 13);
        let g = lattice_graph(&rows, &cols, Lattice::VisiumHex).unwrap();
        let (x, y) = visium_isometric_coords(&rows, &cols, 100.0).unwrap();

        for v in 0..g.node_count() {
            for (u, _) in g.neighbors(v) {
                let d = ((x[u] - x[v]).powi(2) + (y[u] - y[v]).powi(2)).sqrt();
                assert!(
                    (d - 100.0).abs() < 1e-9,
                    "neighbour distance {d}, expected the 100 µm pitch"
                );
            }
        }

        // and nothing else is that close, so radius at 1.1x pitch is exactly the lattice
        let v = (0..g.node_count())
            .find(|&v| rows[v] == 4 && cols[v] == 6)
            .unwrap();
        let within = (0..g.node_count())
            .filter(|&u| u != v && ((x[u] - x[v]).powi(2) + (y[u] - y[v]).powi(2)).sqrt() < 110.0)
            .count();
        assert_eq!(within, 6);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(matches!(
            lattice_graph(&[0, 1], &[0], Lattice::Square4),
            Err(ClusteringError::CoordinateLengthMismatch {
                got: 1,
                expected: 2
            })
        ));
        assert!(matches!(
            lattice_graph(&[3, 3], &[7, 7], Lattice::Square4),
            Err(ClusteringError::DuplicateLatticePosition { row: 3, col: 7, .. })
        ));
    }

    #[test]
    fn handles_degenerate_input() {
        let empty = lattice_graph(&[], &[], Lattice::Square4).unwrap();
        assert_eq!(empty.node_count(), 0);
        assert_eq!(empty.edge_count(), 0);

        let single = lattice_graph(&[5], &[9], Lattice::VisiumHex).unwrap();
        assert_eq!(single.node_count(), 1);
        assert_eq!(single.edge_count(), 0);

        // far apart, nothing adjacent
        let scattered = lattice_graph(&[0, 100, 200], &[0, 100, 200], Lattice::Square8).unwrap();
        assert_eq!(scattered.edge_count(), 0);
    }

    /// Internally-connected communities on a lattice means spatially contiguous domains.
    #[test]
    fn leiden_domains_are_spatially_contiguous() {
        use crate::community_search::leiden::{LeidenConfig, ObjectiveKind, leiden};

        let (rows, cols) = full_square(40, 40);
        let g = lattice_graph(&rows, &cols, Lattice::Square4).unwrap();
        for resolution in [0.05, 0.1, 0.5] {
            let c = leiden(
                &g,
                &LeidenConfig {
                    objective: ObjectiveKind::Cpm { resolution },
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                crate::testdata::disconnected_community_count(&g, c.labels()),
                0,
                "resolution {resolution} produced a spatially split domain"
            );
            assert!(
                c.n_clusters() > 1,
                "resolution {resolution} found one domain"
            );
        }
    }

    /// Top of the `u32` range must not wrap on a negative offset.
    #[test]
    fn extreme_coordinates_do_not_wrap() {
        let rows = [0u32, 0, u32::MAX, u32::MAX];
        let cols = [0u32, u32::MAX, 0, u32::MAX];
        let g = lattice_graph(&rows, &cols, Lattice::Square8).unwrap();
        assert_eq!(g.edge_count(), 0, "corners of the space are not adjacent");
    }

    /// Visium HD at 2 µm. Sources disagree on the extent (3350² vs 3250²); it's only a
    /// benchmark size, and the builder never assumes one.
    ///
    /// ```text
    /// cargo test --release --no-default-features visium_hd_scale -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "takes a few seconds and allocates ~1 GB"]
    fn visium_hd_scale() {
        let side = 3250u32;
        let t = std::time::Instant::now();
        let (rows, cols) = full_square(side, side);
        let coords = t.elapsed();

        let t = std::time::Instant::now();
        let g = lattice_graph(&rows, &cols, Lattice::Square4).unwrap();
        let build = t.elapsed();

        println!(
            "{}x{} = {} bins, {} edges\n  coords {:.2?}, graph {:.2?}, {:.1} MB",
            side,
            side,
            g.node_count(),
            g.edge_count(),
            coords,
            build,
            g.memory_bytes() as f64 / 1e6
        );
        assert_eq!(g.node_count(), (side * side) as usize);
        assert_eq!(g.edge_count() as u32, 2 * side * (side - 1));
    }
}
