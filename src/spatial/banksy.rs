//! BANKSY-style feature augmentation.
//!
//! Augments each cell's expression with a summary of its neighbourhood, so ordinary Leiden
//! finds spatial domains. Nothing here knows about clustering.
//!
//! Formulas come from `banksy-py` (`main.py`, `embed_banksy.py`) rather than the paper, and
//! are pinned by fixtures in `tests/fixtures/banksy_reference.json`. Two things the paper
//! doesn't make obvious: the λ budget halves with each harmonic rather than splitting evenly
//! ([`scale_factors`]), and every block is z-scored per column before scaling.
//!
//! The weight matrix is row-normalised and so asymmetric — an operator, not a graph.

use rayon::prelude::*;

use crate::error::{ClusteringError, Result};
use crate::spatial::grid::{Grid, distance_sq};

/// How neighbour weights fall off with distance, before row normalisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Decay {
    /// Every neighbour counts the same.
    Uniform,
    /// `1 / r`.
    Reciprocal,
    /// `1 / r²`.
    ReciprocalSquared,
    /// `exp(-(r / median_r)²)`, `median_r` per cell. Scale-free, and the reference default.
    #[default]
    ScaledGaussian,
    /// `exp(-((rank + 1) · 1.5 / k)²)` by distance rank. Ignores distances, so density-free.
    Ranked,
}

/// Row-normalised directed neighbour weights.
///
/// Not a graph — normalisation makes it asymmetric. An operator applied to a feature matrix.
pub struct NeighbourhoodOperator {
    node_ptrs: Vec<usize>,
    neighbours: Vec<u32>,
    /// Row-normalised magnitudes, summing to 1 per cell.
    weights: Vec<f64>,
    /// Azimuth from each cell to each of its neighbours, `atan2(dy, dx)`.
    theta: Vec<f64>,
    n_cells: usize,
    /// Harmonic order this operator was built for.
    m: usize,
}

impl NeighbourhoodOperator {
    /// Builds the operator for harmonic `m` over `k` nearest neighbours.
    ///
    /// `k` is explicit per harmonic because the references disagree: R uses `k_geom[m]`,
    /// Python silently uses `k * (m + 1)`. R produced the published DLPFC numbers, so that's
    /// the convention here — pass `k * (m + 1)` yourself to match Python.
    ///
    /// Excludes the cell itself; its own expression is a separate block.
    ///
    /// # Errors
    ///
    /// If `k` is zero or a coordinate isn't finite.
    pub fn for_harmonic(coords: &[[f64; 2]], k: usize, m: usize, decay: Decay) -> Result<Self> {
        Self::knn(coords, k, decay, m)
    }

    fn knn(coords: &[[f64; 2]], k: usize, decay: Decay, m: usize) -> Result<Self> {
        if k == 0 {
            return Err(ClusteringError::InvalidConfig(
                "k must be at least 1".into(),
            ));
        }
        for (i, p) in coords.iter().enumerate() {
            if !p[0].is_finite() || !p[1].is_finite() {
                return Err(ClusteringError::InvalidConfig(format!(
                    "cell {i} has a non-finite coordinate"
                )));
            }
        }
        let n = coords.len();
        if n == 0 {
            return Ok(Self {
                node_ptrs: vec![0],
                neighbours: Vec::new(),
                weights: Vec::new(),
                theta: Vec::new(),
                n_cells: 0,
                m,
            });
        }

        let want = k.min(n.saturating_sub(1));
        let grid = Grid::new(coords, typical_spacing(coords, want.max(1)));

        // sorted by (distance, id) so ties don't depend on visit order
        let found: Vec<Vec<(u32, f64)>> = coords
            .par_iter()
            .enumerate()
            .map(|(i, p)| {
                let centre = grid.cell_of(p);
                let mut cand: Vec<(u32, f64)> = Vec::with_capacity(want * 2);
                for ring in 0..=grid.max_ring(&centre) {
                    grid.for_each_in_ring(&centre, ring, |j| {
                        if j as usize != i {
                            cand.push((j, distance_sq(p, &coords[j as usize])));
                        }
                    });
                    if cand.len() >= want {
                        cand.sort_unstable_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0))
                        });
                        cand.truncate(want);
                        let safe = ring as f64 * grid.cell_size();
                        if cand[cand.len() - 1].1 <= safe * safe {
                            break;
                        }
                    }
                }
                cand.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
                cand.truncate(want);
                cand
            })
            .collect();

        let mut node_ptrs = Vec::with_capacity(n + 1);
        node_ptrs.push(0usize);
        let mut total = 0;
        for row in &found {
            total += row.len();
            node_ptrs.push(total);
        }

        let mut neighbours = Vec::with_capacity(total);
        let mut weights = Vec::with_capacity(total);
        let mut theta = Vec::with_capacity(total);

        for (i, row) in found.iter().enumerate() {
            let distances: Vec<f64> = row.iter().map(|&(_, d2)| d2.sqrt()).collect();
            let mut w = decay_weights(&distances, decay);

            // before any phase factor, like the reference: magnitudes sum to 1, the
            // complex weights don't
            let sum: f64 = w.iter().sum();
            if sum > 0.0 {
                for x in &mut w {
                    *x /= sum;
                }
            }

            for (slot, &(j, _)) in row.iter().enumerate() {
                neighbours.push(j);
                weights.push(w[slot]);
                let (dx, dy) = (
                    coords[j as usize][0] - coords[i][0],
                    coords[j as usize][1] - coords[i][1],
                );
                theta.push(dy.atan2(dx));
            }
        }

        Ok(Self {
            node_ptrs,
            neighbours,
            weights,
            theta,
            n_cells: n,
            m,
        })
    }

    /// Number of cells.
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// Harmonic order this operator computes.
    pub fn harmonic(&self) -> usize {
        self.m
    }

    /// Applies this operator's harmonic to a row-major `n_cells × n_features` matrix.
    ///
    /// `m = 0` is the weighted neighbourhood mean. For `m >= 1` weights carry a phase
    /// `exp(i·m·θ)` and the result is the magnitude of the complex sum — large where a gene
    /// varies with direction, near zero where the neighbourhood looks the same all round.
    ///
    /// # Errors
    ///
    /// If `features` doesn't match the cell count.
    pub fn apply(&self, features: &[f64], n_features: usize) -> Result<Vec<f64>> {
        let m = self.m;
        if n_features == 0 {
            return Ok(Vec::new());
        }
        if features.len() != self.n_cells * n_features {
            return Err(ClusteringError::InvalidConfig(format!(
                "features has {} entries, expected {} cells x {n_features}",
                features.len(),
                self.n_cells
            )));
        }

        let mut out = vec![0.0f64; self.n_cells * n_features];
        out.par_chunks_mut(n_features)
            .enumerate()
            .for_each(|(i, row)| {
                let span = self.node_ptrs[i]..self.node_ptrs[i + 1];
                #[allow(clippy::reversed_empty_ranges)]
                if m == 0 {
                    for e in span {
                        let (w, j) = (self.weights[e], self.neighbours[e] as usize);
                        let src = &features[j * n_features..(j + 1) * n_features];
                        for (o, &x) in row.iter_mut().zip(src) {
                            *o += w * x;
                        }
                    }
                } else {
                    // centre on this cell's neighbourhood first. The phase sum is generally
                    // non-zero, so skipping this mixes in the overall level. Lives in the
                    // reference's matrix builder, not its weight construction.
                    let mut centre = vec![0.0f64; n_features];
                    for e in span.clone() {
                        let (w, j) = (self.weights[e], self.neighbours[e] as usize);
                        let src = &features[j * n_features..(j + 1) * n_features];
                        for (c, &x) in centre.iter_mut().zip(src) {
                            *c += w * x;
                        }
                    }

                    let mut im = vec![0.0f64; n_features];
                    for e in span {
                        let (w, j) = (self.weights[e], self.neighbours[e] as usize);
                        let angle = m as f64 * self.theta[e];
                        let (re_w, im_w) = (w * angle.cos(), w * angle.sin());
                        let src = &features[j * n_features..(j + 1) * n_features];
                        for (((o, i_acc), &x), &c) in row
                            .iter_mut()
                            .zip(im.iter_mut())
                            .zip(src)
                            .zip(centre.iter())
                        {
                            *o += re_w * (x - c);
                            *i_acc += im_w * (x - c);
                        }
                    }
                    for (o, &i_acc) in row.iter_mut().zip(im.iter()) {
                        *o = (*o * *o + i_acc * i_acc).sqrt();
                    }
                }
            });
        Ok(out)
    }
}

/// Unnormalised weights for one cell's neighbours.
fn decay_weights(distances: &[f64], decay: Decay) -> Vec<f64> {
    match decay {
        Decay::Uniform => vec![1.0; distances.len()],
        Decay::Reciprocal => distances.iter().map(|&r| 1.0 / r).collect(),
        Decay::ReciprocalSquared => distances.iter().map(|&r| 1.0 / (r * r)).collect(),
        Decay::ScaledGaussian => {
            let mut sorted = distances.to_vec();
            sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            // numpy's median: mean of the middle two on even counts
            let n = sorted.len();
            let median = if n == 0 {
                0.0
            } else if n % 2 == 1 {
                sorted[n / 2]
            } else {
                0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
            };
            distances
                .iter()
                .map(|&r| (-(r / median).powi(2)).exp())
                .collect()
        }
        Decay::Ranked => {
            let k = distances.len();
            // rank by distance, then a fixed decreasing profile in that order
            let mut order: Vec<usize> = (0..k).collect();
            order.sort_unstable_by(|&a, &b| {
                distances[a]
                    .partial_cmp(&distances[b])
                    .unwrap()
                    .then(a.cmp(&b))
            });
            let mut w = vec![0.0; k];
            for (rank, &idx) in order.iter().enumerate() {
                let t = (rank + 1) as f64 * 1.5 / k as f64;
                w[idx] = (-t * t).exp();
            }
            w
        }
    }
}

/// Column z-scores on the population variance `E[x²] − E[x]²`.
///
/// The reference divides by `n`, not `n − 1`. Zero-variance columns come out zero, not NaN.
fn zscore_columns(matrix: &mut [f64], n_rows: usize, n_cols: usize) {
    if n_rows == 0 || n_cols == 0 {
        return;
    }
    for c in 0..n_cols {
        let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
        for r in 0..n_rows {
            let x = matrix[r * n_cols + c];
            sum += x;
            sum_sq += x * x;
        }
        let n = n_rows as f64;
        let mean = sum / n;
        let variance = sum_sq / n - mean * mean;
        let sd = variance.max(0.0).sqrt();
        for r in 0..n_rows {
            let slot = &mut matrix[r * n_cols + c];
            *slot = if sd > 0.0 { (*slot - mean) / sd } else { 0.0 };
        }
    }
}

/// Block scale factors for `max_m` harmonics at neighbourhood contribution `lambda`.
///
/// `max_m + 2` factors: own expression first, then one per harmonic. Each harmonic gets half
/// the weight of the one before, so at `max_m = 1` the mean takes ⅔ of λ and the gradient ⅓.
pub fn scale_factors(max_m: usize, lambda: f64) -> Vec<f64> {
    let n_blocks = max_m + 1;
    let mut squared = vec![0.0f64; n_blocks + 1];
    squared[0] = 1.0 - lambda;

    let denom: f64 = (0..n_blocks).map(|k| 1.0 / 2f64.powi(k as i32 + 1)).sum();
    for k in 0..n_blocks {
        squared[k + 1] = 1.0 / 2f64.powi(k as i32 + 1) / denom * lambda;
    }
    squared.iter().map(|&s| s.max(0.0).sqrt()).collect()
}

/// Assembles the augmented matrix from the own-expression and harmonic blocks.
///
/// Blocks are z-scored per column then scaled, so λ mixes standardised blocks. Result is
/// `n_cells × (n_features · (max_m + 2))`, row-major, ready to PCA and cluster.
///
/// # Errors
///
/// If `lambda` is outside `[0, 1]` or a block has the wrong length.
pub fn banksy_matrix(
    own: &[f64],
    harmonics: &[Vec<f64>],
    n_cells: usize,
    n_features: usize,
    lambda: f64,
) -> Result<Vec<f64>> {
    if !(0.0..=1.0).contains(&lambda) || !lambda.is_finite() {
        return Err(ClusteringError::InvalidConfig(format!(
            "lambda must be in [0, 1], got {lambda}"
        )));
    }
    let expected = n_cells * n_features;
    if own.len() != expected {
        return Err(ClusteringError::InvalidConfig(format!(
            "own block has {} entries, expected {expected}",
            own.len()
        )));
    }
    for (m, h) in harmonics.iter().enumerate() {
        if h.len() != expected {
            return Err(ClusteringError::InvalidConfig(format!(
                "harmonic {m} has {} entries, expected {expected}",
                h.len()
            )));
        }
    }

    let factors = scale_factors(harmonics.len().saturating_sub(1), lambda);
    let n_blocks = harmonics.len() + 1;
    let width = n_features * n_blocks;
    let mut out = vec![0.0f64; n_cells * width];

    for (b, block) in std::iter::once(own)
        .chain(harmonics.iter().map(|h| h.as_slice()))
        .enumerate()
    {
        let mut scaled = block.to_vec();
        zscore_columns(&mut scaled, n_cells, n_features);
        let f = factors[b];
        for r in 0..n_cells {
            let dst = r * width + b * n_features;
            for c in 0..n_features {
                out[dst + c] = f * scaled[r * n_features + c];
            }
        }
    }
    Ok(out)
}

/// One-call BANKSY augmentation.
///
/// `lambda = 0.2`, `k = 18` are the reference's Visium domain settings; `max_m = 1` adds the
/// gradient term.
///
/// # Errors
///
/// If an argument is out of range or the feature matrix doesn't match the cell count.
pub fn banksy_augment(
    coords: &[[f64; 2]],
    features: &[f64],
    n_features: usize,
    k: usize,
    max_m: usize,
    lambda: f64,
    decay: Decay,
) -> Result<Vec<f64>> {
    let harmonics: Vec<Vec<f64>> = (0..=max_m)
        .map(|m| {
            NeighbourhoodOperator::for_harmonic(coords, k, m, decay)?.apply(features, n_features)
        })
        .collect::<Result<_>>()?;
    banksy_matrix(features, &harmonics, coords.len(), n_features, lambda)
}

/// Cell size putting roughly `k` points per grid cell.
fn typical_spacing(points: &[[f64; 2]], k: usize) -> f64 {
    let (mut min, mut max) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in points {
        for d in 0..2 {
            min[d] = min[d].min(p[d]);
            max[d] = max[d].max(p[d]);
        }
    }
    let area = (max[0] - min[0]).max(f64::MIN_POSITIVE) * (max[1] - min[1]).max(f64::MIN_POSITIVE);
    let cells = (points.len() as f64 / (k.max(1) as f64)).max(1.0);
    (area / cells).sqrt().max(f64::MIN_POSITIVE)
}
