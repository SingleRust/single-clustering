//! Uniform-grid spatial index.
//!
//! Tissue is close to uniformly dense, so fixed cells beat a tree: queries are O(points
//! nearby), duplicates and collinear runs are ordinary inputs, and nothing is approximate.
//!
//! It also replaces `kiddo` 5.2.2, which was wrong here. Its `KdTree` panics once more than
//! 32 points share an axis value — every Visium row does — and its `ImmutableKdTree` returns
//! ids that don't match the coordinates it measured.

/// Points bucketed into cells of a fixed size, stored CSR-style.
pub(crate) struct Grid<const D: usize> {
    cell: f64,
    min: [f64; D],
    dims: [usize; D],
    /// `starts[c]..starts[c + 1]` is cell `c`'s slice of `items`.
    starts: Vec<u32>,
    items: Vec<u32>,
}

impl<const D: usize> Grid<D> {
    /// Buckets `points` into cells of roughly `target_cell`, enlarging if that would need
    /// more cells than points.
    pub(crate) fn new(points: &[[f64; D]], target_cell: f64) -> Self {
        debug_assert!(!points.is_empty());
        let mut min = [f64::INFINITY; D];
        let mut max = [f64::NEG_INFINITY; D];
        for p in points {
            for d in 0..D {
                min[d] = min[d].min(p[d]);
                max[d] = max[d].max(p[d]);
            }
        }

        // ~2 cells per point max, so sparse data gets coarser cells not a huge empty index
        let mut cell = target_cell.max(f64::MIN_POSITIVE);
        let mut dims = [0usize; D];
        loop {
            let mut total = 1usize;
            let mut overflow = false;
            for d in 0..D {
                let span = max[d] - min[d];
                dims[d] = ((span / cell).floor() as usize).saturating_add(1);
                match total.checked_mul(dims[d]) {
                    Some(t) => total = t,
                    None => {
                        overflow = true;
                        break;
                    }
                }
            }
            if !overflow && total <= points.len().saturating_mul(2).max(64) {
                break;
            }
            cell *= 2.0;
        }

        let n_cells: usize = dims.iter().product();
        let mut starts = vec![0u32; n_cells + 1];
        let index_of = |p: &[f64; D]| -> usize {
            let mut idx = 0usize;
            for d in 0..D {
                let c = (((p[d] - min[d]) / cell).floor() as usize).min(dims[d] - 1);
                idx = idx * dims[d] + c;
            }
            idx
        };

        for p in points {
            starts[index_of(p) + 1] += 1;
        }
        for c in 0..n_cells {
            starts[c + 1] += starts[c];
        }
        let mut items = vec![0u32; points.len()];
        let mut cursor: Vec<u32> = starts[..n_cells].to_vec();
        for (i, p) in points.iter().enumerate() {
            let c = index_of(p);
            items[cursor[c] as usize] = i as u32;
            cursor[c] += 1;
        }

        Self {
            cell,
            min,
            dims,
            starts,
            items,
        }
    }

    #[inline]
    pub(crate) fn cell_size(&self) -> f64 {
        self.cell
    }

    /// Which cell a point falls in, per axis.
    #[inline]
    pub(crate) fn cell_of(&self, p: &[f64; D]) -> [usize; D] {
        let mut out = [0usize; D];
        for d in 0..D {
            out[d] = (((p[d] - self.min[d]) / self.cell).floor() as usize).min(self.dims[d] - 1);
        }
        out
    }

    /// Largest Chebyshev ring that could still hold a point, from `centre`.
    pub(crate) fn max_ring(&self, centre: &[usize; D]) -> usize {
        centre
            .iter()
            .zip(&self.dims)
            .fold(0usize, |r, (&c, &dim)| r.max(c).max(dim - 1 - c))
    }

    /// Calls `f` for every point at Chebyshev distance exactly `ring` from `centre`.
    ///
    /// One ring at a time, so a kNN search can stop once its k-th distance beats everything
    /// left unvisited.
    pub(crate) fn for_each_in_ring(
        &self,
        centre: &[usize; D],
        ring: usize,
        mut f: impl FnMut(u32),
    ) {
        // clamp to the grid, then odometer over the box, skipping the interior (already
        // visited as a smaller ring)
        let mut lo = [0usize; D];
        let mut hi = [0usize; D];
        for d in 0..D {
            lo[d] = centre[d].saturating_sub(ring);
            hi[d] = (centre[d] + ring).min(self.dims[d] - 1);
        }

        let mut at = lo;
        loop {
            let on_surface = ring == 0
                || (0..D).any(|d| {
                    (at[d] == centre[d].wrapping_sub(ring) && centre[d] >= ring)
                        || at[d] == centre[d] + ring
                });
            if on_surface {
                let idx = at
                    .iter()
                    .zip(&self.dims)
                    .fold(0usize, |acc, (&a, &dim)| acc * dim + a);
                for &item in &self.items[self.starts[idx] as usize..self.starts[idx + 1] as usize] {
                    f(item);
                }
            }

            // odometer step
            let mut d = D;
            loop {
                if d == 0 {
                    return;
                }
                d -= 1;
                if at[d] < hi[d] {
                    at[d] += 1;
                    break;
                }
                at[d] = lo[d];
            }
        }
    }
}

#[inline]
pub(crate) fn distance_sq<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut s = 0.0;
    for d in 0..D {
        let diff = a[d] - b[d];
        s += diff * diff;
    }
    s
}
