// Similar to the implementation presented here, but without the UMAP implementation: https://github.com/scverse/scanpy/blob/main/src/scanpy/neighbors/_common.py#L73

use nalgebra_sparse::CsrMatrix;
use ndarray::Array2;
use single_utilities::traits::FloatOpsTS;

pub struct GaussianConnectivity<T>
where
    T: FloatOpsTS,
{
    knn: bool,
    min_weight_threshold: T,
}

impl<T> GaussianConnectivity<T>
where
    T: FloatOpsTS + 'static,
{
    pub fn new(knn: bool) -> Self {
        Self {
            knn,
            min_weight_threshold: T::from_f64(1e-14).unwrap(),
        }
    }

    pub fn compute_connectivities(
        &self,
        distances: &CsrMatrix<T>,
        n_neighbors: usize,
    ) -> CsrMatrix<T> {
        let (knn_indices, knn_distances) = self.extract_knn_from_sparse(distances, n_neighbors);

        let sigmas = self.compute_adaptive_sigmas(&knn_distances);

        self.compute_gaussian_weights(distances, &sigmas, &knn_indices, n_neighbors)
    }

    fn extract_knn_from_sparse(
        &self,
        distances: &CsrMatrix<T>,
        n_neighbors: usize,
    ) -> (Array2<usize>, Array2<T>) {
        let n_obs = distances.nrows();
        let mut knn_indices = Array2::<usize>::zeros((n_obs, n_neighbors));
        let mut knn_distances = Array2::zeros((n_obs, n_neighbors));

        for i in 0..n_obs {
            let mut neighbors: Vec<(usize, T)> = Vec::new();

            for (col, &dist) in distances
                .row(i)
                .col_indices()
                .iter()
                .zip(distances.row(i).values().iter())
            {
                if *col != i || dist > T::zero() {
                    neighbors.push((*col, dist));
                }
            }

            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            neighbors.truncate(n_neighbors);

            for (j, &(neighbor_idx, dist)) in neighbors.iter().enumerate() {
                knn_indices[[i, j]] = neighbor_idx;
                knn_distances[[i, j]] = dist;
            }
        }

        (knn_indices, knn_distances)
    }

    fn compute_adaptive_sigmas(&self, knn_distances: &Array2<T>) -> Vec<T> {
        let n_obs = knn_distances.nrows();
        let mut sigmas = Vec::with_capacity(n_obs);

        for i in 0..n_obs {
            let sigma_sq = if self.knn {
                let mut distances_sq: Vec<T> = knn_distances
                    .row(i)
                    .iter()
                    .filter(|&&d| d > T::zero())
                    .map(|&d| d * d)
                    .collect();

                if distances_sq.is_empty() {
                    T::from_f64(1.0).unwrap()
                } else {
                    distances_sq
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let median_idx = distances_sq.len() / 2;
                    distances_sq[median_idx]
                }
            } else {
                let last_dist = knn_distances[[i, knn_distances.ncols() - 1]];
                (last_dist * last_dist) / T::from_f64(4.0).unwrap()
            };

            sigmas.push(sigma_sq.sqrt());
        }

        sigmas
    }

    fn compute_gaussian_weights(
        &self,
        distances: &CsrMatrix<T>,
        sigmas: &[T],
        knn_indices: &Array2<usize>,
        n_neighbors: usize,
    ) -> CsrMatrix<T> {
        let n_obs = distances.nrows();
        let mut triplets = Vec::new();

        if self.knn {
            let mut processed_pairs = std::collections::HashSet::new();

            for i in 0..n_obs {
                for j in 0..n_neighbors {
                    let neighbor_idx = knn_indices[[i, j]];

                    let pair = if i < neighbor_idx {
                        (i, neighbor_idx)
                    } else {
                        (neighbor_idx, i)
                    };
                    if processed_pairs.contains(&pair) {
                        continue;
                    }
                    processed_pairs.insert(pair);

                    if let Some(dist_sq) = distances.get_entry(i, neighbor_idx) {
                        let dist_sq = dist_sq.into_value();
                        let weight = self.compute_gaussian_weight(i, neighbor_idx, dist_sq, sigmas);

                        if weight > self.min_weight_threshold {
                            triplets.push((i, neighbor_idx, weight));
                            if i != neighbor_idx {
                                triplets.push((neighbor_idx, i, weight)); // Ensure symmetry
                            }
                        }
                    }
                }
            }
        } else {
            // For dense: compute all pairwise weights above threshold
            for i in 0..n_obs {
                for j in i..n_obs {
                    // Only upper triangle, then make symmetric
                    if let Some(dist) = distances.get_entry(i, j) {
                        let dist = dist.into_value();
                        let weight = self.compute_gaussian_weight(i, j, dist, sigmas);

                        if weight > self.min_weight_threshold {
                            triplets.push((i, j, weight));
                            if i != j {
                                triplets.push((j, i, weight)); // Ensure symmetry
                            }
                        }
                    }
                }
            }
        }

        // Convert to CSR matrix
        let rows: Vec<usize> = triplets.iter().map(|(r, _, _)| *r).collect();
        let cols: Vec<usize> = triplets.iter().map(|(_, c, _)| *c).collect();
        let data: Vec<T> = triplets.iter().map(|(_, _, v)| *v).collect();

        let coo = nalgebra_sparse::CooMatrix::try_from_triplets(n_obs, n_obs, rows, cols, data)
            .expect("Failed to create COO matrix");
        CsrMatrix::from(&coo)
    }

    fn compute_gaussian_weight(&self, i: usize, j: usize, dist_sq: T, sigmas: &[T]) -> T {
        let sigma_i = sigmas[i];
        let sigma_j = sigmas[j];

        let sigma_i_sq = sigma_i * sigma_i;
        let sigma_j_sq = sigma_j * sigma_j;

        // Symmetric Gaussian kernel formula from scanpy
        let numerator = T::from_f64(2.0).unwrap() * sigma_i * sigma_j;
        let denominator = sigma_i_sq + sigma_j_sq;

        if denominator > T::zero() {
            let normalization = (numerator / denominator).sqrt();
            let exponential = (-dist_sq / denominator).exp();
            normalization * exponential
        } else {
            T::zero()
        }
    }
}
