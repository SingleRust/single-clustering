use kiddo::traits::DistanceMetric;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use ndarray::ArrayViewD;
use single_utilities::traits::FloatOpsTS;

pub struct NeighborResult<T> {
    pub distances: CsrMatrix<T>,
    pub connectivities: CsrMatrix<T>,
}

pub fn knn_arrayd_kiddo_gaussian<T, const K: usize, D>(
    data: ArrayViewD<T>,
    k: u64,
) -> anyhow::Result<NeighborResult<T>>
where
    T: FloatOpsTS + 'static,
    D: DistanceMetric<T, K>,
{
    if data.ndim() != 2 {
        return Err(anyhow::anyhow!(
            "The input array has to have two dimensions."
        )); // TODO error message fix
    }

    let shape = data.shape();
    let n_samples = shape[0] as u64;
    let n_features = shape[1] as u64;

    if (n_features as usize) < K {
        return Err(anyhow::anyhow!(
            "The data must have at least K features in order to be used for KNN calculation"
        ));
    }

    let mut kdtree: kiddo::KdTree<T, K> = kiddo::KdTree::new();

    for i in 0..n_samples {
        let mut point_array = [T::zero(); K];
        for j in 0..K {
            point_array[j] = *data.get([i as usize, j]).unwrap_or(&T::zero());
        }
        kdtree.add(&point_array, i);
    }

    let mut knn_indices = Vec::with_capacity(n_samples as usize);
    let mut knn_distances_sq = Vec::with_capacity(n_samples as usize);

    for i in 0..n_samples {
        let mut query_array = [T::zero(); K];
        for j in 0..K {
            query_array[j] = *data.get([i as usize, j]).unwrap_or(&T::zero());
        }

        let neighbors = kdtree.nearest_n::<D>(&query_array, (k + 1) as usize);
        let mut indices = Vec::with_capacity(k as usize + 1);
        let mut distances_sq = Vec::with_capacity(k as usize + 1);

        for neighbor in neighbors.iter() {
            indices.push(neighbor.item as usize);
            distances_sq.push(neighbor.distance);
        }

        knn_indices.push(indices);
        knn_distances_sq.push(distances_sq);
    }

    let mut distance_triplets = Vec::new();

    for i in 0..n_samples as usize {
        for (idx, &j) in knn_indices[i].iter().enumerate() {
            distance_triplets.push((i, j, knn_distances_sq[i][idx]));
        }
    }

    let mut sigmas_sq = Vec::with_capacity(n_samples as usize);

    for i in 0..n_samples as usize {
        let mut dist_wo_self: Vec<T> = knn_distances_sq[i]
            .iter()
            .filter(|&&d| d > T::zero())
            .copied()
            .collect();

        let sigma = if dist_wo_self.is_empty() {
            T::one()
        } else {
            dist_wo_self.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median_idx = dist_wo_self.len() / 2;
            dist_wo_self[median_idx]
        };
        sigmas_sq.push(sigma);
    }

    let mut connectivity_triplets = Vec::new();
    let min_weight = T::from_f64(1e-14).unwrap();

    for i in 0..n_samples as usize {
        for &j in knn_indices[i].iter().skip(1) {
            if i <= j {
                // place here upper triangle restriction
                let dist_sq = if let Some(pos) = knn_indices[i].iter().position(|&x| x == j) {
                    knn_distances_sq[i][pos]
                } else {
                    continue;
                };

                let sigma_i_sq = sigmas_sq[i];
                let sigma_j_sq = sigmas_sq[j];
                let sigma_i = sigma_i_sq.sqrt();
                let sigma_j = sigma_j_sq.sqrt();
                let num = T::from(2).unwrap() * sigma_i * sigma_j;
                let den = sigma_i_sq + sigma_j_sq;

                let weight = if den > T::zero() {
                    let normalization = (num / den).sqrt();
                    let exponential = (-dist_sq / den).exp();
                    normalization * exponential
                } else {
                    T::zero()
                };

                if weight > min_weight {
                    connectivity_triplets.push((i, j, weight));
                    if i != j {
                        connectivity_triplets.push((j, i, weight)); // symmetry with just one computation step
                    }
                }
            }
        }
    }

    let distances_coo = CooMatrix::try_from_triplets(
        n_samples as usize,
        n_samples as usize,
        distance_triplets.iter().map(|&(i, _, _)| i).collect(),
        distance_triplets.iter().map(|&(_, j, _)| j).collect(),
        distance_triplets.iter().map(|&(_, _, v)| v).collect(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create distance COO matrix: {}", e))?;

    let connectivities_coo = CooMatrix::try_from_triplets(
        n_samples as usize,
        n_samples as usize,
        connectivity_triplets.iter().map(|&(i, _, _)| i).collect(),
        connectivity_triplets.iter().map(|&(_, j, _)| j).collect(),
        connectivity_triplets.iter().map(|&(_, _, v)| v).collect(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create connectivity COO matrix: {}", e))?;

    Ok(NeighborResult {
        distances: CsrMatrix::from(&distances_coo),
        connectivities: CsrMatrix::from(&connectivities_coo),
    })
}
