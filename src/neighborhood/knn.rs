use kiddo::KdTree;
use kiddo::traits::DistanceMetric;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use ndarray::ArrayViewD;
use single_utilities::traits::FloatOpsTS;

pub fn knn_arrayd<T, const K: usize, D>(
    data: ArrayViewD<T>,
    k: u64,
) -> anyhow::Result<CsrMatrix<T>>
where
    T: FloatOpsTS + 'static,
    D: DistanceMetric<T, K>
{
    if data.ndim() != 2 {
        return Err(anyhow::anyhow!("The input array has to be two dimensional in order to be used to build a knn network."));
    }

    let shape = data.shape();
    let n_samples = shape[0] as u64;
    let n_features = shape[1] as u64;

    if (n_features as usize) < K {
        return Err(anyhow::anyhow!("The data must have at least K features in order to be used for building a knn network."))
    }

    let mut kdtree: KdTree<T, K> = KdTree::new();

    for i in 0..n_samples {
        let mut point_array = [T::zero(); K];
        for j in 0..K {
            point_array[j] = *data.get([i as usize, j]).unwrap_or(&T::zero());
        }
        kdtree.add(&point_array, i);
    }

    let mut all_distances = Vec::with_capacity((n_samples * k) as usize);
    for i in 0..n_samples {
        let mut query_array = [T::zero(); K];
        for j in 0..K {
            query_array[j] = *data.get([i as usize, j]).unwrap_or(&T::zero());
        }
        let neighbors = kdtree.nearest_n::<D>(&query_array, (k+1) as usize);
        for neighbor in neighbors.iter().skip(1) {
            all_distances.push(neighbor.distance);
        }
    }

    all_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_idx = all_distances.len() / 2;
    let global_sigma = if all_distances.is_empty() {
        T::from_f64(1.0).unwrap()
    } else {
        all_distances[median_idx] / T::from_f64(f64::ln(k as f64)).unwrap()
    };

    let mut triplets = Vec::with_capacity((n_samples * k) as usize);

    for i in 0..n_samples {
        let mut query_array = [T::zero(); K];
        for j in 0..K {
            query_array[j] = *data.get([i as usize, j]).unwrap_or(&T::zero());
        }

        let neighbors = kdtree.nearest_n::<D>(&query_array, (k+1) as usize);

        for neighbor in neighbors.iter().skip(1) {
            if i <= neighbor.item {
                let weight = (-neighbor.distance / global_sigma).exp();
                let min_weight = T::from_f64(1e-6).unwrap();
                if weight > min_weight {
                    triplets.push((i as usize, neighbor.item as usize, weight));
                }
            }
        }
    }

    let coo = CooMatrix::try_from_triplets(
        n_samples as usize,
        n_samples as usize,
        triplets.iter().map(|&(i, _, _)| i).collect(),
        triplets.iter().map(|&(_, j, _)| j).collect(),
        triplets.iter().map(|&(_, _, v)| v).collect(),
    )
        .map_err(|e| anyhow::anyhow!("Failed to create COO matrix: {}", e))?;

    Ok(CsrMatrix::from(&coo))
}