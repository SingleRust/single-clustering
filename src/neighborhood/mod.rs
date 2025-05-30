use kiddo::SquaredEuclidean;
use ndarray::{Array2, ArrayD};
use single_utilities::traits::FloatOpsTS;
use crate::neighborhood::knn::{knn_arrayd_kdtree, knn_arrayd_kiddo};
use crate::network::{network_from_gaussian_connectivity, Network};

pub mod knn;
pub mod connectivity;

pub fn compute_neighborhood<T>(
    data: &ArrayD<T>,               // Your original data
    n_pca_components: usize,       // PCA dimensions
    n_neighbors: usize,            // k for k-NN
    knn: bool,                     // Use k-NN graph or dense
) -> Network<T, T>
where
    T: FloatOpsTS + 'static,
{
    // Step 2: Compute k-NN distance matrix in PCA space
    let distance_matrix = knn_arrayd_kiddo::<T, 3, SquaredEuclidean>(
        data.view(),
        n_neighbors as u64
    ).expect("Failed to compute k-NN");

    let node_weights = vec![T::from_f64(1.0).unwrap(); data.nrows()];

    network_from_gaussian_connectivity(distance_matrix, node_weights, n_neighbors, knn)
}