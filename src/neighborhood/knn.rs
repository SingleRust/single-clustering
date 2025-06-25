use nalgebra_sparse::{CooMatrix, CsrMatrix};
use ndarray::ArrayViewD;
use kiddo::traits::DistanceMetric;
use rayon::prelude::*;
use single_utilities::traits::FloatOpsTS;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct NeighborResult<T> {
    pub distances: CsrMatrix<T>,
    pub connectivities: CsrMatrix<T>,
}

pub fn knn_arrayd_kiddo_gaussian<T, const K: usize, D>(
    data: ArrayViewD<T>,
    k: u64,
) -> anyhow::Result<NeighborResult<T>>
where
    T: FloatOpsTS + Send + Sync + 'static,
    D: DistanceMetric<T, K>,
{
    if data.ndim() != 2 {
        return Err(anyhow::anyhow!(
            "The input array must have two dimensions."
        ));
    }

    let shape = data.shape();
    let n_samples = shape[0];
    let n_features = shape[1];

    if n_features < K {
        return Err(anyhow::anyhow!(
            "The data must have at least K features for KNN calculation"
        ));
    }

    // Build KD-tree
    let mut kdtree: kiddo::KdTree<T, K> = kiddo::KdTree::new();
    
    // Batch add points to KD-tree (more efficient than individual adds)
    let points: Vec<([T; K], u64)> = (0..n_samples)
        .map(|i| {
            let mut point = [T::zero(); K];
            for j in 0..K {
                point[j] = *data.get([i, j]).unwrap_or(&T::zero());
            }
            (point, i as u64)
        })
        .collect();
    
    for (point, idx) in points.iter() {
        kdtree.add(point, *idx);
    }

    // Parallel KNN search
    let knn_results: Vec<(Vec<usize>, Vec<T>)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let mut query = [T::zero(); K];
            for j in 0..K {
                query[j] = *data.get([i, j]).unwrap_or(&T::zero());
            }
            
            let neighbors = kdtree.nearest_n::<D>(&query, (k + 1) as usize);
            
            let indices: Vec<usize> = neighbors.iter().map(|n| n.item as usize).collect();
            let distances_sq: Vec<T> = neighbors.iter().map(|n| n.distance).collect();
            
            (indices, distances_sq)
        })
        .collect();

    // Extract results
    let (knn_indices, knn_distances_sq): (Vec<_>, Vec<_>) = knn_results.into_iter().unzip();

    // Build distance matrix triplets
    let mut distance_triplets = Vec::with_capacity(n_samples * (k as usize + 1));
    for i in 0..n_samples {
        for (idx, &j) in knn_indices[i].iter().enumerate() {
            distance_triplets.push((i, j, knn_distances_sq[i][idx]));
        }
    }

    // Compute sigmas in parallel
    let sigmas: Vec<T> = knn_distances_sq
        .par_iter()
        .map(|distances| {
            let mut dist_wo_self: Vec<T> = distances
                .iter()
                .filter(|&&d| d > T::zero())
                .copied()
                .collect();
            
            if dist_wo_self.is_empty() {
                T::one()
            } else {
                // Use nth_element for O(n) median finding instead of O(n log n) sort
                let mid = dist_wo_self.len() / 2;
                let (_, median, _) = dist_wo_self.select_nth_unstable(mid);
                median.sqrt()
            }
        })
        .collect();

    // Pre-compute connectivity triplets capacity
    let estimated_connectivity_edges = n_samples * k as usize * 2;
    
    // Use atomic counter for thread-safe capacity tracking
    let triplet_count = AtomicUsize::new(0);
    
    // Parallel connectivity computation with chunking
    let chunk_size = (n_samples + rayon::current_num_threads() - 1) / rayon::current_num_threads();
    
    let connectivity_chunks: Vec<Vec<(usize, usize, T)>> = (0..n_samples)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|chunk| {
            let mut local_triplets = Vec::with_capacity(chunk.len() * k as usize * 2);
            let min_weight = T::from_f64(1e-14).unwrap();
            
            for i in chunk {
                // Only process upper triangle and diagonal
                for (neighbor_idx, &j) in knn_indices[i].iter().enumerate().skip(1) {
                    if i <= j {
                        let dist_sq = knn_distances_sq[i][neighbor_idx];
                        
                        let sigma_i = sigmas[i];
                        let sigma_j = sigmas[j];
                        let sigma_i_sq = sigma_i * sigma_i;
                        let sigma_j_sq = sigma_j * sigma_j;
                        
                        let den = sigma_i_sq + sigma_j_sq;
                        
                        if den > T::zero() {
                            let num = T::from(2).unwrap() * sigma_i * sigma_j;
                            let normalization = (num / den).sqrt();
                            let exponential = (-dist_sq / den).exp();
                            let weight = normalization * exponential;
                            
                            if weight > min_weight {
                                local_triplets.push((i, j, weight));
                                if i != j {
                                    local_triplets.push((j, i, weight));
                                }
                            }
                        }
                    }
                }
            }
            
            triplet_count.fetch_add(local_triplets.len(), Ordering::Relaxed);
            local_triplets
        })
        .collect();
    
    // Flatten connectivity triplets
    let mut connectivity_triplets = Vec::with_capacity(triplet_count.load(Ordering::Relaxed));
    for chunk in connectivity_chunks {
        connectivity_triplets.extend(chunk);
    }

    // Create sparse matrices efficiently
    let distances = create_csr_from_triplets(n_samples, distance_triplets)?;
    let connectivities = create_csr_from_triplets(n_samples, connectivity_triplets)?;

    Ok(NeighborResult {
        distances,
        connectivities,
    })
}

// Helper function to create CSR matrix from triplets more efficiently
fn create_csr_from_triplets<T>(
    n: usize,
    triplets: Vec<(usize, usize, T)>,
) -> anyhow::Result<CsrMatrix<T>>
where
    T: FloatOpsTS,
{
    if triplets.is_empty() {
        return Ok(CsrMatrix::zeros(n, n));
    }
    
    // Sort triplets by (row, col) for better CSR construction
    let mut sorted_triplets = triplets;
    sorted_triplets.sort_unstable_by_key(|&(r, c, _)| (r, c));
    
    // Extract components
    let row_indices: Vec<usize> = sorted_triplets.iter().map(|&(r, _, _)| r).collect();
    let col_indices: Vec<usize> = sorted_triplets.iter().map(|&(_, c, _)| c).collect();
    let values: Vec<T> = sorted_triplets.into_iter().map(|(_, _, v)| v).collect();
    
    // Create COO matrix
    let coo = CooMatrix::try_from_triplets(n, n, row_indices, col_indices, values)
        .map_err(|e| anyhow::anyhow!("Failed to create COO matrix: {}", e))?;
    
    Ok(CsrMatrix::from(&coo))
}

// Alternative implementation using pre-allocated buffers for even better performance
pub fn knn_arrayd_kiddo_gaussian_optimized<T, const K: usize, D>(
    data: ArrayViewD<T>,
    k: u64,
) -> anyhow::Result<NeighborResult<T>>
where
    T: FloatOpsTS + Send + Sync + 'static,
    D: DistanceMetric<T, K>,
{
    if data.ndim() != 2 {
        return Err(anyhow::anyhow!(
            "The input array must have two dimensions."
        ));
    }

    let shape = data.shape();
    let n_samples = shape[0];
    let n_features = shape[1];

    if n_features < K {
        return Err(anyhow::anyhow!(
            "The data must have at least K features for KNN calculation"
        ));
    }

    // Build KD-tree with bulk loading if supported
    let mut kdtree: kiddo::KdTree<T, K> = kiddo::KdTree::new();
    
    // Pre-allocate all points
    let mut points = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut point = [T::zero(); K];
        unsafe {
            // Use unsafe for better performance if we trust the bounds
            for j in 0..K {
                point[j] = *data.uget([i, j]);
            }
        }
        points.push((point, i as u64));
    }
    
    // Bulk insert
    for (point, idx) in &points {
        kdtree.add(point, *idx);
    }

    // Pre-allocate all result vectors
    let mut knn_indices = vec![Vec::with_capacity(k as usize + 1); n_samples];
    let mut knn_distances_sq = vec![Vec::with_capacity(k as usize + 1); n_samples];
    
    // Parallel KNN search with pre-allocated buffers
    knn_indices
        .par_iter_mut()
        .zip(knn_distances_sq.par_iter_mut())
        .enumerate()
        .for_each(|(i, (indices, distances))| {
            let neighbors = kdtree.nearest_n::<D>(&points[i].0, (k + 1) as usize);
            
            for neighbor in neighbors {
                indices.push(neighbor.item as usize);
                distances.push(neighbor.distance);
            }
        });

    // Use the connectivity module if available
    let connectivity_computer = GaussianConnectivity::<T>::new(true);
    let distance_matrix = build_distance_csr(&knn_indices, &knn_distances_sq, n_samples)?;
    let connectivities = connectivity_computer.compute_connectivities(&distance_matrix, k as usize);

    Ok(NeighborResult {
        distances: distance_matrix,
        connectivities,
    })
}

// Helper to build distance CSR matrix
fn build_distance_csr<T>(
    knn_indices: &[Vec<usize>],
    knn_distances_sq: &[Vec<T>],
    n_samples: usize,
) -> anyhow::Result<CsrMatrix<T>>
where
    T: FloatOpsTS,
{
    let nnz: usize = knn_indices.iter().map(|v| v.len()).sum();
    
    let mut row_offsets = vec![0; n_samples + 1];
    let mut col_indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    
    for i in 0..n_samples {
        row_offsets[i + 1] = row_offsets[i] + knn_indices[i].len();
        col_indices.extend(&knn_indices[i]);
        values.extend(&knn_distances_sq[i]);
    }
    
    CsrMatrix::try_from_csr_data(n_samples, n_samples, row_offsets, col_indices, values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {}", e))
}