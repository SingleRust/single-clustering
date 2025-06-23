use kiddo::traits::DistanceMetric;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use ndarray::ArrayViewD;
use rayon::prelude::*;
use single_utilities::traits::FloatOpsTS;
use std::time::Instant;

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
        return Err(anyhow::anyhow!("The input array must have two dimensions."));
    }

    let shape = data.shape();
    let n_samples = shape[0];
    let n_features = shape[1];

    if n_features < K {
        return Err(anyhow::anyhow!(
            "The data must have at least K features for KNN calculation"
        ));
    }

    let start_time = Instant::now();

    let kdtree_start = Instant::now();
    let mut kdtree: kiddo::KdTree<T, K> = kiddo::KdTree::new();

    for i in 0..n_samples {
        let mut point = [T::zero(); K];
        for j in 0..K {
            point[j] = *data.get([i, j]).unwrap_or(&T::zero());
        }
        kdtree.add(&point, i as u64);
    }
    println!("KD-tree construction: {:?}", kdtree_start.elapsed());

    let knn_search_start = Instant::now();

    let knn_search_start = Instant::now();
    let (knn_indices, knn_distances_sq) =
        knn_search_large_batches::<T, K, D>(&kdtree, data, n_samples, k);
    println!("KNN search: {:?}", knn_search_start.elapsed());

    let distance_matrix_start = Instant::now();
    let mut distance_triplets = Vec::with_capacity(n_samples * (k as usize + 1));
    for i in 0..n_samples {
        for (idx, &j) in knn_indices[i].iter().enumerate() {
            distance_triplets.push((i, j, knn_distances_sq[i][idx]));
        }
    }
    println!(
        "Distance matrix triplets: {:?}",
        distance_matrix_start.elapsed()
    );

    let sigma_start = Instant::now();
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
                let mid = dist_wo_self.len() / 2;
                dist_wo_self.select_nth_unstable_by(mid, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                dist_wo_self[mid].sqrt()
            }
        })
        .collect();
    println!("Sigma computation: {:?}", sigma_start.elapsed());

    let connectivity_start = Instant::now();
    let min_weight = T::from_f64(1e-14).unwrap();
    let chunk_size = n_samples.div_ceil(rayon::current_num_threads());

    let connectivity_chunks: Vec<Vec<(usize, usize, T)>> = (0..n_samples)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|chunk| {
            let mut local_triplets = Vec::with_capacity(chunk.len() * k as usize * 2);

            for i in chunk {
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

            local_triplets
        })
        .collect();

    let mut connectivity_triplets = Vec::new();
    for chunk in connectivity_chunks {
        connectivity_triplets.extend(chunk);
    }
    println!(
        "Connectivity computation: {:?}",
        connectivity_start.elapsed()
    );

    let matrix_creation_start = Instant::now();
    let distances = create_csr_from_triplets(n_samples, distance_triplets)?;
    let connectivities = create_csr_from_triplets(n_samples, connectivity_triplets)?;
    println!("CSR matrix creation: {:?}", matrix_creation_start.elapsed());

    println!("Total KNN computation time: {:?}", start_time.elapsed());
    println!(
        "Created distance matrix: {} x {}, nnz: {}",
        distances.nrows(),
        distances.ncols(),
        distances.nnz()
    );
    println!(
        "Created connectivity matrix: {} x {}, nnz: {}",
        connectivities.nrows(),
        connectivities.ncols(),
        connectivities.nnz()
    );

    Ok(NeighborResult {
        distances,
        connectivities,
    })
}

fn create_csr_from_triplets<T>(
    n: usize,
    triplets: Vec<(usize, usize, T)>,
) -> anyhow::Result<CsrMatrix<T>>
where
    T: FloatOpsTS + 'static,
{
    if triplets.is_empty() {
        return Ok(CsrMatrix::zeros(n, n));
    }

    let row_indices: Vec<usize> = triplets.iter().map(|(r, _, _)| *r).collect();
    let col_indices: Vec<usize> = triplets.iter().map(|(_, c, _)| *c).collect();
    let values: Vec<T> = triplets.into_iter().map(|(_, _, v)| v).collect();

    let coo = CooMatrix::try_from_triplets(n, n, row_indices, col_indices, values)
        .map_err(|e| anyhow::anyhow!("Failed to create COO matrix: {}", e))?;

    Ok(CsrMatrix::from(&coo))
}

pub fn knn_search_large_batches<T, const K: usize, D>(
    kdtree: &kiddo::KdTree<T, K>,
    data: ArrayViewD<T>,
    n_samples: usize,
    k: u64,
) -> (Vec<Vec<usize>>, Vec<Vec<T>>)
where
    T: FloatOpsTS + Send + Sync + 'static,
    D: DistanceMetric<T, K>,
{
    // Use fewer threads to reduce memory bandwidth contention
    // Rule of thumb: Use ~1 thread per memory channel (typically 2-8 for modern CPUs)
    let memory_threads = std::cmp::min(64, rayon::current_num_threads());
    let chunk_size = (n_samples + memory_threads - 1) / memory_threads;

    println!(
        "Using {} threads with chunk size {} to reduce memory contention",
        memory_threads, chunk_size
    );

    // Process with controlled parallelism
    let results: Vec<Vec<(Vec<usize>, Vec<T>)>> = rayon::ThreadPoolBuilder::new()
        .num_threads(memory_threads)
        .build()
        .unwrap()
        .install(|| {
            (0..n_samples)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(batch_idx, chunk)| {
                    let batch_start = Instant::now();
                    let mut batch_results = Vec::with_capacity(chunk.len());

                    for &i in chunk {
                        // Build query
                        let mut query = [T::zero(); K];
                        for j in 0..K {
                            query[j] = *data.get([i, j]).unwrap_or(&T::zero());
                        }

                        // Search
                        let neighbors = kdtree.nearest_n::<D>(&query, (k + 1) as usize);

                        // Collect results
                        let mut indices = Vec::with_capacity(neighbors.len());
                        let mut distances = Vec::with_capacity(neighbors.len());

                        for neighbor in neighbors {
                            indices.push(neighbor.item as usize);
                            distances.push(neighbor.distance);
                        }

                        batch_results.push((indices, distances));
                    }

                    let batch_time = batch_start.elapsed();
                    println!(
                        "Batch {} took {:?} ({:.2} µs per query)",
                        batch_idx,
                        batch_time,
                        batch_time.as_micros() as f64 / chunk.len() as f64
                    );

                    batch_results
                })
                .collect()
        });

    // Flatten results
    let mut knn_indices = Vec::with_capacity(n_samples);
    let mut knn_distances = Vec::with_capacity(n_samples);

    for batch in results {
        for (indices, distances) in batch {
            knn_indices.push(indices);
            knn_distances.push(distances);
        }
    }

    (knn_indices, knn_distances)
}
