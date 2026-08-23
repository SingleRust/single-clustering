//! kNN graph construction with Gaussian connectivity weights.
//!
//! Two backends: an exact kd-tree for small inputs, approximate HNSW for large ones.
//! [`knn_arrayd_adaptive`] picks between them. Both produce the same pair of matrices —
//! distances, and the Gaussian connectivities `scanpy` clusters on.

use hnsw_rs::{
    hnsw::{Hnsw, Neighbour},
    prelude::DistL2,
};
use kiddo::traits::DistanceMetric;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use ndarray::ArrayViewD;
use rayon::prelude::*;
use single_utilities::traits::FloatOpsTS;

/// Distance and connectivity matrices from a kNN search.
pub struct NeighborResult<T> {
    /// Distances to each node's k nearest neighbours.
    pub distances: CsrMatrix<T>,
    /// Gaussian-weighted connectivities, which is what clustering wants.
    pub connectivities: CsrMatrix<T>,
}

/// Exact kNN via a kd-tree. Good up to ~250k points.
///
/// `D` must be a *squared* metric — [`kiddo::float::distance::SquaredEuclidean`] is the one
/// this was written for.
pub fn knn_arrayd_kiddo_gaussian<T, const K: usize, D>(
    data: ArrayViewD<T>,
    k: u64,
) -> anyhow::Result<NeighborResult<T>>
where
    T: FloatOpsTS + Send + Sync + 'static,
    D: DistanceMetric<T, K>,
{
    let n_samples = check_shape::<T, K>(&data)?;

    let mut kdtree: kiddo::KdTree<T, K> = kiddo::KdTree::new();
    for i in 0..n_samples {
        kdtree.add(&point_at::<T, K>(&data, i), i as u64);
    }

    // kd-tree hands back squared distances; everything downstream wants real ones
    let (indices, distances): (Vec<Vec<usize>>, Vec<Vec<T>>) = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            kdtree
                .nearest_n::<D>(&point_at::<T, K>(&data, i), (k + 1) as usize)
                .into_iter()
                .map(|h| (h.item as usize, h.distance.sqrt()))
                .unzip()
        })
        .unzip();

    gaussian_result(n_samples, k, &indices, &distances)
}

/// Approximate kNN via HNSW. Used past ~250k points, where the kd-tree stops being viable.
pub fn knn_arrayd_hnswlib_gaussian<T, const K: usize>(
    data: ArrayViewD<T>,
    k: u64,
) -> anyhow::Result<NeighborResult<T>>
where
    T: FloatOpsTS + Send + Sync + 'static,
    hnsw_rs::prelude::DistL2: hnsw_rs::prelude::Distance<T>,
{
    let n_samples = check_shape::<T, K>(&data)?;
    let n_features = data.shape()[1];
    let (max_nb_connection, nb_layer, ef_construction, ef_search) =
        get_optimal_hnsw_params(n_samples, k as usize);

    let mut hnsw = Hnsw::<T, DistL2>::new(
        max_nb_connection,
        n_samples,
        nb_layer,
        ef_construction,
        DistL2,
    );
    hnsw.set_extend_candidates(false);
    hnsw.modify_level_scale(0.5);

    let rows: Vec<Vec<T>> = (0..n_samples)
        .into_par_iter()
        .map(|i| (0..n_features).map(|j| data[[i, j]]).collect())
        .collect();
    let slices: Vec<(&[T], usize)> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.as_slice(), i))
        .collect();
    hnsw.parallel_insert_slice(&slices);
    hnsw.set_searching_mode(true);

    let hits: Vec<Vec<Neighbour>> = hnsw.parallel_search(&rows, (k + 1) as usize, ef_search);
    let (indices, distances): (Vec<Vec<usize>>, Vec<Vec<T>>) = hits
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|h| (h.d_id, T::from(h.distance).unwrap()))
                .unzip()
        })
        .unzip();

    gaussian_result(n_samples, k, &indices, &distances)
}

/// Picks the kd-tree below 250k points and HNSW above it.
pub fn knn_arrayd_adaptive<T, const K: usize, D>(
    data: ArrayViewD<T>,
    k: u64,
) -> anyhow::Result<NeighborResult<T>>
where
    T: FloatOpsTS + Send + Sync + 'static,
    D: DistanceMetric<T, K>,
    hnsw_rs::prelude::DistL2: hnsw_rs::prelude::Distance<T>,
{
    const HNSW_THRESHOLD: usize = 250_000;
    let n_samples = check_shape::<T, K>(&data)?;

    if n_samples > HNSW_THRESHOLD {
        knn_arrayd_hnswlib_gaussian::<T, K>(data, k)
    } else {
        knn_arrayd_kiddo_gaussian::<T, K, D>(data, k)
    }
}

/// HNSW parameters by dataset size, tuned to land near PyNNDescent's recall.
pub fn get_optimal_hnsw_params(n_samples: usize, k: usize) -> (usize, usize, usize, usize) {
    let max_nb_connection = match n_samples {
        n if n < 10_000 => 16,
        n if n < 100_000 => 24,
        n if n < 1_000_000 => 32,
        _ => 48,
    };
    let nb_layer = 16.min((n_samples as f32).ln().trunc() as usize);
    let ef_construction = match n_samples {
        n if n < 10_000 => 100,
        n if n < 1_000_000 => 200,
        n if n < 2_500_000 => 300,
        _ => 400,
    };
    (
        max_nb_connection,
        nb_layer,
        ef_construction,
        (k * 2).clamp(30, 200),
    )
}

fn check_shape<T, const K: usize>(data: &ArrayViewD<T>) -> anyhow::Result<usize> {
    if data.ndim() != 2 {
        return Err(anyhow::anyhow!("The input array must have two dimensions."));
    }
    if data.shape()[1] < K {
        return Err(anyhow::anyhow!(
            "The data must have at least K features for KNN calculation"
        ));
    }
    Ok(data.shape()[0])
}

fn point_at<T: FloatOpsTS, const K: usize>(data: &ArrayViewD<T>, i: usize) -> [T; K] {
    let mut p = [T::zero(); K];
    for (j, slot) in p.iter_mut().enumerate() {
        *slot = *data.get([i, j]).unwrap_or(&T::zero());
    }
    p
}

/// Turns kNN hits into the two output matrices.
///
/// The connectivity kernel is scanpy's: each node gets a bandwidth `sigma` from the median
/// distance to its neighbours, and an edge is weighted
/// `sqrt(2·σi·σj / (σi² + σj²)) · exp(-d² / (σi² + σj²))`.
fn gaussian_result<T>(
    n_samples: usize,
    k: u64,
    indices: &[Vec<usize>],
    distances: &[Vec<T>],
) -> anyhow::Result<NeighborResult<T>>
where
    T: FloatOpsTS + Send + Sync + 'static,
{
    let mut distance_triplets = Vec::with_capacity(n_samples * (k as usize + 1));
    for i in 0..n_samples {
        for (slot, &j) in indices[i].iter().enumerate() {
            distance_triplets.push((i, j, distances[i][slot]));
        }
    }

    let sigmas: Vec<T> = distances
        .par_iter()
        .map(|row| {
            let mut nonzero: Vec<T> = row.iter().filter(|&&d| d > T::zero()).copied().collect();
            if nonzero.is_empty() {
                return T::one();
            }
            let mid = nonzero.len() / 2;
            nonzero.select_nth_unstable_by(mid, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            nonzero[mid]
        })
        .collect();

    let min_weight = T::from_f64(1e-14).unwrap();
    let two = T::from(2).unwrap();
    let connectivity_triplets: Vec<(usize, usize, T)> = (0..n_samples)
        .into_par_iter()
        .fold(Vec::new, |mut acc, i| {
            // skip(1) drops the self hit; `i <= j` keeps each pair once
            for (slot, &j) in indices[i].iter().enumerate().skip(1) {
                if i > j {
                    continue;
                }
                let d = distances[i][slot];
                let den = sigmas[i] * sigmas[i] + sigmas[j] * sigmas[j];
                if den <= T::zero() {
                    continue;
                }
                let weight = (two * sigmas[i] * sigmas[j] / den).sqrt() * (-(d * d) / den).exp();
                if weight > min_weight {
                    acc.push((i, j, weight));
                    if i != j {
                        acc.push((j, i, weight));
                    }
                }
            }
            acc
        })
        .reduce(Vec::new, |mut a, b| {
            a.extend(b);
            a
        });

    Ok(NeighborResult {
        distances: csr_from_triplets(n_samples, distance_triplets)?,
        connectivities: csr_from_triplets(n_samples, connectivity_triplets)?,
    })
}

fn csr_from_triplets<T>(n: usize, triplets: Vec<(usize, usize, T)>) -> anyhow::Result<CsrMatrix<T>>
where
    T: FloatOpsTS + 'static,
{
    if triplets.is_empty() {
        return Ok(CsrMatrix::zeros(n, n));
    }
    let rows: Vec<usize> = triplets.iter().map(|(r, _, _)| *r).collect();
    let cols: Vec<usize> = triplets.iter().map(|(_, c, _)| *c).collect();
    let values: Vec<T> = triplets.into_iter().map(|(_, _, v)| v).collect();

    let coo = CooMatrix::try_from_triplets(n, n, rows, cols, values)
        .map_err(|e| anyhow::anyhow!("Failed to create COO matrix: {}", e))?;
    Ok(CsrMatrix::from(&coo))
}
