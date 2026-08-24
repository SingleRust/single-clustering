//! One switch for every rayon call in the crate.
//!
//! Order-preserving either way, so results don't depend on the `parallel` feature or on how
//! many threads it uses. `hnsw_rs` threads internally regardless; that one isn't ours.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Maps `0..n`, in parallel when `parallel` is on.
pub(crate) fn map_range<U, F>(n: usize, f: F) -> Vec<U>
where
    U: Send,
    F: Fn(usize) -> U + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n).map(f).collect()
    }
}

/// Fills `out` one `chunk`-sized row at a time, in parallel when `parallel` is on.
pub(crate) fn fill_chunks<T, F>(out: &mut [T], chunk: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        out.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(i, row)| f(i, row));
    }
    #[cfg(not(feature = "parallel"))]
    {
        out.chunks_mut(chunk)
            .enumerate()
            .for_each(|(i, row)| f(i, row));
    }
}
