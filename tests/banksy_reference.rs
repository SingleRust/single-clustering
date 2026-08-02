//! Differential tests for the BANKSY feature augmentation.
//!
//! Fixtures come from `banksy-py` itself via `tools/gen_banksy_fixtures.py` — the reference's
//! own `generate_spatial_weights_fixed_nbrs` and `concatenate_all`, not a reimplementation of
//! them. So this checks against what BANKSY computes rather than against a reading of the
//! paper. Fixtures are committed; the suite needs no Python.
//!
//! Three things this pins that a formula transcribed from the paper would get wrong: the λ
//! budget halves with each harmonic rather than splitting evenly, every block is z-scored
//! before scaling, and the gradient term is centred on its own neighbourhood first.
//!
//! Tolerance is 1e-6, not tighter, because the reference stores azimuths as `float32`
//! (`theta_data = np.zeros_like(..., dtype=np.float32)`). We carry f64 throughout, so the
//! residual is the reference's precision, not ours.

use serde_json::Value;
use single_clustering::spatial::banksy::{Decay, NeighbourhoodOperator, banksy_matrix};

fn fixtures() -> Value {
    let raw = include_str!("fixtures/banksy_reference.json");
    serde_json::from_str(raw).expect("fixture JSON should parse")
}

fn coords(v: &Value) -> Vec<[f64; 2]> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let a = p.as_array().unwrap();
            [a[0].as_f64().unwrap(), a[1].as_f64().unwrap()]
        })
        .collect()
}

/// Row-major flatten of a nested JSON matrix.
fn matrix(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn decay_of(name: &str) -> Decay {
    match name {
        "uniform" => Decay::Uniform,
        "reciprocal" => Decay::Reciprocal,
        "reciprocal_squared" => Decay::ReciprocalSquared,
        "scaled_gaussian" => Decay::ScaledGaussian,
        "ranked" => Decay::Ranked,
        other => panic!("unknown decay {other}"),
    }
}

/// Largest absolute difference, and where.
fn max_abs_diff(a: &[f64], b: &[f64]) -> (f64, usize) {
    assert_eq!(a.len(), b.len(), "matrix shapes differ");
    let mut worst = (0.0f64, 0usize);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}

/// Points on a regular grid have many exactly-equidistant neighbours, and nothing specifies
/// which of them a k-nearest search returns. Our tie-break is (distance, index); sklearn's is
/// unspecified, so that case is compared structurally elsewhere rather than value-for-value.
fn has_distance_ties(name: &str) -> bool {
    name.starts_with("grid")
}

/// Every harmonic must match the reference elementwise.
#[test]
fn harmonics_match_the_reference() {
    let f = fixtures();
    let mut checked = 0;

    for c in f["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        if has_distance_ties(name) {
            continue;
        }
        let pts = coords(&c["locations"]);
        let features = matrix(&c["features"]);
        let n_features = c["n_genes"].as_u64().unwrap() as usize;
        let ks: Vec<usize> = c["k_per_harmonic"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let decay = decay_of(c["decay"].as_str().unwrap());

        for (m, expected) in c["harmonics"].as_array().unwrap().iter().enumerate() {
            let op = NeighbourhoodOperator::for_harmonic(&pts, ks[m], m, decay).unwrap();
            let got = op.apply(&features, n_features).unwrap();
            let want = matrix(expected);

            let (diff, at) = max_abs_diff(&got, &want);
            assert!(
                diff < 1e-6,
                "{name} harmonic {m}: max |ours - reference| = {diff:e} at entry {at} \
                 (ours {}, reference {})",
                got[at],
                want[at]
            );
            checked += 1;
        }
    }
    assert!(checked >= 12, "only {checked} harmonics compared");
}

/// The assembled matrix — scaling, z-scoring and block layout together.
#[test]
fn augmented_matrix_matches_the_reference() {
    let f = fixtures();
    let mut checked = 0;

    for c in f["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        if has_distance_ties(name) {
            continue;
        }
        let pts = coords(&c["locations"]);
        let features = matrix(&c["features"]);
        let n_cells = c["n_cells"].as_u64().unwrap() as usize;
        let n_features = c["n_genes"].as_u64().unwrap() as usize;
        let ks: Vec<usize> = c["k_per_harmonic"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let max_m = c["max_m"].as_u64().unwrap() as usize;
        let decay = decay_of(c["decay"].as_str().unwrap());

        let harmonics: Vec<Vec<f64>> = (0..=max_m)
            .map(|m| {
                NeighbourhoodOperator::for_harmonic(&pts, ks[m], m, decay)
                    .unwrap()
                    .apply(&features, n_features)
                    .unwrap()
            })
            .collect();

        for (lambda_str, expected) in c["banksy"].as_object().unwrap() {
            let lambda: f64 = lambda_str.parse().unwrap();
            let got = banksy_matrix(&features, &harmonics, n_cells, n_features, lambda).unwrap();
            let want = matrix(expected);

            let (diff, at) = max_abs_diff(&got, &want);
            assert!(
                diff < 1e-6,
                "{name} at lambda {lambda}: max |ours - reference| = {diff:e} at entry {at} \
                 (ours {}, reference {})",
                got[at],
                want[at]
            );
            checked += 1;
        }
    }
    assert!(checked >= 10, "only {checked} matrices compared");
}

/// On a regular grid the neighbour *set* is ambiguous, but the neighbourhood mean over a
/// symmetric lattice should still land close — a large gap would mean something worse than a
/// tie-break difference.
#[test]
fn grid_case_agrees_up_to_tie_breaking() {
    let f = fixtures();
    for c in f["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        if !has_distance_ties(name) {
            continue;
        }
        let pts = coords(&c["locations"]);
        let features = matrix(&c["features"]);
        let n_features = c["n_genes"].as_u64().unwrap() as usize;
        let k = c["k_per_harmonic"][0].as_u64().unwrap() as usize;
        let decay = decay_of(c["decay"].as_str().unwrap());

        let op = NeighbourhoodOperator::for_harmonic(&pts, k, 0, decay).unwrap();
        let got = op.apply(&features, n_features).unwrap();
        let want = matrix(&c["harmonics"][0]);

        // Features are standard normal and weights sum to 1, so a mean is O(1/sqrt(k)).
        let (diff, _) = max_abs_diff(&got, &want);
        assert!(
            diff < 1.0,
            "{name}: means differ by {diff:e}, too much for tie-breaking alone"
        );
    }
}

/// λ = 0 must leave only the own-expression block carrying signal, λ = 1 only the
/// neighbourhood blocks. Checked against the reference, so the convention is theirs.
#[test]
fn lambda_endpoints_behave() {
    let f = fixtures();
    let c = f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "uniform_small")
        .expect("uniform_small carries lambda 0 and 1");

    let pts = coords(&c["locations"]);
    let features = matrix(&c["features"]);
    let n_cells = c["n_cells"].as_u64().unwrap() as usize;
    let n_features = c["n_genes"].as_u64().unwrap() as usize;
    let ks: Vec<usize> = c["k_per_harmonic"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let max_m = c["max_m"].as_u64().unwrap() as usize;
    let decay = decay_of(c["decay"].as_str().unwrap());

    let harmonics: Vec<Vec<f64>> = (0..=max_m)
        .map(|m| {
            NeighbourhoodOperator::for_harmonic(&pts, ks[m], m, decay)
                .unwrap()
                .apply(&features, n_features)
                .unwrap()
        })
        .collect();

    let width = n_features * (max_m + 2);

    let at_zero = banksy_matrix(&features, &harmonics, n_cells, n_features, 0.0).unwrap();
    let nbr_energy: f64 = (0..n_cells)
        .flat_map(|r| (n_features..width).map(move |c| (r, c)))
        .map(|(r, c)| at_zero[r * width + c].abs())
        .sum();
    assert!(
        nbr_energy < 1e-9,
        "lambda 0 left {nbr_energy:e} in the neighbour blocks"
    );

    let at_one = banksy_matrix(&features, &harmonics, n_cells, n_features, 1.0).unwrap();
    let own_energy: f64 = (0..n_cells)
        .flat_map(|r| (0..n_features).map(move |c| (r, c)))
        .map(|(r, c)| at_one[r * width + c].abs())
        .sum();
    assert!(
        own_energy < 1e-9,
        "lambda 1 left {own_energy:e} in the own block"
    );
}

/// The halving rule, stated explicitly so it cannot drift.
#[test]
fn scale_factors_halve_with_each_harmonic() {
    use single_clustering::spatial::banksy::scale_factors;

    // max_m = 0: one neighbour block takes all of lambda.
    let f = scale_factors(0, 0.2);
    assert!((f[0] - 0.8f64.sqrt()).abs() < 1e-12);
    assert!((f[1] - 0.2f64.sqrt()).abs() < 1e-12);

    // max_m = 1: two-thirds to the mean, one-third to the gradient — not half each.
    let f = scale_factors(1, 0.2);
    assert!((f[0] - 0.8f64.sqrt()).abs() < 1e-12);
    assert!((f[1] - (2.0 / 3.0 * 0.2f64).sqrt()).abs() < 1e-12);
    assert!((f[2] - (1.0 / 3.0 * 0.2f64).sqrt()).abs() < 1e-12);
    assert!(
        (f[1] / f[2] - 2f64.sqrt()).abs() < 1e-12,
        "the mean block should carry twice the gradient's variance"
    );

    // Squared factors always partition 1.
    for max_m in 0..4 {
        for lambda in [0.0, 0.2, 0.5, 0.8, 1.0] {
            let total: f64 = scale_factors(max_m, lambda).iter().map(|f| f * f).sum();
            assert!(
                (total - 1.0).abs() < 1e-12,
                "max_m {max_m}, lambda {lambda}"
            );
        }
    }
}
