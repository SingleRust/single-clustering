//! Stability and validity of the spatial neighbour graphs.
//!
//! The unit tests prove each builder produces the graph it claims to. That is necessary but
//! not sufficient: what matters on real data is whether what you get out survives what real
//! data does to you — cells missed by segmentation, coordinates off by a fraction of a cell,
//! a parameter chosen slightly differently.
//!
//! # What a bare spatial graph can and cannot do
//!
//! It carries no expression, so the only structure it encodes is spatial: which cells are
//! near which. Two consequences, and both are asserted here.
//!
//! It reliably recovers **physical separation** — detached fragments, tissue masked apart,
//! anything with a real gap. That shows up as connected components, needs no resolution
//! choice, and is stable under every perturbation tried below.
//!
//! It does **not** find boundaries inside continuous tissue. There is nothing for the
//! objective to anchor to, so it returns a tiling whose blob size is set by the resolution.
//! The tiling is spatially contiguous and looks exactly like a result, but it is arbitrary,
//! and therefore unstable: resample and the boundaries move. That is asserted too, so the
//! limitation cannot quietly stop being true.
//!
//! So the resolution-free question — "which cells could possibly be in the same domain" — is
//! the one a spatial graph answers. Choosing among the domains inside a connected region
//! needs expression, which arrives with neighbourhood feature averaging.

mod common;

use common::{adjusted_rand_index, disconnected_community_count};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use single_clustering::community_search::leiden::{LeidenConfig, ObjectiveKind, leiden};
use single_clustering::network::CSRNetwork;
use single_clustering::spatial::{
    HullPruning, Lattice, SpatialWeight, Symmetry, delaunay_graph, knn_graph, lattice_graph,
    radius_graph,
};

/// Four tissue islands with clear space between them.
fn separated_tissue(seed: u64) -> (Vec<[f64; 2]>, Vec<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centres = [[20.0, 20.0], [80.0, 20.0], [20.0, 80.0], [80.0, 80.0]];
    let radius = 18.0;

    let mut pts = Vec::new();
    let mut truth = Vec::new();
    for (i, c) in centres.iter().enumerate() {
        for _ in 0..700 {
            let a = rng.random::<f64>() * std::f64::consts::TAU;
            let r = radius * rng.random::<f64>().sqrt();
            pts.push([c[0] + r * a.cos(), c[1] + r * a.sin()]);
            truth.push(i);
        }
    }
    (pts, truth)
}

/// Featureless tissue at one density — nothing for a spatial objective to find.
fn homogeneous_tissue(seed: u64) -> Vec<[f64; 2]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..2500)
        .map(|_| [rng.random::<f64>() * 100.0, rng.random::<f64>() * 100.0])
        .collect()
}

/// Connected component of each node — the resolution-free structure a spatial graph encodes.
fn components(graph: &CSRNetwork) -> Vec<usize> {
    let n = graph.node_count();
    let mut label = vec![usize::MAX; n];
    let mut next = 0;
    for start in 0..n {
        if label[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        label[start] = next;
        while let Some(v) = stack.pop() {
            for (u, _) in graph.neighbors(v) {
                if label[u] == usize::MAX {
                    label[u] = next;
                    stack.push(u);
                }
            }
        }
        next += 1;
    }
    label
}

fn cluster(graph: &CSRNetwork, resolution: f64) -> Vec<usize> {
    let config = LeidenConfig {
        objective: ObjectiveKind::Cpm { resolution },
        ..Default::default()
    };
    leiden(graph, &config).unwrap().labels().to_vec()
}

const BUILDERS: [&str; 3] = ["radius", "knn", "delaunay"];

fn build(name: &str, pts: &[[f64; 2]]) -> CSRNetwork {
    match name {
        "radius" => radius_graph(pts, 4.0, SpatialWeight::Uniform, None).unwrap(),
        "knn" => knn_graph(pts, 8, Symmetry::Union, SpatialWeight::Uniform).unwrap(),
        _ => delaunay_graph(pts, HullPruning::default(), SpatialWeight::Uniform).unwrap(),
    }
}

/// Validity: separated tissue must come back as separate components — exactly, and with no
/// resolution to choose.
#[test]
fn every_builder_separates_disconnected_tissue() {
    let (pts, truth) = separated_tissue(1);
    for name in BUILDERS {
        let agreement = adjusted_rand_index(&components(&build(name, &pts)), &truth);
        assert!(
            (agreement - 1.0).abs() < 1e-9,
            "{name}: components do not match the islands (ARI {agreement:.4})"
        );
    }
}

/// Validity: a domain may subdivide an island, but must never span two. That holds at every
/// resolution, which is what makes it the meaningful guarantee.
#[test]
fn no_domain_ever_spans_separated_tissue() {
    let (pts, truth) = separated_tissue(7);
    for name in BUILDERS {
        let graph = build(name, &pts);
        for resolution in [0.01, 0.05, 0.2, 1.0] {
            let labels = cluster(&graph, resolution);
            let mut owner = std::collections::HashMap::new();
            for (v, &c) in labels.iter().enumerate() {
                let island = *owner.entry(c).or_insert(truth[v]);
                assert_eq!(
                    island, truth[v],
                    "{name} at resolution {resolution}: a domain spans two islands"
                );
            }
        }
    }
}

/// Stability: segmentation misses cells, so dropping some must not change the structure.
#[test]
fn structure_survives_subsampling() {
    let (pts, _) = separated_tissue(2);
    for name in BUILDERS {
        let full = components(&build(name, &pts));

        for drop in [0.05, 0.15, 0.3] {
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            let keep: Vec<usize> = (0..pts.len())
                .filter(|_| rng.random::<f64>() > drop)
                .collect();
            let sub: Vec<[f64; 2]> = keep.iter().map(|&i| pts[i]).collect();
            let restricted: Vec<usize> = keep.iter().map(|&i| full[i]).collect();

            let agreement = adjusted_rand_index(&components(&build(name, &sub)), &restricted);
            assert!(
                agreement > 0.99,
                "{name}: dropping {:.0}% of cells changed the structure ({agreement:.4})",
                drop * 100.0
            );
        }
    }
}

/// Stability: segmentation centroids are not exact, so jitter below cell spacing must not
/// move anything.
#[test]
fn structure_survives_coordinate_jitter() {
    let (pts, _) = separated_tissue(3);
    for name in BUILDERS {
        let reference = components(&build(name, &pts));

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let jittered: Vec<[f64; 2]> = pts
            .iter()
            .map(|p| {
                [
                    p[0] + (rng.random::<f64>() - 0.5) * 0.5,
                    p[1] + (rng.random::<f64>() - 0.5) * 0.5,
                ]
            })
            .collect();

        let agreement = adjusted_rand_index(&components(&build(name, &jittered)), &reference);
        assert!(
            agreement > 0.99,
            "{name}: coordinate jitter changed the structure ({agreement:.4})"
        );
    }
}

/// The builders define "neighbour" differently, but must agree about real structure.
#[test]
fn builders_agree_on_real_structure() {
    let (pts, _) = separated_tissue(5);
    let labels: Vec<(&str, Vec<usize>)> = BUILDERS
        .iter()
        .map(|&name| (name, components(&build(name, &pts))))
        .collect();

    for i in 0..labels.len() {
        for j in (i + 1)..labels.len() {
            let agreement = adjusted_rand_index(&labels[i].1, &labels[j].1);
            assert!(
                agreement > 0.99,
                "{} and {} disagree ({agreement:.4})",
                labels[i].0,
                labels[j].0
            );
        }
    }
}

/// Stability: neighbouring parameter values must give the same structure. A cliff would mean
/// the parameter is load-bearing in a way users cannot reason about.
#[test]
fn parameters_do_not_change_the_structure() {
    let (pts, truth) = separated_tissue(4);

    for radius in [2.5f64, 3.0, 3.5, 4.0, 5.0] {
        let g = radius_graph(&pts, radius, SpatialWeight::Uniform, None).unwrap();
        let agreement = adjusted_rand_index(&components(&g), &truth);
        assert!(
            agreement > 0.99,
            "radius {radius} broke the structure ({agreement:.4})"
        );
    }
    for k in [4usize, 6, 8, 12, 16] {
        let g = knn_graph(&pts, k, Symmetry::Union, SpatialWeight::Uniform).unwrap();
        let agreement = adjusted_rand_index(&components(&g), &truth);
        assert!(
            agreement > 0.99,
            "k {k} broke the structure ({agreement:.4})"
        );
    }
    for factor in [1.5f64, 2.0, 2.5, 3.0] {
        let g = delaunay_graph(
            &pts,
            HullPruning::Adaptive { factor },
            SpatialWeight::Uniform,
        )
        .unwrap();
        let agreement = adjusted_rand_index(&components(&g), &truth);
        assert!(
            agreement > 0.99,
            "factor {factor} broke the structure ({agreement:.4})"
        );
    }
}

/// The documented limitation, asserted so it cannot quietly stop being true.
///
/// On uniform tissue there is nothing spatial to find, so the result is an arbitrary tiling:
/// contiguous, plausible-looking, and not reproducible under resampling. Anyone clustering a
/// bare spatial graph on continuous tissue is reading tea leaves, and this is the evidence.
#[test]
fn homogeneous_tissue_gives_an_arbitrary_unstable_tiling() {
    let pts = homogeneous_tissue(4);
    let graph = build("delaunay", &pts);
    let reference = cluster(&graph, 0.1);

    assert_eq!(
        components(&graph).iter().max(),
        Some(&0),
        "one tissue region"
    );
    assert!(
        reference.iter().max().unwrap() > &3,
        "yet the clustering carves it into many blobs"
    );
    assert_eq!(
        disconnected_community_count(&graph, &reference),
        0,
        "each of them contiguous, which is what makes the tiling look like a result"
    );

    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let keep: Vec<usize> = (0..pts.len())
        .filter(|_| rng.random::<f64>() > 0.1)
        .collect();
    let sub: Vec<[f64; 2]> = keep.iter().map(|&i| pts[i]).collect();
    let restricted: Vec<usize> = keep.iter().map(|&i| reference[i]).collect();
    let agreement = adjusted_rand_index(&cluster(&build("delaunay", &sub), 0.1), &restricted);

    assert!(
        agreement < 0.8,
        "a tiling of featureless tissue reproduced at ARI {agreement:.3}; if that now holds \
         up, the limitation documented in this file no longer applies and the module docs \
         need revisiting"
    );
}

/// Whatever the builder or resolution, a domain must be one connected region of tissue.
#[test]
fn domains_are_always_spatially_contiguous() {
    let (pts, _) = separated_tissue(6);
    for name in BUILDERS {
        let graph = build(name, &pts);
        for resolution in [0.01, 0.05, 0.1, 0.3, 1.0] {
            let labels = cluster(&graph, resolution);
            assert_eq!(
                disconnected_community_count(&graph, &labels),
                0,
                "{name} at resolution {resolution} split a domain across space"
            );
        }
    }
}

/// The lattice path: tissue detection masking a band must split the capture area.
#[test]
fn lattice_masking_splits_the_capture_area_cleanly() {
    let (mut rows, mut cols, mut truth) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..78u32 {
        if (30..48).contains(&r) {
            continue;
        }
        for i in 0..64u32 {
            rows.push(r);
            cols.push(2 * i + (r % 2));
            truth.push(usize::from(r >= 48));
        }
    }
    let g = lattice_graph(&rows, &cols, Lattice::VisiumHex).unwrap();
    let agreement = adjusted_rand_index(&components(&g), &truth);
    assert!(
        (agreement - 1.0).abs() < 1e-9,
        "the masked-apart regions were not separated ({agreement:.4})"
    );
    assert_eq!(disconnected_community_count(&g, &cluster(&g, 0.05)), 0);
}
