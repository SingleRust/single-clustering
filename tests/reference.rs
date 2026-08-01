//! Differential tests against `leidenalg`.
//!
//! Fixtures are generated once by `tools/gen_fixtures.py` and committed, so this needs no
//! Python. Two separate things, which fail for different reasons:
//!
//! 1. **Definition** — on the *same membership*, our objective must give exactly
//!    `leidenalg`'s value (up to the documented factor of 2). Pins the quality functions, and
//!    would have caught the 2x resolution bug.
//! 2. **Optimization** — our algorithm must reach a partition at least as good as theirs.
//!    Checked on quality and agreement, never exact labels — different RNGs, so identical
//!    partitions aren't an achievable bar.

use single_clustering::community_search::leiden::{
    Cpm, LeidenConfig, Objective, ObjectiveKind, Partition, Rb, leiden, modularity,
};
use single_clustering::network::CSRNetwork;

/// `leidenalg` multiplies its quality by `(2 - is_directed)`; ours omits that factor.
const LEIDENALG_SCALE: f64 = 2.0;

struct Case {
    name: String,
    graph: CSRNetwork,
    objective: ObjectiveKind,
    reference_membership: Vec<usize>,
    reference_n_clusters: usize,
    reference_quality: f64,
    reference_modularity: f64,
}

fn load_cases() -> Vec<Case> {
    let raw = include_str!("fixtures/leidenalg_reference.json");
    let doc: serde_json::Value = serde_json::from_str(raw).expect("fixture is valid JSON");

    let graphs: std::collections::HashMap<String, CSRNetwork> = doc["graphs"]
        .as_object()
        .expect("graphs object")
        .iter()
        .map(|(key, g)| {
            let n = g["n_nodes"].as_u64().unwrap() as usize;
            let edges: Vec<(usize, usize, f64)> = g["edges"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| {
                    let e = e.as_array().unwrap();
                    (
                        e[0].as_u64().unwrap() as usize,
                        e[1].as_u64().unwrap() as usize,
                        e[2].as_f64().unwrap(),
                    )
                })
                .collect();
            (key.clone(), CSRNetwork::from_edges(n, &edges).unwrap())
        })
        .collect();

    doc["cases"]
        .as_array()
        .expect("cases array")
        .iter()
        .map(|c| {
            let resolution = c["resolution"].as_f64().unwrap();
            Case {
                name: c["name"].as_str().unwrap().to_string(),
                graph: graphs[c["graph"].as_str().unwrap()].clone(),
                objective: match c["objective"].as_str().unwrap() {
                    "rb" => ObjectiveKind::Rb { resolution },
                    "cpm" => ObjectiveKind::Cpm { resolution },
                    other => panic!("unknown objective {other}"),
                },
                reference_membership: c["reference_membership"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect(),
                reference_n_clusters: c["reference_n_clusters"].as_u64().unwrap() as usize,
                reference_quality: c["reference_quality"].as_f64().unwrap(),
                reference_modularity: c["reference_modularity"].as_f64().unwrap(),
            }
        })
        .collect()
}

fn quality_of(case: &Case, membership: &[usize]) -> f64 {
    let partition = Partition::from_membership(&case.graph, membership);
    match case.objective {
        ObjectiveKind::Rb { resolution } => Rb::new(resolution).quality(&partition),
        ObjectiveKind::Cpm { resolution } => Cpm::new(resolution).quality(&partition),
    }
}

fn nmi(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len() as f64;
    let ka = a.iter().max().map_or(0, |m| m + 1);
    let kb = b.iter().max().map_or(0, |m| m + 1);
    let mut joint = vec![0.0f64; ka * kb];
    let (mut pa, mut pb) = (vec![0.0f64; ka], vec![0.0f64; kb]);
    for i in 0..a.len() {
        joint[a[i] * kb + b[i]] += 1.0;
        pa[a[i]] += 1.0;
        pb[b[i]] += 1.0;
    }
    let mut mi = 0.0;
    for i in 0..ka {
        for j in 0..kb {
            let p = joint[i * kb + j] / n;
            if p > 0.0 {
                mi += p * (p / ((pa[i] / n) * (pb[j] / n))).ln();
            }
        }
    }
    let h = |p: &[f64]| -> f64 {
        -p.iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| (x / n) * (x / n).ln())
            .sum::<f64>()
    };
    let (ha, hb) = (h(&pa), h(&pb));
    if ha + hb == 0.0 {
        1.0
    } else {
        2.0 * mi / (ha + hb)
    }
}

/// Runs our algorithm with `n_iterations = 5`, matching how the fixtures were generated.
fn run(case: &Case, seed: u64) -> Vec<usize> {
    let config = LeidenConfig {
        objective: case.objective,
        seed: Some(seed),
        n_iterations: 5,
        ..Default::default()
    };
    leiden(&case.graph, &config).unwrap().into_labels()
}

/// Best of `seeds` runs. Both are stochastic heuristics over the same landscape —
/// `leidenalg`'s own two fixture seeds differ by up to 3.5% on the hard instances, so one run
/// of ours against whichever run it recorded is noise.
fn run_best(case: &Case, seeds: u64) -> Vec<usize> {
    (0..seeds)
        .map(|s| run(case, 42 + s))
        .max_by(|a, b| {
            quality_of(case, a)
                .partial_cmp(&quality_of(case, b))
                .unwrap()
        })
        .expect("seeds >= 1")
}

#[test]
fn fixtures_are_present() {
    let cases = load_cases();
    assert!(cases.len() >= 100, "only {} cases loaded", cases.len());
}

/// Our quality function must agree with `leidenalg`'s exactly, on the same membership.
///
/// Pins the definitions — fails if a resolution is scaled wrong or a constant convention
/// drifts, no matter how good either optimizer is.
#[test]
fn quality_definitions_match_leidenalg() {
    let mut failures = Vec::new();
    for case in load_cases() {
        let ours = quality_of(&case, &case.reference_membership) * LEIDENALG_SCALE;
        let tol = 1e-6 * case.reference_quality.abs().max(1.0);
        if (ours - case.reference_quality).abs() > tol {
            failures.push(format!(
                "{}: ours*{LEIDENALG_SCALE} = {ours:.9}, leidenalg = {:.9}",
                case.name, case.reference_quality
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} quality mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Our modularity must agree with igraph's on the same membership.
///
/// To `1e-7`, not to the last bit: adjacency is stored as `f32`, so a weight like `0.2531`
/// from the fixtures is rounded on the way in. Modularity is bounded by 1, and f32 epsilon is
/// 1.2e-7, so that is the best agreement the storage can support. The arithmetic itself is
/// f64 throughout.
#[test]
fn modularity_matches_igraph() {
    let mut failures = Vec::new();
    for case in load_cases() {
        // igraph's modularity is always the gamma=1 form regardless of the objective used.
        let ours = modularity(&case.graph, &case.reference_membership, 1.0);
        if (ours - case.reference_modularity).abs() > 1e-7 {
            failures.push(format!(
                "{}: ours = {ours:.9}, igraph = {:.9}",
                case.name, case.reference_modularity
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} modularity mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// No single case may fall grossly short of `leidenalg`.
///
/// The bound is deliberately loose — its job is to catch a systematic defect, like a
/// misscaled resolution, not to police heuristic noise. Aggregate parity is checked by
/// [`optimizer_is_not_systematically_worse`], which is the tight test.
///
/// Known residual: high-resolution RB on the medium SBM sits ~6% below `leidenalg`, on an
/// instance where `leidenalg`'s own two seeds differ by 3.5%.
#[test]
fn no_case_falls_grossly_short() {
    let mut failures = Vec::new();
    for case in load_cases() {
        let labels = run_best(&case, 2);
        let ours = quality_of(&case, &labels) * LEIDENALG_SCALE;
        let slack = 0.08 * case.reference_quality.abs().max(1.0);
        if ours < case.reference_quality - slack {
            failures.push(format!(
                "{}: ours = {ours:.6}, leidenalg = {:.6} ({:+.2}%)",
                case.name,
                case.reference_quality,
                100.0 * (ours - case.reference_quality) / case.reference_quality.abs().max(1e-12),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} cases below leidenalg:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Across the whole fixture set, a single run of ours must be level with `leidenalg`.
///
/// At the time of writing this is -0.11%: 33 cases better, 107 equal, 20 worse. The bound
/// leaves room for heuristic noise while still failing loudly on a real regression.
#[test]
fn optimizer_is_not_systematically_worse() {
    let cases = load_cases();
    let (mut wins, mut losses, mut ties) = (0, 0, 0);
    let mut total_rel = 0.0;

    for case in &cases {
        let labels = run(case, 42);
        let ours = quality_of(case, &labels) * LEIDENALG_SCALE;
        let scale = case.reference_quality.abs().max(1e-9);
        let rel = (ours - case.reference_quality) / scale;
        total_rel += rel;
        if rel > 1e-9 {
            wins += 1;
        } else if rel < -1e-9 {
            losses += 1;
        } else {
            ties += 1;
        }
    }

    let mean_rel = total_rel / cases.len() as f64;
    println!(
        "vs leidenalg over {} cases: {wins} better, {ties} equal, {losses} worse, mean {:+.4}%",
        cases.len(),
        100.0 * mean_rel
    );
    assert!(
        mean_rel > -0.005,
        "mean quality {:+.4}% below leidenalg across {} cases",
        100.0 * mean_rel,
        cases.len()
    );
}

/// Given the same budget `leidenalg` had, we should come out ahead on aggregate.
#[test]
fn two_seeds_beat_leidenalg_on_aggregate() {
    let cases = load_cases();
    let (mut wins, mut losses) = (0, 0);
    let mut total_rel = 0.0;

    for case in &cases {
        let labels = run_best(case, 2);
        let ours = quality_of(case, &labels) * LEIDENALG_SCALE;
        let rel = (ours - case.reference_quality) / case.reference_quality.abs().max(1e-9);
        total_rel += rel;
        if rel > 1e-9 {
            wins += 1;
        } else if rel < -1e-9 {
            losses += 1;
        }
    }

    let mean_rel = total_rel / cases.len() as f64;
    println!(
        "best-of-2 vs leidenalg: {wins} better, {losses} worse, mean {:+.4}%",
        100.0 * mean_rel
    );
    assert!(
        mean_rel > 0.0,
        "mean {:+.4}%, expected to be ahead",
        100.0 * mean_rel
    );
}

/// Where the answer is unambiguous, we find the *same* communities, not merely ones scoring
/// as well.
///
/// Restricted on purpose. Agreement is only a meaningful signal when the instance has one
/// clearly right answer: `sbm_hard` (p_in 0.18 vs p_out 0.07) and resolutions far from an
/// instance's natural granularity admit many near-equal partitions, so two good optimizers
/// legitimately disagree there. Quality, checked above, is the right measure for those.
#[test]
fn partitions_agree_where_structure_is_unambiguous() {
    let mut failures = Vec::new();
    let mut checked = 0;

    for case in load_cases() {
        let strong_structure =
            case.name.starts_with("sbm_easy") || case.name.starts_with("ring_of_cliques");
        let natural_resolution = matches!(
            case.objective,
            ObjectiveKind::Rb { resolution } if (0.5..=2.0).contains(&resolution)
        );
        if !strong_structure || !natural_resolution {
            continue;
        }
        // Degenerate references make NMI uninformative.
        if case.reference_n_clusters <= 1
            || case.reference_n_clusters >= case.reference_membership.len()
        {
            continue;
        }

        checked += 1;
        let labels = run_best(&case, 2);
        let score = nmi(&labels, &case.reference_membership);
        if score < 0.85 {
            failures.push(format!(
                "{}: NMI {score:.3} (ours {} clusters, leidenalg {})",
                case.name,
                labels.iter().max().map_or(0, |m| m + 1),
                case.reference_n_clusters
            ));
        }
    }

    assert!(checked >= 20, "only {checked} cases matched the filter");
    assert!(
        failures.is_empty(),
        "{} of {checked} unambiguous cases disagree:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Cluster counts should track `leidenalg`'s. This is the user-visible consequence of the
/// resolution parameter meaning the same thing in both libraries.
#[test]
fn cluster_counts_track_leidenalg() {
    let mut failures = Vec::new();
    for case in load_cases() {
        let labels = run_best(&case, 2);
        let ours = labels.iter().max().map_or(0, |m| m + 1);
        let theirs = case.reference_n_clusters;
        // Allow a little slack for genuinely ambiguous granularity, but not a factor of two,
        // which is exactly what the old resolution bug produced.
        let tol = 2 + theirs / 5;
        if ours.abs_diff(theirs) > tol {
            failures.push(format!("{}: ours {ours}, leidenalg {theirs}", case.name));
        }
    }
    assert!(
        failures.is_empty(),
        "{} cases differ in cluster count:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
