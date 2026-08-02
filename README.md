# single-clustering

⚠️ **Development Status**: This library is under active development. APIs may change between
versions; see the changelog for 0.7.0, which is a breaking rewrite of the Leiden core.

A Rust library for community detection and graph clustering, focused on being correct,
reproducible, and fast enough for single-cell-scale graphs.

## Features

- **Efficient network representation**: CSR (compressed sparse row) storage with the
  igraph/leidenalg weight conventions, including correct self-loop handling so that a
  partition's quality is invariant under aggregation
- **Leiden algorithm** with the refinement phase, so communities are internally connected
- **Quality functions**: Reichardt–Bornholdt configuration model (modularity at
  `resolution = 1.0`, matching `scanpy`) and CPM, which has no resolution limit
- **Reproducible**: a fixed seed gives bit-for-bit identical results
- **Linear scaling**: 100k nodes / 840k edges in ~110 ms single-threaded
- **Compact**: 16 bytes per undirected edge, and a zero-copy path for connectivity matrices
  you already hold
- **k-NN graph construction** from high-dimensional data (optional `knn` feature)

## Accuracy

Correctness is checked against the implementations people actually use, not against itself.
Fixtures generated from `leidenalg` and `banksy-py` are committed, so the test suite needs no
Python.

**Against `leidenalg`** — 160 committed fixtures spanning karate, LFR at several mixing
parameters, SBM and a real k-NN graph, across resolutions and seeds, for both RB and CPM:

| | modularity vs `leidenalg` |
|---|---|
| single seed | −0.09% mean (70 better, 70 equal, 20 worse) |
| best of 2 seeds | +0.42% mean |

Both libraries are stochastic heuristics over the same landscape — `leidenalg`'s own two
fixture seeds differ by up to 3.5% on the harder instances — so a single-run comparison is
mostly noise, and the sign of the mean is not meaningful. What the fixtures do pin exactly is
the *definition*: on the same membership our quality function reproduces `leidenalg`'s value
to floating point, which is what caught the 0.6.x factor-of-two resolution bug.

**Against `igraph`**: +0.03% mean modularity.

**Against brute force** — every set partition enumerated on graphs small enough to allow it,
so this compares to the true optimum rather than to another heuristic:

| | reaches the exact optimum |
|---|---|
| this crate | 93.5% |
| `igraph` | 94.4–96.3% |

**Against `banksy-py`**: the feature augmentation reproduces the reference elementwise to
**1e-6** across five decay kernels, harmonics 0–2, and λ from 0 to 1. The residual is the
reference's own precision — it stores azimuths as `float32`.

### On real data

PBMC3k (2,638 cells, 15-NN), clustering the *same* connectivity matrix `scanpy` did, so
every difference is the algorithm rather than graph construction:

| resolution | clusters (ours / scanpy) | modularity (ours / scanpy) |
|---|---|---|
| 0.1 | 3 / 3 | 0.9511 / 0.9511 |
| 0.5 | 6 / 5 | 0.8063 / 0.8048 |
| 1.0 | 8 / 8 | 0.6715 / 0.6696 |
| 2.0 | 20 / 19 | 0.5304 / 0.5315 |
| 4.0 | 50 / 49 | 0.4217 / 0.4240 |

Cluster counts track `scanpy` within one at every resolution, so resolution means the same
thing in both. Against the authors' cell-type annotations — the only measure here about
biology rather than about matching another implementation — **ours scores ARI 0.8599 and
`scanpy` 0.8609**, both peaking at resolution 0.75.

## Performance

Timings are against `scanpy`/`igraph`, which is a C implementation called from Python, not
interpreted code — so these are not interpreter-overhead numbers.

| workload | this crate | `scanpy` / `igraph` |
|---|---|---|
| PBMC3k, per resolution | 7–11 ms | 22–24 ms |
| synthetic k-NN graphs | — | **1.33× slower** (geometric mean) |

Single-threaded scaling on synthetic k-NN graphs:

| nodes | edges | time |
|---|---|---|
| 20k | 170k | 18 ms |
| 100k | 840k | 111 ms |
| 3M | 25M | 6.8 s |
| 8M | 57M | 20.1 s |

Spatial graph construction:

| | |
|---|---|
| Visium HD lattice, 10.5M bins / 21.1M edges | 954 ms, 591 MB |
| 500k cells, k-NN (k=6) | 299 ms |
| 500k cells, radius | 80 ms |

## Usage

```rust
use single_clustering::network::CSRNetwork;
use single_clustering::community_search::leiden::{leiden, modularity, LeidenConfig, ObjectiveKind};

# fn main() -> single_clustering::Result<()> {
// Build a graph from an edge list. Each undirected edge is given once.
let edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0), (3, 5, 1.0)];
let graph = CSRNetwork::from_edges(6, &edges)?;

let config = LeidenConfig {
    objective: ObjectiveKind::Rb { resolution: 1.0 },
    seed: Some(42),
    ..Default::default()
};

let clustering = leiden(&graph, &config)?;

println!("{} communities", clustering.n_clusters());
for (node, &label) in clustering.labels().iter().enumerate() {
    println!("node {node} is in community {label}");
}
println!("modularity: {:.4}", modularity(&graph, clustering.labels(), 1.0));
# Ok(())
# }
```

### Large graphs

`CSRNetwork::from_edges` needs the caller's edge list resident alongside the graph. When the
data is already a connectivity matrix — as it is coming out of a k-NN step —
`from_csr_parts` takes ownership of the CSR buffers instead, so nothing proportional to the
edge count is copied:

```rust,no_run
use single_clustering::network::CSRNetwork;
# fn main() -> single_clustering::Result<()> {
# let (node_ptrs, neighbors, weights) = (vec![0usize, 1, 2], vec![1u32, 0], vec![1.0f32, 1.0]);
// node_ptrs: row offsets; neighbors: column indices; weights: values.
// Must be the full symmetric adjacency. Rows need not be sorted.
let graph = CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None)?;
println!("{:.1} GB", graph.memory_bytes() as f64 / 1e9);
# Ok(())
# }
```

Measured peak memory, clustering included:

| nodes | edges | via `from_edges` | via `from_csr_parts` |
|---|---|---|---|
| 4M | 28M | 2.0 GB | 1.4 GB |
| 8M | 57M | 3.7 GB | 2.6 GB |

The `from_edges` figures include the caller's edge list, which `from_csr_parts` never needs.

Adjacency is stored as `u32` ids and `f32` weights (16 bytes per undirected edge); all
arithmetic is `f64`. That caps graphs at ~4.29 billion nodes and means quality values agree
across aggregation levels to `f32` precision rather than exactly — a tradeoff that measurably
costs nothing in cluster quality, and none at all when the input was `f32` to begin with, as
k-NN connectivities normally are.

### Choosing a resolution

`ObjectiveKind::Rb { resolution }` means the same thing it does in `scanpy` and `leidenalg`:
higher values give more, smaller communities, and `1.0` is standard modularity. If you are
porting parameters from a Python pipeline, they carry over directly.

`ObjectiveKind::Cpm { resolution }` measures communities in node weight rather than degree and
has no resolution limit, which makes it better behaved when sweeping resolution on large
graphs.

## Installation

```toml
[dependencies]
single-clustering = "0.7"
```

The k-NN graph construction is behind the default-on `knn` feature. To build just the
clustering core — useful in CI, or on targets where the HNSW stack does not compile:

```toml
single-clustering = { version = "0.7", default-features = false }
```

## Current status

- ✅ **Leiden algorithm**: local moving, refinement, and aggregation
- ✅ **CSR network representation**
- ✅ **Quality functions**: RB configuration model and CPM
- ✅ **Reproducibility**: deterministic under a fixed seed
- ✅ **Spatial neighbour graphs**: Visium/HD lattice, radius, k-NN, Delaunay with adaptive
  pruning, graph fusion, and per-sample construction for multi-slice experiments
- ✅ **BANKSY feature augmentation**: neighbourhood mean and azimuthal gradient, so ordinary
  Leiden finds spatial domains
- 🚧 **Louvain**: available as `LeidenConfig { refine: false, .. }`; no separate entry point
- 🚧 **Benchmarks**: `cargo run --release --example scaling`
- ❌ **Parallel local moving**: planned, deliberately deferred until the sequential path is
  measured. Profiling puts 94% of a pass in level 0, of which local moving is 65% and
  refinement 29%, so that is where it would go
- ❌ **DBSCAN / HDBSCAN**: planned
- ❌ **Python bindings**: PyO3 integration (planned)

### Not yet measured

Spatial *domain detection* has no comparative result. The pieces exist and each is verified
against its reference, but the end-to-end number — ARI against manual annotations on the
DLPFC benchmark — has not been produced. Until it has, nothing here should be read as a claim
about spatial domain accuracy. The target is pre-registered in
[`docs/spatial-benchmark.md`](docs/spatial-benchmark.md), along with what the established
methods score and which of their published numbers reproduce independently.

## Contributing

This project is in active development. Contributions, bug reports, and feature requests are
welcome!

## License

This crate is licensed under the BSD 3-Clause License.
