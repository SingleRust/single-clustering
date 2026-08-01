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
- 🚧 **Louvain**: available as `LeidenConfig { refine: false, .. }`; no separate entry point
- 🚧 **Benchmarks**: `cargo run --release --example scaling`
- ❌ **Parallel local moving**: planned, deliberately deferred until the sequential path is
  measured
- ❌ **DBSCAN / HDBSCAN / spatial-aware clustering**: planned
- ❌ **Python bindings**: PyO3 integration (planned)

## Contributing

This project is in active development. Contributions, bug reports, and feature requests are
welcome!

## License

This crate is licensed under the BSD 3-Clause License.
