# single-clustering

⚠️ **Development Status**: This library is currently under heavy development and should **not be considered production ready**. APIs may change significantly between versions.

A Rust library for community detection and graph clustering algorithms with a focus on performance and flexibility.

## Features

- **Efficient Network Representation**: CSR (Compressed Sparse Row) format for optimal memory usage and performance
- **Community Detection Algorithms**:
    - **Leiden Algorithm**: State-of-the-art method with guaranteed well-connected communities
    - **Louvain Method**: Classic modularity optimization (work-in-progress)
- **Quality Functions**: Multiple partition quality metrics
    - Modularity optimization
    - Reichardt-Bornholdt (RB) configuration model with tunable resolution
- **Flexible Architecture**: Generic trait-based design supporting different network types and grouping strategies
- **Performance Optimized**: Caching strategies and efficient data structures for large networks

## Usage

```rust
use single_clustering::network::CSRNetwork;
use single_clustering::community_search::leiden::{LeidenOptimizer, LeidenConfig};
use single_clustering::community_search::leiden::partition::ModularityPartition;

// Create a CSR network from your data
let network = CSRNetwork::new(edges, weights, node_count);

// Configure the Leiden algorithm
let config = LeidenConfig {
    max_iterations: 100,
    tolerance: 1e-6,
    seed: Some(42),
    ..Default::default()
};

// Initialize the optimizer
let mut optimizer = LeidenOptimizer::new(config);

// Find communities using modularity optimization
let partition: ModularityPartition<f64, _> = optimizer.find_partition(network)?;

// Access results
for node in 0..partition.node_count() {
    println!("Node {} is in community {}", node, partition.membership(node));
}
println!("Modularity: {:.4}", partition.quality());
```

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
single-clustering = "0.6.0"
```

## Current Status

- ✅ **Leiden Algorithm**: Core implementation with modularity and RB quality functions
- ✅ **CSR Network Representation**: Efficient storage for large graphs
- ✅ **Quality Functions**: Modularity and Reichardt-Bornholdt implementations
- 🚧 **Louvain Algorithm**: Basic implementation (work-in-progress)
- 🚧 **Documentation**: API documentation and examples (ongoing)
- ❌ **Benchmarks**: Performance testing suite (planned)
- ❌ **Python Bindings**: PyO3 integration (planned)

## Contributing

This project is in active development. Contributions, bug reports, and feature requests are welcome!

## License

This crate is licensed under the BSD 3-Clause License.