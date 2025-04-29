# single-clustering

A Rust library for community detection and graph clustering algorithms.

## Features

- **Network Analysis**: Efficient graph representation and manipulation for clustering tasks
- **Community Detection**: Implementation of state-of-the-art algorithms
    - Louvain method for community detection
    - Leiden algorithm (enhanced version of Louvain)
- **Flexible Grouping**: Abstract trait system for creating and managing node clusters
- **Performance**: Parallel computation support via Rayon
- **K-NN Graph Creation**: Build networks from high-dimensional data points

## Usage

```rust
use single_clustering::network::Network;
use single_clustering::network::grouping::VectorGrouping;
use single_clustering::community_search::leiden::Leiden;

// Create a network from your data
let network = Network::new_from_graph(graph);

// Initialize clustering (each node in its own cluster)
let mut clustering = VectorGrouping::create_isolated(network.nodes());

// Run Leiden algorithm (resolution parameter, randomness parameter, optional seed)
let mut leiden = Leiden::new(1.0, 0.01, Some(42));
leiden.iterate(&network, &mut clustering);

// Access clustering results
for node in 0..network.nodes() {
    println!("Node {} belongs to cluster {}", node, clustering.get_group(node));
}
```

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
single-clustering = "0.1.0"
```

## Performance Considerations

The library offers multiple implementations optimized for different scenarios:
- `StandardLocalMoving`: Basic implementation of the moving algorithm
- `FastLocalMoving`: Optimized version with better memory usage
- Parallel implementations of various operations for large networks

## License

This crate is licensed under the MIT License.