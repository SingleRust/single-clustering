#!/usr/bin/env python3
"""Times igraph's Leiden on the graph written by `examples/vs_igraph.rs`.

    cargo run --release --no-default-features --example vs_igraph -- 200000 /tmp/bench
    uv run tools/igraph_bench.py /tmp/bench

Graph construction is excluded on both sides — only the clustering call is timed, so this
compares the algorithms rather than the surrounding pipeline.
"""

import sys
import time

import igraph as ig

directory = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench"
edges = []
with open(f"{directory}/edges.tsv") as fh:
    for line in fh:
        a, b = line.split()
        edges.append((int(a), int(b)))

n = max(max(a, b) for a, b in edges) + 1
build = time.perf_counter()
g = ig.Graph(n=n, edges=edges, directed=False)
build = time.perf_counter() - build
print(f"{n} nodes, {len(edges)} edges (igraph build {build:.3f}s, excluded)")

# Best of five, matching the Rust side. n_iterations=2 is our default; igraph's own default
# is 2 as well.
best, part = float("inf"), None
g.community_leiden(objective_function="modularity", resolution=1.0, n_iterations=2)
for _ in range(5):
    t = time.perf_counter()
    p = g.community_leiden(objective_function="modularity", resolution=1.0, n_iterations=2)
    best = min(best, time.perf_counter() - t)
    part = p

print(f"igraph {best:.4f}s  Q {g.modularity(part.membership):.4f}  k {len(part)}")
