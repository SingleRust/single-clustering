#!/usr/bin/env python3
"""Generate reference clusterings from `leidenalg` for the differential test suite.

Run once; the generated JSON is committed, so CI never needs Python.

    uv venv --python 3.12 .refvenv
    uv pip install --python .refvenv/bin/python leidenalg python-igraph
    .refvenv/bin/python tools/gen_fixtures.py

`leidenalg` wheels are unreliable on Python 3.14 — pin 3.12.

Each fixture records the graph, the objective, and, for the partition leidenalg found:

* ``reference_quality`` — leidenalg's own quality value. Our objective must agree with it
  exactly (up to the documented factor of 2) *for the same membership*, which is what pins
  the definitions rather than merely the outcome.
* ``reference_modularity`` — igraph's modularity of that membership, used to check that our
  optimizer reaches partitions as good as theirs.
"""

import json
import pathlib
import random

import igraph as ig
import leidenalg as la

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"

KARATE = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11),
    (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13),
    (1, 17), (1, 19), (1, 21), (1, 30), (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27),
    (2, 28), (2, 32), (3, 7), (3, 12), (3, 13), (4, 6), (4, 10), (5, 6), (5, 10), (5, 16),
    (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32), (14, 33), (15, 32),
    (15, 33), (18, 32), (18, 33), (19, 33), (20, 32), (20, 33), (22, 32), (22, 33),
    (23, 25), (23, 27), (23, 29), (23, 32), (23, 33), (24, 25), (24, 27), (24, 31),
    (25, 31), (26, 29), (26, 33), (27, 33), (28, 31), (28, 33), (29, 32), (29, 33),
    (30, 32), (30, 33), (31, 32), (31, 33), (32, 33),
]


def sbm(n_per, blocks, p_in, p_out, seed):
    rng = random.Random(seed)
    n = n_per * blocks
    truth = [i // n_per for i in range(n)]
    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < (p_in if truth[i] == truth[j] else p_out)
    ]
    return n, edges, truth


def ring_of_cliques(n_cliques, clique_size):
    """Cliques in a ring — the classic resolution-limit case, where RB and CPM disagree."""
    edges, n = [], n_cliques * clique_size
    for c in range(n_cliques):
        base = c * clique_size
        edges += [
            (base + i, base + j)
            for i in range(clique_size)
            for j in range(i + 1, clique_size)
        ]
        edges.append((base, ((c + 1) % n_cliques) * clique_size))
    return n, edges


def weighted(edges, seed):
    """Attach non-unit weights, so the fixtures also cover the weighted path."""
    rng = random.Random(seed)
    return [round(0.25 + 3.5 * rng.random(), 4) for _ in edges]


def make_case(graph_key, n, edges, weights, kind, resolution, seed):
    g = ig.Graph(n=n, edges=edges)
    g.es["weight"] = weights

    cls = la.RBConfigurationVertexPartition if kind == "rb" else la.CPMVertexPartition
    part = la.find_partition(
        g, cls, weights="weight", resolution_parameter=resolution, seed=seed, n_iterations=5
    )
    membership = list(part.membership)

    return {
        "name": f"{graph_key}/{kind}/res={resolution}/seed={seed}",
        # Graphs are stored once under `graphs` and referenced by key: the same graph is
        # reused across ~20 configurations, and inlining it made the fixture 15x larger.
        "graph": graph_key,
        "objective": kind,
        "resolution": resolution,
        "seed": seed,
        "reference_membership": membership,
        "reference_n_clusters": len(set(membership)),
        "reference_quality": float(part.quality()),
        "reference_modularity": float(g.modularity(membership, weights="weight")),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    graphs = {}

    graphs["karate"] = (34, KARATE)
    for tag, args in {
        "sbm_easy": (30, 4, 0.40, 0.02, 1),
        "sbm_medium": (40, 5, 0.25, 0.04, 2),
        "sbm_hard": (50, 4, 0.18, 0.07, 3),
    }.items():
        n, edges, _ = sbm(*args)
        graphs[tag] = (n, edges)
    graphs["ring_of_cliques"] = ring_of_cliques(12, 6)

    stored, cases = {}, []
    for gname, (n, edges) in graphs.items():
        for wtag, weights in (
            ("unit", [1.0] * len(edges)),
            ("weighted", weighted(edges, 7)),
        ):
            graph_key = f"{gname}/{wtag}"
            stored[graph_key] = {
                "n_nodes": n,
                "edges": [[int(a), int(b), float(w)] for (a, b), w in zip(edges, weights)],
            }
            for kind, resolutions in (
                ("rb", [0.25, 0.5, 1.0, 2.0, 4.0]),
                ("cpm", [0.02, 0.05, 0.1]),
            ):
                for res in resolutions:
                    for seed in (1, 2):
                        cases.append(
                            make_case(graph_key, n, edges, weights, kind, res, seed)
                        )

    path = OUT / "leidenalg_reference.json"
    path.write_text(json.dumps({"graphs": stored, "cases": cases}, separators=(",", ":")))
    print(f"wrote {len(cases)} cases to {path}")
    print(f"leidenalg {la.version}, igraph {ig.__version__}")


if __name__ == "__main__":
    main()
