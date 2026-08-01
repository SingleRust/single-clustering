#!/usr/bin/env python3
"""Generate evaluation graphs and leidenalg reference results.

Writes, per case:
  <out>/<name>.edges   "u v w" per line
  <out>/<name>.truth   ground-truth label per line (if any)
and a single results.json with leidenalg's membership, quality and wall time.
"""
import json, pathlib, random, sys, time

import igraph as ig
import leidenalg as la
import networkx as nx

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def write_case(name, n, edges, truth=None):
    (OUT / f"{name}.edges").write_text("".join(f"{a} {b} 1.0\n" for a, b in edges))
    if truth is not None:
        (OUT / f"{name}.truth").write_text("".join(f"{t}\n" for t in truth))
    return {"name": name, "n_nodes": n, "n_edges": len(edges)}


def lfr(n, mu, seed, avg_deg=15, max_deg=50, tau1=2.5, tau2=1.5, min_c=20, max_c=100):
    """Lancichinetti-Fortunato-Radicchi benchmark: the standard test for community detection.

    `mu` is the mixing parameter - the fraction of each node's edges that go outside its
    community. mu=0.1 is easy, mu=0.6+ is at or past the detectability limit."""
    g = nx.LFR_benchmark_graph(
        n, tau1, tau2, mu, average_degree=avg_deg, max_degree=max_deg,
        min_community=min_c, max_community=max_c, seed=seed, max_iters=500,
    )
    g.remove_edges_from(nx.selfloop_edges(g))
    comms = {}
    truth = [0] * n
    for v, data in g.nodes(data=True):
        key = frozenset(data["community"])
        if key not in comms:
            comms[key] = len(comms)
        truth[v] = comms[key]
    return list(g.edges()), truth


def knn_blobs(n, blocks, k, seed):
    """Points in 2D blobs joined to their k nearest neighbours - the shape of a
    single-cell neighbourhood graph."""
    rng = random.Random(seed)
    per = n // blocks
    import math
    centers = [(60 * math.cos(2 * math.pi * b / blocks), 60 * math.sin(2 * math.pi * b / blocks))
               for b in range(blocks)]
    pts, truth = [], []
    for i in range(n):
        b = min(i // per, blocks - 1)
        c = centers[b]
        pts.append((c[0] + rng.uniform(-8, 8), c[1] + rng.uniform(-8, 8)))
        truth.append(b)

    cell = 4.0
    grid = {}
    for i, p in enumerate(pts):
        grid.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(i)

    edges = set()
    for i, p in enumerate(pts):
        gx, gy = int(p[0] // cell), int(p[1] // cell)
        cand = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                cand.extend(j for j in grid.get((gx + dx, gy + dy), ()) if j != i)
        cand.sort(key=lambda j: (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)
        for j in cand[:k]:
            edges.add((i, j) if i < j else (j, i))
    return sorted(edges), truth


def run_leidenalg(n, edges, resolution, n_iterations, seed):
    g = ig.Graph(n=n, edges=edges)
    g.es["weight"] = [1.0] * len(edges)
    t0 = time.perf_counter()
    part = la.find_partition(
        g, la.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=resolution, seed=seed, n_iterations=n_iterations,
    )
    elapsed = time.perf_counter() - t0
    return list(part.membership), float(part.quality()), elapsed


cases = []

# --- LFR: mixing parameter sweep, the standard benchmark -------------------------
for mu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    for seed in [1, 2]:
        name = f"lfr_n2000_mu{mu}_s{seed}"
        try:
            edges, truth = lfr(2000, mu, seed)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        meta = write_case(name, 2000, edges, truth)
        memb, q, t = run_leidenalg(2000, edges, 1.0, 5, 42)
        meta.update(kind="lfr", mu=mu, resolution=1.0,
                    reference_membership=memb, reference_quality=q, reference_seconds=t,
                    reference_n_clusters=len(set(memb)), truth_n_clusters=len(set(truth)))
        cases.append(meta)
        print(f"  {name}: {len(edges)} edges, leidenalg {len(set(memb))} clusters in {t:.2f}s")

# --- scale: kNN-shaped graphs at single-cell sizes --------------------------------
for n in [10_000, 50_000, 200_000]:
    name = f"knn_n{n}"
    edges, truth = knn_blobs(n, 10, 15, 5)
    meta = write_case(name, n, edges, truth)
    memb, q, t = run_leidenalg(n, edges, 1.0, 2, 42)
    meta.update(kind="knn", resolution=1.0,
                reference_membership=memb, reference_quality=q, reference_seconds=t,
                reference_n_clusters=len(set(memb)), truth_n_clusters=len(set(truth)))
    cases.append(meta)
    print(f"  {name}: {len(edges)} edges, leidenalg {len(set(memb))} clusters in {t:.2f}s")

(OUT / "results.json").write_text(json.dumps({"cases": cases}))
print(f"wrote {len(cases)} cases")
