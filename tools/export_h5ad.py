#!/usr/bin/env python3
"""Export a scanpy neighbourhood graph plus reference clusterings for the Rust test harness.

Writes into `<out>`:

* ``indptr.bin`` / ``indices.bin`` / ``data.bin`` - the connectivities matrix as CSR, raw
  little-endian (u64 / u32 / f32), so the Rust harness needs no extra dependency
* ``reference.json``  - scanpy's Leiden labels per resolution, plus author annotations,
                        cluster counts and wall times
* ``meta.json``       - provenance: dataset, versions, n_neighbors

Usage:
    python tools/export_h5ad.py --dataset pbmc3k --out <dir>
    python tools/export_h5ad.py --h5ad path/to/data.h5ad --out <dir>

The graph is exactly what `CSRNetwork::from_csr_parts` consumes, so the Rust side clusters
the *same* graph scanpy did - any difference in output is the algorithm, not the input.
"""

import argparse
import json
import pathlib
import time

import numpy as np
import scanpy as sc
import scipy.sparse as sp

RESOLUTIONS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]


def load(args):
    """Returns (adata, description, annotation column or None)."""
    if args.h5ad:
        adata = sc.read_h5ad(args.h5ad)
        return adata, f"h5ad:{args.h5ad}", args.annotation

    if args.dataset == "pbmc3k":
        # Raw counts; we run the standard preprocessing ourselves so the graph is built the
        # way a real pipeline would build it.
        adata = sc.datasets.pbmc3k()
        annotated = sc.datasets.pbmc3k_processed()
        # pbmc3k_processed is filtered; carry its author annotations across by barcode.
        adata = adata[adata.obs_names.isin(annotated.obs_names)].copy()
        adata.obs["cell_type"] = annotated.obs.loc[adata.obs_names, "louvain"].values
        return adata, "scanpy:pbmc3k", "cell_type"

    if args.dataset == "pbmc68k":
        adata = sc.datasets.pbmc68k_reduced()
        adata.obs["cell_type"] = adata.obs["bulk_labels"].values
        return adata, "scanpy:pbmc68k_reduced", "cell_type"

    raise SystemExit(f"unknown dataset {args.dataset}")


def preprocess(adata, n_neighbors, n_pcs):
    """Standard scanpy pipeline, skipping steps the data already has."""
    if "connectivities" in adata.obsp:
        print("  reusing existing neighbourhood graph")
        return adata

    if adata.X.min() >= 0 and adata.X.max() > 50:  # looks like raw counts
        print("  normalising + log1p + HVG + PCA")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata, max_value=10)

    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1, adata.n_obs - 1))

    print(f"  building {n_neighbors}-NN graph")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    return adata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pbmc3k")
    ap.add_argument("--h5ad")
    ap.add_argument("--annotation", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--n-pcs", type=int, default=50)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading...")
    adata, description, annotation = load(args)
    adata = preprocess(adata, args.n_neighbors, args.n_pcs)

    conn = sp.csr_matrix(adata.obsp["connectivities"])
    conn.sort_indices()
    n = conn.shape[0]
    print(f"  {n} cells, {conn.nnz} directed entries "
          f"({conn.nnz / n:.1f} per cell), dtype {conn.data.dtype}")

    asym = abs(conn - conn.T).max()
    print(f"  max |A - A^T| = {asym:.3e} (must be ~0 for the Rust side to accept it)")

    conn.indptr.astype("<u8").tofile(out / "indptr.bin")
    conn.indices.astype("<u4").tofile(out / "indices.bin")
    conn.data.astype("<f4").tofile(out / "data.bin")

    reference = {}
    for res in RESOLUTIONS:
        t0 = time.perf_counter()
        sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}",
                     flavor="igraph", n_iterations=2, directed=False)
        elapsed = time.perf_counter() - t0
        labels = adata.obs[f"leiden_{res}"].astype(int).tolist()
        reference[str(res)] = {
            "labels": labels,
            "n_clusters": len(set(labels)),
            "seconds": elapsed,
        }
        print(f"  scanpy leiden res={res}: {len(set(labels))} clusters in {elapsed:.2f}s")

    payload = {"resolutions": reference}
    if annotation and annotation in adata.obs:
        codes = adata.obs[annotation].astype("category")
        payload["annotation"] = {
            "name": annotation,
            "labels": codes.cat.codes.tolist(),
            "categories": list(codes.cat.categories.astype(str)),
        }
        print(f"  author annotation '{annotation}': "
              f"{len(codes.cat.categories)} types")

    (out / "reference.json").write_text(json.dumps(payload))
    (out / "meta.json").write_text(json.dumps({
        "dataset": description,
        "n_cells": int(n),
        "n_entries": int(conn.nnz),
        "n_neighbors": args.n_neighbors,
        "max_asymmetry": float(asym),
        "scanpy": sc.__version__,
    }))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
