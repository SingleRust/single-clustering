#!/usr/bin/env python3
"""Reference fixtures for the BANKSY feature augmentation, from `banksy-py` itself.

Calls the reference implementation's own functions rather than reimplementing them, so the
Rust side is checked against what BANKSY actually computes and not against a reading of it.

    uv venv && uv pip install banksy-py
    uv run tools/gen_banksy_fixtures.py tests/fixtures/banksy_reference.json

Fixtures are committed, so the test suite needs no Python.
"""

import json
import sys

import numpy as np
import anndata
from banksy.embed_banksy import create_nbr_matrix
from banksy.main import concatenate_all, generate_spatial_weights_fixed_nbrs

DECAYS = ["uniform", "reciprocal", "reciprocal_squared", "scaled_gaussian", "ranked"]


def case(name, n_cells, n_genes, k, max_m, lambdas, decay, seed):
    """One synthetic dataset put through the reference, with every intermediate captured."""
    rng = np.random.default_rng(seed)

    if name.startswith("grid"):
        side = int(np.sqrt(n_cells))
        xs, ys = np.meshgrid(np.arange(side), np.arange(side))
        locations = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
        n_cells = locations.shape[0]
    else:
        locations = rng.uniform(0, 50, size=(n_cells, 2))

    features = rng.normal(size=(n_cells, n_genes))

    # Go through the reference's own matrix builder rather than a raw matmul: the AGF
    # centring lives in create_nbr_matrix, not in the weight construction, so a matmul
    # against the weights silently omits it.
    weights = {}
    for m in range(max_m + 1):
        w, _dist, _theta = generate_spatial_weights_fixed_nbrs(
            locations,
            m=m,
            num_neighbours=k,
            decay_type=decay,
            verbose=False,
        )
        weights[m] = w

    adata = anndata.AnnData(features.copy())
    banksy_dict = {decay: {"weights": weights}}
    nbr_matrices = create_nbr_matrix(adata, banksy_dict, decay, max_m, verbose=False)
    harmonics = [nbr_matrices[m] for m in range(max_m + 1)]

    # Python derives the harmonic-m neighbourhood as k*(m+1); record what it actually used
    # so the Rust side can be given the same k explicitly.
    k_per_harmonic = [int(np.diff(weights[m].indptr).max()) for m in range(max_m + 1)]

    out = {
        "name": name,
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "k": int(k),
        "max_m": int(max_m),
        "decay": decay,
        "k_per_harmonic": k_per_harmonic,
        "locations": locations.tolist(),
        "features": features.tolist(),
        # As create_nbr_matrix returns them: H_0 is real and signed, and the m >= 1
        # harmonics have already had their magnitude taken inside it. Applying abs() here
        # would flip the sign of every negative H_0 entry.
        "harmonics": [np.asarray(h).tolist() for h in harmonics],
        "banksy": {},
    }

    for lam in lambdas:
        mat_list = [features] + harmonics
        combined = concatenate_all(mat_list, lam, adata=None)
        out["banksy"][str(lam)] = np.asarray(combined).tolist()

    return out


def main():
    cases = [
        case("uniform_small", 60, 4, 6, 1, [0.0, 0.2, 0.5, 1.0], "scaled_gaussian", 1),
        case("uniform_mid", 200, 8, 12, 1, [0.2, 0.8], "scaled_gaussian", 2),
        case("grid_regular", 100, 5, 8, 1, [0.2], "scaled_gaussian", 3),
        case("mean_only", 120, 6, 10, 0, [0.2, 0.5], "scaled_gaussian", 4),
        case("harmonic_two", 120, 4, 10, 2, [0.2], "scaled_gaussian", 5),
    ]
    # Every decay type, so none of the kernels can drift unnoticed.
    #
    # `ranked` is capped at max_m = 0: the reference crashes for m > 0, because it builds a
    # weight profile of length `num_neighbours` while the row holds `num_neighbours * (m+1)`
    # entries. An upstream bug, not a usage error -- so there is no reference behaviour to
    # match above m = 0.
    for decay in DECAYS:
        max_m = 0 if decay == "ranked" else 1
        cases.append(case(f"decay_{decay}", 80, 3, 7, max_m, [0.2], decay, 11))

    # To a file, not stdout: the reference print()s unconditionally, so stdout is not clean.
    out = sys.argv[1] if len(sys.argv) > 1 else "banksy_reference.json"
    with open(out, "w") as fh:
        json.dump({"cases": cases}, fh)
    print(f"\nwrote {out}", file=sys.stderr)
    for c in cases:
        print(
            f"  {c['name']:22} n={c['n_cells']:4} genes={c['n_genes']} "
            f"k={c['k']} m={c['max_m']} decay={c['decay']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
