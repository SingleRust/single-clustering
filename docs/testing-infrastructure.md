# SingleRust testing & benchmarking infrastructure

Org-level design notes. Scoped to what one maintainer can run and keep running, with no
institutional compute and no recurring engineering budget.

> Lives in `single-clustering` for now; belongs in a `SingleRust/.github` org repo once the
> reusable workflows exist.

## Constraints that shape everything

- **Compute is not the scarce resource.** GitHub Actions is free and unlimited for public
  repos on Linux runners. *Maintainer attention* is the budget.
- Therefore: every CI test must be **deterministic, fast, and diagnostic on failure**. A flaky
  test is worse than no test — there is nobody else to triage it, and you learn to ignore red.
- Anything slow, large, or hardware-sensitive moves **out of CI** and onto rented ephemeral
  compute, on a schedule.

---

## P0 — the publish gate (live hazard)

`.github/workflows/publish.yml`:

```yaml
on:
  push:
    branches: [main]     # ← these are OR, not AND
    tags: ['v*']
```

Every push to `main` triggers a crates.io publish, with **no test job gating it**. crates.io
has 0.6.1 (7.8k downloads); `Cargo.toml` says 0.7.0. Pushing the Leiden rewrite publishes it —
untested and unrecallable (crates.io versions yank, never replace).

**Fix:** trigger on `tags: ['v*']` only, add `needs: [test]`, replace the EOL
`actions/checkout@v2` and archived `actions-rs/toolchain`. Blocks committing 0.7.0.

---

## Test tiers

| Tier | Content | Where | Cadence | Cost |
|---|---|---|---|---|
| 0 | Committed fixtures, <1 MB | CI | every push | seconds |
| 1 | Seeded synthetic, ≤1M nodes | CI | nightly | minutes |
| 2 | **Real data, committed exports** | CI | every push | ~80 ms |
| 3 | 50–75M cells, perf + peak RSS | rented instance | weekly / on tag | €, not free |

### Tier 2 — real data belongs in CI

Commit the **export**, not the `.h5ad`. Measured for pbmc3k:

```
indptr.bin  21 KB   indices.bin 260 KB   data.bin 260 KB
reference.json 75 KB (scanpy labels × 8 resolutions + author cell types)
                                          616 KB raw → 350 KB in git
```

No Python, no scanpy, no download in CI. pbmc68k_reduced is ~6 MB by the same math — still
committable, worth adding as a second size point.

Promote `examples/real_data.rs` → `tests/real_data.rs`. Assertions that hold with margin:

| Assertion | Measured | Note |
|---|---|---|
| cluster count within ±1 of scanpy, every resolution | 3/3 … 50/49 | |
| **modularity ≥ scanpy − 0.005**, every resolution | ±0.0015 | tightest, implementation-independent |
| ARI vs author cell types ≥ 0.85 | 0.8599 | the biology gate |
| ARI vs scanpy ≥ 0.95, **res ≤ 0.75 only** | 0.985–1.000 | |
| symmetry passes, no self-loops, seed → identical labels | ✓ | |

Deliberately **not** gated: ARI vs scanpy at fine resolution (0.65 at res=1.5). That is genuine
near-degeneracy, not a defect — gating it buys a flaky test.

---

## The oracle: committed differential fixtures

The pattern that earned its keep. It caught the two Leiden bugs nothing else reached
(`n_iterations` continuation semantics, and the level-loop early break) — neither was
reachable by invariant or property tests.

**Python lives in `tools/`, runs on a laptop, and never runs in CI.** 208 KB of committed JSON
is a permanent oracle with zero ongoing cost and no dependency rot.

Generalizes org-wide — every crate has a Python counterpart that *is* ground truth:

| Crate | Oracle |
|---|---|
| single-clustering | leidenalg / igraph / scanpy |
| anndata-rs | anndata (currently done the expensive way: 4 versions installed per CI run) |
| Anndata-Memory | anndata semantics |

Convention: `tools/gen_fixtures.py` + `uv` + **committed lockfile** + `just regen-fixtures`
that regenerates and shows the diff. The lockfile is what makes this reproducible in two years.

---

## Shared infrastructure

1. **`SingleRust/.github` org repo** with reusable workflows. Each crate's CI becomes ~6 lines
   calling `rust-ci.yml@main`. Fix once, not four times.
2. **`singlerust-testkit`** dev-dependency crate:
   - ARI / NMI / clustering metrics — *already* duplicated between `tests/common/mod.rs` and
     `examples/real_data.rs`; that spreads
   - fixture loading + schema
   - deterministic synthetic generators (SBM, LFR, synthetic AnnData with known structure)
   - cached dataset fetcher keyed by content hash

---

## Benchmark platform (ephemeral OVH)

```
weekly cron / workflow_dispatch / tag
  ├─ provision  (ubuntu-latest, free)   create instance; cloud-init carries a JIT
  │                                     runner token; tag purpose=bench, run-id=…
  ├─ benchmark  (self-hosted, run-${{ github.run_id }})
  └─ teardown   (ubuntu-latest, if: always())
```

Label the runner with the **run id** so concurrent runs cannot steal each other's instance.
Provisioning is the easy part. Two things decide whether this works.

### 1. Leak protection — three independent layers

`if: always()` alone is not enough; a cancelled workflow or crashed runner skips it. A leaked
instance bills indefinitely.

1. `if: always()` teardown — the normal path
2. **Self-destruct on the box**: cloud-init `systemd-run --on-active=3h`. Must **delete**, not
   `shutdown` — OVH bills a stopped Public Cloud instance. Needs a narrowly-scoped API key
   that can only delete `purpose=bench` resources.
3. **Reaper**: daily cron on a free runner, deletes any `purpose=bench` instance older than
   6h. ~20 lines; catches everything the other two miss. **Treat as mandatory.**

Plus a hard budget alert on the account.

### 2. Trustworthy numbers

Shared-tenancy variance will exceed the regressions being hunted.

- **Pin the machine exactly** — flavor, region, *image by ID* (never "latest Ubuntu"), kernel.
  Record all of it. If any field differs from the baseline, the tooling **refuses the
  comparison** rather than silently reporting a regression.
- **Ship a calibration microbenchmark in every run** — a fixed CPU-bound kernel and a fixed
  pointer-chase for memory latency. Report `benchmark / calibration`, not raw seconds. This is
  what separates "my code got slower" from "I landed on a degraded host"; without it,
  month-over-month cloud benchmarking is close to unusable.
- **≥5 reps, median + MAD.** If MAD is too wide, **discard the run** rather than publish it.

### Machine notes

- Check whether **OVH Metal Instances** (bare metal, hourly billing) exist in-region — no noisy
  neighbours, so fewer reps and a much tighter calibration ratio. Classic Rise/Advance
  dedicated is cheaper per unit but monthly-committed, which breaks the ephemeral model.
- **Sizing at 75M cells**: graph ~35 GB (12.3 undirected edges/cell measured on real 15-NN),
  plus ~1.5 GB partition state plus aggregation levels. 64 GB tight, **128 GB comfortable**.
- Keep **kNN construction a separate benchmark** — 75M × 50 dims is 15 GB of input before the
  HNSW index; mixing it in stops measuring clustering.
- **Staging**: derived CSR graphs in OVH Object Storage, same region (egress free), versioned
  by content hash. Never re-derive from `.h5ad` per run — that benchmarks scanpy preprocessing.
- Order of magnitude: weekly 2-hour run on a memory-optimised flavour ≈ €10–20/month. Confirm
  against current pricing.

### Platform shape

`workflow_dispatch` inputs (dataset, flavor, crate ref, resolutions) are effectively the
run-submission API today. A platform generalizes that surface — queue, parameterise, and
compare runs — but the artifact contract below is what it should be built around, not a
database.

---

## Results schema

singlerust.com already exists and already hosts benchmarks, so the transport is decided — the
runner should emit whatever that site ingests. What matters here is the **field set**, which is
transport-independent and is the part that is expensive to add retroactively: a run recorded
without `calibration` or without full `machine` provenance cannot be placed in a longitudinal
series later.

Below is the minimum field set, expressed as JSON for concreteness. Map it onto the existing
format rather than adopting it wholesale.

```json
{
  "schema_version": 1,
  "timestamp": "2026-08-01T12:00:00Z",
  "git_sha": "…",
  "crate_version": "0.7.0",
  "machine": {
    "provider": "ovh", "flavor": "…", "region": "…",
    "image_id": "…", "cpu_model": "…", "kernel": "…"
  },
  "calibration": { "cpu_ns": 0, "mem_latency_ns": 0 },
  "results": [
    {
      "name": "leiden/pbmc3k", "n_cells": 2638, "n_edges": 32507,
      "reps": 5, "median_s": 0.0085, "mad_s": 0.0002, "peak_rss_bytes": 0,
      "quality": { "modularity": 0.729, "ari_vs_truth": 0.86 }
    }
  ]
}
```

Whatever the transport, keep a copy of each run as a plain artifact next to the code — a run
should stay readable if the site is rewritten, and it makes `git bisect`-style perf archaeology
possible without querying anything.

---

## Per-crate, by risk

**Anndata-Memory — highest risk in the org.** 11 `unsafe` blocks, concurrency, and a `1.0.7`
version implying a stability promise. Needs `cargo miri test`, and `loom` if there are
hand-rolled synchronisation primitives. Two things look wrong now: it sits on branch
`feat-anndata-sprs-upgrade` while its CI triggers on `master` — **CI is likely not running on
the active branch at all**.

**anndata-rs** — keep the Python version matrix, but first+last of the range on PRs, full
matrix nightly.

**single-clustering** — suite is in good shape (87 tests, 5 integration binaries). Add
`cargo-semver-checks`; it is published and just took a breaking rewrite.

---

## Deliberately skipped

| Not doing | Why |
|---|---|
| Benchmark dashboard / server | committed JSON + static site covers it |
| Coverage gates | vanity metric solo; `cargo-mutants` quarterly says far more |
| Self-hosted persistent runners | leak risk and maintenance for no gain |
| Property tests everywhere | only where the invariant is crisp enough to be diagnostic |

---

## Sequencing

1. **Fix `publish.yml`** — blocks committing 0.7.0
2. **Commit pbmc3k export + `tests/real_data.rs`** — 350 KB, real biology gate on every push
3. Org `.github` repo with reusable `rust-ci.yml`; point all four crates at it
4. Fix Anndata-Memory's CI branch trigger; add `miri`
5. Extract `singlerust-testkit` (once there is a second consumer)
6. Benchmark platform: reaper first, then provision/teardown, then calibration, then the runs

## First sweep

All published numbers get regenerated, so there is no history to preserve and no migration —
the field set is a greenfield decision, and the first run should be **one clean sweep**: all
crates, all datasets, one machine shape, one session. Every number then comparable to every
other, which is the strongest baseline the ephemeral-instance model can produce. Calibration
still matters for everything after it.

Note the numbers currently on singlerust.com are 0.6.1 — quadratic local moving, and a
resolution parameter off by 2×, so that axis does not mean what 0.7.0's does. They measure
code that clustered incorrectly. Worth pulling or marking independently of when the sweep
lands; the crate is public with ~7.8k downloads.
