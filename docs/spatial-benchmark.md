# DLPFC benchmark target

Written **before** implementing spatial domain detection, so the target cannot drift to meet
the result. Everything here is sourced; anything unverified is marked.

Dataset: 10x Visium human dorsolateral prefrontal cortex, Maynard et al. 2021. 12 sections,
3431–4788 spots each, manual annotation of cortical layers L1–L6 + white matter.

## The target

**Median ARI 0.46–0.52 across the 12 sections**, against a **non-spatial baseline of
0.38–0.43**.

The range on the target is not imprecision — it is a real disagreement:

| Source | BANKSY median ARI |
|---|---|
| Self-reported (authors' deposited artifacts) | 0.518 |
| Independent (Hu et al., Genome Biology 2024) | ~0.46 |

Take the lower number as the bar. Method papers grade their own homework.

## Where everyone else lands

Independent, all 12 sections, computed from the BenchmarkST repo's per-section ARI files
(Hu et al., Genome Biology 2024;25(1):212):

| Method | median | mean |
|---|---|---|
| GraphST | 0.559 | 0.557 |
| STAGATE | 0.523 | 0.505 |
| ADEPT | 0.519 | 0.531 |
| **BANKSY** | **~0.46** | — |
| CCST | 0.434 | 0.459 |
| SEDR | 0.426 | 0.414 |
| ConGI | 0.419 | 0.419 |
| SpaGCN | 0.412 | 0.392 |
| conST | 0.394 | 0.395 |
| DeepST | 0.262 | 0.300 |

GraphST is the ceiling set by heavy deep-learning methods. It is **not** our bar — a
feature-augmentation approach that then runs ordinary Leiden should be measured against
BANKSY, which is the same shape.

## The bar that justifies existing

Non-spatial clustering on expression alone, section 151673:

| | ARI |
|---|---|
| Seurat (expression only) | 0.430 |
| BANKSY's own λ=0 arm, same pipeline | 0.382 |
| Leiden / Louvain | **unconfirmed**, stated as below Seurat's 0.430 |

If we do not clear ~0.43 by a clear margin, the spatial machinery is decoration. BANKSY's
claimed gain over its own baseline is ~0.58 vs ~0.38 on that section, i.e. **+0.184**.

## Protocol — the part that decides comparability

**Cluster count.** Sweep Leiden resolution 0.1–1.5. Keep only runs whose cluster count
exactly matches the section's annotated layer count. Report the **median ARI over all
qualifying resolutions** — not the best. Reporting best-of-sweep makes the number
incomparable with every published figure. Fallback when nothing matches: widen to k±1; if
still nothing, report nothing.

**Exclude unannotated spots.** 28 of 3639 on 151673. Small effect (~0.004) but it shifts
everything.

**5 vs 7 clusters.** Most sections have 7 layers; some have 5 (L3, L4, L5, L6, WM). Reported
as the 151669–151672 group — *unverified*, check each section's annotation set directly.

**ARI implementation** is not a source of difference: `sklearn.adjusted_rand_score` and
`mclust::adjustedRandIndex` agree, and an independent from-scratch implementation reproduced
published values to three decimals.

## Two traps

**BANKSY's published ARIs are on smoothed labels, and the paper never says so.** The
deposited script calls `SmoothLabels(k = 6)` before scoring — visible only in code. Comparing
our raw labels against 0.518 would understate us. Either replicate the smoothing or state
plainly that the comparison is unsmoothed-to-smoothed.

**Configuration can swamp method differences.** DeepST scores 0.538 in one independent
benchmark and 0.229 in another, on the same section — a swing of 0.31, far larger than the
gaps between most methods. GraphST, by contrast, reproduces to three decimals across groups.
Budget for this before concluding anything from a single number.

## Matching BANKSY, if we replicate it

Parameters for Visium domain segmentation, confirmed from the benchmark run script and a
quoted reply from BANKSY's author:

- `lambda = 0.2` — for Visium v1/v2 domain segmentation. **Not 0.8**; that is for Visium HD
  and other technologies.
- `k_geom = c(18, 18)` for DLPFC
- `use_agf = TRUE` — the feature matrix is neighbourhood **mean + azimuthal Gabor gradient**.
  A mean-only implementation is not a faithful replication and should not be compared
  against BANKSY's number.
- 2000 HVGs, 20 PCs, SNN graph at k_expr = 50

## Getting the data

`zenodo.org/records/15114362` → `Benchmark_ST_analysis-master.zip`, **18.8 MB**. Contains a
complete section 151673 (`filtered_feature_bc_matrix.h5`, `metadata.tsv`, `spatial/`) **plus
predicted labels from 14 competing methods**, so a single-section comparison costs one small
download and no reimplementation of anyone else's method.

Full source is spatialLIBD, `http://spatial.libd.org/spatialLIBD/`; annotations ship with the
data as `layer_guess` / `layer_guess_reordered` columns in `metadata.tsv`.

## Unconfirmed

Self-reported numbers from the BayesSpace / SpaGCN / STAGATE / GraphST papers (so no
self-vs-independent gap for any of them); exact Leiden/Louvain DLPFC ARI; BANKSY's λ=0 arm
across all 12 sections and whether it too was smoothed; SDMBench (Yuan et al., Nature Methods
2024) numbers, which is the most likely source of a clean independent Leiden figure.
