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

Independent, all 12 sections, mean across slices of each slice's 20-run mean. Computed from
the BenchmarkST repo's raw per-run ARI files (Hu et al., Genome Biology 2024;25(1):212).

| Method | mean over 12 slices | source |
|---|---|---|
| GraphST | 0.557 | raw data |
| ADEPT | 0.530 | raw data |
| STAGATE | 0.503 | raw data |
| BASS | ~0.49 | figure |
| SpatialPCA | ~0.48 | figure |
| **BANKSY** | **0.469** | figure, ±0.01 |
| CCST | 0.451 | raw data |
| PRECAST | ~0.46 | figure |
| ConGI | 0.422 | raw data |
| SEDR | 0.411 | raw data |
| SpaGCN | 0.392 | raw data |
| conST | 0.385 | raw data |
| BayesSpace | ~0.37 | figure |
| DeepST | 0.322 | raw data |
| DR.SC | ~0.33 | figure |

Nine methods have machine-readable per-run results in `ari_results/*.txt`; the rest exist
only inside the Fig. 2a raster and were recovered by colourmap inversion cross-checked
against the bioRxiv preprint's vector figures. **BANKSY is one of the figure-derived ones**
— it is new in the published version, so there is no vector cross-check. Treat 0.469 as
±0.01, not an exact published value.

This benchmark contains **no non-spatial baseline** — no Seurat, Leiden, Louvain or k-means
anywhere in it. It cannot answer the question that matters most to us.

GraphST is the ceiling set by heavy deep-learning methods. It is **not** our bar — a
feature-augmentation approach that then runs ordinary Leiden should be measured against
BANKSY, which is the same shape.

## Self-reported vs independent

Which published numbers survive someone else running them. Self-reported values are the
stored outputs in each tool's own tutorial notebooks.

| Method | section | self-reported | independent | gap |
|---|---|---|---|---|
| GraphST | 151673 | 0.635 | 0.633 (Kang), 0.638 (Hu) | **~0.00** |
| BayesSpace | 151673 | 0.55 | 0.550 (Kang) | **0.00** |
| STAGATE | 151676 | 0.60 | 0.493 (Hu) | **0.11** |
| BANKSY | 12-slice | 0.518 | 0.469 (Hu) | **0.05** |

GraphST and BayesSpace reproduce essentially exactly. STAGATE and BANKSY do not. That is the
single most useful fact here: a published number is not evidence unless someone independent
has reproduced it, and two of the four checked did not survive.

BayesSpace also disagrees *between* independent benchmarks — 0.550 (Kang) vs ~0.40 (Hu) on
the same section — so "independent" is not automatically decisive either.

Two protocol details found in the tools' own tutorials, both of which change results:

* **GraphST's own tutorial hardcodes `n_clusters = 7` for every section.** Run as-published
  on 151669–151672, it asks for 7 clusters against 5-layer truth. The 5/7 split exists only
  in third-party benchmark repos.
* **GraphST filters unannotated spots *after* clustering and spatial refinement**, so those
  spots still train the model and still vote in the refinement step. Hu's benchmark drops
  them before the graph is built. Same nominal rule, different graph.

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

**Exclude unannotated spots — before clustering, not just before scoring.** 28 of 3639 on
151673. The benchmark drops them in `load_DLPFC` (`ad.obs.dropna()`) so they never enter the
graph at all. Dropping them only at scoring time gives a different graph and a different
answer.

**5 vs 7 clusters — verified from code, and not stated in the paper.** Hard-coded
byte-identically across every run script in the benchmark:

```python
[[7,'151507'],[7,'151508'],[7,'151509'],[7,'151510'],
 [5,'151669'],[5,'151670'],[5,'151671'],[5,'151672'],
 [7,'151673'],[7,'151674'],[7,'151675'],[7,'151676']]
```

So **sample 2 (151669–151672) is 5 clusters**, annotated L3–L6 + WM only — no L1 or L2. The
other eight sections are 7. Getting this wrong makes every affected section incomparable,
and no paper states it.

**ARI implementation** is not a source of difference: `sklearn.adjusted_rand_score` and
`mclust::adjustedRandIndex` agree, and an independent from-scratch implementation reproduced
published values to three decimals.

## Two traps

**BANKSY's published ARIs are on smoothed labels, and the paper never says so.** The
deposited script calls `SmoothLabels(k = 6)` before scoring — visible only in code. Comparing
our raw labels against 0.518 would understate us. Either replicate the smoothing or state
plainly that the comparison is unsmoothed-to-smoothed.

**Configuration can swamp method differences.** On section 151673, across two independent
benchmarks: DeepST scores 0.538 and 0.229 — a swing of 0.31, far larger than the gaps
between most methods on the list. BayesSpace scores 0.550 and ~0.40. GraphST, by contrast,
reproduces to three decimals (0.633 vs 0.638). So a method's number says as much about who
ran it as about the method, unless it is one of the reproducible ones. Budget for that before
concluding anything from a single figure — including ours.

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
