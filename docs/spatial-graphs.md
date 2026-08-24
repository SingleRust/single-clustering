# Notes on spatial neighbour graphs

Background for choices in `src/spatial/` that aren't obvious from the code.

## Delaunay: why prune at all

A triangulation covers the *convex hull*, so it invents edges wherever tissue isn't convex:
across ventricles and vessels, around concave boundaries, and between separate fragments on
the same section. Those edges are long, and joining two sides of a hole produces a
plausible-looking but meaningless domain.

The parameter-free subgraphs don't fix it. Gabriel and relative-neighbourhood tests ask
whether other points lie near the edge, and the whole point of a hole is that nothing does,
so a hole-spanning edge passes. Measured on two detached fragments: Gabriel leaves 19 of 42
bridges standing and drops 21% of genuine edges elsewhere; RNG drops 52%. Removing an
artefact edge needs a rule about length.

They may still be worth adding as an option — MST ⊆ RNG ⊆ GG ⊆ Delaunay, both are always
connected, and Lingas (1994) extracts either from an existing triangulation in O(n). Two
things to get right: use the **open** exclusion region, since the closed variant disconnects
on ties and gridded coordinates are made of ties; and expect mean degree ~4 for Gabriel and
~2.56 for RNG against Delaunay's 6.

## What other tools do

- **squidpy** applies no pruning at all. `delaunay=True` returns the raw triangulation, hull
  included. `radius` and `percentile` both default to off, and in 1.6.x a scalar `radius` is
  silently ignored on the Delaunay path.
- **Giotto** is the only one that derives a cutoff automatically, using a global Tukey whisker
  over all edge lengths. Being global, it fails where density varies: on a dense block beside
  24x sparser stroma it shatters the sparse side into ~570 components while barely touching
  the dense one.

That's the case for a local rule, and why `HullPruning::Adaptive` is the default.

## Where the default factor of 2.0 comes from

`calibrate_adaptive_factor` in `src/spatial/delaunay.rs` measures it: 2.0 is the largest
factor that fully separates two detached tissue fragments, which is the one unambiguous
requirement. It costs ~5% of edges on uniform-random points and less on real tissue, whose
cells pack more regularly than a Poisson process.

It's also the canonical value. Zahn's 1971 inconsistent-edge criterion is the same shape — an
edge significantly larger than nearby edge weights on **both sides** — and he reports a factor
of 2 usually makes the separation apparent, with worked examples from 1.3 to 2.6.

Two things it won't do. It can't clear a hole two or three cells across: that needs a factor
near 1.5, which drops ~16% of genuine edges everywhere, and at that width it's arguable
whether the cells either side are neighbours at all. And, as Zahn notes, it doesn't detect
one-way gradients however steep — density ramping smoothly across a section has no local
discontinuity to find.

## Local scale is the 25th percentile, not the median

A rim node's incident edges are mostly the spurious long ones, so its median is inflated by
exactly the artefact the rule is meant to catch. A low quantile reads the real local spacing.

## Reading Visium data

Two things to know before wiring up a loader, both checked against upstream:

- Space Ranger ≥ 2.0 renamed `tissue_positions_list.csv` to `tissue_positions.csv` **and**
  added a header row. Parse the header rather than assuming column positions — scanpy
  branches on the filename to decide.
- `scanpy.read_visium` labels the two pixel columns the wrong way round, then selects them in
  the wrong order. The errors cancel, so `obsm["spatial"]` really is `(x, y)`, but the names
  in `obs` are swapped. Don't trust `pxl_row_in_fullres` / `pxl_col_in_fullres` if you read
  those columns directly.

Neither affects `array_row` / `array_col`, which is what `lattice_graph` wants.
