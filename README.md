# clara

[![tests](https://github.com/AfzalSurti/clara/actions/workflows/tests.yml/badge.svg)](https://github.com/AfzalSurti/clara/actions/workflows/tests.yml)

**CLARA** (Clustering LARge Applications) and **PAM** (Partitioning Around
Medoids, a.k.a. *k*-medoids), implemented **from scratch in pure Python** with
**no third-party dependencies** — only the standard library. Every numerical
routine, the optimiser, the silhouette metric, and even the CSV reader are
hand-written; there is no `numpy`, `scipy`, `scikit-learn`, or `pandas`
anywhere.

```python
from clara import CLARA
model = CLARA(n_clusters=4, random_state=0).fit(X)
model.labels_          # cluster id per point
model.medoids_         # the k representative data points
model.cost_            # mean distance of a point to its medoid
```

```bash
python -m clara data.csv -k 4 --silhouette
```

---

## Contents

1. [Background: why medoids, why sampling](#background)
2. [How it was built from scratch](#how-it-was-built-from-scratch)
3. [The algorithms, in detail](#the-algorithms-in-detail)
4. [Complexity and memory](#complexity-and-memory)
5. [Accuracy and benchmarks](#accuracy-and-benchmarks)
6. [Installation](#installation)
7. [Library API](#library-api)
8. [Command-line interface](#command-line-interface)
9. [Repository layout](#repository-layout)
10. [Testing](#testing)
11. [Design decisions and limitations](#design-decisions-and-limitations)
12. [Possible improvements](#possible-improvements)
13. [References](#references)

---

## Background

### *k*-means vs *k*-medoids

*k*-means represents each cluster by the **mean** of its points and minimises
the sum of *squared* Euclidean distances. The mean is usually not a real data
point, squaring makes it sensitive to outliers, and it only makes sense for
Euclidean space.

***k*-medoids** represents each cluster by an actual data point — a **medoid**
— and minimises the plain (un-squared) sum of dissimilarities:

```
cost(M) =  Σ   min   d(x, m)
          x∈X  m∈M
```

where `M` is the set of `k` medoids and `d` is *any* distance function. Because
the centre must be one of the input points and distances are not squared,
*k*-medoids is **robust to outliers** and works with **any metric** (or a
custom callable), including on non-Euclidean or precomputed-distance data.

### PAM and its cost

**PAM** (Kaufman & Rousseeuw, 1987) is the standard *k*-medoids optimiser. It
finds high-quality solutions but each SWAP iteration costs on the order of
`k·(n−k)·n` distance look-ups, so it becomes impractical past a few thousand
points (see the [benchmarks](#accuracy-and-benchmarks): ~2 minutes for
`n = 1500` on overlapping data).

### CLARA

**CLARA** (Kaufman & Rousseeuw, 1990) makes *k*-medoids scale. Instead of
optimising over all `n` points, it:

1. draws a small random **sample**,
2. runs full PAM on that sample,
3. assigns **every** one of the `n` points to the nearest sampled medoid and
   measures the mean dissimilarity over the whole dataset,
4. repeats for several samples and keeps the medoid set with the lowest
   full-data cost.

Work per sample is `O(sample_size² + n)` — **linear in `n`**. In the
benchmarks, CLARA clusters 8 000 points in under a second while PAM needs
minutes for 1 500.

---

## How it was built from scratch

"From scratch" here means: **no external packages, and no hidden heavy
lifting from the standard library either.** The table below is the complete
list of stdlib imports in the package and what each is used for.

| Module | Used for | Could it have been avoided? |
| --- | --- | --- |
| `math` | `sqrt`, `comb` (benchmark only) | `sqrt` via Newton's method — kept `math.sqrt` for accuracy |
| `random` | `Random(seed).sample` / `.gauss` / `.shuffle` for sampling and synthetic data | a hand-rolled LCG would work; `random` gives reproducibility across platforms |
| `argparse` | CLI flag parsing | manual `sys.argv` parsing — `argparse` is pure-Python stdlib, no reason to |
| `csv` | **not used** — the CSV reader in `clara/dataset.py` is hand-written | — |

Everything else is written out explicitly:

- **Distances** (`clara/distance.py`) — Euclidean, squared-Euclidean,
  Manhattan, Chebyshev, Minkowski, cosine, each a plain loop over coordinates.
- **Distance matrix** — upper triangle only, mirrored, so the distance
  function is called `n(n−1)/2` times instead of `n²`.
- **PAM BUILD and SWAP** (`clara/kmedoids.py`) — greedy construction and
  steepest-descent swapping, no linear algebra.
- **CLARA sampling loop** (`clara/clara.py`).
- **Silhouette** (`clara/metrics.py`) — the full `O(n²)` definition.
- **CSV parser** (`clara/dataset.py`) — a character-by-character scanner that
  handles quoted fields and `""` escaping.
- **Gaussian blobs** (`clara/dataset.py`) — `random.gauss` around chosen
  centres, then shuffled.
- **Adjusted Rand Index** (`examples/benchmark.py`) — from the contingency
  table, using `math.comb`.

The package ships a `py.typed` marker and type hints throughout, and targets
Python 3.8+.

---

## The algorithms, in detail

### Distance functions — `clara/distance.py`

A "point" is any sequence of floats. Names resolve through
`get_distance(name_or_callable)`:

| name(s) | formula |
| --- | --- |
| `euclidean`, `l2` | `sqrt( Σ (xᵢ − yᵢ)² )` |
| `sqeuclidean` | `Σ (xᵢ − yᵢ)²` |
| `manhattan`, `l1`, `cityblock` | `Σ abs(xᵢ − yᵢ)` |
| `chebyshev`, `linf` | `max abs(xᵢ − yᵢ)` |
| `cosine` | `1 − (x·y) / (norm(x)·norm(y))`, range `[0, 2]` |
| `minkowski(a, b, p)` | `( Σ abs(xᵢ − yᵢ)ᵖ )^(1/p)` |

`pairwise_matrix(points, fn)` returns the full symmetric `n × n` matrix,
computing only the upper triangle.

### PAM — `clara/kmedoids.py`

**BUILD phase** (greedy initial medoids):

```
totals[i] = Σ_j d(i, j)
medoids   = { argmin_i totals[i] }                 # the most central point
while |medoids| < k:
    for each candidate c not in medoids:
        gain(c) = Σ_j  max( nearest[j] − d(j, c), 0 )   # j over non-medoids
    add argmax_c gain(c) to medoids
    update nearest[j] = min(nearest[j], d(j, new_medoid))
```

`nearest[j]` is the distance from point `j` to the closest medoid chosen so
far. Each new medoid is the point that pulls the most total distance out of
the current assignment.

**SWAP phase** (steepest descent):

```
repeat up to max_iter times:
    best = current_cost
    for each medoid m, each non-medoid h:
        trial_cost = cost( medoids with m replaced by h )
        track the (m, h) with the lowest trial_cost
    if the best swap improves on current_cost:  apply it
    else:                                        stop
```

This is the straightforward "recompute the objective for every candidate
swap" formulation — easy to read and verify, `O(k·(n−k)·n)` per iteration. It
is fast enough for the small samples CLARA feeds it; for large `n` used
directly it is slow (that is the whole reason CLARA exists).

`KMedoids` also accepts a **precomputed distance matrix**
(`precomputed=True`), in which case `fit` takes the `n × n` matrix instead of
points and no metric is needed.

### CLARA — `clara/clara.py`

```
sample_size = sample_size or min(n, 40 + 2k)          # classic default
best = None
for s in range(n_samples):
    if best is not None and sample_size > k:
        sample = best.medoid_indices + random_fill(sample_size − k)   # force-keep
    else:
        sample = random_indices(sample_size)
    pam        = KMedoids(k, metric).fit(points[sample])
    medoids    = map pam medoids back to global indices
    labels, C  = assign every one of the n points to the nearest medoid
    if best is None or C / n < best.cost:
        best = (medoids, labels, C / n, s)
```

Forcing the running-best medoids into every sample after the first
(Kaufman & Rousseeuw's refinement) means a new sample can only replace the
incumbent if it is genuinely better on the full data — so **more samples never
hurt** the reported cost, they only cost time.

`cost_` is the **mean** dissimilarity over all `n` points (comparable across
dataset sizes); `total_cost_` is `cost_ · n`.

### Silhouette — `clara/metrics.py`

For each point `i` with cluster `Cᵢ`:

```
a(i) = mean distance from i to the other points in Cᵢ
b(i) = min over clusters C ≠ Cᵢ of  ( mean distance from i to points in C )
s(i) = (b(i) − a(i)) / max(a(i), b(i))            ∈ [−1, 1]
```

Points in a singleton cluster get `s = 0`, as do all points when there is only
one cluster. `silhouette_score` is the mean of `s(i)`. This is the exact
definition — `O(n²)` distance evaluations, fine for a few thousand points,
not for millions.

### CSV reader — `clara/dataset.py`

`load_csv` scans each line character by character, tracking an `in_quotes`
flag, treating `""` inside a quoted field as a literal `"`, and splitting on
the delimiter only when not inside quotes. It auto-detects a header row (first
row not fully parseable as numbers), can pull out a ground-truth label column
(positive or negative index), validates that every row has the same width, and
reports the offending row number on a parse error. **Limitation:** a quoted
field containing a newline is not supported (rare in numeric data).

---

## Complexity and memory

`n` = points, `k` = clusters, `d` = dimensions, `s` = `sample_size`,
`I` = SWAP iterations actually run.

| Routine | Time | Memory |
| --- | --- | --- |
| `pairwise_matrix` | `O(n² · d)` | `O(n²)` |
| PAM BUILD | `O(k · n²)` | `O(n²)` (matrix) |
| PAM SWAP (per iteration) | `O(k · (n−k) · n)` | `O(n²)` |
| `KMedoids.fit` | `O(n² d + k n² + I k (n−k) n)` | `O(n²)` |
| CLARA (per sample) | `O(s² d + I k s² + s · n · d)` | `O(s²)` |
| `CLARA.fit` | `O(n_samples · (s² d + I k s² + s n d))` → **linear in `n`** | `O(s² + n)` |
| `silhouette_score` | `O(n² · d)` | `O(n)` |

The key line: CLARA's memory is `O(s²)` (a tiny sample matrix) plus `O(n)` for
the labels, never `O(n²)`. That is what lets it run on inputs PAM cannot even
allocate a matrix for.

---

## Accuracy and benchmarks

Reproduce with `python examples/benchmark.py` (all seeds fixed). Timings below
are wall-clock on the development machine (CPython 3.13, Windows); absolute
numbers will differ on yours, the shape of the curves will not. Quality is
reported as **ARI** (Adjusted Rand Index vs. ground-truth labels; 1.0 = exact,
0.0 = chance), **purity**, **silhouette**, and **mean cost** (the objective —
lower is better).

### 1. CLARA vs PAM as `n` grows

Four 2-D Gaussian blobs, `k = 4`, `cluster_std = 1.5` (deliberately
**overlapping**, so even a perfect clustering cannot reach ARI 1.0).

| n | PAM time | PAM cost | PAM ARI | CLARA time | CLARA cost | CLARA ARI | CLARA purity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.28 s | 1.797 | 0.786 | **0.07 s** | 1.815 | 0.767 | 0.900 |
| 500 | 3.42 s | 1.761 | 0.821 | **0.22 s** | 1.792 | 0.826 | 0.928 |
| 1000 | 50.0 s | 1.797 | 0.815 | **0.25 s** | 1.878 | 0.791 | 0.908 |
| 1500 | 123.6 s | 1.790 | 0.812 | **0.26 s** | 1.949 | 0.775 | 0.899 |

PAM time grows super-linearly and passes two minutes by `n = 1500`; CLARA
stays flat at a quarter of a second. CLARA pays for it with a slightly higher
cost and a few points of ARI — the expected speed/quality trade-off.

### 2. CLARA on its own, larger `n`

Five blobs, `k = 5`, `cluster_std = 1.8`, `n_samples = 5`.

| n | CLARA time | mean cost | silhouette | ARI |
| ---: | ---: | ---: | ---: | ---: |
| 2 000 | 0.46 s | 2.360 | 0.477 | 0.683 |
| 4 000 | 0.47 s | 2.279 | 0.497 | 0.766 |
| 8 000 | 0.90 s | 2.271 | — | 0.775 |

Doubling `n` from 4 000 to 8 000 roughly doubles the time (the `s · n`
assignment scan dominates) and, if anything, *improves* quality — a bigger
dataset makes each sampled medoid a better estimate.

### 3. Effect of `n_samples` (n = 3 000, k = 5)

| n_samples | time | mean cost | silhouette | ARI |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.08 s | 2.271 | 0.473 | 0.753 |
| 2 | 0.19 s | 2.271 | 0.473 | 0.753 |
| 3 | 0.32 s | 2.271 | 0.473 | 0.753 |
| 5 | 0.52 s | 2.249 | 0.447 | 0.673 |
| 10 | 1.54 s | 2.208 | 0.460 | 0.740 |
| 20 | 2.01 s | **2.152** | 0.475 | 0.762 |

Mean cost falls monotonically as samples are added (the force-keep rule
guarantees this). Note that lower *cost* does not always mean higher *ARI* on
overlapping data — at `n_samples = 5` the run finds a lower-cost partition that
happens to agree less with the labels. Time scales linearly with `n_samples`.

### 4. Effect of `sample_size` (n = 3 000, k = 5, n_samples = 5)

| sample_size | time | mean cost | ARI |
| ---: | ---: | ---: | ---: |
| 20 | 0.21 s | 2.286 | 0.725 |
| 40 | 0.56 s | 2.280 | 0.753 |
| 50 *(default: 40 + 2k)* | 0.61 s | 2.249 | 0.673 |
| 80 | 0.88 s | 2.209 | 0.731 |
| 150 | 4.01 s | 2.143 | 0.764 |
| 300 | 18.9 s | **2.128** | 0.771 |

Bigger samples give PAM more to work with and lower the cost, but time grows
quadratically in `sample_size` (it is a PAM run). The classic `40 + 2k`
default sits at the knee of the curve; push it to ~150 only if you have the
time budget and cost matters more than latency.

### 5. Robustness to outliers — the headline result

Two tight blobs (`std = 0.6`) at `(0,0)` and `(10,0)`, 300 points, plus **15
extreme outliers** scattered uniformly in `[−100, 100]²`.

```
core-point ARI (ignoring the outliers): 1.000
medoids found:  [ 0.067, -0.240]   and   [10.197, -0.065]
```

The medoids land essentially **on the true centres**, unmoved by outliers two
orders of magnitude away. A mean-based method would have its centroids dragged
far off. This is the practical reason to reach for *k*-medoids.

### 6. Metric comparison (n = 1 500, k = 3, 4-D blobs, `std = 1.0`)

| metric | time | mean cost | ARI |
| --- | ---: | ---: | ---: |
| euclidean | 0.16 s | 1.983 | 1.000 |
| manhattan | 0.15 s | 3.371 | 1.000 |
| chebyshev | 0.15 s | 1.547 | 1.000 |
| cosine | 0.20 s | 0.013 | 1.000 |

On well-separated data every metric recovers the clusters exactly. Costs are
not comparable across metrics — they are on different scales.

### 7. Determinism

With a fixed `random_state`, `CLARA.fit` (and `KMedoids.fit`) produce
**identical medoids and identical labels** across runs. Verified in the
benchmark and in the test suite.

---

## Installation

```bash
git clone https://github.com/AfzalSurti/clara.git
cd clara
pip install -e .        # optional — importable straight from the repo root too
```

Python 3.8+. **No runtime dependencies.**

---

## Library API

### `CLARA(n_clusters=2, metric="euclidean", n_samples=5, sample_size=None, max_iter=100, random_state=None)`

| method | description |
| --- | --- |
| `fit(X)` | cluster `X` (list of equal-length float sequences); returns `self` |
| `fit_predict(X)` | `fit` then return `labels_` |
| `predict(X_new)` | assign unseen points to the nearest fitted medoid |

| attribute (after `fit`) | meaning |
| --- | --- |
| `labels_` | cluster id `0 … k−1` for every input point |
| `medoids_` | the `k` medoid points |
| `medoid_indices_` | their indices into `X` |
| `cost_` | mean dissimilarity of a point to its medoid (full dataset) |
| `total_cost_` | `cost_ · n` |
| `best_sample_` | which sample (0-based) produced the winning medoids |
| `sample_size_` | the sample size actually used |

### `KMedoids(n_clusters=2, metric="euclidean", init="build", max_iter=100, random_state=None, precomputed=False)`

Same `fit` / `fit_predict` / `predict` interface. `init` is `"build"` (PAM
BUILD, deterministic) or `"random"` (seeded). With `precomputed=True`, `fit`
takes a square distance matrix and `predict` is unavailable. Attributes:
`labels_`, `medoids_` (absent if precomputed), `medoid_indices_`, `cost_`
(here the **total**, not the mean), `n_iter_`.

```python
from clara import KMedoids
KMedoids(n_clusters=3, precomputed=True).fit(distance_matrix)
```

### `clara.distance`

`euclidean`, `sqeuclidean`, `manhattan`, `chebyshev`, `cosine`,
`minkowski(a, b, p)`, `get_distance(name_or_callable)`,
`pairwise_matrix(points, fn)`. Pass any `callable(a, b) -> float` as `metric=`.

### `clara.metrics`

`total_cost(X, labels, medoids, metric)`, `average_cost(...)`,
`silhouette_samples(X, labels, metric)`, `silhouette_score(X, labels, metric)`.

### `clara.dataset`

`load_csv(path, delimiter=",", has_header=None, label_column=None) -> (X, labels, header)`,
`write_labeled_csv(path, X, labels, header=None, delimiter=",")`,
`make_blobs(n_samples=300, centers=None, n_features=2, cluster_std=1.0, center_box=(-10, 10), random_state=None) -> (X, y)`.
`centers` may be an int (that many random centres) or a list of points.

---

## Command-line interface

```bash
python -m clara INPUT.csv [options]
# or, after `pip install -e .`:
clara INPUT.csv [options]
```

```
$ python -m clara examples/data/blobs.csv -k 4 --label-column -1 --silhouette
clara 0.1.0
  algorithm      : clara
  input          : examples/data/blobs.csv  (600 points, 2 features)
  metric         : euclidean
  clusters (k)    : 4
  samples         : 5  (size 48, best #3)
  mean cost       : 1.816236
  total cost      : 1089.741698
  cluster sizes   : [150, 150, 150, 150]
  medoid indices  : [194, 120, 465, 173]
    medoid 0 (size 150): [3.481, 20.12]
    medoid 1 (size 150): [-0.3008, 0.08772]
    medoid 2 (size 150): [18, 1.551]
    medoid 3 (size 150): [21.58, 21.92]
  silhouette      : 0.861488
```

| flag | meaning |
| --- | --- |
| `-k`, `--clusters` | number of clusters (default 3) |
| `-a`, `--algorithm` | `clara` (default) or `pam` |
| `-m`, `--metric` | `euclidean`, `manhattan`, `chebyshev`, `cosine`, `sqeuclidean`, `l1`, `l2`, `linf` |
| `--samples` | CLARA: number of samples (default 5) |
| `--sample-size` | CLARA: points per sample (default `min(n, 40 + 2k)`) |
| `--max-iter` | PAM SWAP iteration cap (default 100) |
| `--seed` | random seed for reproducible runs |
| `--label-column` | index of a truth-label column to exclude from features |
| `--header` / `--no-header` | override header auto-detection |
| `--delimiter` | CSV field separator (default `,`) |
| `-o`, `--output` | write the input rows back out with a trailing `cluster` column |
| `--silhouette` | also compute the silhouette score (`O(n²)`) |
| `-q`, `--quiet` | print only the space-separated label list |
| `--version` | print version and exit |

Exit codes: `0` success, `2` bad input / arguments.

---

## Repository layout

```
clara/
  __init__.py     public API (CLARA, KMedoids, __version__)
  distance.py     dissimilarity functions + pairwise matrix
  kmedoids.py     PAM: BUILD + SWAP
  clara.py        CLARA sampling loop
  metrics.py      total / average cost, silhouette
  dataset.py      hand-rolled CSV reader/writer, make_blobs
  cli.py          argparse CLI
  __main__.py     `python -m clara`
tests/            67 unittest cases (stdlib unittest, no pytest needed)
examples/
  quickstart.py   CLARA vs PAM on 2 000 points
  benchmark.py    the full suite behind the numbers above
  data/blobs.csv  600-row sample dataset with a label column
```

---

## Testing

```bash
python -m unittest discover -s tests -v        # 67 cases, ~2 s
```

| file | covers |
| --- | --- |
| `test_distance.py` | each metric's values, symmetry, non-negativity, length checks, the name resolver, the triangular matrix |
| `test_kmedoids.py` | recovers 3 separated blobs, cost matches an independent recompute, SWAP never worsens the BUILD cost, precomputed-matrix path, seeded `random` init, `k == n`, parameter validation |
| `test_clara.py` | recovers 4 clusters on 1 200 points (contingency check + silhouette > 0.8), seed reproducibility, "more samples never worsen cost", cost-attribute consistency, medoids are real input points, `predict` matches `labels_`, Manhattan metric, tiny-`n` fallback, parameter validation |
| `test_metrics.py` | cost on hand-worked examples, silhouette near 1.0 for clean blobs, true labels beat shuffled, single/singleton-cluster edge cases, per-sample range `[−1, 1]`, a fully hand-computed 4-point case |
| `test_dataset.py` | CSV line splitter (quotes, `""`, alt delimiter, trailing empty), header auto-detect, label-column extraction (±index), ragged/non-numeric row errors, blank-line skipping, write→read round trip, `make_blobs` shape/labels/reproducibility |
| `test_cli.py` | end-to-end `clara` and `pam` runs, `--quiet`, `--output` file + header, `--silhouette`, `--label-column`, bad-`k` and missing-file errors, alternate metric |

---

## Design decisions and limitations

- **Naive SWAP.** PAM's SWAP recomputes the objective for every candidate
  swap. It is `O(k·(n−k)·n)` per iteration with a large constant in pure
  Python — see the benchmark (~2 min at `n = 1500`). This is intentional:
  CLARA is the scalable path, and the readable SWAP makes the reference
  implementation easy to trust. Use `CLARA`, not `KMedoids`, past ~1 000
  points.
- **Dense `O(n²)` distance matrix** inside every PAM run. Bounded for CLARA
  (the sample is small); the reason `KMedoids` on raw large `n` is
  memory-heavy.
- **`O(n²)` silhouette.** The exact definition, no approximation. Opt in with
  `--silhouette`; avoid on very large `n`.
- **Pure-Python arithmetic.** No SIMD/vectorisation, so per-distance cost is
  ~10–50× a `numpy` equivalent. Acceptable here by design — the goal is a
  dependency-free, legible implementation.
- **CSV newlines in quoted fields** are not parsed (uncommon for numeric
  data).
- **`make_blobs`** produces only isotropic Gaussian clusters.

## Possible improvements

- **FastPAM1 / FastPAM2** (Schubert & Rousseeuw, 2019) — reduce SWAP to
  `O((n−k)·n)` per iteration by sharing work across medoids and caching each
  point's nearest / second-nearest medoid.
- **CLARANS** — randomised neighbour search instead of fixed samples.
- **k-medoids++ / build-from-sample** initialisation for `KMedoids`.
- **Sparse or on-disk distance handling** for the direct large-`n` case.
- **`fit` on a streaming / chunked assignment** so CLARA's final pass never
  holds all points in memory at once.

## References

- L. Kaufman, P. J. Rousseeuw. *Clustering by means of medoids.* 1987.
- L. Kaufman, P. J. Rousseeuw. *Finding Groups in Data: An Introduction to
  Cluster Analysis.* Wiley, 1990. (PAM, CLARA, silhouette.)
- E. Schubert, P. J. Rousseeuw. *Fast and eager k-medoids clustering.*
  Information Systems, 2019. (FastPAM.)
- P. J. Rousseeuw. *Silhouettes: a graphical aid to the interpretation and
  validation of cluster analysis.* J. Comput. Appl. Math., 1987.

## License

MIT
