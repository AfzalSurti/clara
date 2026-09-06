# clara

**CLARA** (Clustering LARge Applications) and **PAM** (Partitioning Around
Medoids / k-medoids), implemented **from scratch in pure Python** with **no
third-party dependencies** — only the standard library. Even the CSV reader is
hand-rolled.

k-medoids is like k-means, but every cluster centre is an *actual data point*
(a "medoid") and the objective is the sum of pairwise dissimilarities instead
of squared error. That makes it robust to outliers and usable with any
distance function. **PAM** solves it well but costs ~`O(k(n−k)²)` per
iteration — too slow for large `n`. **CLARA** runs PAM on several small random
samples and keeps whichever medoid set best explains the *full* dataset, which
is linear in `n` per sample.

On 2000 points (`examples/quickstart.py`): PAM ≈ 32 s, CLARA ≈ 0.08 s, same
clusters.

## Install

```bash
git clone https://github.com/AfzalSurti/clara.git
cd clara
pip install -e .        # optional — the package is importable as-is from the repo root
```

Python 3.8+. No runtime dependencies.

## Library usage

```python
from clara import CLARA, KMedoids
from clara.dataset import make_blobs
from clara.metrics import silhouette_score

X, y = make_blobs(n_samples=5000, centers=4, cluster_std=1.5, random_state=0)

model = CLARA(n_clusters=4, n_samples=5, random_state=0).fit(X)
model.labels_          # cluster id (0..k-1) for every point
model.medoids_         # the k medoid points
model.medoid_indices_  # their indices into X
model.cost_            # mean distance of a point to its medoid
model.predict(X_new)   # assign unseen points

silhouette_score(X, model.labels_)
```

`KMedoids` has the same interface (`fit` / `fit_predict` / `predict`) and also
accepts a precomputed distance matrix:

```python
KMedoids(n_clusters=3, precomputed=True).fit(distance_matrix)
```

### Distance metrics

`euclidean` (`l2`), `sqeuclidean`, `manhattan` (`l1`, `cityblock`),
`chebyshev` (`linf`), `cosine`, plus `minkowski(a, b, p)`. Pass a name or your
own `callable(a, b) -> float` as `metric=`.

## Command line

```bash
python -m clara examples/data/blobs.csv -k 4 --label-column -1 --silhouette
```

```
  algorithm      : clara
  input          : examples/data/blobs.csv  (600 points, 2 features)
  metric         : euclidean
  clusters (k)    : 4
  samples         : 5  (size 48, best #3)
  mean cost       : 1.816236
  cluster sizes   : [150, 150, 150, 150]
  medoid indices  : [194, 120, 465, 173]
  silhouette      : 0.861488
```

| flag | meaning |
| --- | --- |
| `-k, --clusters` | number of clusters (default 3) |
| `-a, --algorithm` | `clara` (default) or `pam` |
| `-m, --metric` | distance name (default `euclidean`) |
| `--samples` / `--sample-size` | CLARA sampling controls |
| `--max-iter` | PAM SWAP iteration cap |
| `--seed` | reproducible runs |
| `--label-column` | index of a truth-label column to exclude from features |
| `--header` / `--no-header` | override header auto-detection |
| `--delimiter` | CSV separator (default `,`) |
| `-o, --output` | write the input rows back out with a `cluster` column |
| `--silhouette` | also compute the silhouette score (`O(n²)`) |
| `-q, --quiet` | print only the space-separated label list |

## How it works

**PAM** (`clara/kmedoids.py`)
- *BUILD*: first medoid = point with the smallest total distance to all
  others; each next medoid = the point whose addition lowers the assignment
  cost the most.
- *SWAP*: repeatedly apply the single medoid ↔ non-medoid swap that reduces
  total cost the most; stop when none does.

**CLARA** (`clara/clara.py`)
- Draw `n_samples` subsets of `sample_size` points
  (default `min(n, 40 + 2k)`).
- Run PAM on each; assign *all* `n` points to the resulting medoids; score by
  mean dissimilarity.
- Keep the best medoid set. From the 2nd sample on, the running-best medoids
  are forced into the sample, so the result can only improve or hold.

## Project layout

```
clara/
  distance.py   metrics.py   kmedoids.py   clara.py
  dataset.py    cli.py       __main__.py
tests/          examples/
```

## Tests

```bash
python -m unittest discover -s tests -v      # 67 cases
```

## License

MIT
