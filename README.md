# clara

**CLARA** (Clustering LARge Applications) and **PAM** (Partitioning Around
Medoids / k-medoids), implemented **from scratch in pure Python** with **no
third-party dependencies** — only the standard library.

k-medoids is like k-means but each cluster centre is an *actual data point* (a
"medoid"), and the objective is the sum of pairwise dissimilarities rather than
squared error. That makes it robust to outliers and usable with any distance
function. PAM solves it well but costs about `O(k(n-k)^2)` per iteration, which
is too slow for large `n`. **CLARA** fixes this by running PAM on several small
random samples and keeping whichever medoid set best explains the *full*
dataset.

## Status

Built incrementally. What is landed so far:

- [x] `clara.distance` — euclidean, squared-euclidean, manhattan, chebyshev,
      minkowski, cosine; a name resolver; and a triangular pairwise matrix.
- [x] `clara.kmedoids.KMedoids` — PAM (BUILD + SWAP), `fit` / `predict` /
      `fit_predict`, supports a precomputed distance matrix.
- [ ] `clara.clara.CLARA` — sampling wrapper
- [ ] `clara.metrics` — silhouette, total / average cost
- [ ] `clara.dataset` — CSV loader + synthetic blobs
- [ ] `clara.cli` — command line interface

## Install

```bash
git clone https://github.com/AfzalSurti/clara.git
cd clara
pip install -e .          # optional; the package is importable as-is
```

Requires Python 3.8+. No runtime dependencies.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT
