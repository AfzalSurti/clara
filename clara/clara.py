"""CLARA -- Clustering LARge Applications -- from scratch in pure Python.

CLARA (Kaufman & Rousseeuw, 1990) scales PAM to large ``n``:

1. Draw a small random sample of the data.
2. Run PAM (:class:`clara.kmedoids.KMedoids`) on that sample.
3. Assign **every** point in the full dataset to its nearest sampled medoid and
   record the mean dissimilarity.
4. Repeat for several samples; keep the medoid set with the lowest full-data
   mean dissimilarity.

From the second sample onward the current best medoids are forced into the
sample (the standard refinement), so CLARA can only improve or hold.

Cost is ``O(n_samples * (sample_size^2 * ... + n))`` -- linear in ``n`` per
sample instead of PAM's quadratic.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from .distance import Point, get_distance
from .kmedoids import KMedoids


def _assign_all(X: Sequence[Point], medoid_points: Sequence[Point], fn):
    """Nearest-medoid labels (0..k-1) and total distance over all of ``X``."""
    labels = [0] * len(X)
    total = 0.0
    for i, p in enumerate(X):
        best_pos = 0
        best_d = fn(p, medoid_points[0])
        for pos in range(1, len(medoid_points)):
            d = fn(p, medoid_points[pos])
            if d < best_d:
                best_d = d
                best_pos = pos
        labels[i] = best_pos
        total += best_d
    return labels, total


class CLARA:
    """Clustering LARge Applications.

    Parameters
    ----------
    n_clusters : int
        Number of medoids ``k``.
    metric : str | callable
        Anything :func:`clara.distance.get_distance` accepts.
    n_samples : int
        How many samples to draw and run PAM on.
    sample_size : int | None
        Points per sample.  ``None`` -> ``min(n, 40 + 2*k)`` (the classic
        default).  Must be > ``k``.
    max_iter : int
        Passed through to PAM's SWAP phase for each sample.
    random_state : int | None
        Seed for sampling; makes the whole fit reproducible.

    Attributes (set after :meth:`fit`)
    ----------------------------------
    medoid_indices_ : list[int]        -- indices of the best medoids in ``X``
    medoids_ : list[point]             -- the medoid points
    labels_ : list[int]              -- cluster id (0..k-1) for every point
    cost_ : float                    -- best mean dissimilarity over all points
    total_cost_ : float              -- ``cost_ * n``
    best_sample_ : int               -- which sample (0-based) won
    sample_size_ : int               -- the sample size actually used
    """

    def __init__(
        self,
        n_clusters: int = 2,
        metric="euclidean",
        n_samples: int = 5,
        sample_size: Optional[int] = None,
        max_iter: int = 100,
        random_state: Optional[int] = None,
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if sample_size is not None and sample_size <= n_clusters:
            raise ValueError("sample_size must be greater than n_clusters")
        self.n_clusters = n_clusters
        self.metric = metric
        self.n_samples = n_samples
        self.sample_size = sample_size
        self.max_iter = max_iter
        self.random_state = random_state

        self.medoid_indices_: List[int] = []
        self.medoids_: List[Point] = []
        self.labels_: List[int] = []
        self.cost_: float = float("nan")
        self.total_cost_: float = float("nan")
        self.best_sample_: int = -1
        self.sample_size_: int = 0

    def fit(self, X: Sequence[Point]):
        n = len(X)
        if n == 0:
            raise ValueError("X is empty")
        k = self.n_clusters
        if k > n:
            raise ValueError(f"n_clusters={k} exceeds number of samples {n}")

        if self.sample_size is None:
            sample_size = min(n, 40 + 2 * k)
        else:
            sample_size = min(self.sample_size, n)
        if sample_size <= k:
            # only possible when n itself is tiny; fall back to using every point
            sample_size = n
        self.sample_size_ = sample_size

        fn = get_distance(self.metric)
        rng = random.Random(self.random_state)
        all_indices = list(range(n))

        best_cost = None
        best_medoids: List[int] = []
        best_labels: List[int] = []
        best_sample = -1

        for s in range(self.n_samples):
            if best_medoids and sample_size > k:
                forced = list(best_medoids)
                forced_set = set(forced)
                pool = [i for i in all_indices if i not in forced_set]
                extra = rng.sample(pool, sample_size - len(forced))
                sample_idx = forced + extra
            else:
                sample_idx = rng.sample(all_indices, sample_size)

            sub = [X[i] for i in sample_idx]
            pam = KMedoids(
                n_clusters=k,
                metric=self.metric,
                init="build",
                max_iter=self.max_iter,
            ).fit(sub)

            global_medoids = [sample_idx[m] for m in pam.medoid_indices_]
            medoid_points = [X[g] for g in global_medoids]
            labels, total = _assign_all(X, medoid_points, fn)
            mean_cost = total / n

            if best_cost is None or mean_cost < best_cost:
                best_cost = mean_cost
                best_medoids = global_medoids
                best_labels = labels
                best_sample = s

        self.medoid_indices_ = best_medoids
        self.medoids_ = [list(X[g]) for g in best_medoids]
        self.labels_ = best_labels
        self.cost_ = best_cost
        self.total_cost_ = best_cost * n
        self.best_sample_ = best_sample
        return self

    def fit_predict(self, X: Sequence[Point]) -> List[int]:
        self.fit(X)
        return self.labels_

    def predict(self, X: Sequence[Point]) -> List[int]:
        if not self.medoids_:
            raise ValueError("model is not fitted yet")
        fn = get_distance(self.metric)
        labels, _ = _assign_all(X, self.medoids_, fn)
        return labels
