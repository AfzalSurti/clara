"""Partitioning Around Medoids (PAM), a.k.a. k-medoids -- from scratch.

PAM has two phases:

* **BUILD** -- greedily pick ``k`` initial medoids.  The first medoid is the
  point with the smallest total distance to every other point; each further
  medoid is the point that, when added, reduces the assignment cost the most.
* **SWAP** -- repeatedly try replacing one medoid with one non-medoid.  Apply
  the single swap that lowers the total cost the most, and stop when no swap
  helps.

The objective (``cost_``) is the sum over all points of the distance to the
nearest medoid.  Everything is plain Python; the only stdlib import is
``random`` for the optional random initialisation.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from .distance import Point, get_distance, pairwise_matrix

_EPS = 1e-12


def _assign(D: Sequence[Sequence[float]], medoids: Sequence[int]):
    """Return (labels, cost) for the given medoid indices.

    ``labels[i]`` is the position *within ``medoids``* of the nearest medoid to
    point ``i`` (so labels are 0..k-1).  Ties go to the earliest medoid.
    """
    labels = [0] * len(D)
    cost = 0.0
    for i, row in enumerate(D):
        best_pos = 0
        best_dist = row[medoids[0]]
        for pos in range(1, len(medoids)):
            d = row[medoids[pos]]
            if d < best_dist:
                best_dist = d
                best_pos = pos
        labels[i] = best_pos
        cost += best_dist
    return labels, cost


def _build_init(D: Sequence[Sequence[float]], k: int) -> List[int]:
    """PAM BUILD: greedy construction of ``k`` medoids."""
    n = len(D)
    totals = [sum(row) for row in D]
    first = min(range(n), key=lambda i: totals[i])
    medoids = [first]
    chosen = {first}
    # nearest[j] = distance from point j to the closest medoid picked so far
    nearest = [D[j][first] for j in range(n)]

    while len(medoids) < k:
        best_gain = None
        best_cand = None
        for c in range(n):
            if c in chosen:
                continue
            gain = 0.0
            Dc = D[c]
            for j in range(n):
                if j in chosen:
                    continue
                delta = nearest[j] - Dc[j]
                if delta > 0.0:
                    gain += delta
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_cand = c
        if best_cand is None:  # fewer distinct candidates than k
            for c in range(n):
                if c not in chosen:
                    best_cand = c
                    break
        medoids.append(best_cand)
        chosen.add(best_cand)
        Dbc = D[best_cand]
        for j in range(n):
            if Dbc[j] < nearest[j]:
                nearest[j] = Dbc[j]
    return medoids


def _random_init(n: int, k: int, rng: random.Random) -> List[int]:
    return rng.sample(range(n), k)


def _swap_phase(D, medoids: List[int], max_iter: int):
    """PAM SWAP: apply the best cost-reducing medoid/non-medoid swap until none.

    This is the straightforward "recompute the cost for every candidate swap"
    formulation -- O(max_iter * k * (n-k) * n).  It is easy to verify and fast
    enough for the small samples CLARA feeds it; for big inputs use :class:`CLARA`.
    """
    n = len(D)
    medoids = list(medoids)
    _, current_cost = _assign(D, medoids)
    n_iter = 0

    while n_iter < max_iter:
        n_iter += 1
        chosen = set(medoids)
        best_cost = current_cost
        best_swap = None
        for mi in range(len(medoids)):
            for h in range(n):
                if h in chosen:
                    continue
                trial = medoids[:]
                trial[mi] = h
                _, c = _assign(D, trial)
                if c < best_cost - _EPS:
                    best_cost = c
                    best_swap = (mi, h)
        if best_swap is None:
            break
        mi, h = best_swap
        medoids[mi] = h
        current_cost = best_cost

    return medoids, current_cost, n_iter


class KMedoids:
    """k-medoids clustering via PAM.

    Parameters
    ----------
    n_clusters : int
        Number of medoids ``k``.
    metric : str | callable
        Anything :func:`clara.distance.get_distance` accepts.  Ignored when
        ``precomputed=True``.
    init : {"build", "random"}
        Initial medoid selection strategy.
    max_iter : int
        Maximum SWAP iterations.
    random_state : int | None
        Seed for the ``"random"`` init.
    precomputed : bool
        If True, ``fit`` expects a square distance matrix instead of points.

    Attributes (set after :meth:`fit`)
    ----------------------------------
    medoid_indices_ : list[int]   -- indices of the chosen medoids in ``X``
    medoids_ : list[point]        -- the medoid points (absent if precomputed)
    labels_ : list[int]          -- cluster id (0..k-1) per input point
    cost_ : float                -- sum of distances to the nearest medoid
    n_iter_ : int                -- SWAP iterations actually run
    """

    def __init__(
        self,
        n_clusters: int = 2,
        metric="euclidean",
        init: str = "build",
        max_iter: int = 100,
        random_state: Optional[int] = None,
        precomputed: bool = False,
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if init not in ("build", "random"):
            raise ValueError("init must be 'build' or 'random'")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        self.n_clusters = n_clusters
        self.metric = metric
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.precomputed = precomputed

        self.medoid_indices_: List[int] = []
        self.medoids_: List[Point] = []
        self.labels_: List[int] = []
        self.cost_: float = float("nan")
        self.n_iter_: int = 0

    # -- internal -----------------------------------------------------------
    def _distance_matrix(self, X):
        if self.precomputed:
            n = len(X)
            for row in X:
                if len(row) != n:
                    raise ValueError("precomputed matrix must be square")
            return [list(map(float, row)) for row in X]
        fn = get_distance(self.metric)
        return pairwise_matrix(X, fn)

    # -- public API -------------------------------------------------------
    def fit(self, X: Sequence):
        n = len(X)
        if n == 0:
            raise ValueError("X is empty")
        if self.n_clusters > n:
            raise ValueError(
                f"n_clusters={self.n_clusters} exceeds number of samples {n}"
            )

        D = self._distance_matrix(X)

        if self.init == "build":
            init_medoids = _build_init(D, self.n_clusters)
        else:
            rng = random.Random(self.random_state)
            init_medoids = _random_init(n, self.n_clusters, rng)

        medoids, cost, n_iter = _swap_phase(D, init_medoids, self.max_iter)
        labels, cost = _assign(D, medoids)

        self.medoid_indices_ = list(medoids)
        self.labels_ = labels
        self.cost_ = cost
        self.n_iter_ = n_iter
        if not self.precomputed:
            self.medoids_ = [list(X[i]) for i in medoids]
        return self

    def fit_predict(self, X: Sequence) -> List[int]:
        self.fit(X)
        return self.labels_

    def predict(self, X: Sequence[Point]) -> List[int]:
        """Assign new points to the nearest fitted medoid."""
        if self.precomputed:
            raise ValueError("predict is unavailable when precomputed=True")
        if not self.medoids_:
            raise ValueError("model is not fitted yet")
        fn = get_distance(self.metric)
        out = []
        for p in X:
            best_pos, best_d = 0, fn(p, self.medoids_[0])
            for pos in range(1, len(self.medoids_)):
                d = fn(p, self.medoids_[pos])
                if d < best_d:
                    best_d, best_pos = d, pos
            out.append(best_pos)
        return out
