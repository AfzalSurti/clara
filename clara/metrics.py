"""Cluster-quality metrics, from scratch in pure Python.

* :func:`total_cost` / :func:`average_cost` -- the k-medoids objective (sum /
  mean distance of each point to its assigned medoid).
* :func:`silhouette_samples` / :func:`silhouette_score` -- the classic
  silhouette, ``s(i) = (b - a) / max(a, b)``, where ``a`` is the mean
  within-cluster distance and ``b`` the mean distance to the nearest other
  cluster.  This is ``O(n^2)`` distance evaluations -- fine for a few thousand
  points, not for millions.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .distance import Point, get_distance


def total_cost(
    X: Sequence[Point],
    labels: Sequence[int],
    medoids: Sequence[Point],
    metric="euclidean",
) -> float:
    """Sum over points of the distance to the medoid named by ``labels[i]``."""
    if len(X) != len(labels):
        raise ValueError("X and labels must have the same length")
    fn = get_distance(metric)
    total = 0.0
    for p, lab in zip(X, labels):
        total += fn(p, medoids[lab])
    return total


def average_cost(
    X: Sequence[Point],
    labels: Sequence[int],
    medoids: Sequence[Point],
    metric="euclidean",
) -> float:
    if not X:
        raise ValueError("X is empty")
    return total_cost(X, labels, medoids, metric) / len(X)


def _group_by_label(labels: Sequence[int]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)
    return groups


def silhouette_samples(
    X: Sequence[Point], labels: Sequence[int], metric="euclidean"
) -> List[float]:
    """Per-point silhouette values in ``[-1, 1]``.

    Points in a singleton cluster get ``0.0`` (the conventional choice), as do
    all points when there is only one cluster.
    """
    if len(X) != len(labels):
        raise ValueError("X and labels must have the same length")
    fn = get_distance(metric)
    n = len(X)
    groups = _group_by_label(labels)
    other_labels = sorted(groups)
    out = [0.0] * n

    for i in range(n):
        li = labels[i]
        own = groups[li]
        if len(own) <= 1:
            out[i] = 0.0
            continue

        a = 0.0
        for j in own:
            if j != i:
                a += fn(X[i], X[j])
        a /= (len(own) - 1)

        b = None
        for lab in other_labels:
            if lab == li:
                continue
            grp = groups[lab]
            mean_d = 0.0
            for j in grp:
                mean_d += fn(X[i], X[j])
            mean_d /= len(grp)
            if b is None or mean_d < b:
                b = mean_d

        if b is None:  # only one cluster present
            out[i] = 0.0
        else:
            denom = a if a > b else b
            out[i] = 0.0 if denom == 0.0 else (b - a) / denom
    return out


def silhouette_score(
    X: Sequence[Point], labels: Sequence[int], metric="euclidean"
) -> float:
    """Mean of :func:`silhouette_samples`."""
    samples = silhouette_samples(X, labels, metric)
    if not samples:
        raise ValueError("X is empty")
    return sum(samples) / len(samples)
