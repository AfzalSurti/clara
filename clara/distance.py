"""Distance / dissimilarity functions, written from scratch in pure Python.

A "point" here is just a sequence of floats (``list`` or ``tuple``).  Every
function in this module takes two points of equal length and returns a single
non-negative float.  No third-party libraries are used anywhere.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence

Point = Sequence[float]
DistanceFn = Callable[[Point, Point], float]


def _check_same_length(a: Point, b: Point) -> None:
    if len(a) != len(b):
        raise ValueError(
            "points must have the same number of dimensions, got "
            f"{len(a)} and {len(b)}"
        )


def euclidean(a: Point, b: Point) -> float:
    """Straight-line (L2) distance."""
    _check_same_length(a, b)
    total = 0.0
    for x, y in zip(a, b):
        diff = x - y
        total += diff * diff
    return math.sqrt(total)


def sqeuclidean(a: Point, b: Point) -> float:
    """Squared Euclidean distance (cheaper: no square root)."""
    _check_same_length(a, b)
    total = 0.0
    for x, y in zip(a, b):
        diff = x - y
        total += diff * diff
    return total


def manhattan(a: Point, b: Point) -> float:
    """City-block (L1) distance."""
    _check_same_length(a, b)
    total = 0.0
    for x, y in zip(a, b):
        total += abs(x - y)
    return total


def chebyshev(a: Point, b: Point) -> float:
    """Chebyshev (L-infinity) distance: the largest per-axis gap."""
    _check_same_length(a, b)
    largest = 0.0
    for x, y in zip(a, b):
        gap = abs(x - y)
        if gap > largest:
            largest = gap
    return largest


def minkowski(a: Point, b: Point, p: float = 2.0) -> float:
    """Generalised Minkowski distance with order ``p`` (p >= 1)."""
    if p < 1:
        raise ValueError("Minkowski order p must be >= 1")
    _check_same_length(a, b)
    total = 0.0
    for x, y in zip(a, b):
        total += abs(x - y) ** p
    return total ** (1.0 / p)


def cosine(a: Point, b: Point) -> float:
    """Cosine *distance* = 1 - cosine similarity.  Range [0, 2].

    If either vector has zero magnitude the similarity is undefined; we treat
    two zero vectors as identical (distance 0) and a zero vs. non-zero pair as
    maximally dissimilar (distance 1).
    """
    _check_same_length(a, b)
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 and norm_b == 0.0:
        return 0.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    similarity = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    # Guard against tiny floating point overshoot outside [-1, 1].
    if similarity > 1.0:
        similarity = 1.0
    elif similarity < -1.0:
        similarity = -1.0
    return 1.0 - similarity


_NAMED: dict = {
    "euclidean": euclidean,
    "l2": euclidean,
    "sqeuclidean": sqeuclidean,
    "manhattan": manhattan,
    "l1": manhattan,
    "cityblock": manhattan,
    "chebyshev": chebyshev,
    "linf": chebyshev,
    "cosine": cosine,
}


def get_distance(name_or_fn) -> DistanceFn:
    """Resolve a distance either from a known name or pass a callable through."""
    if callable(name_or_fn):
        return name_or_fn
    try:
        return _NAMED[str(name_or_fn).lower()]
    except KeyError:
        raise ValueError(
            f"unknown distance {name_or_fn!r}; known names: "
            + ", ".join(sorted(_NAMED))
        )


def pairwise_matrix(points: Sequence[Point], distance: DistanceFn) -> List[List[float]]:
    """Full symmetric distance matrix.

    Only the upper triangle is computed and then mirrored, so we call the
    distance function ``n * (n - 1) / 2`` times instead of ``n**2``.
    """
    n = len(points)
    matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        pi = points[i]
        row_i = matrix[i]
        for j in range(i + 1, n):
            d = distance(pi, points[j])
            row_i[j] = d
            matrix[j][i] = d
    return matrix
