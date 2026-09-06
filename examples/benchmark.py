"""Reproducible benchmark suite for the numbers quoted in the README.

Run from the repo root:  python examples/benchmark.py

Everything is deterministic (fixed seeds).  Timings are wall-clock on the
machine you run it on and will differ from the README's, but the *relative*
behaviour (CLARA flat in n, PAM quadratic, medoids unmoved by outliers) holds.
"""

from __future__ import annotations

import os
import sys
import time
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clara import CLARA, KMedoids
from clara.dataset import make_blobs
from clara.metrics import silhouette_score

echo = lambda *a: print(*a, flush=True)


def purity(truth, pred):
    groups: dict = {}
    for t, p in zip(truth, pred):
        groups.setdefault(p, {})
        groups[p][t] = groups[p].get(t, 0) + 1
    return sum(max(c.values()) for c in groups.values()) / len(truth)


def adjusted_rand_index(truth, pred):
    idx: dict = {}
    a: dict = {}
    b: dict = {}
    for t, p in zip(truth, pred):
        idx[(t, p)] = idx.get((t, p), 0) + 1
        a[t] = a.get(t, 0) + 1
        b[p] = b.get(p, 0) + 1
    n = len(truth)
    sum_ij = sum(comb(v, 2) for v in idx.values())
    sum_a = sum(comb(v, 2) for v in a.values())
    sum_b = sum(comb(v, 2) for v in b.values())
    expected = sum_a * sum_b / comb(n, 2)
    maximum = (sum_a + sum_b) / 2
    return (sum_ij - expected) / (maximum - expected) if maximum != expected else 1.0


def bench_scaling():
    echo("=== 1. CLARA vs PAM, scaling in n (k=4, std=1.5) ===")
    echo(f"{'n':>6} | {'PAM s':>8} {'PAMcost':>8} {'PAMari':>7} | "
         f"{'CLARA s':>8} {'CLcost':>8} {'CLari':>7} {'purity':>7}")
    for n in (200, 500, 1000, 1500):
        X, y = make_blobs(n_samples=n, centers=4, n_features=2, cluster_std=1.5, random_state=0)
        t = time.perf_counter()
        pam = KMedoids(n_clusters=4, random_state=0).fit(X)
        tp = time.perf_counter() - t
        t = time.perf_counter()
        cl = CLARA(n_clusters=4, n_samples=5, random_state=0).fit(X)
        tc = time.perf_counter() - t
        echo(f"{n:>6} | {tp:8.3f} {pam.cost_ / n:8.4f} {adjusted_rand_index(y, pam.labels_):7.3f} | "
             f"{tc:8.3f} {cl.cost_:8.4f} {adjusted_rand_index(y, cl.labels_):7.3f} {purity(y, cl.labels_):7.3f}")


def bench_clara_large():
    echo("\n=== 2. CLARA alone, larger n (k=5, std=1.8, n_samples=5) ===")
    echo(f"{'n':>6} | {'CLARA s':>8} {'meancost':>9} {'silhou':>8} {'ARI':>6}")
    for n in (2000, 4000, 8000):
        X, y = make_blobs(n_samples=n, centers=5, n_features=2, cluster_std=1.8, random_state=2)
        t = time.perf_counter()
        cl = CLARA(n_clusters=5, n_samples=5, random_state=0).fit(X)
        dt = time.perf_counter() - t
        sil = silhouette_score(X, cl.labels_) if n <= 4000 else float("nan")
        echo(f"{n:>6} | {dt:8.3f} {cl.cost_:9.4f} {sil:8.4f} {adjusted_rand_index(y, cl.labels_):6.3f}")


def bench_n_samples():
    echo("\n=== 3. Effect of n_samples (n=3000, k=5, std=1.8) ===")
    X, y = make_blobs(n_samples=3000, centers=5, n_features=2, cluster_std=1.8, random_state=3)
    echo(f"{'n_samples':>9} | {'time s':>7} {'meancost':>9} {'silhou':>8} {'ARI':>6}")
    for ns in (1, 2, 3, 5, 10, 20):
        t = time.perf_counter()
        cl = CLARA(n_clusters=5, n_samples=ns, random_state=1).fit(X)
        dt = time.perf_counter() - t
        echo(f"{ns:>9} | {dt:7.3f} {cl.cost_:9.4f} {silhouette_score(X, cl.labels_):8.4f} "
             f"{adjusted_rand_index(y, cl.labels_):6.3f}")


def bench_sample_size():
    echo("\n=== 4. Effect of sample_size (n=3000, k=5, n_samples=5) ===")
    X, y = make_blobs(n_samples=3000, centers=5, n_features=2, cluster_std=1.8, random_state=3)
    echo(f"{'sample_size':>11} | {'time s':>7} {'meancost':>9} {'ARI':>6}")
    for ss in (20, 40, 50, 80, 150, 300):
        t = time.perf_counter()
        cl = CLARA(n_clusters=5, n_samples=5, sample_size=ss, random_state=1).fit(X)
        dt = time.perf_counter() - t
        echo(f"{ss:>11} | {dt:7.3f} {cl.cost_:9.4f} {adjusted_rand_index(y, cl.labels_):6.3f}")


def bench_outliers():
    import random

    echo("\n=== 5. Robustness to outliers (2 tight blobs + 15 extreme outliers) ===")
    rng = random.Random(0)
    X, y = make_blobs(n_samples=300, centers=[(0, 0), (10, 0)], cluster_std=0.6, random_state=0)
    for _ in range(15):
        X.append([rng.uniform(-100, 100), rng.uniform(-100, 100)])
        y.append(-1)
    cl = CLARA(n_clusters=2, n_samples=8, random_state=0).fit(X)
    core_truth = [t for t in y if t != -1]
    core_pred = [l for l, t in zip(cl.labels_, y) if t != -1]
    echo(f"  core-point ARI (ignoring outliers): {adjusted_rand_index(core_truth, core_pred):.3f}")
    echo("  medoids (expected near (0,0) and (10,0)):")
    for m in cl.medoids_:
        echo(f"    [{m[0]:7.3f}, {m[1]:7.3f}]")


def bench_metrics():
    echo("\n=== 6. Metric comparison (n=1500, k=3, 4-D, std=1.0) ===")
    X, y = make_blobs(n_samples=1500, centers=3, n_features=4, cluster_std=1.0, random_state=7)
    echo(f"{'metric':>12} | {'time s':>7} {'meancost':>9} {'ARI':>6}")
    for mt in ("euclidean", "manhattan", "chebyshev", "cosine"):
        t = time.perf_counter()
        cl = CLARA(n_clusters=3, metric=mt, n_samples=5, random_state=0).fit(X)
        dt = time.perf_counter() - t
        echo(f"{mt:>12} | {dt:7.3f} {cl.cost_:9.4f} {adjusted_rand_index(y, cl.labels_):6.3f}")


def bench_determinism():
    echo("\n=== 7. Determinism ===")
    X, _ = make_blobs(n_samples=1000, centers=4, random_state=1)
    a = CLARA(n_clusters=4, random_state=42).fit(X)
    b = CLARA(n_clusters=4, random_state=42).fit(X)
    echo(f"  identical medoids across runs: {a.medoid_indices_ == b.medoid_indices_}")
    echo(f"  identical labels across runs : {a.labels_ == b.labels_}")


def main() -> None:
    bench_scaling()
    bench_clara_large()
    bench_n_samples()
    bench_sample_size()
    bench_outliers()
    bench_metrics()
    bench_determinism()


if __name__ == "__main__":
    main()
