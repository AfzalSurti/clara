"""Quickstart: generate blobs, cluster them with CLARA, compare against PAM.

Run from the repo root:  python examples/quickstart.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clara import CLARA, KMedoids
from clara.dataset import make_blobs
from clara.metrics import silhouette_score


def purity(truth, pred):
    groups = {}
    for t, p in zip(truth, pred):
        groups.setdefault(p, {})
        groups[p][t] = groups[p].get(t, 0) + 1
    correct = sum(max(counts.values()) for counts in groups.values())
    return correct / len(truth)


def main() -> None:
    X, y = make_blobs(
        n_samples=2000,
        centers=[(0, 0), (25, 0), (0, 25), (25, 25), (12, 12)],
        cluster_std=1.6,
        random_state=7,
    )
    k = 5
    print(f"{len(X)} points, {len(X[0])} features, {k} true clusters\n")

    t0 = time.perf_counter()
    clara = CLARA(n_clusters=k, n_samples=5, random_state=0).fit(X)
    t_clara = time.perf_counter() - t0

    t0 = time.perf_counter()
    pam = KMedoids(n_clusters=k, random_state=0).fit(X)
    t_pam = time.perf_counter() - t0

    print(f"{'':10} {'time (s)':>10} {'mean cost':>12} {'silhouette':>12} {'purity':>9}")
    for name, model, t in (("CLARA", clara, t_clara), ("PAM", pam, t_pam)):
        labels = model.labels_
        mc = getattr(model, "cost_", None)
        if name == "PAM":
            mc = model.cost_ / len(X)
        sil = silhouette_score(X, labels)
        print(f"{name:10} {t:10.3f} {mc:12.4f} {sil:12.4f} {purity(y, labels):9.3f}")

    print(f"\nCLARA medoids (indices): {clara.medoid_indices_}")
    print("CLARA medoid points:")
    for i, m in enumerate(clara.medoids_):
        print(f"  {i}: [" + ", ".join(f"{v:7.3f}" for v in m) + "]")


if __name__ == "__main__":
    main()
