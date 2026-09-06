"""Command-line interface: ``clara [options] input.csv``.

Runs CLARA (default) or plain PAM on a numeric CSV, prints a summary, and
optionally writes the input back out with a ``cluster`` column.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from . import __version__
from .clara import CLARA
from .dataset import load_csv, write_labeled_csv
from .kmedoids import KMedoids
from .metrics import average_cost, silhouette_score


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clara",
        description="CLARA / PAM k-medoids clustering, pure Python, no dependencies.",
    )
    p.add_argument("input", help="path to a numeric CSV file")
    p.add_argument("-k", "--clusters", type=int, default=3, help="number of clusters (default 3)")
    p.add_argument(
        "-a", "--algorithm", choices=("clara", "pam"), default="clara",
        help="clara (sampled, scales to large n) or pam (exact, small n only)",
    )
    p.add_argument(
        "-m", "--metric", default="euclidean",
        help="euclidean, manhattan, chebyshev, cosine, sqeuclidean, l1, l2, linf",
    )
    p.add_argument("--samples", type=int, default=5, help="CLARA: number of samples (default 5)")
    p.add_argument("--sample-size", type=int, default=None, help="CLARA: points per sample")
    p.add_argument("--max-iter", type=int, default=100, help="PAM SWAP iteration cap")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    p.add_argument("--delimiter", default=",", help="CSV field delimiter (default ',')")
    p.add_argument(
        "--label-column", type=int, default=None,
        help="index of a ground-truth label column to exclude from features",
    )
    p.add_argument("--no-header", action="store_true", help="treat the first row as data")
    p.add_argument("--header", action="store_true", help="treat the first row as a header")
    p.add_argument("-o", "--output", default=None, help="write input rows + cluster column here")
    p.add_argument(
        "--silhouette", action="store_true",
        help="also compute the silhouette score (O(n^2); slow for large n)",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="only print the label list")
    p.add_argument("--version", action="version", version=f"clara {__version__}")
    return p


def _resolve_header_flag(args) -> Optional[bool]:
    if args.no_header and args.header:
        raise SystemExit("clara: --header and --no-header are mutually exclusive")
    if args.no_header:
        return False
    if args.header:
        return True
    return None


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    has_header = _resolve_header_flag(args)

    try:
        X, truth, header = load_csv(
            args.input,
            delimiter=args.delimiter,
            has_header=has_header,
            label_column=args.label_column,
        )
    except (OSError, ValueError) as exc:
        print(f"clara: {exc}", file=sys.stderr)
        return 2

    n = len(X)
    if args.clusters < 1 or args.clusters > n:
        print(f"clara: -k must be between 1 and {n}", file=sys.stderr)
        return 2

    if args.algorithm == "clara":
        model = CLARA(
            n_clusters=args.clusters,
            metric=args.metric,
            n_samples=args.samples,
            sample_size=args.sample_size,
            max_iter=args.max_iter,
            random_state=args.seed,
        ).fit(X)
        mean_cost = model.cost_
        medoid_indices = model.medoid_indices_
        medoids = model.medoids_
        labels = model.labels_
    else:
        model = KMedoids(
            n_clusters=args.clusters,
            metric=args.metric,
            max_iter=args.max_iter,
            random_state=args.seed,
        ).fit(X)
        labels = model.labels_
        medoid_indices = model.medoid_indices_
        medoids = model.medoids_
        mean_cost = average_cost(X, labels, medoids, args.metric)

    if args.quiet:
        print(" ".join(str(v) for v in labels))
        return 0

    if args.output:
        try:
            write_labeled_csv(args.output, X, labels, header, args.delimiter)
        except OSError as exc:
            print(f"clara: cannot write {args.output}: {exc}", file=sys.stderr)
            return 2

    sizes: List[int] = [0] * args.clusters
    for lab in labels:
        sizes[lab] += 1

    print(f"clara {__version__}")
    print(f"  algorithm      : {args.algorithm}")
    print(f"  input          : {args.input}  ({n} points, {len(X[0])} features)")
    print(f"  metric         : {args.metric}")
    print(f"  clusters (k)    : {args.clusters}")
    if args.algorithm == "clara":
        print(f"  samples         : {args.samples}  (size {model.sample_size_}, best #{model.best_sample_})")
    print(f"  mean cost       : {mean_cost:.6f}")
    print(f"  total cost      : {mean_cost * n:.6f}")
    print(f"  cluster sizes   : {sizes}")
    print(f"  medoid indices  : {medoid_indices}")
    for ci, m in enumerate(medoids):
        pretty = "[" + ", ".join(f"{v:.4g}" for v in m) + "]"
        print(f"    medoid {ci} (size {sizes[ci]}): {pretty}")
    if args.silhouette:
        if len(set(labels)) < 2:
            print("  silhouette      : n/a (only one cluster)")
        else:
            print(f"  silhouette      : {silhouette_score(X, labels, args.metric):.6f}")
    if args.output:
        print(f"  written         : {args.output}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    raise SystemExit(run(argv))


if __name__ == "__main__":
    main()
