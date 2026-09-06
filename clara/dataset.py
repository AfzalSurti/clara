"""Data loading and synthetic data -- from scratch, no third-party libraries.

Includes a small hand-rolled CSV reader (quoted fields and doubled-quote
escaping are handled; quoted fields containing a newline are not) so the whole
package really is dependency-free.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

Row = List[str]


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def _split_csv_line(line: str, delimiter: str) -> Row:
    fields: Row = []
    buf: List[str] = []
    in_quotes = False
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < n and line[i + 1] == '"':
                    buf.append('"')
                    i += 2
                    continue
                in_quotes = False
                i += 1
                continue
            buf.append(ch)
            i += 1
        else:
            if ch == '"':
                in_quotes = True
                i += 1
            elif ch == delimiter:
                fields.append("".join(buf))
                buf = []
                i += 1
            else:
                buf.append(ch)
                i += 1
    fields.append("".join(buf))
    return fields


def load_csv(
    path: str,
    delimiter: str = ",",
    has_header: Optional[bool] = None,
    label_column: Optional[int] = None,
) -> Tuple[List[List[float]], Optional[List[str]], Optional[List[str]]]:
    """Read a numeric CSV into ``(X, labels, header)``.

    Parameters
    ----------
    delimiter : str
        Single-character field separator.
    has_header : bool | None
        ``None`` auto-detects: the first row is treated as a header when it is
        not fully parseable as numbers.
    label_column : int | None
        Column index to pull out as ground-truth labels (kept out of ``X``).
        Negative indices count from the end.

    Returns
    -------
    X : list[list[float]]
    labels : list[str] | None
    header : list[str] | None
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        raw_lines = [ln.rstrip("\r\n") for ln in fh]
    rows = [_split_csv_line(ln, delimiter) for ln in raw_lines if ln.strip() != ""]
    if not rows:
        raise ValueError(f"{path}: no data rows")

    if has_header is None:
        first_numeric = True
        for f in rows[0]:
            try:
                float(f)
            except ValueError:
                first_numeric = False
                break
        has_header = not first_numeric

    header: Optional[List[str]] = None
    if has_header:
        header = rows[0]
        rows = rows[1:]
    if not rows:
        raise ValueError(f"{path}: header present but no data rows")

    width = len(rows[0])
    lbl_idx: Optional[int] = None
    if label_column is not None:
        lbl_idx = label_column if label_column >= 0 else width + label_column
        if not (0 <= lbl_idx < width):
            raise ValueError(f"label_column {label_column} out of range for width {width}")

    X: List[List[float]] = []
    labels: List[str] = []
    for r, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"{path}: row {r + 1} has {len(row)} fields, expected {width}"
            )
        if lbl_idx is not None:
            labels.append(row[lbl_idx])
        feats = [v for j, v in enumerate(row) if j != lbl_idx]
        try:
            X.append([float(v) for v in feats])
        except ValueError as exc:
            raise ValueError(f"{path}: row {r + 1} has a non-numeric feature: {exc}")

    if header is not None and lbl_idx is not None:
        header = [h for j, h in enumerate(header) if j != lbl_idx]

    return X, (labels if lbl_idx is not None else None), header


def write_labeled_csv(
    path: str,
    X: Sequence[Sequence[float]],
    labels: Sequence[int],
    header: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    cluster_field: str = "cluster",
) -> None:
    """Write ``X`` back out with a trailing integer ``cluster`` column."""
    def esc(value: str) -> str:
        s = str(value)
        if delimiter in s or '"' in s or "\n" in s:
            return '"' + s.replace('"', '""') + '"'
        return s

    lines: List[str] = []
    if header is not None:
        lines.append(delimiter.join(esc(h) for h in list(header) + [cluster_field]))
    for row, lab in zip(X, labels):
        cells = [repr(float(v)) if isinstance(v, float) else str(v) for v in row]
        lines.append(delimiter.join(esc(c) for c in cells + [str(lab)]))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------
def make_blobs(
    n_samples: int = 300,
    centers: Optional[Sequence[Sequence[float]]] = None,
    n_features: int = 2,
    cluster_std: float = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    random_state: Optional[int] = None,
) -> Tuple[List[List[float]], List[int]]:
    """Isotropic Gaussian blobs.  Returns ``(X, y)`` with integer labels.

    If ``centers`` is an int it is the number of random centres; if it is a
    list of points those are the centres; ``None`` -> 3 random centres.
    """
    rng = random.Random(random_state)

    if centers is None:
        centers = 3
    if isinstance(centers, int):
        lo, hi = center_box
        center_pts = [
            [rng.uniform(lo, hi) for _ in range(n_features)] for _ in range(centers)
        ]
    else:
        center_pts = [list(map(float, c)) for c in centers]
        n_features = len(center_pts[0])

    k = len(center_pts)
    base = n_samples // k
    counts = [base] * k
    for i in range(n_samples - base * k):
        counts[i] += 1

    X: List[List[float]] = []
    y: List[int] = []
    for ci, c in enumerate(center_pts):
        for _ in range(counts[ci]):
            X.append([c[d] + rng.gauss(0.0, cluster_std) for d in range(n_features)])
            y.append(ci)

    order = list(range(len(X)))
    rng.shuffle(order)
    X = [X[i] for i in order]
    y = [y[i] for i in order]
    return X, y
