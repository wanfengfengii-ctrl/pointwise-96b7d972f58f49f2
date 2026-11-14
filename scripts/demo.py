"""Demo: duration-weighted quantiles and windows that contain missing data.

Prints the effective duration of each window, showing that gaps between
intervals carry no weight, that a window entirely inside a gap returns None,
and that a partially covered window is weighted only by its covered portion.

Usage:  python3 scripts/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duration_quantile import Index  # noqa: E402

INTERVALS = [
    [0, 4, 10],     # 4 units at value 10
    [6, 10, 20],    # 4 units at value 20, but [4, 6) is missing
    [12, 14, 10],   # 2 units at value 10, [10, 12) missing
    [14, 20, 30],   # 6 units at value 30, touching the previous interval
]

QUERIES = [
    (2, 8, 1, 2),   # covers 2 of 10 and 2 of 20; effective 4 -> answer 10
    (4, 6, 1, 2),   # entirely inside the first gap -> None
    (11, 13, 1, 1), # only [12,13) exists -> covers 1 -> 10
    (0, 20, 1, 2),  # full span, but only 16 of the 20 units exist
    (0, 20, 1, 1),  # all existing data -> highest value 30
    (0, 20, 7, 10), # threshold 11.2: 10 has 6, +20 has 4 (10 total) -> 30
    (0, 20, 3, 10), # threshold 4.8 of 16 -> value 10 (6 units)
    (5, 6, 1, 1),   # gap only -> None
    (5, 13, 1, 2),  # covered [6,10) + [12,13) = 5, threshold 2.5 -> 20
    (-5, 1, 3, 4),  # half-covered left edge, threshold 0.75 -> 10
]


def covered(intervals, qs, qe):
    total = 0
    for s, e, _ in intervals:
        a = max(s, qs)
        b = min(e, qe)
        if a < b:
            total += b - a
    return total


def main():
    print("signal intervals [start, end) -> value:")
    for row in INTERVALS:
        print("  [%2d, %2d) -> %d" % tuple(row))
    print("missing: [4,6), [10,12); total window [0,20) is 20 units wide,"
          " effective duration is %d" % covered(INTERVALS, 0, 20))
    print()

    idx = Index(INTERVALS)
    print("query(start, end, p, q) -> covered duration, result")
    for qs, qe, p, q in QUERIES:
        cov = covered(INTERVALS, qs, qe)
        result = idx.query(qs, qe, p, q)
        note = "none (missing data only)" if result is None else str(result)
        print("  query(%2d, %2d, %d/%d) -> covered=%2d  answer=%s"
              % (qs, qe, p, q, cov, note))

    print()
    print("Same index re-queried (immutability / repeatability):")
    print("  query(2, 8, 1, 2) =", idx.query(2, 8, 1, 2))
    print("  query(2, 8, 1, 2) =", idx.query(2, 8, 1, 2))

    print()
    print("Weighting check: window [0,20) is missing [4,6) and [10,12), so the")
    print("  effective duration is 16 and the 1/2 threshold is 8 units.")
    print("  Value 10 holds 6 units ([0,4) + [12,14)) < 8, value 20 adds 4 more")
    print("  for a cumulative 10 >= 8, so the answer is 20:",
          idx.query(0, 20, 1, 2))


if __name__ == "__main__":
    main()
