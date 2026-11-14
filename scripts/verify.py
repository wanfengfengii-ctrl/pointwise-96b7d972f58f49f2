"""Brute-force reference check of duration_quantile.Index.

Builds small random signals, answers random queries with the indexed structure
and with a straightforward per-window accumulator, and fails loudly on any
mismatch.  Also exercises the documented worked example and the validation
rules.  Runs with the standard library only.

Usage:  python3 scripts/verify.py [--seed N] [--trials N]
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duration_quantile import Index  # noqa: E402


def brute(intervals, qs, qe, p, q):
    """Reference implementation: scan the window, weigh by covered duration."""
    acc = {}
    covered = 0
    for s, e, v in intervals:
        a = s if s > qs else qs
        b = e if e < qe else qe
        if a < b:
            d = b - a
            covered += d
            acc[v] = acc.get(v, 0) + d
    if covered == 0:
        return None
    target = covered * p
    running = 0
    for v in sorted(acc):
        running += acc[v]
        if running * q >= target:  # exact integer comparison
            return v
    raise AssertionError("unreachable: accumulator never reached target")


def random_signal(rng, n_max=12, value_pool=(-3, -1, 0, 1, 2, 5, 9)):
    n = rng.randint(0, n_max)
    cur = rng.randint(-10, 5)
    intervals = []
    for _ in range(n):
        cur += rng.randint(0, 3)  # 0 creates an exact adjacency, >0 a gap
        start = cur
        end = start + rng.randint(1, 6)
        intervals.append([start, end, rng.choice(value_pool)])
        cur = end
    return intervals, cur


def check_validation():
    def bad(fn, label):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("expected ValueError: " + label)

    bad(lambda: Index([[1, 1, 0]]), "empty interval")
    bad(lambda: Index([[3, 2, 0]]), "start >= end")
    bad(lambda: Index([[0, 2, 1], [2, 4, 1], [1, 5, 1]]), "starts not increasing")
    bad(lambda: Index([[0, 3, 1], [2, 5, 1]]), "overlapping intervals")
    bad(lambda: Index([[0, 2, True]]), "bool value")
    bad(lambda: Index([[0, 2, 1.0]]), "float value")
    bad(lambda: Index(5), "non-sequence")
    bad(lambda: Index([[0, 2]]), "wrong arity")

    idx = Index([[0, 2, 1], [4, 6, 2]])
    bad(lambda: idx.query(2, 2, 1, 1), "empty window")
    bad(lambda: idx.query(3, 2, 1, 1), "reversed window")
    bad(lambda: idx.query(0, 2, 0, 1), "p == 0")
    bad(lambda: idx.query(0, 2, 3, 2), "p > q")
    bad(lambda: idx.query(0, 2, 1, False), "bool query argument")
    bad(lambda: idx.query(0.5, 2, 1, 1), "float query argument")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--trials", type=int, default=1500)
    args = ap.parse_args()

    check_validation()
    print("validation: OK")

    # Documented worked example.
    example = [[0, 4, 10], [6, 10, 20]]
    snapshot = [list(row) for row in example]
    idx = Index(example)
    assert idx.query(2, 8, 1, 2) == 10
    assert idx.query(4, 6, 1, 2) is None
    assert example == snapshot, "input list was mutated"
    print("worked example: OK  query(2,8,1,2)=10  query(4,6,1,2)=None")
    print("repeatable queries:", idx.query(2, 8, 1, 2), idx.query(2, 8, 1, 4),
          idx.query(2, 8, 3, 4), idx.query(0, 10, 1, 1))

    rng = random.Random(args.seed)
    checked = 0
    for trial in range(args.trials):
        intervals, _ = random_signal(rng)
        original = [list(row) for row in intervals]
        idx = Index(intervals)
        assert intervals == original, "Index mutated its input"
        for _ in range(20):
            qs = rng.randint(-15, 20)
            qe = rng.randint(qs + 1, 40)
            q = rng.randint(1, 6)
            p = rng.randint(1, q)
            got = idx.query(qs, qe, p, q)
            want = brute(intervals, qs, qe, p, q)
            if got != want:
                raise AssertionError(
                    "mismatch intervals=%r query=(%d,%d,%d,%d) got=%r want=%r"
                    % (intervals, qs, qe, p, q, got, want)
                )
            checked += 1

    print("randomized brute-force comparison: %d queries over %d signals: OK"
          % (checked, args.trials))

    # Large-magnitude coordinates and a non-trivial sample ratio.
    big = 1 << 80
    big_idx = Index([[0, big, 7], [big, 2 * big, 9]])
    assert big_idx.query(0, 2 * big, 1, 2) == 7
    assert big_idx.query(big, 2 * big, 1, 1) == 9
    print("big-integer coordinates: OK")


if __name__ == "__main__":
    main()
