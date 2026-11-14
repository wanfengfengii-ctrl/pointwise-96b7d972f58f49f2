"""Measured 100k-interval build and 100k-query benchmark for duration_quantile.

Reports wall-clock build time, per-query time for 100,000 randomized queries,
peak resident memory, and a spot check of query answers against a brute-force
reference on the same 100k-interval signal.

Usage:  python3 scripts/bench.py [--intervals 100000] [--queries 100000]
"""

import argparse
import os
import random
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duration_quantile import Index  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import brute  # noqa: E402


def make_signal(rng, n):
    cur = 0
    intervals = []
    for _ in range(n):
        cur += rng.randint(0, 2)  # occasional exact adjacency, mostly gaps
        start = cur
        end = start + rng.randint(1, 40)
        intervals.append([start, end, rng.randint(-10**6, 10**6)])
        cur = end
    return intervals, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", type=int, default=100000)
    ap.add_argument("--queries", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    intervals, end_of_signal = make_signal(rng, args.intervals)

    t0 = time.perf_counter()
    idx = Index(intervals)
    build_s = time.perf_counter() - t0

    queries = []
    for _ in range(args.queries):
        a = rng.randint(0, end_of_signal)
        b = rng.randint(a + 1, end_of_signal + 1)
        q = rng.choice((1, 2, 3, 4, 5, 10, 100))
        p = rng.randint(1, q)
        queries.append((a, b, p, q))

    t0 = time.perf_counter()
    checksum = 0
    none_count = 0
    for a, b, p, q in queries:
        r = idx.query(a, b, p, q)
        if r is None:
            none_count += 1
        else:
            checksum += r
    query_s = time.perf_counter() - t0

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print("intervals:            %d" % len(intervals))
    print("signal extent:        [0, %d)" % end_of_signal)
    print("build time:           %.3f s" % build_s)
    print("queries:              %d" % args.queries)
    print("query time:           %.3f s total, %.2f us/query"
          % (query_s, query_s / args.queries * 1e6))
    print("answers:              checksum=%d  none=%d" % (checksum, none_count))
    print("peak RSS:             %.1f MB" % rss_mb)

    # Independent spot check on the big signal.
    sample_rng = random.Random(args.seed + 1)
    for _ in range(100):
        a = sample_rng.randint(0, end_of_signal)
        b = sample_rng.randint(a + 1, end_of_signal + 1)
        q = sample_rng.randint(1, 7)
        p = sample_rng.randint(1, q)
        got = idx.query(a, b, p, q)
        want = brute(intervals, a, b, p, q)
        if got != want:
            raise AssertionError("bench spot check mismatch: %r" % ((a, b, p, q),))
    print("spot check vs brute force (100 windows on the 100k signal): OK")


if __name__ == "__main__":
    main()
