"""Windowed duration quantiles for irregularly sampled integer intervals.

An input signal is a list of disjoint half-open intervals ``[start, end)``, each
carrying one integer ``value``.  Stretches not covered by any interval are
"missing data" and contribute no weight at all.

For a query window ``[qs, qe)`` and a pair of integers ``0 < p <= q``, the
effective duration of the window is the total length of the parts of the
intervals that fall inside it.  The answer is the smallest signal value ``v``
for which the duration spent at values ``<= v`` first reaches
``effective * p / q``; if the window touches no interval at all the answer is
``None``.

All comparisons are exact integer comparisons (``accumulated * q >=
effective * p``) -- no floating point is involved anywhere.

Public API::

    idx = Index(intervals)              # validate once, build the index
    idx.query(start, end, p, q)         # -> int | None

Complexity (n = number of intervals, m = number of distinct values):

* build:  ``O(n log n)`` time, ``O(n log n)`` space
  (value compression + one persistent segment-tree version per interval);
* query:  ``O(log n)`` time -- two bisections over the interval endpoints and
  at most three descents through the value tree.  A query never scans the
  intervals that a window covers.
"""

from array import array
from bisect import bisect_left, bisect_right

__all__ = ["Index"]

_INT64_MAX = (1 << 63) - 1


def _is_int(x):
    """True for genuine integers; ``bool`` is explicitly rejected."""
    return isinstance(x, int) and not isinstance(x, bool)


class Index:
    """Queryable index over a duration-weighted irregular signal.

    ``Index(intervals)`` validates ``intervals`` and builds immutable prefix
    structures.  The input sequence is only read, never modified, and the very
    same index can answer any number of queries.
    """

    __slots__ = (
        "_starts",
        "_ends",
        "_prefix",
        "_roots",
        "_left",
        "_right",
        "_sums",
        "_distinct",
        "_ranks",
        "_m",
    )

    # ---------------------------------------------------------------- build

    def __init__(self, intervals):
        if isinstance(intervals, (list, tuple)):
            items = intervals
        else:
            try:
                items = list(intervals)
            except TypeError:
                raise ValueError(
                    "intervals must be a list of [start, end, value] triples"
                ) from None

        n = len(items)
        starts = [0] * n
        ends = [0] * n
        values = [0] * n
        prefix = [0] * (n + 1)

        prev_start = None
        prev_end = None
        total = 0

        for i in range(n):
            item = items[i]
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError(
                    "intervals[%d] must be a [start, end, value] triple" % i
                )
            s, e, v = item
            if not (_is_int(s) and _is_int(e) and _is_int(v)):
                raise ValueError(
                    "intervals[%d]: start, end and value must be non-bool integers" % i
                )
            if s >= e:
                raise ValueError("intervals[%d]: start must be less than end" % i)
            if prev_start is not None:
                if s <= prev_start:
                    raise ValueError(
                        "intervals[%d]: starts must be strictly increasing" % i
                    )
                if s < prev_end:
                    raise ValueError("intervals[%d]: intervals must not overlap" % i)
            starts[i] = s
            ends[i] = e
            values[i] = v
            total += e - s
            prefix[i + 1] = total
            prev_start = s
            prev_end = e

        # Coordinate compression of the values.  Every query answer is one of
        # these values, so ranks can replace values in the tree.
        distinct = sorted(set(values))
        m = len(distinct)
        rank_of = {v: r for r, v in enumerate(distinct)}
        ranks = [rank_of[v] for v in values]

        # Persistent segment tree over value ranks.  Version i (roots[i]) holds
        # the full durations of intervals [0, i) keyed by value rank.  Each
        # insertion copies only the O(log m) nodes on one root-to-leaf path.
        left = array("i", [0])
        right = array("i", [0])
        # Node sums never exceed the total duration; use a compact int64 array
        # whenever that is safe, otherwise fall back to Python integers.
        sums = array("q", [0]) if total <= _INT64_MAX else [0]
        roots = array("i", [0])
        lapp = left.append
        rapp = right.append
        sapp = sums.append
        roots_app = roots.append
        counter = 1

        for i in range(n):
            pos = ranks[i]
            w = ends[i] - starts[i]
            prev = roots[-1]

            new = counter
            counter += 1
            lapp(left[prev])
            rapp(right[prev])
            sapp(sums[prev] + w)
            roots_app(new)

            cur_new = new
            cur_prev = prev
            lo_r = 0
            hi_r = m - 1
            while lo_r < hi_r:
                mid = (lo_r + hi_r) >> 1
                if pos <= mid:
                    child_prev = left[cur_prev]
                    child_new = counter
                    counter += 1
                    lapp(left[child_prev])
                    rapp(right[child_prev])
                    sapp(sums[child_prev] + w)
                    left[cur_new] = child_new
                    cur_new = child_new
                    cur_prev = child_prev
                    hi_r = mid
                else:
                    child_prev = right[cur_prev]
                    child_new = counter
                    counter += 1
                    lapp(left[child_prev])
                    rapp(right[child_prev])
                    sapp(sums[child_prev] + w)
                    right[cur_new] = child_new
                    cur_new = child_new
                    cur_prev = child_prev
                    lo_r = mid + 1

        self._starts = starts
        self._ends = ends
        self._prefix = prefix
        self._roots = roots
        self._left = left
        self._right = right
        self._sums = sums
        self._distinct = distinct
        self._ranks = ranks
        self._m = m

    # ---------------------------------------------------------------- query

    def query(self, start, end, p, q):
        """Return the duration-weighted p/q quantile of ``[start, end)``.

        ``start``, ``end``, ``p``, ``q`` must be non-bool integers with
        ``start < end`` and ``0 < p <= q``.  Returns ``None`` when the window
        is entirely missing data.
        """
        if not (_is_int(start) and _is_int(end) and _is_int(p) and _is_int(q)):
            raise ValueError("query arguments must be non-bool integers")
        if start >= end:
            raise ValueError("query start must be less than end")
        if p <= 0:
            raise ValueError("p must be positive")
        if p > q:
            raise ValueError("p must not exceed q")

        starts = self._starts
        if not starts:
            return None

        ends = self._ends
        # First interval ending after ``start``; ends are strictly increasing.
        lo = bisect_right(ends, start)
        # First interval starting at or after ``end``.
        hi = bisect_left(starts, end)
        if lo >= hi:
            return None

        f = lo
        l = hi - 1
        sf = starts[f]
        ef = ends[f]
        head = sf if sf > start else start
        tail = ef if ef < end else end
        ovf = tail - head

        corr = []
        if f == l:
            covered = ovf
            excess = (ef - sf) - ovf
            if excess:
                corr.append((self._ranks[f], excess))
        else:
            sl = starts[l]
            el = ends[l]
            head_l = sl if sl > start else start
            tail_l = el if el < end else end
            ovl = tail_l - head_l
            if l > f + 1:
                inter = self._prefix[l] - self._prefix[f + 1]
            else:
                inter = 0
            covered = ovf + inter + ovl
            excess_f = (ef - sf) - ovf
            excess_l = (el - sl) - ovl
            if excess_f:
                corr.append((self._ranks[f], excess_f))
            if excess_l:
                corr.append((self._ranks[l], excess_l))
            corr.sort()

        # The tree stores full interval durations for the index range [lo, hi).
        # Values are returned by descending to the first rank whose cumulative
        # duration satisfies the threshold.
        root_hi = self._roots[hi]
        root_lo = self._roots[lo]
        target = covered * p
        m = self._m
        distinct = self._distinct
        correction = 0
        region_lo = 0
        qq = q

        for rank_limit, excess in corr:
            if region_lo < rank_limit:
                required = (target + correction * qq + qq - 1) // qq
                rank = self._first_rank(required, root_hi, root_lo)
                if rank < rank_limit:
                    return distinct[rank]
            correction += excess
            region_lo = rank_limit

        required = (target + correction * qq + qq - 1) // qq
        rank = self._first_rank(required, root_hi, root_lo)
        return distinct[rank]

    def _first_rank(self, target, root_hi, root_lo):
        """Smallest value rank whose cumulative weight reaches ``target``.

        The weight of a rank range is ``version(hi) - version(lo)``.  Both
        versions are traversed together, so this is a single O(log m) descent
        without any per-node prefix query.
        """
        left = self._left
        right = self._right
        sums = self._sums
        a = root_hi
        b = root_lo
        lo = 0
        hi = self._m - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            la = left[a]
            lb = left[b]
            w = sums[la] - sums[lb]
            if w >= target:
                a = la
                b = lb
                hi = mid
            else:
                target -= w
                a = right[a]
                b = right[b]
                lo = mid + 1
        return lo
