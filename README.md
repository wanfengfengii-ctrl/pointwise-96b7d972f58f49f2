# duration_quantile

Duration-weighted window quantiles over an irregularly sampled integer signal,
using only the Python standard library.

```python
from duration_quantile import Index

idx = Index([[0, 4, 10], [6, 10, 20]])
idx.query(2, 8, 1, 2)   # 10
idx.query(4, 6, 1, 2)   # None  (the window is entirely missing data)
```

## Semantics

* `intervals` is a list of `[start, end, value]` triples of non-bool integers.
  Each interval is half-open, `start < end`; starts strictly increase and
  intervals never overlap. Gaps between intervals are missing data. Equal
  values may appear in any number of intervals.
* A window `[start, end)` is clipped against the intervals. Its **effective
  duration** is the total length of the interval portions inside it; missing
  stretches contribute neither to the weight nor to the denominator.
* `query(start, end, p, q)` requires `start < end` and `0 < p <= q`. It returns
  the smallest signal value whose cumulative covered duration first reaches
  `effective * p / q`, or `None` if the window covers no data.
* Every threshold comparison is exact integer arithmetic
  (`cumulative * q >= effective * p`); no floating point is used.
* Invalid input raises `ValueError`. The input list is never modified and one
  `Index` can answer any number of queries.

## Complexity

With `n` intervals and `m` distinct values:

| phase | time | space |
| --- | --- | --- |
| build (`Index(...)`) | `O(n log n)` | `O(n log n)` |
| query | `O(log n)` | `O(1)` extra |

Build validates the input in `O(n)`, compresses the values in `O(n log n)`, and
then inserts each interval into a **persistent segment tree** over value ranks,
copying one root-to-leaf path (`O(log m)` nodes) per interval. Version `i` of
the tree stores the full durations of intervals `[0, i)` aggregated by value.

A query uses two bisections over the interval endpoints to find the slice of
intervals intersecting the window (`lo`, `hi`) and the overall covered
duration (a prefix-sum difference plus clipping of the two boundary intervals).
The window's value distribution is then the difference of tree versions `hi`
and `lo`; a single paired descent finds the first value whose weight reaches
the threshold. The boundary intervals' out-of-window durations are subtracted
at their ranks during that same descent (at most two correction points), so a
query never scans the covered intervals — it is `O(log n)` regardless of how
much of the signal the window spans.

## Verification and measurements

```sh
python3 scripts/verify.py     # worked example + randomized brute-force reference
python3 scripts/demo.py       # missing-data/gap windows
python3 scripts/bench.py      # 100k intervals, 100k queries
```

`.pointwise-runtime.json` contains the `setup`, `verify` and `demo` argv arrays.

### Worked example

`[[0,4,10],[6,10,20]]`, window `[2,8)`: covered durations 2 (value 10) + 2
(value 20) = 4, threshold 4×1/2 = 2 → value **10**. Window `[4,6)` is the gap →
**None**. `scripts/verify.py` asserts both, checks the input is unchanged and
repeats the query on the same index.

### Measured (this machine, CPython 3.11, `python3 scripts/bench.py`)

* 100,000 intervals: build ≈ 0.85 s, peak RSS ≈ 87 MB.
* 100,000 randomized queries: ≈ 2.5 s total, ≈ 25 µs/query.
* 100 windows on the 100k signal matched a brute-force scan exactly.
* Additional local stress runs: 2.34 million exhaustive small-window queries
  and 30,000 randomized queries matched the reference implementation, including
  worst cases with 100k distinct values and 100k equal values, and coordinates
  up to 2^200.

### Missing-data demo (`scripts/demo.py`)

Signal `[0,4)=10, [6,10)=20, [12,14)=10, [14,20)=30`; gaps `[4,6)` and
`[10,12)`. The full window `[0,20)` has width 20 but effective duration 16, so
the 1/2 quantile threshold is 8: value 10 holds 6 and value 20 adds 4 → answer
20 (not 30, which a width-based denominator would give). A window inside a gap,
e.g. `[4,6)` or `[5,6)`, returns `None`.

## Files

* `duration_quantile.py` — implementation (`Index`).
* `scripts/verify.py` — brute-force reference check and validation tests.
* `scripts/demo.py` — gap/missing-data demonstration.
* `scripts/bench.py` — 100k-interval / 100k-query benchmark and spot check.
* `.pointwise-runtime.json` — runtime descriptor.
