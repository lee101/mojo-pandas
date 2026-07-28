# mojo-pandas

`mojo-pandas` is a standalone experimental port of compute-heavy pandas
columnar kernels to Mojo. It provides pandas-compatible `Series` and
`DataFrame` subclasses: construction, indexing, and ordinary container
behavior come from pandas, while covered numeric operations execute in a
single compiled Mojo shared library.

This is useful when an application has large, contiguous numeric columns and
wants pandas-shaped calls without moving the data into a separate query
engine. It is not a replacement for all of pandas.

## Covered subset

- `Series.groupby` and `DataFrame.groupby` over `float64` payload columns with
  one row-aligned or named key:
  `sum`, `mean`, `min`, `max`, `count`, `prod`, `var`, `std`, `size`,
  lists of these reductions, and reduction `transform`
- `Series.sort_values` and one-column `DataFrame.sort_values` for `int64` and
  `float64`, including stable ties, ascending/descending order, and NaN
  placement
- `DataFrame.merge` and `mojopandas.merge` for inner and left hash joins on one
  non-null integer column, including duplicate keys, suffixes, `indicator`,
  `validate`, and sorted output
- fixed-size trailing `float64` `Series.rolling` reductions: `sum`, `mean`, `min`,
  `max`, `count`, `var`, and `std`, with `min_periods` and `step`
- sorted datetime-like, `float64` `Series.resample` reductions with pandas bin semantics,
  including empty bins, `closed`, `label`, `origin`, and `offset`

Unsupported variants of the accelerated entry points raise
`NotImplementedError` or `TypeError` rather than silently taking a different
kernel path. Other methods inherited by the subclasses still execute ordinary
pandas code and are not Mojo ports. The main kernel omissions are
heterogeneous grouped payloads, multiple group or join keys,
right/outer/cross joins, nullable join keys, multi-column sorting,
expanding/exponentially weighted or offset windows, centered and weighted
rolling windows, and DataFrame rolling and resampling. Pandas I/O, reshape,
string, categorical, and extension-array APIs are not accelerated.

## Install and build

The repository pins the tested Mojo nightly and carries pandas as the parity
oracle:

```bash
pixi install
pixi run build
pixi run test
```

The build writes `dist/libmojo-pandas.so`. Importing the Python package also
rebuilds it when the Mojo source is newer.

## Usage

```python
import mojopandas as mpd

sales = mpd.DataFrame(
    {
        "store": [2, 1, 2, 1],
        "revenue": [12.5, 8.0, 9.5, 11.0],
    }
)

totals = sales.groupby("store").sum()
smoothed = mpd.Series([3.0, 4.0, 8.0, 6.0]).rolling(
    3, min_periods=1
).mean()

print(totals)
print(smoothed)
```

Save the example and run it with `pixi run python example.py` from the
repository root.

## Benchmarks

Run benchmarks only through `pixi run bench`; that task holds a
machine-wide lock to avoid overlap with other jobs. Times below are
end-to-end medians and include Python dispatch, conversion of covered numeric
columns to contiguous views where necessary, output construction, and
group/bin setup.

Measured on an Intel Xeon E5-2697 v4 at 2.30GHz (`x86_64`), Python 3.13.14,
and pandas 3.0.3:

| Kernel | Rows | pandas (ms) | Mojo (ms) | Speedup |
|---|---:|---:|---:|---:|
| groupby sum (2 cols) | 1,000,000 | 135.872 | 86.433 | 1.57x |
| groupby std (2 cols) | 1,000,000 | 120.594 | 76.497 | 1.58x |
| stable sort | 500,000 | 122.814 | 119.972 | 1.02x |
| left hash join | 500,000 | 236.046 | 76.118 | 3.10x |
| rolling mean (w=128) | 1,000,000 | 26.722 | 15.427 | 1.73x |
| rolling min (w=128) | 1,000,000 | 33.127 | 22.146 | 1.50x |
| resample hourly sum | 500,000 | 8.361 | 7.173 | 1.17x |

Speedup is `pandas time / mojo-pandas time`; values below 1.00x mean the Mojo
path is slower. Results are a snapshot of one run, not a general performance
guarantee.

There is no GPU path.

## How it works

Python and NumPy own every input, output, and scratch allocation. The ctypes
layer passes each contiguous buffer to Mojo as an integer address plus explicit
shape metadata. The exported C ABI reconstructs typed
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside non-parametric exports,
so no Python object or allocator crosses the boundary.

Grouped values use pandas' native column-major `float64` block views without a
layout copy; group codes, sort indices, and join keys are contiguous `int64`.
Grouped reductions make one column-streaming pass (two for variance), rolling
sums and moments use constant-time window updates, and rolling min/max uses a
monotonic deque. Stable float sorting generates normalized IEEE keys with SIMD
and uses original row positions as tie breakers. Large integer sort passes can
run in parallel. Joins use an open-addressed integer hash table with linked
duplicate rows. Resampling asks pandas only for its exact datetime bin codes
and then performs the reductions in the same Mojo groupby kernel.
