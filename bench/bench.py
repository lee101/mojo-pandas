"""End-to-end benchmarks against pandas on identical inputs."""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojopandas as mpd


def timed(fn, repeats=5):
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        if result is None:
            raise RuntimeError("benchmark unexpectedly returned None")
        samples.append(elapsed)
    return statistics.median(samples) * 1000


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def assert_equal(actual, expected):
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(pd.Series(actual), expected, rtol=1e-9, atol=1e-9)
    else:
        pd.testing.assert_frame_equal(
            pd.DataFrame(actual), expected, check_frame_type=False, rtol=1e-9, atol=1e-9
        )


def main():
    rng = np.random.default_rng(2026)
    cases = []

    n = 1_000_000
    group_frame = pd.DataFrame(
        {
            "key": rng.integers(0, 4096, n, dtype=np.int64),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )
    mojo_group_frame = mpd.DataFrame(group_frame)
    cases.append(
        (
            "groupby sum (2 cols)",
            n,
            lambda: group_frame.groupby("key").sum(),
            lambda: mojo_group_frame.groupby("key").sum(),
        )
    )
    cases.append(
        (
            "groupby std (2 cols)",
            n,
            lambda: group_frame.groupby("key").std(),
            lambda: mojo_group_frame.groupby("key").std(),
        )
    )

    n_sort = 500_000
    sort_frame = pd.DataFrame(
        {"key": rng.normal(size=n_sort), "value": rng.normal(size=n_sort)}
    )
    mojo_sort_frame = mpd.DataFrame(sort_frame)
    cases.append(
        (
            "stable sort",
            n_sort,
            lambda: sort_frame.sort_values("key", kind="stable"),
            lambda: mojo_sort_frame.sort_values("key", kind="stable"),
        )
    )

    n_left = 500_000
    n_right = 250_000
    left = pd.DataFrame(
        {
            "key": rng.integers(0, n_right * 2, n_left, dtype=np.int64),
            "x": rng.normal(size=n_left),
        }
    )
    right = pd.DataFrame(
        {"key": np.arange(n_right, dtype=np.int64), "y": rng.normal(size=n_right)}
    )
    mojo_left = mpd.DataFrame(left)
    cases.append(
        (
            "left hash join",
            n_left,
            lambda: left.merge(right, on="key", how="left", sort=False),
            lambda: mojo_left.merge(right, on="key", how="left", sort=False),
        )
    )

    n_roll = 1_000_000
    rolling_values = rng.normal(size=n_roll)
    rolling_values[::101] = np.nan
    rolling_series = pd.Series(rolling_values)
    mojo_rolling_series = mpd.Series(rolling_values)
    cases.append(
        (
            "rolling mean (w=128)",
            n_roll,
            lambda: rolling_series.rolling(128, min_periods=64).mean(),
            lambda: mojo_rolling_series.rolling(128, min_periods=64).mean(),
        )
    )
    cases.append(
        (
            "rolling min (w=128)",
            n_roll,
            lambda: rolling_series.rolling(128, min_periods=64).min(),
            lambda: mojo_rolling_series.rolling(128, min_periods=64).min(),
        )
    )

    n_time = 500_000
    time_index = pd.date_range("2020-01-01", periods=n_time, freq="min")
    time_values = rng.normal(size=n_time)
    time_series = pd.Series(time_values, index=time_index)
    mojo_time_series = mpd.Series(time_values, index=time_index)
    cases.append(
        (
            "resample hourly sum",
            n_time,
            lambda: time_series.resample("1h").sum(),
            lambda: mojo_time_series.resample("1h").sum(),
        )
    )

    rows = []
    for name, size, pandas_fn, mojo_fn in cases:
        assert_equal(mojo_fn(), pandas_fn())
        pandas_ms = timed(pandas_fn)
        mojo_ms = timed(mojo_fn)
        rows.append((name, size, pandas_ms, mojo_ms, pandas_ms / mojo_ms))

    print(f"Machine: {cpu_name()} ({platform.machine()}), Python {platform.python_version()}, pandas {pd.__version__}")
    print()
    print("| Kernel | Rows | pandas (ms) | Mojo (ms) | Speedup |")
    print("|---|---:|---:|---:|---:|")
    for name, size, pandas_ms, mojo_ms, speedup in rows:
        print(f"| {name} | {size:,} | {pandas_ms:.3f} | {mojo_ms:.3f} | {speedup:.2f}x |")


if __name__ == "__main__":
    main()
