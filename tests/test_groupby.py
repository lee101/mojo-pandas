import numpy as np
import pandas as pd
import pytest

import mojopandas as mpd
from mojopandas import _kernels


@pytest.fixture
def grouped_series():
    rng = np.random.default_rng(12)
    values = rng.normal(size=300)
    values[::17] = np.nan
    keys = pd.Series(rng.choice(["alpha", "beta", "gamma"], size=len(values)), name="key")
    return mpd.Series(values, name="value"), pd.Series(values, name="value"), keys


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "count", "prod", "var", "std"])
def test_series_groupby_reductions_match_pandas(grouped_series, op):
    ours, theirs, keys = grouped_series
    actual = getattr(ours.groupby(keys), op)()
    expected = getattr(theirs.groupby(keys), op)()
    pd.testing.assert_series_equal(pd.Series(actual), expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("dropna", [True, False])
def test_groupby_order_and_null_keys(sort, dropna):
    keys = pd.Series(["b", None, "a", "b", None], name="key")
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    actual = mpd.Series(values, name="v").groupby(
        keys, sort=sort, dropna=dropna
    ).sum()
    expected = pd.Series(values, name="v").groupby(
        keys, sort=sort, dropna=dropna
    ).sum()
    pd.testing.assert_series_equal(pd.Series(actual), expected)


@pytest.fixture
def grouped_frame():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "key": rng.integers(0, 12, size=500),
            "x": rng.normal(size=500),
            "y": rng.normal(size=500),
        }
    )
    frame.loc[::19, "x"] = np.nan
    return mpd.DataFrame(frame), frame


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "count", "var", "std", "prod"])
def test_dataframe_groupby_reductions_match_pandas(grouped_frame, op):
    ours, theirs = grouped_frame
    actual = getattr(ours.groupby("key"), op)()
    expected = getattr(theirs.groupby("key"), op)()
    pd.testing.assert_frame_equal(
        pd.DataFrame(actual), expected, check_frame_type=False, rtol=1e-11, atol=1e-11
    )


def test_groupby_min_count(grouped_series):
    ours, theirs, keys = grouped_series
    actual_sum = ours.groupby(keys).sum(min_count=200)
    expected_sum = theirs.groupby(keys).sum(min_count=200)
    actual_min = ours.groupby(keys).min(min_count=200)
    expected_min = theirs.groupby(keys).min(min_count=200)
    pd.testing.assert_series_equal(pd.Series(actual_sum), expected_sum)
    pd.testing.assert_series_equal(pd.Series(actual_min), expected_min)


def test_groupby_as_index_false(grouped_frame):
    ours, theirs = grouped_frame
    actual = ours.groupby("key", as_index=False).mean()
    expected = theirs.groupby("key", as_index=False).mean()
    pd.testing.assert_frame_equal(pd.DataFrame(actual), expected, check_frame_type=False)


def test_groupby_column_selection(grouped_frame):
    ours, theirs = grouped_frame
    actual = ours.groupby("key")["x"].sum()
    expected = theirs.groupby("key")["x"].sum()
    pd.testing.assert_series_equal(pd.Series(actual), expected)


def test_groupby_aggregate_list(grouped_frame):
    ours, theirs = grouped_frame
    actual = ours.groupby("key").agg(["sum", "mean", "std"])
    expected = theirs.groupby("key").agg(["sum", "mean", "std"])
    pd.testing.assert_frame_equal(
        pd.DataFrame(actual), expected, check_frame_type=False, rtol=1e-11, atol=1e-11
    )


def test_groupby_transform(grouped_frame):
    ours, theirs = grouped_frame
    actual = ours.groupby("key").transform("mean")
    expected = theirs.groupby("key").transform("mean")
    pd.testing.assert_frame_equal(pd.DataFrame(actual), expected, check_frame_type=False)


def test_groupby_size_includes_null_values():
    frame = pd.DataFrame({"key": [1, 1, 2], "x": [1.0, np.nan, np.nan]})
    actual = mpd.DataFrame(frame).groupby("key").size()
    expected = frame.groupby("key").size()
    pd.testing.assert_series_equal(pd.Series(actual), expected)


def test_dataframe_groupby_uses_fortran_contiguous_view():
    frame = mpd.DataFrame(
        {
            "key": np.arange(9) % 3,
            "a": np.arange(9, dtype=np.float64),
            "b": np.arange(9, dtype=np.float64) + 10,
            "c": np.arange(9, dtype=np.float64) + 20,
        }
    )
    grouped = frame.groupby("key")
    matrix = grouped._matrix()
    assert matrix.flags.f_contiguous
    assert np.shares_memory(matrix, frame["a"].to_numpy())
    expected = pd.DataFrame(frame).groupby("key").std()
    actual = grouped.std()
    pd.testing.assert_frame_equal(
        pd.DataFrame(actual), expected, check_frame_type=False
    )


@pytest.mark.parametrize("size", [262_143, 262_147])
def test_groupby_std_parallel_threshold(size):
    rng = np.random.default_rng(91)
    frame = pd.DataFrame(
        {
            "key": np.arange(size, dtype=np.int64) % 257,
            "x": rng.normal(size=size),
            "y": rng.normal(size=size),
        }
    )
    frame.loc[::101, "x"] = np.nan
    actual = mpd.DataFrame(frame).groupby("key").std()
    expected = frame.groupby("key").std()
    pd.testing.assert_frame_equal(
        pd.DataFrame(actual), expected, check_frame_type=False,
        rtol=1e-11, atol=1e-11,
    )


def test_groupby_boundary_rejects_mismatched_lengths_and_dtypes():
    with pytest.raises(ValueError, match="same row count"):
        _kernels.groupby_reduce(
            np.array([0, 0], dtype=np.int64),
            np.array([1.0], dtype=np.float64),
            1,
            "sum",
        )
    with pytest.raises(TypeError, match="float64"):
        _kernels.groupby_reduce(
            np.array([0], dtype=np.int64),
            np.array([1.0], dtype=np.float32),
            1,
            "sum",
        )


def test_empty_groupby_matches_pandas():
    actual = mpd.Series([], dtype=np.float64).groupby(
        np.array([], dtype=np.int64)
    ).sum()
    expected = pd.Series([], dtype=np.float64).groupby(
        np.array([], dtype=np.int64)
    ).sum()
    pd.testing.assert_series_equal(pd.Series(actual), expected)
