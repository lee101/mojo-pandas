import numpy as np
import pandas as pd
import pytest

import mojopandas as mpd


@pytest.fixture
def window_series():
    rng = np.random.default_rng(17)
    values = rng.normal(size=250)
    values[::13] = np.nan
    return mpd.Series(values, name="signal"), pd.Series(values, name="signal")


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "count", "var", "std"])
def test_rolling_reductions_match_pandas(window_series, op):
    ours, theirs = window_series
    actual = getattr(ours.rolling(17, min_periods=5), op)()
    expected = getattr(theirs.rolling(17, min_periods=5), op)()
    pd.testing.assert_series_equal(
        pd.Series(actual), expected, rtol=1e-10, atol=1e-10
    )


def test_rolling_zero_min_periods_and_all_nan():
    values = [np.nan, np.nan, 2.0]
    actual = mpd.Series(values).rolling(2, min_periods=0).sum()
    expected = pd.Series(values).rolling(2, min_periods=0).sum()
    pd.testing.assert_series_equal(pd.Series(actual), expected)


def test_rolling_step():
    values = np.arange(20.0)
    actual = mpd.Series(values).rolling(4, step=3).mean()
    expected = pd.Series(values).rolling(4, step=3).mean()
    pd.testing.assert_series_equal(pd.Series(actual), expected)


def test_rolling_requires_float64_without_narrowing():
    with pytest.raises(TypeError, match="float64"):
        mpd.Series(np.arange(5, dtype=np.float32)).rolling(2).sum()


def test_empty_rolling_matches_pandas():
    actual = mpd.Series([], dtype=np.float64).rolling(2).sum()
    expected = pd.Series([], dtype=np.float64).rolling(2).sum()
    pd.testing.assert_series_equal(pd.Series(actual), expected)


@pytest.mark.parametrize("op", ["mean", "var", "std"])
def test_rolling_moments_are_stable_for_large_offsets(op):
    values = 1e12 + np.sin(np.arange(300.0))
    actual = getattr(mpd.Series(values).rolling(31), op)()
    expected = getattr(pd.Series(values).rolling(31), op)()
    pd.testing.assert_series_equal(
        pd.Series(actual), expected, rtol=2e-4, atol=2e-4
    )


def test_rolling_aggregate_list(window_series):
    ours, theirs = window_series
    actual = ours.rolling(9).agg(["sum", "mean", "max"])
    expected = theirs.rolling(9).agg(["sum", "mean", "max"])
    pd.testing.assert_frame_equal(
        pd.DataFrame(actual), expected, check_frame_type=False, rtol=1e-11, atol=1e-11
    )


@pytest.fixture
def resampled_series():
    index = pd.date_range("2025-01-01", periods=80, freq="7min")
    values = np.sin(np.arange(len(index)) / 7)
    values[::11] = np.nan
    values = np.delete(values, np.s_[25:35])
    index = index.delete(np.s_[25:35])
    return mpd.Series(values, index=index, name="v"), pd.Series(values, index=index, name="v")


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "count", "prod", "var", "std"])
def test_resample_reductions_match_pandas(resampled_series, op):
    ours, theirs = resampled_series
    actual = getattr(ours.resample("30min"), op)()
    expected = getattr(theirs.resample("30min"), op)()
    pd.testing.assert_series_equal(
        pd.Series(actual), expected, rtol=1e-11, atol=1e-11
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"closed": "right", "label": "right"},
        {"origin": "epoch", "offset": "5min"},
    ],
)
def test_resample_bin_options(resampled_series, kwargs):
    ours, theirs = resampled_series
    actual = ours.resample("1h", **kwargs).sum()
    expected = theirs.resample("1h", **kwargs).sum()
    pd.testing.assert_series_equal(pd.Series(actual), expected)


def test_resample_aggregate_list(resampled_series):
    ours, theirs = resampled_series
    actual = ours.resample("1h").agg(["sum", "mean", "count"])
    expected = theirs.resample("1h").agg(["sum", "mean", "count"])
    pd.testing.assert_frame_equal(pd.DataFrame(actual), expected, check_frame_type=False)
