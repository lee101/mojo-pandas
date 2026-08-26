import numpy as np
import pandas as pd
import pytest

import mojopandas as mpd
from mojopandas import _kernels


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_series_float_sort_matches_stable_pandas(ascending, na_position):
    values = [3.0, np.nan, 1.0, 3.0, -2.0, np.nan, 1.0]
    actual = mpd.Series(values, name="v").sort_values(
        ascending=ascending, na_position=na_position, kind="stable"
    )
    expected = pd.Series(values, name="v").sort_values(
        ascending=ascending, na_position=na_position, kind="stable"
    )
    pd.testing.assert_series_equal(pd.Series(actual), expected)


def test_series_integer_sort_ignore_index():
    actual = mpd.Series([4, 1, 4, -2]).sort_values(
        ascending=False, kind="stable", ignore_index=True
    )
    expected = pd.Series([4, 1, 4, -2]).sort_values(
        ascending=False, kind="stable", ignore_index=True
    )
    pd.testing.assert_series_equal(pd.Series(actual), expected)


def test_dataframe_sort_values_matches_pandas():
    frame = pd.DataFrame({"key": [2.0, np.nan, 1.0, 2.0], "v": list("abcd")})
    actual = mpd.DataFrame(frame).sort_values(
        "key", ascending=False, na_position="first", kind="mergesort"
    )
    expected = frame.sort_values(
        "key", ascending=False, na_position="first", kind="mergesort"
    )
    pd.testing.assert_frame_equal(pd.DataFrame(actual), expected, check_frame_type=False)


def test_sort_key_and_inplace():
    actual = mpd.Series([-3.0, 1.0, -2.0])
    expected = pd.Series([-3.0, 1.0, -2.0])
    assert actual.sort_values(key=np.abs, inplace=True, kind="stable") is None
    expected.sort_values(key=np.abs, inplace=True, kind="stable")
    pd.testing.assert_series_equal(pd.Series(actual), expected)


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_float_sort_simd_tail_special_values(ascending, na_position):
    values = np.array(
        [0.0, -0.0, np.inf, -np.inf, np.nan, 3.0, -2.0, np.nan, 3.0]
    )
    actual = mpd.Series(values).sort_values(
        ascending=ascending, na_position=na_position, kind="stable"
    )
    expected = pd.Series(values).sort_values(
        ascending=ascending, na_position=na_position, kind="stable"
    )
    pd.testing.assert_series_equal(pd.Series(actual), expected)


@pytest.mark.parametrize("size", [2_047, 2_053])
@pytest.mark.parametrize("ascending", [True, False])
def test_float_sort_radix_threshold_and_simd_tail(size, ascending):
    values = (np.arange(size, dtype=np.float64)[::-1] % 127) - 63
    values[::97] = np.nan
    values[1::211] = -0.0
    actual = _kernels.argsort(values, ascending=ascending, na_position="last")
    expected = pd.Series(values).sort_values(
        ascending=ascending, na_position="last", kind="stable"
    ).index.to_numpy(dtype=np.int64)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("size", [262_143, 262_147])
def test_integer_sort_parallel_threshold(size):
    values = (np.arange(size, dtype=np.int64)[::-1] % 8191) - 4096
    actual = _kernels.argsort(values)
    expected = np.argsort(values, kind="stable")
    np.testing.assert_array_equal(actual, expected)


@pytest.fixture
def merge_frames():
    left = pd.DataFrame(
        {"key": [2, 1, 2, 4], "value": [20.0, 10.0, 21.0, 40.0], "shared": [1.0, 2.0, 3.0, 4.0]}
    )
    right = pd.DataFrame(
        {"key": [2, 2, 3], "other": [200.0, 201.0, 300.0], "shared": [5.0, 6.0, 7.0]}
    )
    return left, right


@pytest.mark.parametrize("how", ["inner", "left"])
def test_hash_join_matches_pandas(merge_frames, how):
    left, right = merge_frames
    actual = mpd.DataFrame(left).merge(right, on="key", how=how, sort=False)
    expected = left.merge(right, on="key", how=how, sort=False)
    pd.testing.assert_frame_equal(pd.DataFrame(actual), expected, check_frame_type=False)


def test_hash_join_sort_suffixes_and_indicator(merge_frames):
    left, right = merge_frames
    actual = mpd.merge(
        mpd.DataFrame(left),
        right,
        on="key",
        how="left",
        sort=True,
        suffixes=("_left", "_right"),
        indicator="source",
    )
    expected = pd.merge(
        left,
        right,
        on="key",
        how="left",
        sort=True,
        suffixes=("_left", "_right"),
        indicator="source",
    )
    pd.testing.assert_frame_equal(pd.DataFrame(actual), expected, check_frame_type=False)


def test_hash_join_validate(merge_frames):
    left, right = merge_frames
    with pytest.raises(pd.errors.MergeError):
        mpd.DataFrame(left).merge(right, on="key", validate="one_to_one")


def test_hash_join_rejects_unknown_validate(merge_frames):
    left, right = merge_frames
    with pytest.raises(ValueError, match="not a valid argument"):
        mpd.DataFrame(left).merge(right, on="key", validate="anything")


@pytest.mark.parametrize("dtype", [np.int32, np.uint64])
def test_sort_and_join_reject_silent_integer_narrowing(dtype):
    with pytest.raises(TypeError, match="without narrowing"):
        _kernels.argsort(np.array([2, 1], dtype=dtype))
    with pytest.raises(TypeError, match="int64"):
        _kernels.join_indexers(
            np.array([1], dtype=dtype), np.array([1], dtype=np.int64), "inner"
        )


@pytest.mark.parametrize("how", ["inner", "left"])
def test_hash_join_empty_inputs_match_pandas(how):
    left = pd.DataFrame({"key": np.array([], dtype=np.int64), "x": np.array([], dtype=float)})
    right = pd.DataFrame({"key": np.array([], dtype=np.int64), "y": np.array([], dtype=float)})
    actual = mpd.DataFrame(left).merge(right, on="key", how=how)
    expected = left.merge(right, on="key", how=how)
    pd.testing.assert_frame_equal(pd.DataFrame(actual), expected, check_frame_type=False)
