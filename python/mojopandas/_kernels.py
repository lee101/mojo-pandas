"""NumPy-facing wrappers around the raw Mojo exports."""

from __future__ import annotations

import numpy as np

from ._lib import addr, lib


OPS = {"sum": 0, "mean": 1, "min": 2, "max": 3, "count": 4, "var": 5, "std": 6, "prod": 7}
ROLLING_OPS = {"sum": 0, "mean": 1, "min": 2, "max": 3, "var": 4, "std": 5, "count": 6}


def groupby_reduce(codes, values, ngroups: int, op: str, *, ddof: int = 1, min_count: int = 0):
    codes = np.asarray(codes)
    values = np.asarray(values)
    if codes.ndim != 1 or codes.dtype != np.dtype(np.int64):
        raise TypeError("group codes must be a one-dimensional int64 array")
    if values.dtype != np.dtype(np.float64) or values.ndim not in (1, 2):
        raise TypeError("group values must be a one- or two-dimensional float64 array")
    if op not in OPS:
        raise ValueError(f"unknown groupby operation: {op!r}")
    if not isinstance(ngroups, (int, np.integer)) or ngroups < 0:
        raise ValueError("ngroups must be a non-negative integer")
    if len(codes) != values.shape[0]:
        raise ValueError("group codes and values must have the same row count")
    codes = np.ascontiguousarray(codes)
    if values.ndim == 1:
        values = values[:, None]
    values = np.asfortranarray(values)
    shape = (ngroups, values.shape[1])
    result = np.full(shape, np.nan, dtype=np.float64, order="F")
    counts = np.zeros(shape, dtype=np.int64, order="F")
    if not result.size or not len(codes):
        if op in {"sum", "count"} and min_count <= 0:
            result.fill(0.0)
        elif op == "prod" and min_count <= 0:
            result.fill(1.0)
        return result, counts
    if op in {"var", "std"}:
        means = np.empty(shape, dtype=np.float64, order="F")
        lib().mp_groupby_var(
            addr(codes), addr(values), addr(result), addr(means), addr(counts),
            len(codes), values.shape[1], ngroups, ddof, op == "std",
        )
    else:
        lib().mp_groupby_reduce(
            addr(codes), addr(values), addr(result), addr(counts),
            len(codes), values.shape[1], ngroups, OPS[op], ddof, min_count,
        )
    return result, counts


def argsort(values, *, ascending: bool = True, na_position: str = "last"):
    array = np.asarray(values)
    if array.ndim != 1:
        raise TypeError("Mojo sort requires a one-dimensional array")
    if array.dtype not in (np.dtype(np.int64), np.dtype(np.float64)):
        raise TypeError("Mojo sort supports int64 and float64 keys without narrowing")
    if na_position not in {"first", "last"}:
        raise ValueError("na_position must be 'first' or 'last'")
    idx = np.empty(len(array), dtype=np.int64)
    if not len(array):
        return idx
    work = np.empty(len(array), dtype=np.int64)
    if array.dtype == np.dtype(np.int64) and not np.ma.isMaskedArray(array):
        data = np.ascontiguousarray(array)
        lib().mp_argsort_i64(addr(data), addr(idx), addr(work), len(data), ascending)
    elif array.dtype == np.dtype(np.float64):
        data = np.ascontiguousarray(array)
        keys = np.empty(len(data), dtype=np.uint64)
        lib().mp_argsort_f64(
            addr(data), addr(idx), addr(work), addr(keys), len(data), ascending,
            na_position == "first",
        )
    else:
        raise TypeError("Mojo sort supports int64 and float64 keys")
    return idx


def rolling_reduce(values, window: int, min_periods: int, op: str, *, ddof: int = 1):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype != np.dtype(np.float64):
        raise TypeError("rolling values must be a one-dimensional float64 array")
    if op not in ROLLING_OPS:
        raise ValueError(f"unknown rolling operation: {op!r}")
    if not isinstance(window, (int, np.integer)) or window <= 0:
        raise ValueError("window must be a positive integer")
    if not isinstance(min_periods, (int, np.integer)) or not 0 <= min_periods <= window:
        raise ValueError("min_periods must satisfy 0 <= min_periods <= window")
    data = np.ascontiguousarray(values)
    result = np.full(len(data), np.nan, dtype=np.float64)
    if not len(data):
        return result
    queue = np.empty(max(len(data), 1), dtype=np.int64)
    lib().mp_rolling_reduce(
        addr(data), addr(result), addr(queue), len(data), window, min_periods,
        ROLLING_OPS[op], ddof,
    )
    return result


def join_indexers(left_keys, right_keys, how: str):
    left_keys = np.asarray(left_keys)
    right_keys = np.asarray(right_keys)
    if (
        left_keys.ndim != 1
        or right_keys.ndim != 1
        or left_keys.dtype != np.dtype(np.int64)
        or right_keys.dtype != np.dtype(np.int64)
    ):
        raise TypeError("join keys must be one-dimensional int64 arrays without narrowing")
    if how not in {"inner", "left"}:
        raise ValueError("how must be 'inner' or 'left'")
    left = np.ascontiguousarray(left_keys)
    right = np.ascontiguousarray(right_keys)
    if not len(left):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    if not len(right):
        if how == "inner":
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy()
        return np.arange(len(left), dtype=np.int64), np.full(len(left), -1, dtype=np.int64)
    capacity = 2
    while capacity < max(2, len(right) * 2):
        capacity *= 2
    used = np.zeros(capacity, dtype=np.int64)
    slot_keys = np.empty(capacity, dtype=np.int64)
    heads = np.empty(capacity, dtype=np.int64)
    tails = np.empty(capacity, dtype=np.int64)
    next_rows = np.empty(max(len(right), 1), dtype=np.int64)
    left_join = how == "left"
    count = lib().mp_join_count(
        addr(left), addr(right), addr(used), addr(slot_keys), addr(heads), addr(tails),
        addr(next_rows), len(left), len(right), capacity, left_join,
    )
    left_idx = np.empty(count, dtype=np.int64)
    right_idx = np.empty(count, dtype=np.int64)
    written = lib().mp_join_fill(
        addr(left), addr(right), addr(left_idx), addr(right_idx), addr(used), addr(slot_keys),
        addr(heads), addr(tails), addr(next_rows), len(left), len(right), capacity, left_join,
    )
    if written != count:
        raise RuntimeError("join kernel produced an inconsistent result size")
    if how == "inner" and count and np.all(left[left_idx] == left[left_idx[0]]):
        order = np.argsort(right_idx, kind="stable")
        left_idx = left_idx[order]
        right_idx = right_idx[order]
    return left_idx, right_idx
