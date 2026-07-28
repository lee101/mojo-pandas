"""Pandas-compatible objects backed by Mojo kernels on covered numeric paths."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import _kernels


_REDUCTIONS = {"sum", "mean", "min", "max", "count", "var", "std", "prod"}


def _factorize(by, *, sort: bool, dropna: bool):
    keys = pd.Index(by)
    codes, uniques = pd.factorize(keys, sort=sort, use_na_sentinel=dropna)
    uniques = pd.Index(uniques, name=getattr(by, "name", None))
    return np.asarray(codes, dtype=np.int64), uniques


class _GroupByBase:
    def __init__(self, source, by, *, as_index=True, sort=True, dropna=True):
        if callable(by) or isinstance(by, dict):
            raise NotImplementedError("callable and mapping groupers are outside the covered subset")
        if len(by) != len(source):
            raise ValueError("Grouper and axis must be same length")
        self.source = source
        self.codes, self.uniques = _factorize(by, sort=sort, dropna=dropna)
        self.as_index = as_index
        self.key_name = getattr(by, "name", None)

    def _matrix(self):
        raise NotImplementedError

    def _wrap(self, values, op):
        raise NotImplementedError

    def _reduce(self, op, *, ddof=1, min_count=0):
        matrix = self._matrix()
        reduced, _ = _kernels.groupby_reduce(
            self.codes, matrix, len(self.uniques), op, ddof=ddof, min_count=min_count
        )
        return self._wrap(reduced, op)

    def sum(self, numeric_only=False, min_count=0, engine=None, engine_kwargs=None):
        return self._reduce("sum", min_count=min_count)

    def mean(self, numeric_only=False, engine=None, engine_kwargs=None):
        return self._reduce("mean")

    def min(self, numeric_only=False, min_count=-1, engine=None, engine_kwargs=None):
        return self._reduce("min", min_count=min_count)

    def max(self, numeric_only=False, min_count=-1, engine=None, engine_kwargs=None):
        return self._reduce("max", min_count=min_count)

    def count(self):
        return self._reduce("count")

    def prod(self, numeric_only=False, min_count=0):
        return self._reduce("prod", min_count=min_count)

    def var(self, ddof=1, engine=None, engine_kwargs=None, numeric_only=False):
        return self._reduce("var", ddof=ddof)

    def std(self, ddof=1, engine=None, engine_kwargs=None, numeric_only=False):
        return self._reduce("std", ddof=ddof)

    def size(self):
        ones = np.ones((len(self.codes), 1), dtype=np.float64)
        values, _ = _kernels.groupby_reduce(self.codes, ones, len(self.uniques), "sum")
        result = pd.Series(values[:, 0].astype(np.int64), index=self.uniques)
        if self.as_index:
            return result
        return DataFrame({self.key_name or "index": self.uniques, "size": result.to_numpy()})

    def agg(self, func=None, *args, **kwargs):
        if isinstance(func, str):
            if func not in _REDUCTIONS:
                raise NotImplementedError(f"aggregation {func!r} is outside the covered subset")
            return getattr(self, func)(*args, **kwargs)
        if isinstance(func, Sequence) and not isinstance(func, (str, bytes)):
            names = list(func)
            if not names or any(name not in _REDUCTIONS for name in names):
                raise NotImplementedError("only lists of named covered reductions are supported")
            return self._aggregate_list(names)
        raise NotImplementedError("callable and named aggregations are outside the covered subset")

    aggregate = agg

    def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        if not isinstance(func, str) or func not in _REDUCTIONS - {"count"}:
            raise NotImplementedError("transform supports named numeric reductions")
        matrix = self._matrix()
        reduced, _ = _kernels.groupby_reduce(
            self.codes,
            matrix,
            len(self.uniques),
            func,
            ddof=kwargs.get("ddof", 1),
            min_count=kwargs.get("min_count", 0),
        )
        transformed = np.full(matrix.shape, np.nan)
        keep = self.codes >= 0
        transformed[keep] = reduced[self.codes[keep]]
        return self._wrap_transform(transformed)


class SeriesGroupBy(_GroupByBase):
    def _matrix(self):
        if self.source.dtype != np.dtype("float64"):
            raise TypeError("Mojo groupby reductions require float64 values")
        return np.asarray(self.source.to_numpy(), dtype=np.float64)[:, None]

    def _wrap(self, values, op):
        data = values[:, 0]
        if op == "count":
            data = data.astype(np.int64)
        result = Series(data, index=self.uniques, name=self.source.name)
        if self.as_index:
            return result
        return DataFrame(
            {self.key_name or "index": self.uniques.to_numpy(), self.source.name or 0: data}
        )

    def _aggregate_list(self, names):
        columns = {name: np.asarray(getattr(self, name)()) for name in names}
        return DataFrame(columns, index=self.uniques)

    def _wrap_transform(self, values):
        return Series(values[:, 0], index=self.source.index, name=self.source.name)


class DataFrameGroupBy(_GroupByBase):
    def __init__(self, source, by, *, key_column=None, **kwargs):
        self.key_column = key_column
        self.columns = [column for column in source.columns if column != key_column]
        super().__init__(source, by, **kwargs)
        if key_column is not None:
            self.key_name = key_column
            self.uniques = self.uniques.rename(key_column)

    def __getitem__(self, key):
        if isinstance(key, (list, tuple, pd.Index)):
            selected = DataFrame(self.source.loc[:, list(key)])
            grouped = DataFrameGroupBy.__new__(DataFrameGroupBy)
            grouped.source = selected
            grouped.codes = self.codes
            grouped.uniques = self.uniques
            grouped.as_index = self.as_index
            grouped.key_name = self.key_name
            grouped.key_column = None
            grouped.columns = list(key)
            return grouped
        grouped = SeriesGroupBy.__new__(SeriesGroupBy)
        grouped.source = Series(self.source[key])
        grouped.codes = self.codes
        grouped.uniques = self.uniques
        grouped.as_index = self.as_index
        grouped.key_name = self.key_name
        return grouped

    def _matrix(self):
        if not self.columns:
            raise ValueError("no value columns to aggregate")
        selected = self.source.loc[:, self.columns]
        if any(dtype != np.dtype("float64") for dtype in selected.dtypes):
            raise TypeError("Mojo groupby reductions require float64 value columns")
        return np.asfortranarray(selected.to_numpy(), dtype=np.float64)

    def _wrap(self, values, op):
        if op == "count":
            values = values.astype(np.int64)
        result = DataFrame(values, index=self.uniques, columns=self.columns)
        if self.as_index:
            return result
        result = result.reset_index()
        if result.columns[0] != (self.key_name or "index"):
            result = result.rename(columns={result.columns[0]: self.key_name or "index"})
        return DataFrame(result)

    def _aggregate_list(self, names):
        pieces = [pd.DataFrame(getattr(self, name)()) for name in names]
        combined = pd.concat(pieces, axis=1, keys=names)
        combined = combined.swaplevel(0, 1, axis=1)
        combined = combined.reindex(
            columns=pd.MultiIndex.from_product([self.columns, names])
        )
        return DataFrame(combined)

    def _wrap_transform(self, values):
        return DataFrame(values, index=self.source.index, columns=self.columns)


class MojoRolling:
    def __init__(
        self,
        source,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        closed=None,
        step=None,
        method="single",
    ):
        if not isinstance(window, (int, np.integer)) or window <= 0:
            raise NotImplementedError("Mojo rolling supports positive fixed-size integer windows")
        if center or win_type is not None or on is not None or closed is not None or method != "single":
            raise NotImplementedError("centered, weighted, labelled, and custom-bound rolling is uncovered")
        if min_periods is None:
            min_periods = int(window)
        if min_periods < 0 or min_periods > window:
            raise ValueError("min_periods must satisfy 0 <= min_periods <= window")
        if step is not None and (not isinstance(step, int) or step <= 0):
            raise ValueError("step must be a positive integer")
        self.source = source
        self.window = int(window)
        self.min_periods = int(min_periods)
        self.step = step

    def _reduce(self, op, ddof=1):
        values = np.asarray(self.source)
        if values.dtype != np.dtype("float64"):
            raise TypeError("Mojo rolling reductions require float64 values")
        result = _kernels.rolling_reduce(
            values, self.window, self.min_periods, op, ddof=ddof
        )
        wrapped = Series(result, index=self.source.index, name=self.source.name)
        return wrapped if self.step is None else wrapped.iloc[:: self.step]

    def sum(self, numeric_only=False, engine=None, engine_kwargs=None):
        return self._reduce("sum")

    def mean(self, numeric_only=False, engine=None, engine_kwargs=None):
        return self._reduce("mean")

    def min(self, numeric_only=False, engine=None, engine_kwargs=None):
        return self._reduce("min")

    def max(self, numeric_only=False, engine=None, engine_kwargs=None):
        return self._reduce("max")

    def count(self, numeric_only=False):
        return self._reduce("count")

    def var(self, ddof=1, numeric_only=False, engine=None, engine_kwargs=None):
        return self._reduce("var", ddof=ddof)

    def std(self, ddof=1, numeric_only=False, engine=None, engine_kwargs=None):
        return self._reduce("std", ddof=ddof)

    def agg(self, func, *args, **kwargs):
        if isinstance(func, str) and func in _REDUCTIONS - {"prod"}:
            return getattr(self, func)(*args, **kwargs)
        if isinstance(func, Sequence) and not isinstance(func, (str, bytes)):
            return DataFrame(
                {name: np.asarray(getattr(self, name)(*args, **kwargs)) for name in func},
                index=getattr(self, func[0])(*args, **kwargs).index,
            )
        raise NotImplementedError("rolling agg supports covered reduction names")

    aggregate = agg


class MojoResampler:
    def __init__(
        self,
        source,
        rule,
        closed=None,
        label=None,
        convention="start",
        on=None,
        level=None,
        origin="start_day",
        offset=None,
        group_keys=False,
    ):
        if on is not None or level is not None:
            raise NotImplementedError("Series resampling on a column or level is outside the subset")
        if convention != "start":
            raise NotImplementedError("only convention='start' is covered")
        if not isinstance(source.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
            raise TypeError("resampling requires a datetime-like index")
        if not source.index.is_monotonic_increasing:
            raise NotImplementedError("Mojo resample currently requires a sorted index")
        if source.dtype != np.dtype("float64"):
            raise TypeError("Mojo resample reductions require float64 values")
        self.source = source
        base = pd.Series(source.to_numpy(), index=source.index, name=source.name)
        pandas_resampler = base.resample(
            rule,
            closed=closed,
            label=label,
            convention=convention,
            origin=origin,
            offset=offset,
            group_keys=group_keys,
        )
        self.codes = np.asarray(pandas_resampler._grouper.ids, dtype=np.int64)
        self.result_index = pandas_resampler._grouper.result_index

    def _reduce(self, op, *, ddof=1, min_count=0):
        values = np.asarray(self.source.to_numpy(), dtype=np.float64)[:, None]
        result, _ = _kernels.groupby_reduce(
            self.codes,
            values,
            len(self.result_index),
            op,
            ddof=ddof,
            min_count=min_count,
        )
        data = result[:, 0]
        if op == "count":
            data = data.astype(np.int64)
        return Series(data, index=self.result_index, name=self.source.name)

    def sum(self, numeric_only=False, min_count=0):
        return self._reduce("sum", min_count=min_count)

    def mean(self, numeric_only=False):
        return self._reduce("mean")

    def min(self, numeric_only=False, min_count=1):
        return self._reduce("min", min_count=min_count)

    def max(self, numeric_only=False, min_count=1):
        return self._reduce("max", min_count=min_count)

    def count(self):
        return self._reduce("count")

    def prod(self, numeric_only=False, min_count=0):
        return self._reduce("prod", min_count=min_count)

    def var(self, ddof=1, numeric_only=False):
        return self._reduce("var", ddof=ddof)

    def std(self, ddof=1, numeric_only=False):
        return self._reduce("std", ddof=ddof)

    def agg(self, func=None, *args, **kwargs):
        if isinstance(func, str) and func in _REDUCTIONS:
            return getattr(self, func)(*args, **kwargs)
        if isinstance(func, Sequence) and not isinstance(func, (str, bytes)):
            return DataFrame(
                {name: np.asarray(getattr(self, name)(*args, **kwargs)) for name in func},
                index=self.result_index,
            )
        raise NotImplementedError("resample agg supports covered reduction names")

    aggregate = agg


class Series(pd.Series):
    @property
    def _constructor(self):
        return Series

    @property
    def _constructor_expanddim(self):
        return DataFrame

    def groupby(
        self,
        by=None,
        level=None,
        *,
        as_index=True,
        sort=True,
        group_keys=True,
        observed=True,
        dropna=True,
    ):
        if by is None or level is not None:
            raise NotImplementedError("Mojo Series.groupby covers an explicit row-aligned grouper")
        return SeriesGroupBy(
            self, by, as_index=as_index, sort=sort, dropna=dropna
        )

    def rolling(
        self,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        closed=None,
        step=None,
        method="single",
    ):
        return MojoRolling(
            self, window, min_periods, center, win_type, on, closed, step, method
        )

    def resample(
        self,
        rule,
        closed=None,
        label=None,
        convention="start",
        on=None,
        level=None,
        origin="start_day",
        offset=None,
        group_keys=False,
    ):
        return MojoResampler(
            self, rule, closed, label, convention, on, level, origin, offset, group_keys
        )

    def sort_values(
        self,
        *,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
        key=None,
    ):
        if axis not in (0, "index"):
            raise ValueError("Series axis must be 0 or 'index'")
        if na_position not in {"first", "last"}:
            raise ValueError("invalid na_position")
        sort_key = self if key is None else key(self)
        idx = _kernels.argsort(
            np.asarray(sort_key), ascending=bool(ascending), na_position=na_position
        )
        result = self.iloc[idx]
        if ignore_index:
            result.index = pd.RangeIndex(len(result))
        if inplace:
            self._update_inplace(result)
            return None
        return result


class DataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return DataFrame

    @property
    def _constructor_sliced(self):
        return Series

    def groupby(
        self,
        by=None,
        level=None,
        *,
        as_index=True,
        sort=True,
        group_keys=True,
        observed=True,
        dropna=True,
    ):
        if level is not None:
            raise NotImplementedError("index-level grouping is outside the covered subset")
        if isinstance(by, str):
            if by not in self.columns:
                raise KeyError(by)
            keys = self[by]
            return DataFrameGroupBy(
                self,
                keys,
                key_column=by,
                as_index=as_index,
                sort=sort,
                dropna=dropna,
            )
        if by is None or isinstance(by, (list, tuple)) and all(x in self.columns for x in by):
            raise NotImplementedError("Mojo DataFrame.groupby covers one column or row-aligned grouper")
        return DataFrameGroupBy(
            self, by, as_index=as_index, sort=sort, dropna=dropna
        )

    def sort_values(
        self,
        by,
        *,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
        key=None,
    ):
        if axis not in (0, "index"):
            raise NotImplementedError("column-axis sorting is outside the covered subset")
        if isinstance(by, (list, tuple)):
            if len(by) != 1:
                raise NotImplementedError("Mojo sort_values covers one sort key")
            by = by[0]
        if not isinstance(ascending, (bool, np.bool_)):
            if len(ascending) != 1:
                raise ValueError("Length of ascending does not match length of by")
            ascending = ascending[0]
        sort_key = self[by]
        if key is not None:
            sort_key = key(sort_key)
        idx = _kernels.argsort(
            np.asarray(sort_key), ascending=bool(ascending), na_position=na_position
        )
        result = self.iloc[idx]
        if ignore_index:
            result.index = pd.RangeIndex(len(result))
        if inplace:
            self._update_inplace(result)
            return None
        return result

    def merge(
        self,
        right,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("_x", "_y"),
        copy=None,
        indicator=False,
        validate=None,
    ):
        return merge(
            self,
            right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            copy=copy,
            indicator=indicator,
            validate=validate,
        )


def _validate_merge(left, right, key, validate):
    if validate is None or validate in {"many_to_many", "m:m"}:
        return
    allowed = {"one_to_one", "1:1", "one_to_many", "1:m", "many_to_one", "m:1"}
    if validate not in allowed:
        raise ValueError(f"{validate!r} is not a valid argument for validate")
    left_unique = not left[key].duplicated().any()
    right_unique = not right[key].duplicated().any()
    if validate in {"one_to_one", "1:1"} and not (left_unique and right_unique):
        raise pd.errors.MergeError("Merge keys are not unique in either dataset")
    if validate in {"one_to_many", "1:m"} and not left_unique:
        raise pd.errors.MergeError("Merge keys are not unique in left dataset")
    if validate in {"many_to_one", "m:1"} and not right_unique:
        raise pd.errors.MergeError("Merge keys are not unique in right dataset")


def _take_column(series, indices):
    values = series.to_numpy()
    missing = indices < 0
    if not missing.any():
        return values[indices]
    if pd.api.types.is_numeric_dtype(values.dtype):
        result = np.full(len(indices), np.nan, dtype=np.result_type(values.dtype, np.float64))
    else:
        result = np.full(len(indices), np.nan, dtype=object)
    keep = ~missing
    result[keep] = values[indices[keep]]
    return result


def merge(
    left,
    right,
    how="inner",
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    sort=False,
    suffixes=("_x", "_y"),
    copy=None,
    indicator=False,
    validate=None,
):
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        raise TypeError("Mojo merge requires DataFrame operands")
    if how not in {"inner", "left"}:
        raise NotImplementedError("Mojo hash join currently covers how='inner' and how='left'")
    if left_on is not None or right_on is not None or left_index or right_index:
        raise NotImplementedError("Mojo hash join covers a shared named column via on=")
    if on is None:
        common = left.columns.intersection(right.columns)
        if len(common) != 1:
            raise NotImplementedError("implicit merge requires exactly one shared column")
        on = common[0]
    if isinstance(on, (list, tuple)):
        if len(on) != 1:
            raise NotImplementedError("Mojo hash join covers one key column")
        on = on[0]
    if on not in left or on not in right:
        raise KeyError(on)
    left_keys = left[on].to_numpy()
    right_keys = right[on].to_numpy()
    if left_keys.dtype != np.dtype("int64") or right_keys.dtype != np.dtype("int64"):
        raise TypeError("Mojo hash join requires non-null int64 key columns")
    _validate_merge(left, right, on, validate)
    left_idx, right_idx = _kernels.join_indexers(left_keys, right_keys, how)
    left_columns = [column for column in left.columns if column != on]
    right_columns = [column for column in right.columns if column != on]
    overlap = set(left_columns).intersection(right_columns)
    if overlap and suffixes[0] is None and suffixes[1] is None:
        raise ValueError(f"columns overlap but no suffix specified: {sorted(overlap)!r}")
    data = {on: left[on].to_numpy()[left_idx]}
    for column in left_columns:
        name = f"{column}{suffixes[0]}" if column in overlap and suffixes[0] is not None else column
        data[name] = left[column].to_numpy()[left_idx]
    for column in right_columns:
        name = f"{column}{suffixes[1]}" if column in overlap and suffixes[1] is not None else column
        if name in data:
            raise ValueError(f"duplicate output column {name!r}")
        data[name] = _take_column(right[column], right_idx)
    if indicator:
        name = "_merge" if indicator is True else indicator
        labels = np.where(right_idx < 0, "left_only", "both")
        data[name] = pd.Categorical(labels, categories=["left_only", "right_only", "both"])
    result = DataFrame(data)
    if sort:
        result = DataFrame(
            pd.DataFrame(result).sort_values(on, kind="stable", ignore_index=True)
        )
    return result
