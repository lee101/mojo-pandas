"""Columnar pandas kernels exposed through a small C ABI."""

from max.algorithm import parallelize
from std.math import iota, isnan, sqrt
from std.memory import stack_allocation
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime UPtr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime GROUPBY_PARALLEL_THRESHOLD = 262_144
comptime SORT_RADIX_THRESHOLD = 2_048


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def up(addr: Int) -> UPtr:
    return UPtr(unsafe_from_address=addr)


def groupby_reduce(
    codes: IPtr,
    values: FPtr,
    dst: FPtr,
    counts: IPtr,
    n: Int,
    ncols: Int,
    ngroups: Int,
    op: Int,
    ddof: Int,
    min_count: Int,
):
    var size = ngroups * ncols
    for k in range(size):
        counts[k] = 0
        if op == 0 or op == 1 or op == 4 or op == 7:
            dst[k] = 1.0 if op == 7 else 0.0

    for j in range(ncols):
        for i in range(n):
            var g = Int(codes[i])
            if g < 0 or g >= ngroups:
                continue
            var v = values[j * n + i]
            if isnan(v):
                continue
            var k = j * ngroups + g
            if op == 2:
                if counts[k] == 0 or v < dst[k]:
                    dst[k] = v
            elif op == 3:
                if counts[k] == 0 or v > dst[k]:
                    dst[k] = v
            elif op == 7:
                dst[k] *= v
            else:
                dst[k] += v
            counts[k] += 1

    if op == 1:
        for k in range(size):
            if counts[k] > 0:
                dst[k] /= Float64(counts[k])
            else:
                dst[k] = 0.0 / Float64(0)
    elif op == 4:
        for k in range(size):
            dst[k] = Float64(counts[k])
    if op == 0 or op == 2 or op == 3 or op == 7:
        for k in range(size):
            if Int(counts[k]) < min_count:
                dst[k] = 0.0 / Float64(0)


def groupby_var_column(
    codes: IPtr,
    values: FPtr,
    dst: FPtr,
    means: FPtr,
    counts: IPtr,
    n: Int,
    ncols: Int,
    ngroups: Int,
    j: Int,
):
    for i in range(n):
        var g = Int(codes[i])
        if g < 0 or g >= ngroups:
            continue
        var v = values[j * n + i]
        if not isnan(v):
            var k = j * ngroups + g
            means[k] += v
            counts[k] += 1

    for g in range(ngroups):
        var k = j * ngroups + g
        if counts[k] > 0:
            means[k] /= Float64(counts[k])

    for i in range(n):
        var g = Int(codes[i])
        if g < 0 or g >= ngroups:
            continue
        var v = values[j * n + i]
        if not isnan(v):
            var k = j * ngroups + g
            var d = v - means[k]
            dst[k] += d * d


def groupby_var(
    codes: IPtr,
    values: FPtr,
    dst: FPtr,
    means: FPtr,
    counts: IPtr,
    n: Int,
    ncols: Int,
    ngroups: Int,
    ddof: Int,
    take_sqrt: Bool,
):
    var size = ngroups * ncols
    comptime W = simd_width_of[DType.float64]()
    var k = 0
    var zeros = SIMD[DType.float64, W](0.0)
    while k + W <= size:
        means.store(k, zeros)
        dst.store(k, zeros)
        k += W
    while k < size:
        means[k] = 0.0
        dst[k] = 0.0
        k += 1
    for k in range(size):
        counts[k] = 0

    @parameter
    def process_column(j: Int):
        groupby_var_column(
            codes, values, dst, means, counts, n, ncols, ngroups, j
        )

    if n >= GROUPBY_PARALLEL_THRESHOLD and ncols > 1:
        parallelize[process_column](ncols, min(ncols, 4))
    else:
        for j in range(ncols):
            groupby_var_column(
                codes, values, dst, means, counts, n, ncols, ngroups, j
            )

    for k in range(size):
        if Int(counts[k]) > ddof:
            dst[k] /= Float64(Int(counts[k]) - ddof)
            if take_sqrt:
                dst[k] = sqrt(dst[k])
        else:
            dst[k] = 0.0 / Float64(0)


@always_inline
def f64_sort_key(value: Float64, ascending: Bool, nan_first: Bool) -> UInt64:
    if isnan(value):
        return UInt64(0) if nan_first else ~UInt64(0)
    var bits: UInt64 = value.to_bits[DType.uint64]()
    if value == 0.0:
        bits = 0
    var sign = UInt64(1) << 63
    var key = ~bits if (bits & sign) != 0 else bits ^ sign
    return key if ascending else ~key


def materialize_f64_sort_keys(
    values: FPtr,
    keys: UPtr,
    n: Int,
    ascending: Bool,
    nan_first: Bool,
):
    comptime W = simd_width_of[DType.float64]()
    comptime SIGN = UInt64(0x8000000000000000)
    comptime EXPONENT = UInt64(0x7ff0000000000000)
    comptime MANTISSA = UInt64(0x000fffffffffffff)
    var words = values.bitcast[UInt64]()
    var sign = SIMD[DType.uint64, W](SIGN)
    var exponent = SIMD[DType.uint64, W](EXPONENT)
    var mantissa = SIMD[DType.uint64, W](MANTISSA)
    var nan_key = SIMD[DType.uint64, W](
        UInt64(0) if nan_first else ~UInt64(0)
    )
    var i = 0
    while i + W <= n:
        var bits = words.load[width=W](i)
        var negative = (bits & sign).ne(0)
        var normalized = negative.select(~bits, bits ^ sign)
        normalized = ((bits & ~sign).eq(0)).select(sign, normalized)
        if not ascending:
            normalized = ~normalized
        var nan = ((bits & exponent).eq(exponent)) & (
            (bits & mantissa).ne(0)
        )
        keys.store(i, nan.select(nan_key, normalized))
        i += W
    while i < n:
        keys[i] = f64_sort_key(values[i], ascending, nan_first)
        i += 1


def radix_argsort_f64(keys: UPtr, idx: IPtr, work: IPtr, n: Int):
    comptime RADIX_SIZE = 256
    var counts = stack_allocation[RADIX_SIZE, Int64]()
    var source = idx
    var destination = work
    for radix_pass in range(8):
        for bucket in range(RADIX_SIZE):
            counts[bucket] = 0
        var shift = UInt64(radix_pass * 8)
        for i in range(n):
            var bucket = Int((keys[Int(source[i])] >> shift) & UInt64(255))
            counts[bucket] += 1
        var offset = 0
        for bucket in range(RADIX_SIZE):
            var size = Int(counts[bucket])
            counts[bucket] = Int64(offset)
            offset += size
        for i in range(n):
            var index = source[i]
            var bucket = Int((keys[Int(index)] >> shift) & UInt64(255))
            destination[counts[bucket]] = index
            counts[bucket] += 1
        var temporary = source
        source = destination
        destination = temporary


def merge_f64_range(
    values: FPtr,
    src: IPtr,
    dst: IPtr,
    n: Int,
    width: Int,
    ascending: Bool,
    nan_first: Bool,
):
    var start = 0
    while start < n:
        var mid = min(start + width, n)
        var end = min(start + width + width, n)
        var left = start
        var right = mid
        var pos = start
        while left < mid and right < end:
            var a = f64_sort_key(
                values[Int(src[left])], ascending, nan_first
            )
            var b = f64_sort_key(
                values[Int(src[right])], ascending, nan_first
            )
            if a <= b:
                dst[pos] = src[left]
                left += 1
            else:
                dst[pos] = src[right]
                right += 1
            pos += 1
        while left < mid:
            dst[pos] = src[left]
            left += 1
            pos += 1
        while right < end:
            dst[pos] = src[right]
            right += 1
            pos += 1
        start += width + width


def argsort_f64(
    values: FPtr,
    idx: IPtr,
    work: IPtr,
    keys: UPtr,
    n: Int,
    ascending: Bool,
    nan_first: Bool,
):
    comptime W = simd_width_of[DType.int64]()
    var i = 0
    while i + W <= n:
        idx.store(i, iota[DType.int64, W](Int64(i)))
        i += W
    while i < n:
        idx[i] = Int64(i)
        i += 1
    if n >= SORT_RADIX_THRESHOLD:
        materialize_f64_sort_keys(values, keys, n, ascending, nan_first)
        radix_argsort_f64(keys, idx, work, n)
        return
    var src = idx
    var dst_ptr = work
    var sorted_in_idx = True
    var width = 1
    while width < n:
        merge_f64_range(values, src, dst_ptr, n, width, ascending, nan_first)
        var tmp = src
        src = dst_ptr
        dst_ptr = tmp
        sorted_in_idx = not sorted_in_idx
        width *= 2
    if not sorted_in_idx:
        i = 0
        while i + W <= n:
            idx.store(i, src.load[width=W](i))
            i += W
        while i < n:
            idx[i] = src[i]
            i += 1


def merge_i64_range(
    values: IPtr,
    src: IPtr,
    dst: IPtr,
    n: Int,
    width: Int,
    range_start: Int,
    range_end: Int,
    ascending: Bool,
):
    var start = range_start
    while start < range_end:
        var mid = min(start + width, n)
        var end = min(start + width + width, n)
        var left = start
        var right = mid
        var pos = start
        while left < mid and right < end:
            var a = values[Int(src[left])]
            var b = values[Int(src[right])]
            var before = a <= b if ascending else a >= b
            if before:
                dst[pos] = src[left]
                left += 1
            else:
                dst[pos] = src[right]
                right += 1
            pos += 1
        while left < mid:
            dst[pos] = src[left]
            left += 1
            pos += 1
        while right < end:
            dst[pos] = src[right]
            right += 1
            pos += 1
        start += width + width


def argsort_i64(values: IPtr, idx: IPtr, work: IPtr, n: Int, ascending: Bool):
    comptime W = simd_width_of[DType.int64]()
    var i = 0
    while i + W <= n:
        idx.store(i, iota[DType.int64, W](Int64(i)))
        i += W
    while i < n:
        idx[i] = Int64(i)
        i += 1
    var src = idx
    var dst_ptr = work
    var sorted_in_idx = True
    var width = 1
    while width < n:
        merge_i64_range(values, src, dst_ptr, n, width, 0, n, ascending)
        var tmp = src
        src = dst_ptr
        dst_ptr = tmp
        sorted_in_idx = not sorted_in_idx
        width *= 2
    if not sorted_in_idx:
        i = 0
        while i + W <= n:
            idx.store(i, src.load[width=W](i))
            i += W
        while i < n:
            idx[i] = src[i]
            i += 1


def rolling_reduce(
    values: FPtr,
    dst: FPtr,
    queue: IPtr,
    n: Int,
    window: Int,
    min_periods: Int,
    op: Int,
    ddof: Int,
):
    var total = 0.0
    var total2 = 0.0
    var moment_mean = 0.0
    var moment2 = 0.0
    var valid = 0
    var head = 0
    var tail = 0
    for i in range(n):
        var v = values[i]
        if not isnan(v):
            total += v
            total2 += v * v
            valid += 1
            var delta = v - moment_mean
            moment_mean += delta / Float64(valid)
            moment2 += delta * (v - moment_mean)
        if i >= window:
            var old = values[i - window]
            if not isnan(old):
                total -= old
                total2 -= old * old
                if valid == 1:
                    moment_mean = 0.0
                    moment2 = 0.0
                else:
                    var new_count = valid - 1
                    var new_mean = (
                        Float64(valid) * moment_mean - old
                    ) / Float64(new_count)
                    moment2 -= (old - moment_mean) * (old - new_mean)
                    moment_mean = new_mean
                valid -= 1

        if op == 2 or op == 3:
            var first = i - window + 1
            while head < tail and Int(queue[head]) < first:
                head += 1
            if not isnan(v):
                while head < tail:
                    var qv = values[Int(queue[tail - 1])]
                    var remove = qv >= v if op == 2 else qv <= v
                    if not remove:
                        break
                    tail -= 1
                queue[tail] = Int64(i)
                tail += 1

        if op == 6:
            if min(i + 1, window) >= min_periods:
                dst[i] = Float64(valid)
            continue
        if valid < min_periods:
            continue
        if op == 0:
            dst[i] = total
        elif op == 1:
            if valid > 0:
                dst[i] = moment_mean
        elif op == 2 or op == 3:
            if head < tail:
                dst[i] = values[Int(queue[head])]
        elif op == 4 or op == 5:
            if valid > ddof:
                var variance = moment2
                if variance < 0.0:
                    variance = 0.0
                variance /= Float64(valid - ddof)
                dst[i] = sqrt(variance) if op == 5 else variance


def hash_slot(key: Int64, used: IPtr, slot_keys: IPtr, capacity: Int) -> Int:
    var h = Int(key) % capacity
    if h < 0:
        h += capacity
    while used[h] != 0 and slot_keys[h] != key:
        h += 1
        if h == capacity:
            h = 0
    return h


def join_build(
    right_keys: IPtr,
    used: IPtr,
    slot_keys: IPtr,
    heads: IPtr,
    tails: IPtr,
    next_rows: IPtr,
    nr: Int,
    capacity: Int,
):
    for s in range(capacity):
        used[s] = 0
        heads[s] = -1
        tails[s] = -1
    for j in range(nr):
        next_rows[j] = -1
        var key = right_keys[j]
        var s = hash_slot(key, used, slot_keys, capacity)
        if used[s] == 0:
            used[s] = 1
            slot_keys[s] = key
            heads[s] = Int64(j)
            tails[s] = Int64(j)
        else:
            next_rows[Int(tails[s])] = Int64(j)
            tails[s] = Int64(j)


def join_count(
    left_keys: IPtr,
    right_keys: IPtr,
    used: IPtr,
    slot_keys: IPtr,
    heads: IPtr,
    tails: IPtr,
    next_rows: IPtr,
    nl: Int,
    nr: Int,
    capacity: Int,
    left_join: Bool,
) -> Int:
    join_build(right_keys, used, slot_keys, heads, tails, next_rows, nr, capacity)
    var total = 0
    for i in range(nl):
        var key = left_keys[i]
        var s = hash_slot(key, used, slot_keys, capacity)
        if used[s] == 0:
            if left_join:
                total += 1
        else:
            var j = Int(heads[s])
            while j >= 0:
                total += 1
                j = Int(next_rows[j])
    return total


def join_fill(
    left_keys: IPtr,
    right_keys: IPtr,
    left_idx: IPtr,
    right_idx: IPtr,
    used: IPtr,
    slot_keys: IPtr,
    heads: IPtr,
    tails: IPtr,
    next_rows: IPtr,
    nl: Int,
    nr: Int,
    capacity: Int,
    left_join: Bool,
) -> Int:
    join_build(right_keys, used, slot_keys, heads, tails, next_rows, nr, capacity)
    var written = 0
    for i in range(nl):
        var key = left_keys[i]
        var s = hash_slot(key, used, slot_keys, capacity)
        if used[s] == 0:
            if left_join:
                left_idx[written] = Int64(i)
                right_idx[written] = -1
                written += 1
        else:
            var j = Int(heads[s])
            while j >= 0:
                left_idx[written] = Int64(i)
                right_idx[written] = Int64(j)
                written += 1
                j = Int(next_rows[j])
    return written


@export("mp_groupby_reduce")
def mp_groupby_reduce(
    codes: Int,
    values: Int,
    dst: Int,
    counts: Int,
    n: Int,
    ncols: Int,
    ngroups: Int,
    op: Int,
    ddof: Int,
    min_count: Int,
) abi("C"):
    groupby_reduce(
        ip(codes), fp(values), fp(dst), ip(counts),
        n, ncols, ngroups, op, ddof, min_count,
    )


@export("mp_groupby_var")
def mp_groupby_var(
    codes: Int,
    values: Int,
    dst: Int,
    means: Int,
    counts: Int,
    n: Int,
    ncols: Int,
    ngroups: Int,
    ddof: Int,
    take_sqrt: Int,
) abi("C"):
    groupby_var(
        ip(codes), fp(values), fp(dst), fp(means), ip(counts),
        n, ncols, ngroups, ddof, take_sqrt != 0,
    )


@export("mp_argsort_f64")
def mp_argsort_f64(
    values: Int,
    idx: Int,
    work: Int,
    keys: Int,
    n: Int,
    ascending: Int,
    nan_first: Int,
) abi("C"):
    argsort_f64(
        fp(values), ip(idx), ip(work), up(keys), n,
        ascending != 0, nan_first != 0,
    )


@export("mp_argsort_i64")
def mp_argsort_i64(
    values: Int, idx: Int, work: Int, n: Int, ascending: Int
) abi("C"):
    argsort_i64(ip(values), ip(idx), ip(work), n, ascending != 0)


@export("mp_rolling_reduce")
def mp_rolling_reduce(
    values: Int,
    dst: Int,
    queue: Int,
    n: Int,
    window: Int,
    min_periods: Int,
    op: Int,
    ddof: Int,
) abi("C"):
    rolling_reduce(fp(values), fp(dst), ip(queue), n, window, min_periods, op, ddof)


@export("mp_join_count")
def mp_join_count(
    left_keys: Int,
    right_keys: Int,
    used: Int,
    slot_keys: Int,
    heads: Int,
    tails: Int,
    next_rows: Int,
    nl: Int,
    nr: Int,
    capacity: Int,
    left_join: Int,
) abi("C") -> Int:
    return join_count(
        ip(left_keys), ip(right_keys), ip(used), ip(slot_keys), ip(heads),
        ip(tails), ip(next_rows), nl, nr, capacity, left_join != 0,
    )


@export("mp_join_fill")
def mp_join_fill(
    left_keys: Int,
    right_keys: Int,
    left_idx: Int,
    right_idx: Int,
    used: Int,
    slot_keys: Int,
    heads: Int,
    tails: Int,
    next_rows: Int,
    nl: Int,
    nr: Int,
    capacity: Int,
    left_join: Int,
) abi("C") -> Int:
    return join_fill(
        ip(left_keys), ip(right_keys), ip(left_idx), ip(right_idx),
        ip(used), ip(slot_keys), ip(heads), ip(tails), ip(next_rows),
        nl, nr, capacity, left_join != 0,
    )
