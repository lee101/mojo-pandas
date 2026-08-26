"""ctypes loader for the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-pandas.so")
I = ctypes.c_int64

_SIGNATURES = {
    "mp_groupby_reduce": ([I] * 10, None),
    "mp_groupby_var": ([I] * 10, None),
    "mp_argsort_f64": ([I] * 7, None),
    "mp_argsort_i64": ([I] * 5, None),
    "mp_rolling_reduce": ([I] * 8, None),
    "mp_join_count": ([I] * 11, I),
    "mp_join_fill": ([I] * 13, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "kernels.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _loaded


def addr(array) -> int:
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("cannot pass a null array pointer to Mojo")
    return address
