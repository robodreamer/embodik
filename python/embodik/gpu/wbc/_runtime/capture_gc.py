"""Defer automatic cyclic collection while CUDA streams are capturing."""

from __future__ import annotations

import gc
import threading
from contextlib import contextmanager
from typing import Iterator

_lock = threading.Lock()
_depth = 0
_restore_enabled = False


@contextmanager
def suspend_capture_gc() -> Iterator[None]:
    """Avoid unrelated Warp buffer finalizers on a legacy stream mid-capture.

    Nested/concurrent users preserve the original GC policy. Explicit
    ``gc.collect()`` is unaffected; callers must run it outside active capture.
    This guard is for graph creation only, not ordinary graph replay.
    """
    global _depth, _restore_enabled
    with _lock:
        if _depth == 0:
            _restore_enabled = gc.isenabled()
            if _restore_enabled:
                gc.disable()
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0 and _restore_enabled:
                gc.enable()
