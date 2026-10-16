"""Capture-only GC policy is restored, including nesting and exceptions."""

import gc

import pytest

pytest.importorskip("torch")

from embodik.gpu.wbc._runtime.capture_gc import suspend_capture_gc


def test_nested_capture_guard_restores_original_gc_policy():
    original = gc.isenabled()
    with suspend_capture_gc():
        assert not gc.isenabled()
        with suspend_capture_gc():
            assert not gc.isenabled()
        assert not gc.isenabled()
    assert gc.isenabled() == original


def test_capture_exception_restores_policy():
    original = gc.isenabled()
    with pytest.raises(RuntimeError):
        with suspend_capture_gc():
            raise RuntimeError("capture failed")
    assert gc.isenabled() == original


def test_guard_preserves_an_explicitly_disabled_collector():
    original = gc.isenabled()
    gc.disable()
    try:
        with suspend_capture_gc():
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        if original:
            gc.enable()
