"""CPU-only regression for interleaved Warp device-stream scopes."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

pytest.importorskip("torch")

from embodik.gpu.wbc._runtime.newton_com import _scoped_com_stream


class FakeWarp:
    current = "original"

    class ScopedStream:
        def __init__(self, stream, *, sync_enter):
            assert not sync_enter
            self.stream = stream

        def __enter__(self):
            self.saved = FakeWarp.current
            FakeWarp.current = self.stream

        def __exit__(self, *args):
            FakeWarp.current = self.saved


def test_com_stream_scopes_do_not_interleave_or_leave_a_transient_default():
    FakeWarp.current = "original"
    first_entered, second_attempted, release_first, second_entered = (Event() for _ in range(4))

    def first():
        with _scoped_com_stream(FakeWarp, "first"):
            first_entered.set()
            assert release_first.wait(5)
            assert FakeWarp.current == "first"

    def second():
        assert first_entered.wait(5)
        second_attempted.set()
        with _scoped_com_stream(FakeWarp, "second"):
            second_entered.set()
            assert FakeWarp.current == "second"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(first), executor.submit(second)]
        try:
            assert second_attempted.wait(5)
            assert not second_entered.wait(0.05)
        finally:
            release_first.set()
        for future in futures:
            future.result(timeout=5)
    assert second_entered.is_set()
    assert FakeWarp.current == "original"


def test_com_stream_scope_restores_on_error_and_allows_nested_use():
    FakeWarp.current = "original"
    with pytest.raises(RuntimeError):
        with _scoped_com_stream(FakeWarp, "outer"):
            with _scoped_com_stream(FakeWarp, "inner"):
                assert FakeWarp.current == "inner"
            assert FakeWarp.current == "outer"
            raise RuntimeError("interrupted")
    assert FakeWarp.current == "original"
