#!/usr/bin/env python3
"""Regression tests for examples copied out of the source tree."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest


pytest.importorskip("embodik.interactive_ik")


def test_ai_worker_example_imports_from_copied_examples_layout(monkeypatch) -> None:
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples_dir))

    module_globals = runpy.run_path(
        str(examples_dir / "12_ai_worker_constraint_teleop.py"),
        run_name="embodik_ai_worker_import_check",
    )

    assert "main" in module_globals
