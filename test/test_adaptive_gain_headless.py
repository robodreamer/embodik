"""Headless integration gate for adaptive gain tuning (panda)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import pytest

pytest.importorskip("robot_descriptions")

from example_helpers.adaptive_gain_harness import run_all_scenarios


def test_adaptive_gain_headless_harness_passes() -> None:
    report = run_all_scenarios(max_steps=2, seed=0)
    assert report["passed"], report
