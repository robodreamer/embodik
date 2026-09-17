#!/usr/bin/env python3
"""Compatibility entry point for the numbered parallel GPU WBC example."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        Path(__file__).with_name("10_parallel_trajectory_tracking.py"),
        run_name="__main__",
    )
