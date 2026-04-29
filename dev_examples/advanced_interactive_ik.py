#!/usr/bin/env python3
"""Developer-only advanced interactive IK launcher.

This file intentionally lives outside ``examples/``. The pip package includes
and copies public examples from ``examples/``; ``dev_examples/`` is for clone-only
workflows where maintainers want the full tuning/debug surfaces.

Usage:
    pixi run demo-advanced-ik
    pixi run python dev_examples/advanced_interactive_ik.py basic -- --robot iiwa
    pixi run python dev_examples/advanced_interactive_ik.py collision -- --robot iiwa
    pixi run python dev_examples/advanced_interactive_ik.py teleop
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR = _REPO_ROOT / "examples"


def _load_example(filename: str):
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))

    path = _EXAMPLES_DIR / filename
    module_name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone-only launcher for advanced EmbodiK IK dev surfaces."
    )
    parser.add_argument(
        "surface",
        choices=("basic", "collision", "teleop"),
        nargs="?",
        default="basic",
        help="Advanced surface to launch.",
    )
    parser.add_argument(
        "example_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded after an optional '--'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forwarded = list(args.example_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    if args.surface == "basic":
        module = _load_example("../dev_examples/advanced_basic_ik.py")
        sys.argv = ["advanced_basic_ik.py", *forwarded]
        module.main(module.parse_args())
        return
    elif args.surface == "collision":
        module = _load_example("02_collision_aware_IK.py")
        sys.argv = ["02_collision_aware_IK.py", *forwarded]
    else:
        module = _load_example("03_teleop_ik.py")
        sys.argv = ["03_teleop_ik.py", *forwarded]

    module.main()


if __name__ == "__main__":
    main()
