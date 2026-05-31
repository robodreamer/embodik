#!/usr/bin/env python3
"""Headless validation: fixed vs auto-tuned pos/rot gains on panda IK.

Usage:
    pixi run python scripts/validate_adaptive_gain_tuning.py
    pixi run python scripts/validate_adaptive_gain_tuning.py --seed 1 --max-steps 2

Writes JSON to scripts/results/adaptive_gain_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from example_helpers.adaptive_gain_harness import run_all_scenarios
from example_helpers.adaptive_gain_tuning import AdaptiveGainTuningConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate adaptive gain tuning")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--position-gain", type=float, default=10.0)
    parser.add_argument("--orientation-gain", type=float, default=10.0)
    parser.add_argument("--pos-ref", type=float, default=0.05)
    parser.add_argument("--rot-ref", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = AdaptiveGainTuningConfig(
        position_reference_m=float(args.pos_ref),
        orientation_reference_rad=float(args.rot_ref),
    )
    report = run_all_scenarios(
        base_position_gain=float(args.position_gain),
        base_orientation_gain=float(args.orientation_gain),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        tuning_config=cfg,
    )

    out_dir = REPO_ROOT / "scripts" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or out_dir / f"adaptive_gain_{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Overall passed: {report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
