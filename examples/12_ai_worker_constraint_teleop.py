#!/usr/bin/env python3
"""Public AI Worker dual-arm teleop entrypoint.

This entrypoint runs the public AI Worker dual-arm constraint teleop demo
without machine-local path assumptions.

URDF sources:
- explicit ``--urdf`` / ``--collision-urdf``
- a local clone of the public ``ROBOTIS-GIT/ai_worker`` repository via
  ``--ai-worker-root``
- automatic cached download from the public ``ROBOTIS-GIT/ai_worker`` repo

Examples
--------
python 12_ai_worker_constraint_teleop.py \
    --variant sg2 \
    --ai-worker-root /path/to/ROBOTIS-GIT/ai_worker

python 12_ai_worker_constraint_teleop.py \
    --variant bg2 \
    --urdf /path/to/base.urdf \
    --collision-urdf /path/to/collision.urdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from example_helpers.public_ai_worker_paths import resolve_public_ai_worker_urdf_paths
    import example_helpers.ai_worker_constraint_teleop_app as worker_impl
except ModuleNotFoundError as exc:
    if exc.name != "example_helpers" and not str(exc.name).startswith("example_helpers."):
        raise
    from examples.example_helpers.public_ai_worker_paths import resolve_public_ai_worker_urdf_paths
    import examples.example_helpers.ai_worker_constraint_teleop_app as worker_impl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("sg2", "bg2"), default="sg2")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument(
        "--ai-worker-root",
        type=str,
        default=None,
        help="Path to a local clone of https://github.com/ROBOTIS-GIT/ai_worker",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Explicit follower URDF path. Overrides --ai-worker-root auto-resolution.",
    )
    parser.add_argument(
        "--collision-urdf",
        type=str,
        default=None,
        help="Optional collision-specialized URDF path. Defaults to the base URDF.",
    )
    parser.add_argument(
        "--print-resolved-paths",
        action="store_true",
        help="Print resolved URDF inputs before launching the app.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urdf_path, collision_urdf_path = resolve_public_ai_worker_urdf_paths(
        variant=args.variant,
        ai_worker_root=args.ai_worker_root,
        urdf=args.urdf,
        collision_urdf=args.collision_urdf,
        allow_bundled_base_fallback=False,
    )

    if args.print_resolved_paths:
        print(f"[ai-worker] variant={args.variant}")
        print(f"[ai-worker] urdf={urdf_path}")
        print(f"[ai-worker] collision_urdf={collision_urdf_path or urdf_path}")

    worker_impl.resolve_ffw_urdf_path = lambda _variant: urdf_path
    worker_impl.resolve_generated_ffw_collision_urdf_path = (
        lambda _variant: collision_urdf_path
    )
    worker_impl.parse_args = lambda: argparse.Namespace(variant=args.variant, port=args.port)
    worker_impl.main()


if __name__ == "__main__":
    main()
