#!/usr/bin/env python3
"""Default robot-geometry-view selection for the shared bimanual app.

The app normally shows collision meshes when the visual and collision URDFs are
the same file (and it has collision geometry). An entrypoint can override the
default via ``COMMON_BIMANUAL_DEFAULT_GEOMETRY_VIEW`` -- the alpha-wheelbase
example uses this to show the visual meshes by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.example_helpers.common_bimanual_teleop_app import _resolve_default_geometry_view

URDF = Path("/tmp/model.urdf")
COLLISION = Path("/tmp/model_collision.urdf")


def test_override_visual_wins_over_auto_collision() -> None:
    # Same path + collision geometry would auto-pick "Collision"; override wins.
    view = _resolve_default_geometry_view(
        override="Visual", urdf_path=URDF, collision_urdf_path=URDF, has_collision_geometry=True
    )
    assert view == "Visual"


def test_auto_picks_collision_when_same_urdf_has_collision_geometry() -> None:
    view = _resolve_default_geometry_view(
        override=None, urdf_path=URDF, collision_urdf_path=URDF, has_collision_geometry=True
    )
    assert view == "Collision"


def test_auto_picks_visual_when_collision_urdf_differs() -> None:
    view = _resolve_default_geometry_view(
        override=None,
        urdf_path=URDF,
        collision_urdf_path=COLLISION,
        has_collision_geometry=True,
    )
    assert view == "Visual"


def test_auto_picks_visual_without_collision_geometry() -> None:
    view = _resolve_default_geometry_view(
        override=None, urdf_path=URDF, collision_urdf_path=URDF, has_collision_geometry=False
    )
    assert view == "Visual"


def test_invalid_override_is_ignored() -> None:
    view = _resolve_default_geometry_view(
        override="Nonsense", urdf_path=URDF, collision_urdf_path=URDF, has_collision_geometry=True
    )
    assert view == "Collision"
