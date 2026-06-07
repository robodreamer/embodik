#!/usr/bin/env python3
"""Coverage for headless common-bimanual solver fixture helpers."""

from __future__ import annotations

import numpy as np

from examples.example_helpers.common_bimanual_solver_fixture import _fallback_support_polygon


class _Pose:
    def __init__(self, translation: np.ndarray) -> None:
        self.translation = np.asarray(translation, dtype=float)


class _BaseOnlyRobot:
    def get_frame_pose(self, frame_name: str) -> _Pose:
        assert frame_name == "base_link"
        return _Pose(np.array([1.0, 2.0, 0.0], dtype=float))


def test_fallback_support_polygon_uses_base_frame_center() -> None:
    polygon = _fallback_support_polygon(_BaseOnlyRobot(), ("base_link", "arm_base_link"))

    assert polygon.shape == (4, 2)
    np.testing.assert_allclose(polygon.mean(axis=0), np.array([1.0, 2.0]))
    np.testing.assert_allclose(np.ptp(polygon, axis=0), np.array([0.7, 0.5]))
