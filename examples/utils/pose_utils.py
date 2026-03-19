"""Shared pose helpers for examples."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class PoseUtils:
    """Helpers for common SE(3) matrix construction patterns in examples."""

    @staticmethod
    def make_pose_matrix(
        rotation: np.ndarray | None = None,
        translation: np.ndarray | Sequence[float] | None = None,
    ) -> np.ndarray:
        """Build a 4x4 homogeneous pose matrix from optional rotation/translation."""
        pose = np.eye(4)
        if rotation is not None:
            pose[:3, :3] = np.asarray(rotation, dtype=float)
        if translation is not None:
            pose[:3, 3] = np.asarray(translation, dtype=float)
        return pose
