#!/usr/bin/env python3
"""Shared visualization helpers for example-side Viser integrations."""

from __future__ import annotations

import numpy as np


def make_visual_config_mapper(robot, urdf_vis):
    """Return a q->visual mapping function based on actuated joint names."""
    actuated_names = list(getattr(urdf_vis._urdf, "actuated_joint_names", []))
    name_to_q_idx: dict[str, int] = {}
    if hasattr(robot, "get_joint_config_index"):
        for name in robot.get_joint_names():
            try:
                name_to_q_idx[name] = int(robot.get_joint_config_index(name))
            except Exception:
                pass
    if not name_to_q_idx:
        robot_joint_names = list(robot.get_joint_names())
        name_to_q_idx = {name: idx for idx, name in enumerate(robot_joint_names)}

    def map_q(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if not actuated_names:
            return q
        cfg_vec = np.zeros(len(actuated_names), dtype=float)
        for i, joint_name in enumerate(actuated_names):
            idx = name_to_q_idx.get(joint_name)
            if idx is not None and idx < q.size:
                cfg_vec[i] = q[idx]
        return cfg_vec

    return map_q
