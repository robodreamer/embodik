#!/usr/bin/env python3
"""Headless tests for sungjoon G1 port helper behaviors.

These tests intentionally target pure helper logic so we can validate
port behavior deterministically without interactive Viser loops.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repository root is on sys.path so tests can import example modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.incubating.sungjoon_port_phase1 import g1_viser_utils as g1_utils


def test_resolve_frames_for_g1_site_mode_prefers_expected_names() -> None:
    frame_names = [
        "pelvis",
        "right_palm",
        "right_palm_top",
        "right_palm_palmar",
        "right_palm_front",
    ]
    frames = g1_utils.resolve_frames_for_g1_site_mode(frame_names)
    assert frames == [
        "right_palm",
        "right_palm_top",
        "right_palm_palmar",
        "right_palm_front",
    ]


def test_resolve_frames_for_g1_base_mode_contains_required_keys() -> None:
    frame_names = [
        "right_palm",
        "left_palm",
        "imu_in_torso",
        "right_ankle",
        "left_ankle",
    ]
    frame_map = g1_utils.resolve_frames_for_g1_base_mode(frame_names)
    assert frame_map["right_palm"] == "right_palm"
    assert frame_map["left_palm"] == "left_palm"
    assert frame_map["imu_in_torso"] == "imu_in_torso"
    assert frame_map["right_ankle"] == "right_ankle"
    assert frame_map["left_ankle"] == "left_ankle"


def test_build_three_point_mode_targets_rotate_local_offsets() -> None:
    p_target = np.array([1.0, 2.0, 3.0], dtype=float)
    # 90 deg yaw: x->y
    R_target = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    local_offsets = [
        np.zeros(3),
        np.array([0.1, 0.0, 0.0], dtype=float),
        np.array([0.0, 0.1, 0.0], dtype=float),
        np.array([0.0, 0.0, 0.1], dtype=float),
    ]
    poses = g1_utils.build_sungjoon_three_point_targets(p_target, R_target, local_offsets)
    assert len(poses) == 4
    np.testing.assert_allclose(poses[0][:3, 3], np.array([1.0, 2.0, 3.0]), atol=1e-9)
    np.testing.assert_allclose(poses[1][:3, 3], np.array([1.0, 2.1, 3.0]), atol=1e-9)
    np.testing.assert_allclose(poses[2][:3, 3], np.array([0.9, 2.0, 3.0]), atol=1e-9)
    np.testing.assert_allclose(poses[3][:3, 3], np.array([1.0, 2.0, 3.1]), atol=1e-9)


def test_build_embodik_6d_target_pose() -> None:
    p_target = np.array([0.2, -0.1, 0.5], dtype=float)
    R_target = np.eye(3, dtype=float)
    pose = g1_utils.build_embodik_6d_target_pose(p_target, R_target)
    np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(pose[:3, 3], p_target, atol=1e-12)


class _FakeRobot:
    def __init__(self) -> None:
        self._meta = {
            "waist_yaw_joint": (0, 1),
            "right_shoulder_pitch_joint": (1, 1),
            "left_hip_pitch_joint": (2, 2),
            "left_knee_joint": (4, 1),
        }
        self.nv = 5
        self.is_floating_base = True

    def get_joint_names(self):  # noqa: D401
        return list(self._meta.keys())

    def get_joint_velocity_index(self, name: str) -> int:
        return self._meta[name][0]

    def get_joint_velocity_size(self, name: str) -> int:
        return self._meta[name][1]


def test_compute_arm_only_excluded_velocity_indices_expands_multidof_ranges() -> None:
    robot = _FakeRobot()
    allowed = {"waist_yaw_joint", "right_shoulder_pitch_joint"}
    excluded = g1_utils.compute_excluded_velocity_indices(robot, allowed)
    # left_hip_pitch_joint has nv size 2 -> indices [2, 3], left_knee_joint -> [4]
    assert excluded == [2, 3, 4]


def test_resolve_frames_for_g1_site_mode_raises_when_missing() -> None:
    with pytest.raises(ValueError):
        g1_utils.resolve_frames_for_g1_site_mode(["right_palm"])


def test_resolve_frames_for_g1_site_mode_fallback_selects_aux_frames() -> None:
    frame_names = [
        "right_wrist_yaw_link",
        "right_palm_force_sensor",
        "right_index_2",
        "right_middle_2",
        "right_thumb_4",
        "right_index_2_joint",
    ]
    frames = g1_utils.resolve_frames_for_g1_site_mode(frame_names)
    assert len(frames) == 4
    assert frames[0] in {"right_palm_force_sensor", "right_wrist_yaw_link"}
    assert all("joint" not in f for f in frames[1:])


def test_compute_support_polygon_from_feet_bounds() -> None:
    right_xy = np.array([0.10, -0.05], dtype=float)
    left_xy = np.array([0.12, 0.08], dtype=float)
    poly = g1_utils.compute_support_polygon_from_feet(
        right_xy,
        left_xy,
        toe_pad=0.02,
        side_pad=0.01,
    )
    assert poly.shape == (4, 2)
    x_min, x_max = np.min(poly[:, 0]), np.max(poly[:, 0])
    y_min, y_max = np.min(poly[:, 1]), np.max(poly[:, 1])
    assert x_min == pytest.approx(0.08)
    assert x_max == pytest.approx(0.14)
    assert y_min == pytest.approx(-0.06)
    assert y_max == pytest.approx(0.09)


def test_compute_support_polygon_from_foot_poses_reflects_orientation() -> None:
    r_pose = np.eye(4, dtype=float)
    l_pose = np.eye(4, dtype=float)
    r_pose[:3, 3] = np.array([0.1, -0.1, 0.0], dtype=float)
    l_pose[:3, 3] = np.array([0.1, 0.1, 0.0], dtype=float)
    # Left foot yaw 90 deg to ensure non-axis-aligned aggregate hull.
    l_pose[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    poly = g1_utils.compute_support_polygon_from_foot_poses(
        r_pose,
        l_pose,
        foot_length=0.2,
        foot_width=0.1,
        toe_pad=0.0,
        side_pad=0.0,
    )
    assert poly.shape[1] == 2
    assert poly.shape[0] >= 4
    # Convex hull should cover both foot centers.
    x_min, x_max = np.min(poly[:, 0]), np.max(poly[:, 0])
    y_min, y_max = np.min(poly[:, 1]), np.max(poly[:, 1])
    assert x_min <= 0.1 <= x_max
    assert y_min <= -0.1 <= y_max
    assert y_min <= 0.1 <= y_max


def test_polygon_segments_xy_closes_loop() -> None:
    poly = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    seg = g1_utils.polygon_segments_xy(poly, z=0.123)
    assert seg.shape == (4, 2, 3)
    np.testing.assert_allclose(seg[0, 0], np.array([0.0, 0.0, 0.123]))
    np.testing.assert_allclose(seg[3, 1], np.array([0.0, 0.0, 0.123]))


def test_get_retargeting_presets_contains_expected_profiles() -> None:
    presets = g1_utils.get_retargeting_presets()
    assert "neutral" in presets
    assert "reach_forward" in presets
    assert "dual_reach" in presets
    assert "cross_body" in presets
    assert "right_palm" in presets["neutral"]


def test_retarget_position_from_anchor_applies_rotation_and_scale() -> None:
    anchor_p = np.array([1.0, 2.0, 3.0], dtype=float)
    anchor_R = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    local = np.array([0.2, 0.0, 0.1], dtype=float)
    world = g1_utils.retarget_position_from_anchor(anchor_p, anchor_R, local, scale=0.5)
    np.testing.assert_allclose(world, np.array([1.0, 2.1, 3.05]), atol=1e-12)


def test_build_site_mode_target_payloads_counts_and_names() -> None:
    p_target = np.array([0.1, 0.2, 0.3], dtype=float)
    R_target = np.eye(3, dtype=float)
    local_offsets = [np.zeros(3), np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.0]), np.array([0.0, 0.0, 0.1])]

    payloads_3p = g1_utils.build_site_mode_target_payloads(
        "Sungjoon 3-point",
        p_target,
        R_target,
        local_offsets,
        position_gain=10.0,
        orientation_gain=10.0,
    )
    assert len(payloads_3p) == 4
    assert [p["task_name"] for p in payloads_3p] == ["g1_site_0", "g1_site_1", "g1_site_2", "g1_site_3"]
    assert all(float(p["orientation_gain"]) == 0.0 for p in payloads_3p)

    payloads_6d = g1_utils.build_site_mode_target_payloads(
        "EmbodiK 6D",
        p_target,
        R_target,
        local_offsets,
        position_gain=10.0,
        orientation_gain=2.0,
    )
    assert len(payloads_6d) == 2
    assert [p["task_name"] for p in payloads_6d] == ["g1_pose_pos", "g1_pose_ori"]
    assert float(payloads_6d[1]["orientation_gain"]) == pytest.approx(2.0)


def test_build_fullbody_target_payloads_switches_foot_orientation_targets() -> None:
    pose = np.eye(4, dtype=float)
    target_poses = {
        "right_palm": pose.copy(),
        "left_palm": pose.copy(),
        "imu_in_torso": pose.copy(),
        "right_ankle": pose.copy(),
        "left_ankle": pose.copy(),
    }
    p_pos_only = g1_utils.build_fullbody_target_payloads(
        "Sungjoon Position-only",
        target_poses,
        position_gain=15.0,
        foot_orientation_gain=8.0,
    )
    assert len(p_pos_only) == 5
    assert all("_pos" in p["task_name"] for p in p_pos_only)

    p_6d = g1_utils.build_fullbody_target_payloads(
        "EmbodiK 6D Feet",
        target_poses,
        position_gain=15.0,
        foot_orientation_gain=8.0,
    )
    assert len(p_6d) == 7
    names = [p["task_name"] for p in p_6d]
    assert "g1_base_right_ankle_ori" in names
    assert "g1_base_left_ankle_ori" in names


def test_collision_arm_only_option_values() -> None:
    excluded, zero_idx = g1_utils.collision_arm_only_option_values(
        arm_only_enabled=True,
        floating_base=True,
        excluded_joint_indices=[8, 9, 10],
    )
    assert excluded == [8, 9, 10]
    assert zero_idx == [0, 1, 2, 3, 4, 5]

    excluded2, zero_idx2 = g1_utils.collision_arm_only_option_values(
        arm_only_enabled=False,
        floating_base=True,
        excluded_joint_indices=[8, 9, 10],
    )
    assert excluded2 == []
    assert zero_idx2 == []


def test_compute_feet_center_returns_midpoint() -> None:
    r = np.array([0.2, -0.1, -0.8], dtype=float)
    l = np.array([0.0, 0.1, -0.8], dtype=float)
    c = g1_utils.compute_feet_center(r, l)
    np.testing.assert_allclose(c, np.array([0.1, 0.0, -0.8]), atol=1e-12)


def test_ground_floating_base_from_feet_center_shifts_by_negative_center() -> None:
    q = np.array([0.5, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.1], dtype=float)
    r = np.array([0.3, -0.2, -0.9], dtype=float)
    l = np.array([0.1, 0.2, -0.7], dtype=float)
    q_new = g1_utils.ground_floating_base_from_feet_center(q, r, l)
    np.testing.assert_allclose(q_new[:3], q[:3] - np.array([0.2, 0.0, -0.8]), atol=1e-12)
    np.testing.assert_allclose(q_new[3:], q[3:], atol=1e-12)


def test_make_visual_config_mapper_uses_joint_config_indices_when_available() -> None:
    class _FakeRobotCfg:
        def get_joint_names(self):
            return ["j0", "j1"]

        def get_joint_config_index(self, name: str) -> int:
            return {"j0": 7, "j1": 8}[name]

    class _FakeUrdf:
        actuated_joint_names = ["j0", "j1"]

    class _FakeVis:
        _urdf = _FakeUrdf()

    mapper = g1_utils.make_visual_config_mapper(_FakeRobotCfg(), _FakeVis())
    q = np.array([10, 11, 12, 13, 14, 15, 16, 0.25, -0.5], dtype=float)
    mapped = mapper(q)
    np.testing.assert_allclose(mapped, np.array([0.25, -0.5]), atol=1e-12)
