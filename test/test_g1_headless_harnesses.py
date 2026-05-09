import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
G1_EXAMPLE = REPO_ROOT / "examples/13_unitree_g1_retargeting_ik.py"
G1_BENCHMARK = REPO_ROOT / "examples/harnesses/g1_four_gizmo_ik_benchmark.py"


def _run_example(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180.0,
    )


def test_g1_reset_smoke_recovers_first_move() -> None:
    result = _run_example(str(G1_EXAMPLE), "--headless-reset-smoke")

    assert "[13][headless-reset-smoke] ok" in result.stdout
    assert "status=SUCCESS" in result.stdout


def test_g1_com_smoke_configures_fractional_margin() -> None:
    result = _run_example(str(G1_EXAMPLE), "--headless-com-smoke")

    assert "[13][headless-com-smoke] ok" in result.stdout
    assert "proximity=" in result.stdout


def test_g1_base_mode_prefers_palm_center_frames() -> None:
    from examples.example_helpers.g1_model_utils import (
        create_g1_robot_model,
        resolve_frames_for_g1_base_mode,
    )

    robot = create_g1_robot_model(floating_base=True, reduced_ik=True)
    frame_map = resolve_frames_for_g1_base_mode(robot.get_frame_names())

    assert frame_map["right_palm"] == "right_palm_force_sensor"
    assert frame_map["left_palm"] == "left_palm_force_sensor"


def test_g1_com_geometry_helpers_use_inner_polygon_margin() -> None:
    import numpy as np

    from examples.example_helpers.g1_model_utils import (
        com_min_slack,
        com_slack_color,
        shrink_polygon_xy,
    )

    polygon = np.array(
        [
            [-0.2, -0.1],
            [0.2, -0.1],
            [0.2, 0.1],
            [-0.2, 0.1],
        ],
        dtype=float,
    )
    center = np.array([0.0, 0.0], dtype=float)
    inner = shrink_polygon_xy(polygon, 0.2)

    assert inner.shape == polygon.shape
    assert com_min_slack(polygon, center, margin_fraction=0.2) < com_min_slack(polygon, center)
    assert com_slack_color(-1e-3)[0] > com_slack_color(0.05)[0]


def test_g1_retargeting_clips_include_diverse_merge_ready_motions() -> None:
    import numpy as np

    from examples.example_helpers.g1_model_utils import (
        get_retargeting_clips,
        sample_retargeting_clip,
    )

    clips = get_retargeting_clips()
    assert {"alternating_wave", "balance_circle"}.issubset(clips)

    wave_start = sample_retargeting_clip("alternating_wave", 0.0)
    wave_peak = sample_retargeting_clip("alternating_wave", 0.5)
    balance_left = sample_retargeting_clip("balance_circle", 0.6)
    balance_right = sample_retargeting_clip("balance_circle", 1.2)

    assert wave_peak["right_palm"][2] > wave_start["right_palm"][2] + 0.2
    assert np.sign(balance_left["imu_in_torso"][1]) != np.sign(balance_right["imu_in_torso"][1])


def test_g1_delta_retargeting_preserves_measured_neutral_targets() -> None:
    import numpy as np

    from examples.example_helpers.g1_model_utils import (
        build_retargeting_delta_target_poses,
        get_retargeting_presets,
    )

    anchor = np.eye(4, dtype=float)
    neutral_offsets = get_retargeting_presets()["neutral"]
    reference_poses = {}
    for i, key in enumerate(neutral_offsets):
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = np.array([0.1 * i, -0.02 * i, 0.3 + 0.01 * i], dtype=float)
        reference_poses[key] = pose

    targets = build_retargeting_delta_target_poses(
        anchor,
        reference_poses,
        neutral_offsets,
        neutral_offsets,
    )

    for key, pose in reference_poses.items():
        np.testing.assert_allclose(targets[key], pose)


def test_g1_four_gizmo_stress_includes_reset_events() -> None:
    result = _run_example(
        str(G1_EXAMPLE),
        "--headless-gizmo-stress-steps",
        "1",
        "--headless-reset-interval",
        "4",
    )

    assert "statuses=['SUCCESS']" in result.stdout
    assert "reset_count=4" in result.stdout
    assert "reset_recapture=0.000e+00" in result.stdout


def test_g1_single_target_oscillation_tracks_without_stall() -> None:
    result = _run_example(
        str(G1_EXAMPLE),
        "--headless-single-target-oscillation-steps",
        "80",
        "--headless-single-target",
        "all",
    )

    for target in ("right_palm", "left_palm", "right_ankle", "left_ankle", "pelvis"):
        assert f"target={target}" in result.stdout
    assert result.stdout.count("status_counts={'SUCCESS': 80}") == 5
    pelvis_line = next(line for line in result.stdout.splitlines() if "target=pelvis" in line)
    match = re.search(r"max_q_step=([0-9.]+)", pelvis_line)
    assert match is not None
    assert float(match.group(1)) < 0.12


def test_g1_four_gizmo_benchmark_reports_reset_metrics() -> None:
    result = _run_example(
        str(G1_BENCHMARK),
        "--max-steps",
        "2",
        "--reset-interval",
        "8",
        "--quiet",
    )
    metrics = json.loads(result.stdout)

    assert metrics["status_counts"] == {"SUCCESS": metrics["measured_steps"]}
    assert metrics["reset"]["count"] >= 2
    assert metrics["reset"]["max_recapture_error_m"] <= 1e-9
    assert metrics["reset"]["first_move_min_dq_norm"] > 1e-10
    assert metrics["target_error"]["max_position_m"] < 0.01
    assert metrics["wall_time_ms"]["p95"] < 50.0


def test_g1_collision_benchmark_uses_curated_primitive_pairs() -> None:
    result = _run_example(
        str(G1_BENCHMARK),
        "--steps",
        "60",
        "--warmup",
        "10",
        "--max-steps",
        "1",
        "--disable-posture",
        "--include-pelvis",
        "--collision-preset",
        "core",
        "--collision-tuning",
        "balanced",
        "--quiet",
    )
    metrics = json.loads(result.stdout)

    assert metrics["status_counts"] == {"SUCCESS": metrics["measured_steps"]}
    assert metrics["config"]["collision_enabled"] is True
    assert metrics["config"]["collision_tuning"] == "balanced"
    assert 50 <= metrics["config"]["collision_include_pair_count"] <= 100
    assert metrics["collision_urdf"].endswith("_box_collision.urdf")
    assert metrics["wall_time_ms"]["p95"] < 5.0


def test_g1_viewer_urdfs_strip_mimic_tags() -> None:
    from examples.example_helpers.g1_model_utils import (
        prepare_g1_viewer_urdf_path,
        resolve_g1_collision_urdf_path,
        resolve_g1_urdf_path,
    )

    visual_viewer = prepare_g1_viewer_urdf_path(resolve_g1_urdf_path())
    collision_urdf = resolve_g1_collision_urdf_path()

    assert "<mimic" not in visual_viewer.read_text()
    assert "<mimic" not in collision_urdf.read_text()
