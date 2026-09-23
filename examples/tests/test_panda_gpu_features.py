from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace


example = importlib.import_module("02_collision_aware_IK")


def test_gpu_collision_pairs_are_model_derived_and_order_independent():
    robot = SimpleNamespace(
        get_collision_pair_names=lambda: [
            ("base", "link1"),
            ("link1", "link4"),
            ("hand", "link2"),
        ]
    )

    assert example.gpu_collision_include_pairs(
        robot, [("link1", "base"), ("link2", "hand")]
    ) == (("link1", "link4"),)


def test_public_gpu_path_uses_generalized_multiframe_runtime_features():
    source = inspect.getsource(example.run_gui)

    assert "GpuWbcMultiFrameSolver(" in source
    assert "configure_runtime(" in source
    assert "gpu_collision_include_pairs(" in source
