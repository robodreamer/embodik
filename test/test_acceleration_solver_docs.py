import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_acceleration_guide_names_supported_and_unsupported_scope() -> None:
    guide = (ROOT / "docs" / "acceleration_solver.md").read_text()

    for required in (
        "Fixed-base scalar 1-DoF joints",
        "Compatible joint position, velocity, and acceleration box",
        "Fixed-base point or rigid contact",
        "collision_step_certified = False",
        "all-prismatic scalar models",
        "floating bases",
        "wheel rolling/steering constraints",
        "not a hard real-time controller",
        "Inequality Constraint Parity",
        "`FrozenNextVelocityConstraint`",
        "Floating-base position/orientation bounds",
        "position-step priority-policy rows are not imported automatically",
        "allow_state_box_task_fallback = True",
        "state_box_task_fallback_applied",
        "collect_task_diagnostics = False",
        "backend_computation_time_ms",
        "conservative enclosing-sphere lower bound",
        "ambiguous geometry falls back",
    ):
        assert required in guide


def test_acceleration_example_docs_explain_selector_and_collision_contract() -> None:
    basic = (ROOT / "docs" / "examples" / "basic_ik.md").read_text()
    collision = (ROOT / "docs" / "examples" / "collision_aware_ik.md").read_text()
    collision_words = " ".join(collision.split())

    assert "Velocity is the default" in basic
    assert "resets its caller-owned `dq`" in basic
    assert "`solve_with_velocity_collision()`" in collision
    assert "same include/exclude" in collision
    assert "does not claim continuous swept-path certification" in collision_words


def test_acceleration_page_is_in_navigation_and_release_history() -> None:
    navigation = (ROOT / "mkdocs.yml").read_text()
    changelog = (ROOT / "CHANGELOG.md").read_text()
    with (ROOT / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]

    assert "Acceleration Solver: acceleration_solver.md" in navigation
    assert f"## [{version}] - " in changelog
    current_release = changelog.split(f"## [{version}]", maxsplit=1)[1].split("## [", maxsplit=1)[0]
    acceleration_release = changelog.split("## [0.21.0]", maxsplit=1)[1].split("## [", maxsplit=1)[
        0
    ]
    assert current_release.strip()
    assert "fixed-base acceleration-level eSNS API" in acceleration_release
    assert "non-hard-real-time" in acceleration_release


def test_development_guide_defines_pre_one_versioning_contract() -> None:
    guide = (ROOT / "docs" / "development.md").read_text()

    for required in (
        "Versioning Contract",
        "Patch (`0.Y.Z`)",
        "Minor (`0.Y.0`)",
        "new public solvers",
        "`1.0.0`",
        "applies prospectively from `0.21.0`",
    ):
        assert required in guide


def test_example_index_classifies_every_runnable_script() -> None:
    overview = (ROOT / "docs" / "examples" / "index.md").read_text()
    examples = ROOT / "examples"
    scripts = {
        path.relative_to(examples).as_posix()
        for path in examples.rglob("*.py")
        if 'if __name__ == "__main__"' in path.read_text()
        or "if __name__ == '__main__'" in path.read_text()
    }

    missing = sorted(name for name in scripts if f"`{name}`" not in overview)
    assert missing == []
    for status in ("Selectable", "Velocity-only", "Unsupported", "Not applicable"):
        assert status in overview
    assert "Velocity remains the default solver for all IK examples" in overview


def test_centroidal_stability_guide_defines_physical_contracts() -> None:
    guide = " ".join((ROOT / "docs" / "centroidal_stability.md").read_text().split())

    for required in (
        "linear x, y, z; angular x, y, z",
        "kg m/s",
        "kg m^2/s",
        "kg m/s^2",
        "kg m^2/s^2",
        "explicit current `dq`",
        "world or structurally root-fixed",
        "positive vertical force",
        "`supports_dynamic_balance = False`",
        "does not prove contact-force feasibility",
        "`solve_velocity_with_state()`",
        "`CapturePointAccelerationConstraint`",
        "`ZmpAccelerationConstraint`",
        "`compute_centroidal_momentum_matrix_time_variation(q, dq)`",
    ):
        assert required in guide
    assert "compute_centroidal_momentum_matrix_derivative" not in guide


def test_centroidal_release_surface_is_navigable_and_versioned() -> None:
    navigation = (ROOT / "mkdocs.yml").read_text()
    acceleration = (ROOT / "docs" / "acceleration_solver.md").read_text()
    examples = (ROOT / "docs" / "examples" / "index.md").read_text()
    changelog = (ROOT / "CHANGELOG.md").read_text()
    with (ROOT / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]

    assert version == "0.22.0"
    assert "Centroidal Stability: centroidal_stability.md" in navigation
    assert "`04_com_constraint_example.py`" in examples
    assert "`06_bimanual_whole_body_ik.py`" in examples
    assert "Centroidal momentum-rate objective and bounds" in acceleration
    assert "Predicted capture point" in acceleration
    assert "Physical centroidal-rate ZMP" in acceleration
    assert "supports_dynamic_balance" in acceleration
    assert "## [0.22.0] - 2026-07-29" in changelog
    current_release = changelog.split("## [0.22.0]", maxsplit=1)[1].split("## [", maxsplit=1)[0]
    for required in (
        "centroidal momentum",
        "capture point",
        "ZMP",
        "explicit current velocity",
        "fixed-base",
    ):
        assert required in current_release
