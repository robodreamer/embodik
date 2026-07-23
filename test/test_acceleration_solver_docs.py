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


def test_acceleration_page_is_in_navigation_and_unreleased_changelog() -> None:
    navigation = (ROOT / "mkdocs.yml").read_text()
    changelog = (ROOT / "CHANGELOG.md").read_text()

    assert "Acceleration Solver: acceleration_solver.md" in navigation
    unreleased = changelog.split("## [0.20.18]", maxsplit=1)[0]
    assert "fixed-base acceleration-level eSNS API" in unreleased
    assert "non-hard-real-time" in unreleased
