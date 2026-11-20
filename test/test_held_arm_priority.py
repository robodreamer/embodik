"""Keep hold-arm priority for the lifetime of an unfinished input request."""

from types import SimpleNamespace

import pytest

from examples.example_helpers.common_bimanual_teleop_app import (
    _apply_torso_control_priority_policy,
)
from examples.example_helpers.held_arm_priority import HeldArmPriority


def update(policy: HeldArmPriority, **changes: object) -> tuple[int, int]:
    values = dict(
        enabled=True,
        right_active=True,
        left_active=True,
        right_changed=False,
        left_changed=False,
        right_settled=False,
        left_settled=False,
        context=("cpu", "shared", False, False, 1),
    )
    values.update(changes)
    return policy.update(**values)


@pytest.mark.parametrize("side,priorities", [("right", (1, 0)), ("left", (0, 1))])
def test_stationary_marker_keeps_unfinished_request_protected(side, priorities):
    policy = HeldArmPriority()
    assert update(policy, **{f"{side}_changed": True}) == priorities
    for _ in range(20):
        assert update(policy) == priorities
    assert policy.moving_side == side


@pytest.mark.parametrize("side", ["right", "left"])
def test_settling_only_one_hand_does_not_drop_protection(side):
    policy = HeldArmPriority()
    expected = update(policy, **{f"{side}_changed": True})
    assert update(policy, **{f"{side}_settled": True}) == expected
    assert update(policy, right_settled=True, left_settled=True) == (0, 0)
    assert policy.moving_side is None


@pytest.mark.parametrize("side", ["right", "left"])
def test_new_single_hand_command_hands_over_the_role(side):
    policy = HeldArmPriority()
    other = "left" if side == "right" else "right"
    update(policy, **{f"{other}_changed": True})
    expected = (1, 0) if side == "right" else (0, 1)
    assert update(policy, **{f"{side}_changed": True}) == expected
    assert update(policy) == expected


@pytest.mark.parametrize("side", ["right", "left"])
def test_simultaneous_hand_commands_clear_the_single_hand_role(side):
    policy = HeldArmPriority()
    update(policy, **{f"{side}_changed": True})
    assert update(policy, right_changed=True, left_changed=True) == (0, 0)
    assert update(policy) == (0, 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": False},
        {"right_active": False},
        {"left_active": False},
        {"other_input": True},
        {"context": ("gpu", "shared", False, False, 1)},
        {"context": ("cpu", "decoupled", False, False, 1)},
        {"context": ("cpu", "shared", True, False, 1)},
        {"context": ("cpu", "shared", False, True, 1)},
        {"context": ("cpu", "shared", False, False, 2)},
    ],
)
def test_disable_or_context_change_discards_a_stale_role(changes):
    policy = HeldArmPriority()
    assert update(policy, right_changed=True) == (1, 0)
    assert update(policy, **changes) == (0, 0)
    assert policy.moving_side is None


@pytest.mark.parametrize("side", ["right", "left"])
def test_reset_requires_a_fresh_command_to_reacquire_role(side):
    policy = HeldArmPriority()
    update(policy, **{f"{side}_changed": True})
    policy.reset()
    assert update(policy) == (0, 0)
    assert update(policy, **{f"{side}_changed": True}) != (0, 0)


def test_other_input_does_not_acquire_a_role_from_simultaneous_hand_motion():
    policy = HeldArmPriority()
    assert update(policy, right_changed=True, other_input=True) == (0, 0)
    assert update(policy) == (0, 0)


def test_already_settled_input_does_not_leave_a_latched_role():
    policy = HeldArmPriority()
    assert update(policy, right_changed=True, right_settled=True, left_settled=True) == (0, 0)


@pytest.mark.parametrize("priorities", [(0,), (0, 1), (1, 0), ()])
def test_shared_torso_stays_below_every_active_hand(priorities):
    torso = SimpleNamespace(priority=0)
    posture = SimpleNamespace(controlled=[])
    posture.set_controlled_joint_indices = lambda indices: setattr(
        posture, "controlled", list(indices)
    )
    shared = _apply_torso_control_priority_policy(
        torso_task=torso,
        posture_task=posture,
        torso_active=True,
        decouple_torso_and_arms=False,
        posture_controlled_indices=[0, 1, 2],
        lift_posture_indices=[0],
        ee_priorities=priorities,
    )
    assert shared
    assert torso.priority == max(priorities, default=0) + 1
    assert posture.controlled == [1, 2]


@pytest.mark.parametrize("active,decoupled", [(False, False), (True, True)])
def test_inactive_or_decoupled_torso_keeps_original_priority(active, decoupled):
    torso = SimpleNamespace(priority=99)
    posture = SimpleNamespace(controlled=[])
    posture.set_controlled_joint_indices = lambda indices: setattr(
        posture, "controlled", list(indices)
    )
    assert not _apply_torso_control_priority_policy(
        torso_task=torso,
        posture_task=posture,
        torso_active=active,
        decouple_torso_and_arms=decoupled,
        posture_controlled_indices=[0, 1, 2],
        lift_posture_indices=[0],
        ee_priorities=(0, 1),
    )
    assert torso.priority == 0
    assert posture.controlled == [0, 1, 2]
