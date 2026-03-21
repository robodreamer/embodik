"""Stall detection and recovery for EmbodiK solvers.

The stall handler is implemented in C++ inside ``KinematicsSolver`` and runs
automatically on every ``solve_velocity``, ``solve_position_step``, and
``solve_position`` call when enabled. This Python module provides a
convenience wrapper.

When the solver returns non-success (INFEASIBLE or NUMERICAL_ERROR) with
near-zero joint velocities for several consecutive steps, the robot is
*stalled* — the task direction conflicts with collision constraints and
the QP has no feasible motion.

Recovery works by **collision margin relaxation**: when the stall is
collision-bounded, temporarily reduce the collision min_distance so the
feasibility cone opens up. When the solver starts producing healthy motion
again, the margin is restored toward nominal with a ceiling that prevents
it from overshooting above the actual clearance (which would re-create
infeasibility).

The simplest opt-in is via the ``stall_recovery`` flag on options structs::

    # solve_velocity — handler persists across calls:
    result = solver.solve_velocity(q, apply_limits=True, stall_recovery=True)

    # solve_position_step — handler persists across calls so stall counts
    # accumulate in single-step-per-tick loops (e.g. teleop at 100 Hz):
    opts = PositionStepOptions()
    opts.stall_recovery = True
    result = solver.solve_position_step(q, target, "ee", opts)

    # solve_position — handler auto-enables/disables within the call:
    opts = PositionIKOptions()
    opts.stall_recovery = True
    result = solver.solve_position(q, target, "panda_hand", opts)

For explicit control, use ``enable_stall_handler`` / ``configure_stall_handler``
or this convenience wrapper::

    handler = StallHandler(solver, nominal_min_distance=0.04)

    for _ in range(steps):
        result = solver.solve_velocity(q, apply_limits=True)
        q = robot.integrate(q, result.joint_velocities, solver.dt)
        robot.update_configuration(q)

All tuning parameters are documented on ``configure_stall_handler()``.
The defaults are conservative and suitable for interactive teleop loops
running at 50–200 Hz.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodik import KinematicsSolver


class StallHandler:
    """Convenience wrapper around the C++-level stall handler.

    Calling the constructor enables the stall handler on the given solver.
    The handler runs automatically inside every ``solve_velocity``,
    ``solve_position_step``, and ``solve_position`` call — no per-tick
    Python code needed.

    Parameters
    ----------
    solver:
        The ``KinematicsSolver`` to attach to.
    nominal_min_distance:
        The user-intended collision min_distance.
    stall_threshold:
        Consecutive infeasible steps before recovery activates.
    restore_rate:
        Per-step restoration of min_distance as fraction of nominal.
    floor_fraction:
        Minimum min_distance as fraction of nominal.
    """

    def __init__(
        self,
        solver: "KinematicsSolver",
        nominal_min_distance: float,
        stall_threshold: int = 5,
        restore_rate: float = 0.005,
        floor_fraction: float = 0.3,
    ) -> None:
        self._solver = solver
        self._nominal = nominal_min_distance
        solver.enable_stall_handler(nominal_min_distance)
        solver.configure_stall_handler(
            stall_threshold=stall_threshold,
            restore_rate=restore_rate,
            floor_fraction=floor_fraction,
        )

    @property
    def enabled(self) -> bool:
        return self._solver.stall_handler_enabled()

    @property
    def is_relaxed(self) -> bool:
        return self._solver.stall_handler_is_relaxed()

    @property
    def current_min_distance(self) -> float:
        return self._solver.stall_handler_current_min_distance()

    @property
    def consecutive_stall_steps(self) -> int:
        return self._solver.stall_handler_consecutive_stall_steps()

    def disable(self) -> None:
        """Disable the stall handler and restore nominal parameters."""
        self._solver.disable_stall_handler()
