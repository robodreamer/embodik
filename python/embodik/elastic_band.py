"""Elastic band joint limit expansion for EmbodiK solvers.

The elastic band mechanism temporarily expands joint position limit margins
when the solver is overconstrained by joint limits (limit-dominated stalls).
This keeps more DOFs active in the SNS solver, preventing premature task
scale collapse when multiple joints are saturated near their limits.

The mechanism is complementary to the collision stall handler: the stall
handler relaxes collision margins for collision-dominated stalls, while the
elastic band expands joint limits for limit-dominated stalls.

The simplest opt-in is via ``enable_elastic_band``::

    solver.enable_elastic_band(delta_max=0.05)

    for _ in range(steps):
        result = solver.solve_velocity(q, apply_limits=True)
        q = robot.integrate(q, result.joint_velocities, solver.dt)
        q_lower, q_upper = robot.get_joint_limits()
        q = np.clip(q, q_lower, q_upper)  # clamp to nominal limits
        robot.update_kinematics(q)

For explicit control, use this convenience wrapper::

    handler = ElasticBandHandler(solver, delta_max=0.05)

All tuning parameters are documented on ``configure_elastic_band()``.
The defaults are suitable for interactive teleop loops at 50-200 Hz.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from embodik import KinematicsSolver


class ElasticBandHandler:
    """Convenience wrapper around the C++-level elastic band mechanism.

    Calling the constructor enables the elastic band on the given solver.
    The mechanism runs automatically inside every ``solve_velocity`` call.

    Parameters
    ----------
    solver:
        The ``KinematicsSolver`` to attach to.
    delta_max:
        Maximum expansion per joint in radians.
    expand_rate:
        Expansion per stall trigger (radians/step).
    decay_rate:
        Exponential decay per healthy step (0-1).
    stall_threshold:
        Consecutive stalls before expansion starts.
    expand_only_saturated:
        Only expand joints that are actually saturated.
    """

    def __init__(
        self,
        solver: "KinematicsSolver",
        delta_max: float = 0.05,
        expand_rate: float = 0.01,
        decay_rate: float = 0.2,
        stall_threshold: int = 3,
        expand_only_saturated: bool = True,
    ) -> None:
        self._solver = solver
        solver.enable_elastic_band(delta_max)
        solver.configure_elastic_band(
            delta_max=delta_max,
            expand_rate=expand_rate,
            decay_rate=decay_rate,
            stall_threshold=stall_threshold,
            expand_only_saturated=expand_only_saturated,
        )

    @property
    def enabled(self) -> bool:
        return self._solver.elastic_band_enabled()

    @property
    def is_expanded(self) -> bool:
        return self._solver.elastic_band_is_expanded()

    @property
    def max_delta(self) -> float:
        return self._solver.elastic_band_max_delta()

    @property
    def deltas(self) -> np.ndarray:
        return self._solver.elastic_band_deltas()

    def disable(self) -> None:
        """Disable the elastic band and reset expansion state."""
        self._solver.disable_elastic_band()
