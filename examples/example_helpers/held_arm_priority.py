"""Input ownership for the existing hold-non-dragged-arm option.

This selects task priorities only. It never changes targets, configurations,
physical constraints or solver results.
"""

from __future__ import annotations


class HeldArmPriority:
    def __init__(self) -> None:
        self.moving_side: str | None = None
        self._context: tuple | None = None

    def reset(self) -> None:
        self.moving_side = None
        self._context = None

    def update(
        self,
        *,
        enabled: bool,
        right_active: bool,
        left_active: bool,
        right_changed: bool,
        left_changed: bool,
        right_settled: bool,
        left_settled: bool,
        context: tuple,
        other_input: bool = False,
    ) -> tuple[int, int]:
        if context != self._context:
            self.moving_side = None
            self._context = context
        if (
            not enabled
            or not (right_active and left_active)
            or other_input
            or (right_changed and left_changed)
            or (right_settled and left_settled)
        ):
            self.moving_side = None
        elif right_changed:
            self.moving_side = "right"
        elif left_changed:
            self.moving_side = "left"
        # A stationary marker remains a request to finish tracking its target.
        # Absence of another UI event must not drop the held arm's priority.
        return int(self.moving_side == "right"), int(self.moving_side == "left")
