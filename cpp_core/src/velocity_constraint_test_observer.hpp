#pragma once

#include "velocity_collision_constraint_provider.hpp"

namespace embodik::detail {

struct VelocityConstraintTestObserver {
  static std::optional<VelocityCollisionConstraintLinearization>
  linearize_collision_velocity_constraint(KinematicsSolver &solver,
                                          double row_dt) {
    return VelocityCollisionConstraintProvider(solver).linearize(row_dt);
  }

  static std::optional<VelocityCollisionConstraintRows>
  compute_collision_constraint(KinematicsSolver &solver, double row_dt) {
    return VelocityCollisionConstraintProvider(solver).compute(row_dt);
  }

  static std::optional<VelocityCollisionConstraintRows>
  compute_collision_constraint(KinematicsSolver &solver) {
    return VelocityCollisionConstraintProvider(solver).compute();
  }
};

} // namespace embodik::detail
