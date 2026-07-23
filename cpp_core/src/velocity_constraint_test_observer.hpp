#pragma once

#include <embodik/kinematics_solver.hpp>

namespace embodik::detail {

struct VelocityConstraintTestObserver {
  static std::optional<
      KinematicsSolver::CollisionVelocityConstraintLinearization>
  linearize_collision_velocity_constraint(KinematicsSolver &solver,
                                          double row_dt) {
    return solver.linearize_collision_velocity_constraint(row_dt);
  }

  static std::optional<KinematicsSolver::CollisionConstraintResult>
  compute_collision_constraint(KinematicsSolver &solver, double row_dt) {
    return solver.compute_collision_constraint(row_dt);
  }

  static std::optional<KinematicsSolver::CollisionConstraintResult>
  compute_collision_constraint(KinematicsSolver &solver) {
    return solver.compute_collision_constraint();
  }
};

} // namespace embodik::detail
