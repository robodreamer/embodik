#pragma once

#include "velocity_collision_constraint_provider.hpp"
#include <embodik/acceleration_solver.hpp>

namespace embodik::acceleration_solver_internal {
struct VelocityCollisionLiftDiagnostics;
}

namespace embodik::detail {

// Source-private local model; candidate acceptance remains independently checked.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
bool make_sample_distance_cut(
    const RobotModel &robot,
    const VelocityCollisionSampleValidationResult &validation,
    const Eigen::VectorXd &sample_q, const Eigen::VectorXd &anchor_ddq,
    double sample_time, double reserve, AffineAccelerationConstraint *cut,
    acceleration_solver_internal::VelocityCollisionLiftDiagnostics *diagnostics);

} // namespace embodik::detail
