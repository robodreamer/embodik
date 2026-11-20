#include "acceleration_affine_state_constraint.hpp"

#include <cmath>
#include <numeric>

namespace embodik::detail {

AffineStatePreparationResult prepare_affine_state_constraint(
    const AffineStateConstraint &constraint, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq, double dt,
    const Eigen::VectorXd &acceleration_lower,
    const Eigen::VectorXd &acceleration_upper) {
  AffineStatePreparationResult out;
  const auto fail = [&](SolverStatus status, const std::string &message) {
    out.status = status;
    out.message = "affine state constraint '" + constraint.source_id + "': " + message;
    return out;
  };
  if (robot.is_floating_base() || robot.nq() != robot.nv())
    return fail(SolverStatus::kInvalidInput, "requires fixed-base additive scalar joints");
  for (pinocchio::JointIndex index = 1; index < robot.model().joints.size(); ++index) {
    const auto &joint = robot.model().joints[index];
    if (joint.nq() != 1 || joint.nv() != 1)
      return fail(SolverStatus::kInvalidInput, "requires fixed-base additive scalar joints");
  }
  const auto rows = constraint.coefficient_matrix.rows();
  if (rows <= 0 || constraint.coefficient_matrix.cols() != robot.nv() ||
      constraint.offset.size() != rows || q.size() != robot.nq() || dq.size() != robot.nv())
    return fail(SolverStatus::kShapeMismatch, "matrix, offset or state dimensions differ");
  if (!constraint.coefficient_matrix.allFinite() || !constraint.offset.allFinite() ||
      !q.allFinite() || !dq.allFinite())
    return fail(SolverStatus::kNonFiniteInput, "matrix, offset and states must be finite");
  ScalarGeometricConstraintSpecification specification;
  specification.family_label = "affine state constraint";
  specification.source_id = constraint.source_id;
  specification.dimension = rows;
  specification.active_axes.resize(static_cast<std::size_t>(rows));
  std::iota(specification.active_axes.begin(), specification.active_axes.end(), 0);
  specification.state_lower_bounds = constraint.lower_bounds;
  specification.state_upper_bounds = constraint.upper_bounds;
  specification.policy = constraint.policy;
  ScalarGeometricDifferential differential;
  differential.coefficient_matrix = constraint.coefficient_matrix;
  differential.affine_bias = Eigen::VectorXd::Zero(rows);
  differential.coordinates.value = constraint.coefficient_matrix * q + constraint.offset;
  differential.coordinates.rate = constraint.coefficient_matrix * dq;
  differential.coordinates.acceleration = Eigen::VectorXd::Zero(rows);
  auto prepared = prepare_scalar_geometric_constraint(specification, differential, dt,
      robot.nv(), acceleration_lower, acceleration_upper,
      BrakingSupportRequirement::kReachableBoundary);
  if (!prepared.satisfied()) {
    out.status = prepared.status;
    out.message = prepared.message;
    return out;
  }
  out.prepared = PreparedAffineStateConstraint{constraint, differential, std::move(*prepared.prepared)};
  return out;
}

ScalarGeometricConstraintResult validate_affine_state_constraint_acceptance(
    const PreparedAffineStateConstraint &prepared,
    const Eigen::VectorXd &ddq, double dt,
    const Eigen::VectorXd &q_next, const Eigen::VectorXd &dq_next,
    const Eigen::VectorXd &predicted_acceleration_lower,
    const Eigen::VectorXd &predicted_acceleration_upper) {
  const auto linearized = validate_scalar_geometric_linearized_acceptance(prepared.scalar, ddq, dt);
  if (!linearized.satisfied()) return linearized;
  const auto &initial = prepared.current.coordinates;
  const Eigen::VectorXd acceleration = prepared.definition.coefficient_matrix * ddq;
  const auto sample = [&](double time) {
    ScalarGeometricCoordinateSample point;
    point.value = initial.value + time * initial.rate + 0.5 * time * time * acceleration;
    point.rate = initial.rate + time * acceleration;
    point.acceleration = acceleration;
    return point;
  };
  // Endpoints plus every coordinate's quadratic stationary point bound the
  // whole affine path. Rates are linear and accelerations are constant.
  auto checked = validate_scalar_geometric_sample_acceptance(prepared.scalar.specification, sample(0.0), false);
  if (!checked.satisfied()) return checked;
  for (Eigen::Index axis = 0; axis < acceleration.size(); ++axis) {
    if (acceleration(axis) == 0.0) continue;
    const double extremum = -initial.rate(axis) / acceleration(axis);
    if (extremum > 0.0 && extremum < dt) {
      checked = validate_scalar_geometric_sample_acceptance(prepared.scalar.specification, sample(extremum), false);
      if (!checked.satisfied()) return checked;
    }
  }
  auto predicted = prepared.current;
  predicted.coordinates = sample(dt);
  checked = validate_scalar_geometric_sample_acceptance(prepared.scalar.specification, predicted.coordinates, true);
  if (!checked.satisfied()) return checked;
  if (q_next.size() != prepared.definition.coefficient_matrix.cols() ||
      dq_next.size() != q_next.size() || !q_next.allFinite() || !dq_next.allFinite())
    return {SolverStatus::kNumericalError, "affine state integrated endpoint is invalid"};
  // Re-evaluate the actual published endpoint: distributing C over joint
  // integration is not identical under floating-point cancellation.
  predicted.coordinates.value = prepared.definition.coefficient_matrix * q_next + prepared.definition.offset;
  predicted.coordinates.rate = prepared.definition.coefficient_matrix * dq_next;
  checked = validate_scalar_geometric_sample_acceptance(prepared.scalar.specification, predicted.coordinates, true);
  if (!checked.satisfied()) return checked;
  auto support = validate_scalar_geometric_joint_box_support(
      prepared.scalar.specification, predicted, predicted_acceleration_lower,
      predicted_acceleration_upper, "predicted next-state",
      BrakingSupportRequirement::kReachableBoundary, dt);
  if (support.status == SolverStatus::kInfeasible) {
    // This selected candidate lacks the requested next-state support. Another
    // candidate can still satisfy the original problem; no infeasibility proof.
    support.status = SolverStatus::kNoProgress;
  }
  return support;
}

} // namespace embodik::detail
