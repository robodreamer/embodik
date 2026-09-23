#pragma once

#include "acceleration_braking_witness.hpp"
#include "acceleration_state_box.hpp"
#include "collision_constraint_differential.hpp"

#include <embodik/acceleration_solver.hpp>

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace embodik::detail {

struct CompiledAnalyticCollisionConstraint {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<std::size_t> pair_indices;
  std::vector<double> minimum_distances;
  std::vector<CollisionGeometryPair> active_pairs;

  bool satisfied() const {
    return status == SolverStatus::kSuccess && !pair_indices.empty() &&
           pair_indices.size() == minimum_distances.size() &&
           pair_indices.size() == active_pairs.size();
  }
};

struct PreparedAnalyticCollisionConstraint {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  AffineAccelerationConstraint physical_constraint;
  std::vector<AccelerationBrakingWitnessRow> braking_rows;
  std::vector<AccelerationAnalyticCollisionPairDiagnostics> diagnostics;
  std::vector<CollisionPairDifferential> pair_differentials;
  std::uint64_t pair_evaluations = 0;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
  bool has_rows() const {
    return physical_constraint.coefficient_matrix.rows() > 0;
  }
};

struct AnalyticCollisionStepCertificate {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AccelerationAnalyticCollisionPairDiagnostics> diagnostics;
  std::uint64_t pair_evaluations = 0;
  std::uint64_t path_visited_nodes = 0;
  std::uint64_t certified_intervals = 0;
  bool endpoint_validated = false;
  bool step_certified = false;

  bool satisfied() const {
    return status == SolverStatus::kSuccess && endpoint_validated &&
           step_certified;
  }
};

CompiledAnalyticCollisionConstraint compile_analytic_collision_constraint(
    const RobotModel &robot, const CollisionConstraintDefinition &definition,
    const CollisionConstraintAccelerationPolicy &policy);

PreparedAnalyticCollisionConstraint prepare_analytic_collision_constraint(
    const RobotModel &robot, CollisionDifferentialScratch &scratch,
    const CompiledAnalyticCollisionConstraint &compiled,
    const CollisionConstraintAccelerationPolicy &policy,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq, double dt,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    const std::string &state_label);

AnalyticCollisionStepCertificate certify_analytic_collision_step(
    const RobotModel &robot, CollisionDifferentialScratch &scratch,
    const CompiledAnalyticCollisionConstraint &compiled,
    const CollisionConstraintAccelerationPolicy &policy,
    const PreparedAnalyticCollisionConstraint &current,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &accepted_acceleration, double dt,
    const Eigen::VectorXd &q_next, const Eigen::VectorXd &dq_next,
    const Eigen::VectorXd &predicted_joint_acceleration_lower,
    const Eigen::VectorXd &predicted_joint_acceleration_upper);

} // namespace embodik::detail
