#pragma once

#include <embodik/kinematics_solver.hpp>

#include <Eigen/Dense>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace embodik::detail {

struct VelocityCollisionConstraintRows {
  Eigen::MatrixXd jacobian;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
  double distance = std::numeric_limits<double>::infinity();
  std::string object_a;
  std::string object_b;
  Eigen::Vector3d point_a_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d point_b_world = Eigen::Vector3d::Zero();
};

struct VelocityCollisionConstraintLinearization {
  Eigen::MatrixXd coefficient_matrix;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
  double dt = 0.0;
  double distance = std::numeric_limits<double>::infinity();
  std::string object_a;
  std::string object_b;
  Eigen::Vector3d point_a_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d point_b_world = Eigen::Vector3d::Zero();
};

struct VelocityCollisionConstraintAccountingSnapshot {
  std::uint64_t pairs_considered = 0;
  std::uint64_t exact_distance_queries = 0;
};

struct VelocityCollisionSampleValidationResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  bool acceptable = false;
  std::uint64_t samples_checked = 0;
  std::uint64_t allowed_pair_count = 0;
  std::uint64_t pairs_checked = 0;
  std::uint64_t exact_distance_queries = 0;
  std::uint64_t initial_exact_distance_queries = 0;
  std::uint64_t sample_exact_distance_queries = 0;
  std::uint64_t conservative_bound_checks = 0;
  std::uint64_t conservative_bound_certified_pairs = 0;
  std::uint64_t kinematics_updates = 0;
  std::uint64_t geometry_updates = 0;
  bool initial_state_certificate_reused = false;
  std::uint64_t failed_sample_index = 0;
  std::uint64_t failed_pair_catalog_index =
      std::numeric_limits<std::uint64_t>::max();
  std::uint64_t failed_pair_model_index =
      std::numeric_limits<std::uint64_t>::max();
  std::string failed_pair_key;
};

class VelocityCollisionConstraintProvider {
public:
  explicit VelocityCollisionConstraintProvider(KinematicsSolver &solver)
      : solver_(solver) {}

  bool has_enabled_collision_constraint() const {
    return solver_.collision_constraint_.has_value() &&
           solver_.collision_constraint_->enabled;
  }

  std::optional<VelocityCollisionConstraintRows> compute() {
    return convert(solver_.compute_collision_constraint());
  }

  std::optional<VelocityCollisionConstraintRows> compute(double row_dt) {
    return convert(solver_.compute_collision_constraint(row_dt));
  }

  std::optional<VelocityCollisionConstraintLinearization>
  linearize(double row_dt) {
    const auto linearization =
        solver_.linearize_collision_velocity_constraint(row_dt);
    if (!linearization.has_value()) {
      return std::nullopt;
    }
    VelocityCollisionConstraintLinearization out;
    out.coefficient_matrix = linearization->coefficient_matrix;
    out.lower_bounds = linearization->lower_bounds;
    out.upper_bounds = linearization->upper_bounds;
    out.dt = linearization->dt;
    out.distance = linearization->distance;
    out.object_a = linearization->object_a;
    out.object_b = linearization->object_b;
    out.point_a_world = linearization->point_a_world;
    out.point_b_world = linearization->point_b_world;
    return out;
  }

  VelocityCollisionConstraintAccountingSnapshot accounting_snapshot() const {
    VelocityCollisionConstraintAccountingSnapshot snapshot;
    snapshot.pairs_considered = solver_.last_collision_pairs_considered_;
    snapshot.exact_distance_queries =
        solver_.last_collision_exact_distance_queries_;
    return snapshot;
  }

  VelocityCollisionSampleValidationResult validate_samples(
      const Eigen::VectorXd &q_from,
      const std::vector<Eigen::VectorXd> &q_samples) {
    const auto validation =
        solver_.validate_collision_samples(q_from, q_samples);
    VelocityCollisionSampleValidationResult out;
    out.status = validation.status;
    out.message = validation.message;
    out.acceptable = validation.acceptable;
    out.samples_checked = validation.samples_checked;
    out.allowed_pair_count = validation.allowed_pair_count;
    out.pairs_checked = validation.pairs_checked;
    out.exact_distance_queries = validation.exact_distance_queries;
    out.initial_exact_distance_queries =
        validation.initial_exact_distance_queries;
    out.sample_exact_distance_queries =
        validation.sample_exact_distance_queries;
    out.conservative_bound_checks = validation.conservative_bound_checks;
    out.conservative_bound_certified_pairs =
        validation.conservative_bound_certified_pairs;
    out.kinematics_updates = validation.kinematics_updates;
    out.geometry_updates = validation.geometry_updates;
    out.initial_state_certificate_reused =
        validation.initial_state_certificate_reused;
    out.failed_sample_index = validation.failed_sample_index;
    out.failed_pair_catalog_index = validation.failed_pair_catalog_index;
    out.failed_pair_model_index = validation.failed_pair_model_index;
    out.failed_pair_key = validation.failed_pair_key;
    return out;
  }

private:
  static std::optional<VelocityCollisionConstraintRows> convert(
      const std::optional<KinematicsSolver::CollisionConstraintResult> &rows) {
    if (!rows.has_value()) {
      return std::nullopt;
    }
    VelocityCollisionConstraintRows out;
    out.jacobian = rows->jacobian;
    out.lower_bounds = rows->lower_bounds;
    out.upper_bounds = rows->upper_bounds;
    out.distance = rows->distance;
    out.object_a = rows->object_a;
    out.object_b = rows->object_b;
    out.point_a_world = rows->point_a_world;
    out.point_b_world = rows->point_b_world;
    return out;
  }

  KinematicsSolver &solver_;
};

} // namespace embodik::detail
