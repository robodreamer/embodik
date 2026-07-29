#pragma once

#include <embodik/acceleration_solver.hpp>

#include "acceleration_analytic_collision.hpp"
#include "acceleration_allocation_transform.hpp"
#include "acceleration_com_support_polygon_constraint.hpp"
#include "acceleration_fixed_frame_pose_constraint.hpp"
#include "acceleration_relative_pose_constraint.hpp"
#include "acceleration_state_box.hpp"
#include "acceleration_task_differential.hpp"
#include "acceleration_tight_point_constraint.hpp"
#include "frame_kinematic_differential.hpp"
#include "generalized_constraint_set.hpp"
#include "velocity_collision_constraint_provider.hpp"

#include <embodik/ik_baseline.hpp>
#include <embodik/kinematics_solver.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif

namespace embodik {
namespace acceleration_solver_internal {

inline constexpr double kConstraintTolerance = 1e-8;
inline constexpr double kInactiveAffineBackendBound = 1e10;

AccelerationSolverResult failure(SolverStatus status, std::string message);
bool finite_positive_vector(const Eigen::VectorXd &values);
bool fixed_base_scalar_joints_only(const RobotModel &robot);
bool native_collision_options_are_supported(
    const AccelerationSolveOptions &options, std::string *message);
detail::CompiledAnalyticCollisionConstraint make_stored_collision_compilation(
    const std::vector<std::size_t> &pair_indices,
    const std::vector<double> &minimum_distances,
    const std::vector<CollisionGeometryPair> &active_pairs);

template <typename Constraint>
const char *constraint_family_name(const Constraint &) {
  return "constraint";
}

template <>
inline const char *
constraint_family_name<AffineAccelerationConstraint>(
    const AffineAccelerationConstraint &) {
  return "affine acceleration constraint";
}

template <>
inline const char *
constraint_family_name<FrozenNextVelocityConstraint>(
    const FrozenNextVelocityConstraint &) {
  return "frozen next-velocity constraint";
}

inline const char *constraint_family_name(const TaskAccelerationBounds &) {
  return "task acceleration bounds";
}
inline const char *
constraint_family_name(const CentroidalMomentumRateBounds &) {
  return "centroidal momentum-rate bounds";
}
inline const char *constraint_family_name(const ContactAccelerationConstraint &) {
  return "contact acceleration constraint";
}
inline const char *constraint_family_name(const TightPointAccelerationConstraint &) {
  return "tight point constraint";
}
inline const char *constraint_family_name(const TightFramePoseAccelerationConstraint &) {
  return "tight frame pose constraint";
}
inline const char *constraint_family_name(const RelativePoseAccelerationConstraint &) {
  return "relative pose constraint";
}
inline const char *constraint_family_name(const TorsoPoseBoundAccelerationConstraint &) {
  return "torso pose bound constraint";
}
inline const char *constraint_family_name(const ComSupportPolygonAccelerationConstraint &) {
  return "CoM support-polygon constraint";
}

template <typename Constraint>
bool lower_side_active(const Constraint &constraint, Eigen::Index row) {
  return constraint.lower_bound_active.empty() ||
         constraint.lower_bound_active[static_cast<std::size_t>(row)];
}

template <typename Constraint>
bool upper_side_active(const Constraint &constraint, Eigen::Index row) {
  return constraint.upper_bound_active.empty() ||
         constraint.upper_bound_active[static_cast<std::size_t>(row)];
}

template <typename Constraint>
Eigen::Index count_constraint_rows(const std::vector<Constraint> &constraints) {
  Eigen::Index rows = 0;
  for (const auto &constraint : constraints) {
    rows += constraint.coefficient_matrix.rows();
  }
  return rows;
}

bool inactive_bound_magnitude(const Eigen::Ref<const Eigen::RowVectorXd> &row, double *magnitude);
template <typename Constraint>
bool inactive_affine_bound_magnitude(const Constraint &constraint, Eigen::Index row, double *magnitude) {
  return inactive_bound_magnitude(constraint.coefficient_matrix.row(row), magnitude);
}
bool checked_product(double lhs, double rhs, double *out);
bool checked_quotient(double numerator, double denominator, double *out);
bool checked_add(double lhs, double rhs, double *out);
bool checked_scaled_matrix(const Eigen::MatrixXd &matrix, double scale, Eigen::MatrixXd *scaled);
bool checked_dot_row(const Eigen::Ref<const Eigen::RowVectorXd> &row, const Eigen::VectorXd &vector, double *out);
bool checked_shifted_bound(double bound, double bias);

struct ConstraintValidation {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::Index row_count = 0;
};

struct PreparedEffortConstraint {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::MatrixXd mass_matrix;
  Eigen::VectorXd bias;
  Eigen::VectorXd limits;
};

struct InverseDynamicsEvaluation {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::VectorXd torques;
};

struct ContactConstraintAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> constraints;
  Eigen::Index row_count = 0;
};

struct TightPointConstraintAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> constraints;
  std::vector<detail::PreparedTightPointConstraint> prepared_constraints;
  Eigen::Index row_count = 0;
};

struct FixedFramePoseConstraintAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> constraints;
  std::vector<detail::PreparedFixedFramePoseConstraint> prepared_constraints;
  Eigen::Index row_count = 0;
};

struct RelativePoseConstraintAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> constraints;
  std::vector<detail::PreparedRelativePoseConstraint> prepared_constraints;
  Eigen::Index row_count = 0;
};

struct ComSupportPolygonConstraintAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> constraints;
  std::vector<detail::PreparedComSupportPolygonConstraint>
      prepared_constraints;
  Eigen::Index row_count = 0;
};

struct CentroidalMomentumRateObjectiveDiagnosticRecord {
  std::string source_id;
  Eigen::VectorXd target_momentum;
  Eigen::VectorXd reference_momentum_rate;
  Eigen::VectorXd current_momentum;
  Eigen::VectorXd bias_momentum_rate;
  std::vector<bool> selected_axes;
  std::size_t objective_index = 0;
};

struct CentroidalMomentumRateObjectiveAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<CentroidalMomentumRateObjectiveDiagnosticRecord> diagnostics;
};

struct CentroidalMomentumRateBoundsAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> constraints;
  Eigen::Matrix<double, 6, Eigen::Dynamic> centroidal_matrix;
  Eigen::Matrix<double, 6, 1> bias;
  Eigen::Index row_count = 0;
};


VelocitySolverConfig acceleration_backend_config();
Eigen::VectorXd resolve_acceleration_limits(const RobotModel &robot, const AccelerationSolveOptions &options, SolverStatus *status, std::string *message);

struct StateBoxAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::VectorXd lower;
  Eigen::VectorXd upper;
  std::vector<detail::StateBoxCauseSet> lower_causes;
  std::vector<detail::StateBoxCauseSet> upper_causes;
};

struct ObjectiveAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<Eigen::MatrixXd> matrices;
  std::vector<Eigen::VectorXd> targets;
  std::vector<Eigen::VectorXd> biases;
  std::vector<ObjectiveSolveConfig> configs;
  std::vector<std::vector<std::shared_ptr<Task>>> groups;
  std::vector<std::vector<detail::AccelerationTaskDifferential>> differentials;
  std::vector<std::vector<Eigen::VectorXd>> references;
  bool synthetic_hold_objective = false;
};


StateBoxAssembly build_joint_state_box(const RobotModel &robot, const Eigen::VectorXd &q, const Eigen::VectorXd &dq, double dt, const Eigen::VectorXd &accel_limits, const AccelerationSolveOptions &options);
void assemble_objectives(ObjectiveAssembly *assembly, const std::vector<std::shared_ptr<Task>> &tasks, std::vector<std::shared_ptr<Task>> &ordered_tasks, const std::unordered_map<std::string, AccelerationTaskReference> &references, const RobotModel &robot, const Eigen::VectorXd &dq, bool retain_task_differentials);
CentroidalMomentumRateObjectiveAssembly append_centroidal_momentum_rate_objectives(ObjectiveAssembly *objectives, const std::vector<CentroidalMomentumRateObjective> &centroidal_objectives, const RobotModel &robot, std::unordered_set<std::string> *source_ids);
const detail::AccelerationTaskDifferential *find_active_task_differential(const ObjectiveAssembly &objectives, const std::string &task_name);
ConstraintValidation validate_acceleration_constraints(const std::vector<AffineAccelerationConstraint> &affine_constraints, const std::vector<FrozenNextVelocityConstraint> &frozen_constraints, Eigen::Index variable_count, std::unordered_set<std::string> *source_ids);
ConstraintValidation validate_task_acceleration_bounds(const std::vector<TaskAccelerationBounds> &bounds, const ObjectiveAssembly &objectives, std::unordered_set<std::string> *source_ids);
std::vector<AffineAccelerationConstraint> make_task_bound_affine_constraints(const std::vector<TaskAccelerationBounds> &bounds, const ObjectiveAssembly &objectives);
CentroidalMomentumRateBoundsAssembly make_centroidal_momentum_rate_bounds(const RobotModel &robot, const std::vector<CentroidalMomentumRateBounds> &bounds, std::unordered_set<std::string> *source_ids);
ContactConstraintAssembly make_contact_acceleration_constraints(const RobotModel &robot, const std::vector<ContactAccelerationConstraint> &contacts, std::unordered_set<std::string> *source_ids);
TightPointConstraintAssembly make_tight_point_constraints(const RobotModel &robot, const std::vector<TightPointAccelerationConstraint> &tight_points, double dt, const Eigen::VectorXd &joint_acceleration_lower, const Eigen::VectorXd &joint_acceleration_upper, std::unordered_set<std::string> *source_ids);

template <typename Constraint, typename RecordMaker>
FixedFramePoseConstraintAssembly make_fixed_frame_pose_constraints(
    const RobotModel &robot, const std::vector<Constraint> &pose_constraints,
    double dt, const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    std::unordered_set<std::string> *source_ids, RecordMaker make_record) {
  FixedFramePoseConstraintAssembly assembly;
  assembly.constraints.reserve(pose_constraints.size());
  assembly.prepared_constraints.reserve(pose_constraints.size());

  for (const auto &pose_constraint : pose_constraints) {
    const std::string family = constraint_family_name(pose_constraint);
    if (pose_constraint.source_id.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " source_id must not be empty";
      return assembly;
    }
    if (!source_ids->insert(pose_constraint.source_id).second) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = "duplicate acceleration constraint source_id '" +
                         pose_constraint.source_id + "'";
      return assembly;
    }

    detail::FixedFramePoseConstraintRecord record;
    const auto record_result = make_record(pose_constraint, &record);
    if (!record_result.satisfied()) {
      assembly.status = record_result.status;
      assembly.message = record_result.message;
      return assembly;
    }
    const auto prepared = detail::prepare_fixed_frame_pose_constraint(
        record, robot, dt, joint_acceleration_lower,
        joint_acceleration_upper);
    if (!prepared.satisfied()) {
      assembly.status = prepared.status;
      assembly.message = prepared.message;
      return assembly;
    }
    assembly.row_count +=
        prepared.prepared->scalar.state_box.physical_constraint
            .coefficient_matrix
            .rows();
    assembly.constraints.push_back(
        prepared.prepared->scalar.state_box.physical_constraint);
    assembly.prepared_constraints.push_back(std::move(*prepared.prepared));
  }

  return assembly;
}

RelativePoseConstraintAssembly make_relative_pose_constraints(const RobotModel &robot, const std::vector<RelativePoseAccelerationConstraint> &relative_poses, double dt, const Eigen::VectorXd &joint_acceleration_lower, const Eigen::VectorXd &joint_acceleration_upper, std::unordered_set<std::string> *source_ids);
ComSupportPolygonConstraintAssembly make_com_support_polygon_constraints(const RobotModel &robot, const std::vector<ComSupportPolygonAccelerationConstraint> &polygons, double dt, const Eigen::VectorXd &joint_acceleration_lower, const Eigen::VectorXd &joint_acceleration_upper, std::unordered_set<std::string> *source_ids);
Eigen::VectorXd vector_from_solution(const SolverResult &result);
void assign_solution_from_vector(SolverResult *result, const Eigen::VectorXd &solution);
bool zero_excluded_by_bounds(const Eigen::VectorXd &lower, const Eigen::VectorXd &upper);
bool populate_allocation_diagnostics(AccelerationSolverResult *result, const detail::PreparedAccelerationAllocationTransform &allocation, const Eigen::VectorXd &physical_acceleration);
PreparedEffortConstraint prepare_effort_constraint(const RobotModel &robot, const Eigen::VectorXd &q, const Eigen::VectorXd &dq, const EffortConstraintOptions &options);
InverseDynamicsEvaluation evaluate_inverse_dynamics(const RobotModel &robot, const Eigen::VectorXd &q, const Eigen::VectorXd &dq, const Eigen::VectorXd &ddq);
AccelerationSolverResult clear_outputs(AccelerationSolverResult result);
void append_unique(std::vector<int> *indices, int index);
void attribute_saturation(AccelerationSolverResult *result, const StateBoxAssembly &state_box, const Eigen::VectorXd &ddq);

struct LockRows {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::MatrixXd coefficients;
  Eigen::VectorXd bounds;
};


bool insert_lock_indices(const std::vector<int> &indices, const std::string &field_name, Eigen::Index nv, std::unordered_set<int> *seen, SolverStatus *status, std::string *message);
LockRows build_lock_rows(const AccelerationSolveOptions &options, const Eigen::VectorXd &dq, double dt);
detail::GeneralizedConstraintBlock make_affine_constraint_block(const std::vector<AffineAccelerationConstraint> &constraints, Eigen::Index variable_count, Eigen::Index row_count);
detail::GeneralizedConstraintBlock make_lock_constraint_block(const LockRows &locks, Eigen::Index variable_count);
detail::GeneralizedConstraintBlock make_effort_constraint_block(const PreparedEffortConstraint &effort, Eigen::Index variable_count);

struct FrozenTransform {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> transformed_constraints;
  Eigen::Index row_count = 0;
};


FrozenTransform transform_frozen_constraints(const std::vector<FrozenNextVelocityConstraint> &constraints, const Eigen::VectorXd &dq, double dt);
bool accepted_affine_constraints_are_satisfied(const std::vector<AffineAccelerationConstraint> &constraints, const Eigen::VectorXd &ddq);
bool accepted_centroidal_momentum_rate_bounds_are_satisfied(const std::vector<AffineAccelerationConstraint> &constraints, const Eigen::VectorXd &ddq);
bool accepted_frozen_constraints_are_satisfied(const std::vector<FrozenNextVelocityConstraint> &constraints, const Eigen::VectorXd &dq, double dt, const Eigen::VectorXd &ddq);
bool accepted_lock_constraints_are_satisfied(const AccelerationSolveOptions &options, const Eigen::VectorXd &dq, double dt, const Eigen::VectorXd &ddq);

struct VelocityCollisionLiftDiagnostics {
  bool applied = false;
  bool endpoint_validated = false;
  std::uint64_t validation_samples = 0;
  std::uint64_t validation_allowed_pairs = 0;
  std::uint64_t validation_pairs_checked = 0;
  std::uint64_t validation_exact_queries = 0;
  std::uint64_t validation_initial_exact_queries = 0;
  std::uint64_t validation_sample_exact_queries = 0;
  std::uint64_t validation_conservative_checks = 0;
  std::uint64_t validation_conservative_certified_pairs = 0;
  std::uint64_t validation_kinematics_updates = 0;
  std::uint64_t validation_geometry_updates = 0;
  bool validation_initial_certificate_reused = false;
  std::uint64_t pairs_considered = 0;
  std::uint64_t row_pairs = 0;
  std::uint64_t row_exact_queries = 0;
};


void apply_velocity_collision_lift_diagnostics(AccelerationSolverResult *result, const VelocityCollisionLiftDiagnostics &diagnostics);
AccelerationSolverResult velocity_collision_lift_failure(SolverStatus status, const std::string &message, const VelocityCollisionLiftDiagnostics &diagnostics);

} // namespace acceleration_solver_internal

namespace detail {
#if defined(__GNUC__) || defined(__clang__)
struct __attribute__((visibility("hidden"))) AccelerationSolverWorkspace {
#else
struct AccelerationSolverWorkspace {
#endif
  acceleration_solver_internal::ObjectiveAssembly objectives;
  Eigen::MatrixXd state_box_identity;
};
} // namespace detail
} // namespace embodik

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
