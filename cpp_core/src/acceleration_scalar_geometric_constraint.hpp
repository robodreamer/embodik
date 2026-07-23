#pragma once

#include <embodik/acceleration_solver.hpp>

#include "acceleration_state_box.hpp"

#include <Eigen/Core>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace embodik::detail {

struct ScalarGeometricConstraintResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

struct ScalarGeometricCoordinateSample {
  Eigen::VectorXd value;
  Eigen::VectorXd rate;
  Eigen::VectorXd acceleration;
  std::optional<Eigen::Matrix3d> so3_rotation;
};

struct ScalarGeometricConstraintSpecification {
  std::string family_label;
  std::string source_id;
  Eigen::Index dimension = 0;
  std::vector<int> active_axes;
  Eigen::VectorXd state_lower_bounds;
  Eigen::VectorXd state_upper_bounds;
  GeometricConstraintAccelerationPolicy policy;
};

struct ScalarGeometricDifferential {
  Eigen::MatrixXd coefficient_matrix;
  Eigen::VectorXd affine_bias;
  ScalarGeometricCoordinateSample coordinates;
};

struct PreparedScalarGeometricConstraint {
  ScalarGeometricConstraintSpecification specification;
  LinearizedStateBoxResult state_box;
};

struct ScalarGeometricPreparationResult : ScalarGeometricConstraintResult {
  std::optional<PreparedScalarGeometricConstraint> prepared;
};

struct ScalarGeometricPathValidationOptions {
  int minimum_segments = 16;
  int maximum_segments = 512;
  double maximum_joint_tangent_step = 2e-3;
};

struct ScalarGeometricPathPoint {
  int segment_index = 0;
  int segment_count = 0;
  double time = 0.0;
  Eigen::VectorXd tangent;
  Eigen::VectorXd rate;
  ScalarGeometricCoordinateSample coordinates;
};

struct ScalarGeometricPathSegment {
  ScalarGeometricPathPoint previous;
  ScalarGeometricPathPoint current;
};

using ScalarGeometricPathSampleEvaluator =
    std::function<ScalarGeometricConstraintResult(
        const ScalarGeometricPathPoint &path_point,
        ScalarGeometricCoordinateSample *coordinates)>;

using ScalarGeometricPathSegmentValidator =
    std::function<ScalarGeometricConstraintResult(
        const ScalarGeometricPathSegment &segment)>;

ScalarGeometricConstraintResult validate_scalar_geometric_differential(
    const ScalarGeometricConstraintSpecification &specification,
    const ScalarGeometricDifferential &differential,
    Eigen::Index variable_count);

ScalarGeometricPreparationResult prepare_scalar_geometric_constraint(
    const ScalarGeometricConstraintSpecification &specification,
    const ScalarGeometricDifferential &differential, double dt,
    Eigen::Index variable_count,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper);

ScalarGeometricConstraintResult validate_scalar_geometric_joint_box_support(
    const ScalarGeometricConstraintSpecification &specification,
    const ScalarGeometricDifferential &differential,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    const std::string &state_label);

ScalarGeometricConstraintResult validate_scalar_geometric_linearized_acceptance(
    const PreparedScalarGeometricConstraint &prepared,
    const Eigen::VectorXd &accepted_acceleration, double dt);

ScalarGeometricConstraintResult validate_scalar_geometric_sample_acceptance(
    const ScalarGeometricConstraintSpecification &specification,
    const ScalarGeometricCoordinateSample &sample, bool endpoint);

int scalar_geometric_path_segment_count(
    const Eigen::VectorXd &dq, const Eigen::VectorXd &ddq, double dt,
    const ScalarGeometricConstraintSpecification &specification,
    const ScalarGeometricPathValidationOptions &options,
    ScalarGeometricConstraintResult *failure_result);

ScalarGeometricConstraintResult validate_scalar_geometric_path(
    const ScalarGeometricConstraintSpecification &specification,
    const Eigen::VectorXd &dq, const Eigen::VectorXd &ddq, double dt,
    const ScalarGeometricPathValidationOptions &options,
    const ScalarGeometricPathSampleEvaluator &sample_evaluator,
    const ScalarGeometricPathSegmentValidator &segment_validator = {});

ScalarGeometricConstraintResult validate_so3_log_rotation_segment(
    const ScalarGeometricPathSegment &segment);

} // namespace embodik::detail
