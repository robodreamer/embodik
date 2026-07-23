#include "acceleration_scalar_geometric_constraint.hpp"

#include "acceleration_geometric_constraint_policy.hpp"
#include "so3_log_differential.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <pinocchio/spatial/explog.hpp>
#include <unordered_set>
#include <utility>

namespace embodik::detail {
namespace {

constexpr double kConstraintTolerance = 1e-8;

double comparison_tolerance(double lhs, double rhs) {
  return std::max(kConstraintTolerance,
                  1e-12 * (1.0 + std::max(std::abs(lhs), std::abs(rhs))));
}

ScalarGeometricConstraintResult failure(SolverStatus status,
                                        std::string message) {
  ScalarGeometricConstraintResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

std::string constraint_identity(
    const ScalarGeometricConstraintSpecification &specification) {
  return specification.family_label + " '" + specification.source_id + "'";
}

std::string accepted_constraint_identity(
    const ScalarGeometricConstraintSpecification &specification) {
  return "accepted " + specification.family_label + " '" +
         specification.source_id + "'";
}

bool active_axis_is_valid(int axis, Eigen::Index dimension) {
  return axis >= 0 && static_cast<Eigen::Index>(axis) < dimension;
}

ScalarGeometricConstraintResult validate_scalar_geometric_specification(
    const ScalarGeometricConstraintSpecification &specification) {
  if (specification.family_label.empty() || specification.source_id.empty() ||
      specification.dimension <= 0) {
    return failure(SolverStatus::kInvalidInput,
                   "scalar geometric constraint identity is invalid");
  }
  if (specification.state_lower_bounds.size() != specification.dimension ||
      specification.state_upper_bounds.size() != specification.dimension) {
    return failure(SolverStatus::kShapeMismatch,
                   constraint_identity(specification) +
                       " state bounds have inconsistent dimensions");
  }
  if (!specification.state_lower_bounds.allFinite() ||
      !specification.state_upper_bounds.allFinite()) {
    return failure(SolverStatus::kNonFiniteInput,
                   constraint_identity(specification) +
                       " state bounds must contain only finite values");
  }
  if ((specification.state_lower_bounds.array() >
       specification.state_upper_bounds.array())
          .any()) {
    return failure(SolverStatus::kInvalidInput,
                   constraint_identity(specification) +
                       " state lower bounds exceed upper bounds");
  }
  if (specification.active_axes.empty()) {
    return failure(SolverStatus::kInvalidInput,
                   constraint_identity(specification) +
                       " must activate at least one axis");
  }
  std::unordered_set<int> unique_axes;
  for (int axis : specification.active_axes) {
    if (!active_axis_is_valid(axis, specification.dimension)) {
      return failure(SolverStatus::kShapeMismatch,
                     constraint_identity(specification) +
                         " active axes are inconsistent");
    }
    if (!unique_axes.insert(axis).second) {
      return failure(SolverStatus::kInvalidInput,
                     constraint_identity(specification) +
                         " active axes must not contain duplicates");
    }
  }

  const auto policy = validate_geometric_constraint_policy(
      specification.family_label, specification.source_id,
      specification.policy, specification.dimension);
  if (!policy.satisfied()) {
    return failure(policy.status, policy.message);
  }
  return {};
}

} // namespace

ScalarGeometricConstraintResult validate_scalar_geometric_differential(
    const ScalarGeometricConstraintSpecification &specification,
    const ScalarGeometricDifferential &differential,
    Eigen::Index variable_count) {
  const auto specification_validation =
      validate_scalar_geometric_specification(specification);
  if (!specification_validation.satisfied()) {
    return specification_validation;
  }
  if (variable_count <= 0 ||
      differential.coefficient_matrix.rows() != specification.dimension ||
      differential.coefficient_matrix.cols() != variable_count ||
      differential.affine_bias.size() != specification.dimension ||
      differential.coordinates.value.size() != specification.dimension ||
      differential.coordinates.rate.size() != specification.dimension ||
      differential.coordinates.acceleration.size() !=
          specification.dimension) {
    return failure(SolverStatus::kShapeMismatch,
                   constraint_identity(specification) +
                       " differential dimensions are inconsistent");
  }
  if (!differential.coefficient_matrix.allFinite() ||
      !differential.affine_bias.allFinite() ||
      !differential.coordinates.value.allFinite() ||
      !differential.coordinates.rate.allFinite() ||
      !differential.coordinates.acceleration.allFinite()) {
    return failure(SolverStatus::kNumericalError,
                   constraint_identity(specification) +
                       " differential must contain only finite values");
  }
  return {};
}

ScalarGeometricConstraintResult validate_scalar_geometric_joint_box_support(
    const ScalarGeometricConstraintSpecification &specification,
    const ScalarGeometricDifferential &differential,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    const std::string &state_label) {
  const auto specification_validation =
      validate_scalar_geometric_specification(specification);
  if (!specification_validation.satisfied()) {
    return specification_validation;
  }
  if (differential.coefficient_matrix.rows() != specification.dimension ||
      differential.affine_bias.size() != specification.dimension ||
      differential.coefficient_matrix.cols() !=
          joint_acceleration_lower.size() ||
      joint_acceleration_lower.size() != joint_acceleration_upper.size()) {
    return failure(SolverStatus::kShapeMismatch,
                   constraint_identity(specification) +
                       " support-check dimensions are inconsistent");
  }
  if (!differential.coefficient_matrix.allFinite() ||
      !differential.affine_bias.allFinite()) {
    return failure(SolverStatus::kNumericalError,
                   constraint_identity(specification) + " " + state_label +
                       " support differential was non-finite");
  }
  if (!joint_acceleration_lower.allFinite() ||
      !joint_acceleration_upper.allFinite() ||
      (joint_acceleration_lower.array() > joint_acceleration_upper.array())
          .any()) {
    return failure(SolverStatus::kInvalidInput,
                   constraint_identity(specification) + " " + state_label +
                       " joint acceleration support box is invalid");
  }

  for (int axis : specification.active_axes) {
    double minimum = differential.affine_bias(axis);
    double maximum = differential.affine_bias(axis);
    const Eigen::VectorXd row = differential.coefficient_matrix.row(axis);
    for (Eigen::Index index = 0; index < row.size(); ++index) {
      const double coefficient = row(index);
      if (coefficient >= 0.0) {
        minimum += coefficient * joint_acceleration_lower(index);
        maximum += coefficient * joint_acceleration_upper(index);
      } else {
        minimum += coefficient * joint_acceleration_upper(index);
        maximum += coefficient * joint_acceleration_lower(index);
      }
    }
    if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
      return failure(SolverStatus::kNumericalError,
                     constraint_identity(specification) + " " + state_label +
                         " joint-box support was non-finite");
    }

    const double lower_braking =
        specification.policy.lower_braking_accelerations(axis);
    const double upper_braking =
        specification.policy.upper_braking_accelerations(axis);
    if (maximum < lower_braking -
                      comparison_tolerance(maximum, lower_braking)) {
      return failure(SolverStatus::kInfeasible,
                     constraint_identity(specification) + " " + state_label +
                         " axis " + std::to_string(axis) +
                         " lacks lower-side braking support");
    }
    if (minimum > -upper_braking +
                      comparison_tolerance(minimum, -upper_braking)) {
      return failure(SolverStatus::kInfeasible,
                     constraint_identity(specification) + " " + state_label +
                         " axis " + std::to_string(axis) +
                         " lacks upper-side braking support");
    }
  }

  return {};
}

ScalarGeometricPreparationResult prepare_scalar_geometric_constraint(
    const ScalarGeometricConstraintSpecification &specification,
    const ScalarGeometricDifferential &differential, double dt,
    Eigen::Index variable_count,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper) {
  ScalarGeometricPreparationResult result;
  const auto differential_validation =
      validate_scalar_geometric_differential(specification, differential,
                                             variable_count);
  if (!differential_validation.satisfied()) {
    result.status = differential_validation.status;
    result.message = differential_validation.message;
    return result;
  }

  LinearizedStateBoxInput input;
  input.source_id = specification.source_id;
  input.coefficient_matrix.resize(
      static_cast<Eigen::Index>(specification.active_axes.size()),
      variable_count);
  input.affine_bias.resize(
      static_cast<Eigen::Index>(specification.active_axes.size()));
  input.rows.reserve(specification.active_axes.size());
  for (std::size_t row_index = 0; row_index < specification.active_axes.size();
       ++row_index) {
    const int axis = specification.active_axes[row_index];
    input.coefficient_matrix.row(static_cast<Eigen::Index>(row_index)) =
        differential.coefficient_matrix.row(axis);
    input.affine_bias(static_cast<Eigen::Index>(row_index)) =
        differential.affine_bias(axis);

    StateBoxRowInput row;
    row.value = differential.coordinates.value(axis);
    row.rate = differential.coordinates.rate(axis);
    row.state_bounds = {specification.state_lower_bounds(axis),
                        specification.state_upper_bounds(axis), true, true};
    row.rate_bounds = {-specification.policy.rate_limits(axis),
                       specification.policy.rate_limits(axis), true, true};
    row.acceleration_bounds = {-specification.policy.acceleration_limits(axis),
                               specification.policy.acceleration_limits(axis),
                               true, true};
    row.lower_braking_acceleration =
        specification.policy.lower_braking_accelerations(axis);
    row.upper_braking_acceleration =
        specification.policy.upper_braking_accelerations(axis);
    input.rows.push_back(row);
  }

  auto state_box = prepare_linearized_state_box(input, dt, variable_count);
  if (state_box.status != SolverStatus::kSuccess) {
    result.status = state_box.status;
    result.message = constraint_identity(specification) + ": " +
                     state_box.message;
    return result;
  }

  const auto support = validate_scalar_geometric_joint_box_support(
      specification, differential, joint_acceleration_lower,
      joint_acceleration_upper, "current-state");
  if (!support.satisfied()) {
    result.status = support.status;
    result.message = support.message;
    return result;
  }

  PreparedScalarGeometricConstraint prepared;
  prepared.specification = specification;
  prepared.state_box = std::move(state_box);
  result.prepared = std::move(prepared);
  return result;
}

ScalarGeometricConstraintResult validate_scalar_geometric_linearized_acceptance(
    const PreparedScalarGeometricConstraint &prepared,
    const Eigen::VectorXd &accepted_acceleration, double dt) {
  const auto &specification = prepared.specification;
  const auto specification_validation =
      validate_scalar_geometric_specification(specification);
  if (!specification_validation.satisfied()) {
    return specification_validation;
  }
  if (!(dt > 0.0) || !std::isfinite(dt) ||
      !accepted_acceleration.allFinite()) {
    return failure(SolverStatus::kInvalidInput,
                   accepted_constraint_identity(specification) +
                       " validation inputs are invalid");
  }
  const auto &physical_constraint = prepared.state_box.physical_constraint;
  if (physical_constraint.coefficient_matrix.cols() !=
      accepted_acceleration.size()) {
    return failure(SolverStatus::kShapeMismatch,
                   accepted_constraint_identity(specification) +
                       " validation dimensions are inconsistent");
  }
  const Eigen::VectorXd physical =
      physical_constraint.coefficient_matrix * accepted_acceleration +
      physical_constraint.affine_bias;
  if (!physical.allFinite()) {
    return failure(SolverStatus::kNumericalError,
                   accepted_constraint_identity(specification) +
                       " produced non-finite physical acceleration");
  }
  if (physical.size() !=
      static_cast<Eigen::Index>(prepared.state_box.state_rows.size())) {
    return failure(SolverStatus::kShapeMismatch,
                   accepted_constraint_identity(specification) +
                       " validation dimensions are inconsistent");
  }

  for (Eigen::Index row_index = 0; row_index < physical.size(); ++row_index) {
    const auto &row =
        prepared.state_box.state_rows[static_cast<std::size_t>(row_index)];
    const double acceleration = physical(row_index);
    if (acceleration <
        row.acceleration_bounds.lower -
            comparison_tolerance(acceleration, row.acceleration_bounds.lower)) {
      return failure(SolverStatus::kNumericalError,
                     accepted_constraint_identity(specification) +
                         " violated linearized acceleration lower bound");
    }
    if (acceleration >
        row.acceleration_bounds.upper +
            comparison_tolerance(acceleration, row.acceleration_bounds.upper)) {
      return failure(SolverStatus::kNumericalError,
                     accepted_constraint_identity(specification) +
                         " violated linearized acceleration upper bound");
    }

    const double next_rate = row.rate + dt * acceleration;
    if (next_rate <
        row.rate_bounds.lower -
            comparison_tolerance(next_rate, row.rate_bounds.lower)) {
      return failure(SolverStatus::kNumericalError,
                     accepted_constraint_identity(specification) +
                         " violated linearized rate lower bound");
    }
    if (next_rate >
        row.rate_bounds.upper +
            comparison_tolerance(next_rate, row.rate_bounds.upper)) {
      return failure(SolverStatus::kNumericalError,
                     accepted_constraint_identity(specification) +
                         " violated linearized rate upper bound");
    }

    const double endpoint =
        row.value + dt * row.rate + 0.5 * dt * dt * acceleration;
    if (endpoint <
        row.state_bounds.lower -
            comparison_tolerance(endpoint, row.state_bounds.lower)) {
      return failure(SolverStatus::kNumericalError,
                     accepted_constraint_identity(specification) +
                         " violated linearized state lower bound");
    }
    if (endpoint >
        row.state_bounds.upper +
            comparison_tolerance(endpoint, row.state_bounds.upper)) {
      return failure(SolverStatus::kNumericalError,
                     accepted_constraint_identity(specification) +
                         " violated linearized state upper bound");
    }
  }
  return {};
}

ScalarGeometricConstraintResult validate_scalar_geometric_sample_acceptance(
    const ScalarGeometricConstraintSpecification &specification,
    const ScalarGeometricCoordinateSample &sample, bool endpoint) {
  const auto specification_validation =
      validate_scalar_geometric_specification(specification);
  if (!specification_validation.satisfied()) {
    return specification_validation;
  }
  if (sample.value.size() != specification.dimension ||
      sample.rate.size() != specification.dimension ||
      sample.acceleration.size() != specification.dimension) {
    return failure(SolverStatus::kShapeMismatch,
                   accepted_constraint_identity(specification) +
                       " nonlinear sample dimensions are inconsistent");
  }
  if (!sample.value.allFinite() || !sample.rate.allFinite() ||
      !sample.acceleration.allFinite()) {
    return failure(SolverStatus::kNumericalError,
                   accepted_constraint_identity(specification) +
                       " nonlinear sample was non-finite");
  }

  for (int axis : specification.active_axes) {
    const double lower = specification.state_lower_bounds(axis);
    const double upper = specification.state_upper_bounds(axis);
    if (sample.value(axis) <
            lower - comparison_tolerance(sample.value(axis), lower) ||
        sample.value(axis) >
            upper + comparison_tolerance(sample.value(axis), upper)) {
      return failure(SolverStatus::kNumericalError,
                     accepted_constraint_identity(specification) +
                         " nonlinear path violated axis " +
                         std::to_string(axis) + " state bound");
    }
    const double rate_limit = specification.policy.rate_limits(axis);
    if (std::abs(sample.rate(axis)) >
        rate_limit + comparison_tolerance(sample.rate(axis), rate_limit)) {
      return failure(SolverStatus::kNumericalError,
                     accepted_constraint_identity(specification) +
                         " nonlinear path violated axis " +
                         std::to_string(axis) + " rate bound");
    }
    const double acceleration_limit =
        specification.policy.acceleration_limits(axis);
    if (std::abs(sample.acceleration(axis)) >
        acceleration_limit +
            comparison_tolerance(sample.acceleration(axis),
                                 acceleration_limit)) {
      return failure(SolverStatus::kNumericalError,
                     accepted_constraint_identity(specification) +
                         " nonlinear path violated axis " +
                         std::to_string(axis) + " acceleration bound");
    }
    if (!endpoint) {
      continue;
    }
    if (sample.rate(axis) < 0.0) {
      const double margin = sample.value(axis) - lower;
      const double capacity =
          2.0 * specification.policy.lower_braking_accelerations(axis) *
          margin;
      if (margin < -comparison_tolerance(margin, 0.0) ||
          sample.rate(axis) * sample.rate(axis) >
              capacity + comparison_tolerance(
                             sample.rate(axis) * sample.rate(axis),
                             capacity)) {
        return failure(SolverStatus::kNumericalError,
                       accepted_constraint_identity(specification) +
                           " endpoint failed axis " + std::to_string(axis) +
                           " lower braking viability");
      }
    }
    if (sample.rate(axis) > 0.0) {
      const double margin = upper - sample.value(axis);
      const double capacity =
          2.0 * specification.policy.upper_braking_accelerations(axis) *
          margin;
      if (margin < -comparison_tolerance(margin, 0.0) ||
          sample.rate(axis) * sample.rate(axis) >
              capacity + comparison_tolerance(
                             sample.rate(axis) * sample.rate(axis),
                             capacity)) {
        return failure(SolverStatus::kNumericalError,
                       accepted_constraint_identity(specification) +
                           " endpoint failed axis " + std::to_string(axis) +
                           " upper braking viability");
      }
    }
  }
  return {};
}

int scalar_geometric_path_segment_count(
    const Eigen::VectorXd &dq, const Eigen::VectorXd &ddq, double dt,
    const ScalarGeometricConstraintSpecification &specification,
    const ScalarGeometricPathValidationOptions &options,
    ScalarGeometricConstraintResult *failure_result) {
  if (failure_result == nullptr) {
    return 0;
  }
  const auto specification_validation =
      validate_scalar_geometric_specification(specification);
  if (!specification_validation.satisfied()) {
    *failure_result = specification_validation;
    return 0;
  }
  if (dq.size() <= 0 || dq.size() != ddq.size()) {
    *failure_result =
        failure(SolverStatus::kShapeMismatch,
                accepted_constraint_identity(specification) +
                    " nonlinear path state dimensions are inconsistent");
    return 0;
  }
  if (!dq.allFinite() || !ddq.allFinite() || !(dt > 0.0) ||
      !std::isfinite(dt)) {
    *failure_result =
        failure(SolverStatus::kInvalidInput,
                accepted_constraint_identity(specification) +
                    " nonlinear path state inputs are invalid");
    return 0;
  }
  if (options.minimum_segments <= 0 ||
      options.maximum_segments < options.minimum_segments ||
      !(options.maximum_joint_tangent_step > 0.0) ||
      !std::isfinite(options.maximum_joint_tangent_step)) {
    *failure_result = failure(SolverStatus::kInvalidInput,
                              "scalar geometric path options are invalid");
    return 0;
  }
  const Eigen::VectorXd next_velocity = dq + dt * ddq;
  if (!next_velocity.allFinite()) {
    *failure_result = failure(
        SolverStatus::kNumericalError,
        accepted_constraint_identity(specification) +
            " nonlinear path velocity was non-finite");
    return 0;
  }
  const double maximum_speed =
      dq.cwiseAbs().cwiseMax(next_velocity.cwiseAbs()).maxCoeff();
  const double required_segments_value =
      std::ceil(dt * maximum_speed / options.maximum_joint_tangent_step);
  if (!std::isfinite(required_segments_value) ||
      required_segments_value > options.maximum_segments) {
    *failure_result = failure(
        SolverStatus::kNumericalError,
        accepted_constraint_identity(specification) +
            " nonlinear path exceeded validation trust step");
    return 0;
  }
  return std::max(options.minimum_segments,
                  static_cast<int>(required_segments_value));
}

ScalarGeometricConstraintResult validate_scalar_geometric_path(
    const ScalarGeometricConstraintSpecification &specification,
    const Eigen::VectorXd &dq, const Eigen::VectorXd &ddq, double dt,
    const ScalarGeometricPathValidationOptions &options,
    const ScalarGeometricPathSampleEvaluator &sample_evaluator,
    const ScalarGeometricPathSegmentValidator &segment_validator) {
  if (!sample_evaluator) {
    return failure(SolverStatus::kInvalidInput,
                   accepted_constraint_identity(specification) +
                       " nonlinear path evaluator is missing");
  }
  ScalarGeometricConstraintResult segment_count_failure;
  const int segment_count = scalar_geometric_path_segment_count(
      dq, ddq, dt, specification, options, &segment_count_failure);
  if (!segment_count_failure.satisfied()) {
    return segment_count_failure;
  }

  std::optional<ScalarGeometricPathPoint> previous;
  for (int segment = 0; segment <= segment_count; ++segment) {
    ScalarGeometricPathPoint current;
    current.segment_index = segment;
    current.segment_count = segment_count;
    current.time = dt * static_cast<double>(segment) /
                   static_cast<double>(segment_count);
    current.tangent =
        current.time * dq + 0.5 * current.time * current.time * ddq;
    current.rate = dq + current.time * ddq;

    auto evaluated = sample_evaluator(current, &current.coordinates);
    if (!evaluated.satisfied()) {
      return evaluated;
    }

    const auto sample = validate_scalar_geometric_sample_acceptance(
        specification, current.coordinates, segment == segment_count);
    if (!sample.satisfied()) {
      return sample;
    }

    if (previous.has_value() && segment_validator) {
      const auto segment_result =
          segment_validator({previous.value(), current});
      if (!segment_result.satisfied()) {
        return segment_result;
      }
    }
    previous = std::move(current);
  }
  return {};
}

ScalarGeometricConstraintResult validate_so3_log_rotation_segment(
    const ScalarGeometricPathSegment &segment,
    const Eigen::VectorXd *tangent_support_mask) {
  if (segment.previous.coordinates.value.size() < 3 ||
      segment.current.coordinates.value.size() < 3) {
    return failure(SolverStatus::kNumericalError,
                   "SO(3) segment guard coordinate dimensions are invalid");
  }
  if (segment.previous.tangent.size() <= 0 ||
      segment.previous.tangent.size() != segment.current.tangent.size()) {
    return failure(SolverStatus::kNumericalError,
                   "SO(3) segment guard tangent dimensions are invalid");
  }
  if (!segment.previous.coordinates.value.allFinite() ||
      !segment.current.coordinates.value.allFinite() ||
      !segment.previous.tangent.allFinite() ||
      !segment.current.tangent.allFinite()) {
    return failure(SolverStatus::kNumericalError,
                   "SO(3) segment guard inputs must be finite");
  }
  if (tangent_support_mask != nullptr) {
    if (tangent_support_mask->size() != segment.previous.tangent.size() ||
        !tangent_support_mask->allFinite()) {
      return failure(SolverStatus::kNumericalError,
                     "SO(3) segment guard support mask dimensions are invalid");
    }
    for (Eigen::Index index = 0; index < tangent_support_mask->size();
         ++index) {
      const double value = (*tangent_support_mask)(index);
      if (value != 0.0 && value != 1.0) {
        return failure(SolverStatus::kNumericalError,
                       "SO(3) segment guard support mask must be binary");
      }
    }
  }
  if (!segment.previous.coordinates.so3_rotation.has_value() ||
      !segment.current.coordinates.so3_rotation.has_value()) {
    return failure(SolverStatus::kNumericalError,
                   "SO(3) segment guard metadata is missing");
  }
  if (!is_valid_so3_rotation(segment.previous.coordinates.so3_rotation.value()) ||
      !is_valid_so3_rotation(segment.current.coordinates.so3_rotation.value())) {
    return failure(SolverStatus::kNumericalError,
                   "SO(3) segment guard rotations are invalid");
  }
  const Eigen::Vector3d previous_log =
      segment.previous.coordinates.value.tail<3>();
  const Eigen::Vector3d current_log =
      segment.current.coordinates.value.tail<3>();
  const double previous_angle = previous_log.norm();
  const double current_angle = current_log.norm();
  Eigen::VectorXd tangent_delta =
      segment.current.tangent - segment.previous.tangent;
  if (tangent_support_mask != nullptr) {
    tangent_delta = tangent_delta.cwiseProduct(*tangent_support_mask);
  }
  const double tangent_excursion = tangent_delta.cwiseAbs().sum();
  const double endpoint_geodesic =
      pinocchio::log3(segment.previous.coordinates.so3_rotation->transpose() *
                      segment.current.coordinates.so3_rotation.value())
          .norm();
  if (!std::isfinite(endpoint_geodesic) ||
      endpoint_geodesic >
          tangent_excursion + 1e-12 * (1.0 + tangent_excursion)) {
    return failure(SolverStatus::kNumericalError,
                   "SO(3) endpoint geodesic exceeds generalized tangent "
                   "excursion");
  }
  const double excursion_bound =
      std::max(previous_angle, current_angle) + tangent_excursion;
  if (!std::isfinite(excursion_bound) ||
      excursion_bound >= kSo3Pi - kSo3LogMinimumBranchMargin) {
    return failure(SolverStatus::kNumericalError,
                   "SO(3) path segment can enter the near-pi chart guard");
  }
  return {};
}

ScalarGeometricConstraintResult
validate_so3_log_rotation_segment(const ScalarGeometricPathSegment &segment) {
  return validate_so3_log_rotation_segment(segment, nullptr);
}

ScalarGeometricConstraintResult validate_so3_log_rotation_segment(
    const ScalarGeometricPathSegment &segment,
    const Eigen::VectorXd &tangent_support_mask) {
  return validate_so3_log_rotation_segment(segment, &tangent_support_mask);
}

} // namespace embodik::detail
