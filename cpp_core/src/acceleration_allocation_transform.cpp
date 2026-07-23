#include "acceleration_allocation_transform.hpp"

#include <cmath>
#include <utility>

namespace embodik::detail {
namespace {

constexpr double kMaxReliableLogMetricSpread = 1000.0;

AccelerationAllocationMatrixResult matrix_failure(SolverStatus status,
                                                  std::string message) {
  AccelerationAllocationMatrixResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

AccelerationAllocationVectorResult vector_failure(SolverStatus status,
                                                  std::string message) {
  AccelerationAllocationVectorResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

AccelerationAllocationHardRowsResult hard_rows_failure(SolverStatus status,
                                                       std::string message) {
  AccelerationAllocationHardRowsResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

bool prepared_transform_is_usable(
    const PreparedAccelerationAllocationTransform &transform) {
  return transform.status == SolverStatus::kSuccess &&
         transform.reference_acceleration.size() ==
             transform.variable_scale.size() &&
         transform.variable_scale.size() ==
             transform.inverse_variable_scale.size() &&
         transform.reference_acceleration.allFinite() &&
         transform.variable_scale.allFinite() &&
         transform.inverse_variable_scale.allFinite() &&
         (transform.variable_scale.array() > 0.0).all() &&
         (transform.inverse_variable_scale.array() > 0.0).all();
}

Eigen::Index variable_count(
    const PreparedAccelerationAllocationTransform &transform) {
  return transform.variable_scale.size();
}

} // namespace

AccelerationAllocationMatrixResult
PreparedAccelerationAllocationTransform::transform_objective_matrix(
    const Eigen::MatrixXd &physical_matrix) const {
  if (!prepared_transform_is_usable(*this)) {
    return matrix_failure(status, message.empty()
                                      ? "acceleration allocation transform is "
                                        "not prepared"
                                      : message);
  }
  if (physical_matrix.cols() != variable_count(*this)) {
    return matrix_failure(
        SolverStatus::kShapeMismatch,
        "acceleration allocation objective matrix must have nv columns");
  }
  if (!physical_matrix.allFinite()) {
    return matrix_failure(
        SolverStatus::kNonFiniteInput,
        "acceleration allocation objective matrix must be finite");
  }

  AccelerationAllocationMatrixResult result;
  result.matrix = physical_matrix * variable_scale.asDiagonal();
  if (!result.matrix.allFinite()) {
    return matrix_failure(
        SolverStatus::kNumericalError,
        "acceleration allocation objective matrix transform is not finite");
  }
  return result;
}

AccelerationAllocationVectorResult
PreparedAccelerationAllocationTransform::transform_objective_bias(
    const Eigen::MatrixXd &physical_matrix,
    const Eigen::VectorXd &physical_bias) const {
  if (!prepared_transform_is_usable(*this)) {
    return vector_failure(status, message.empty()
                                      ? "acceleration allocation transform is "
                                        "not prepared"
                                      : message);
  }
  if (physical_matrix.cols() != variable_count(*this) ||
      physical_bias.size() != physical_matrix.rows()) {
    return vector_failure(
        SolverStatus::kShapeMismatch,
        "acceleration allocation objective bias dimensions are inconsistent");
  }
  if (!physical_matrix.allFinite() || !physical_bias.allFinite()) {
    return vector_failure(
        SolverStatus::kNonFiniteInput,
        "acceleration allocation objective bias inputs must be finite");
  }

  AccelerationAllocationVectorResult result;
  result.vector = physical_bias + physical_matrix * reference_acceleration;
  if (!result.vector.allFinite()) {
    return vector_failure(
        SolverStatus::kNumericalError,
        "acceleration allocation objective bias transform is not finite");
  }
  return result;
}

AccelerationAllocationHardRowsResult
PreparedAccelerationAllocationTransform::transform_backend_hard_rows(
    const Eigen::MatrixXd &coefficient_matrix,
    const Eigen::VectorXd &lower_shifted,
    const Eigen::VectorXd &upper_shifted) const {
  if (!prepared_transform_is_usable(*this)) {
    return hard_rows_failure(status, message.empty()
                                         ? "acceleration allocation transform "
                                           "is not prepared"
                                         : message);
  }
  if (coefficient_matrix.cols() != variable_count(*this) ||
      lower_shifted.size() != coefficient_matrix.rows() ||
      upper_shifted.size() != coefficient_matrix.rows()) {
    return hard_rows_failure(
        SolverStatus::kShapeMismatch,
        "acceleration allocation hard-row dimensions are inconsistent");
  }
  if (!coefficient_matrix.allFinite() || !lower_shifted.allFinite() ||
      !upper_shifted.allFinite()) {
    return hard_rows_failure(
        SolverStatus::kNonFiniteInput,
        "acceleration allocation hard-row inputs must be finite");
  }

  AccelerationAllocationHardRowsResult result;
  const Eigen::VectorXd reference_shift =
      coefficient_matrix * reference_acceleration;
  result.coefficient_matrix =
      coefficient_matrix * variable_scale.asDiagonal();
  result.lower_bounds = lower_shifted - reference_shift;
  result.upper_bounds = upper_shifted - reference_shift;
  if (!reference_shift.allFinite() || !result.coefficient_matrix.allFinite() ||
      !result.lower_bounds.allFinite() || !result.upper_bounds.allFinite()) {
    return hard_rows_failure(
        SolverStatus::kNumericalError,
        "acceleration allocation hard-row transform is not finite");
  }
  return result;
}

AccelerationAllocationVectorResult
PreparedAccelerationAllocationTransform::reconstruct_physical_acceleration(
    const Eigen::VectorXd &solver_variable) const {
  if (!prepared_transform_is_usable(*this)) {
    return vector_failure(status, message.empty()
                                      ? "acceleration allocation transform is "
                                        "not prepared"
                                      : message);
  }
  if (solver_variable.size() != variable_count(*this)) {
    return vector_failure(
        SolverStatus::kShapeMismatch,
        "acceleration allocation solver variable must have size nv");
  }
  if (!solver_variable.allFinite()) {
    return vector_failure(
        SolverStatus::kNonFiniteInput,
        "acceleration allocation solver variable must be finite");
  }

  AccelerationAllocationVectorResult result;
  result.vector = reference_acceleration +
                  variable_scale.cwiseProduct(solver_variable);
  if (!result.vector.allFinite()) {
    return vector_failure(
        SolverStatus::kNumericalError,
        "acceleration allocation reconstruction is not finite");
  }
  return result;
}

AccelerationAllocationVectorResult
PreparedAccelerationAllocationTransform::inverse_scale_physical_residual(
    const Eigen::VectorXd &physical_residual) const {
  if (!prepared_transform_is_usable(*this)) {
    return vector_failure(status, message.empty()
                                      ? "acceleration allocation transform is "
                                        "not prepared"
                                      : message);
  }
  if (physical_residual.size() != variable_count(*this)) {
    return vector_failure(
        SolverStatus::kShapeMismatch,
        "acceleration allocation physical residual must have size nv");
  }
  if (!physical_residual.allFinite()) {
    return vector_failure(
        SolverStatus::kNonFiniteInput,
        "acceleration allocation physical residual must be finite");
  }

  AccelerationAllocationVectorResult result;
  result.vector = inverse_variable_scale.cwiseProduct(physical_residual);
  if (!result.vector.allFinite()) {
    return vector_failure(
        SolverStatus::kNumericalError,
        "acceleration allocation inverse scaling is not finite");
  }
  return result;
}

PreparedAccelerationAllocationTransform prepare_acceleration_allocation_transform(
    const Eigen::VectorXd &metric_diagonal,
    const Eigen::VectorXd &reference_acceleration,
    Eigen::Index variable_count) {
  PreparedAccelerationAllocationTransform result;
  const auto fail = [&](SolverStatus status, std::string failure_message) {
    result = PreparedAccelerationAllocationTransform{};
    result.status = status;
    result.message = std::move(failure_message);
    return result;
  };

  if (variable_count <= 0 || metric_diagonal.size() != variable_count ||
      reference_acceleration.size() != variable_count) {
    return fail(SolverStatus::kShapeMismatch,
                "acceleration allocation metric and reference must have size "
                "nv");
  }
  if (!metric_diagonal.allFinite() || !reference_acceleration.allFinite()) {
    return fail(SolverStatus::kNonFiniteInput,
                "acceleration allocation metric and reference must be finite");
  }
  if ((metric_diagonal.array() <= 0.0).any()) {
    return fail(SolverStatus::kInvalidInput,
                "acceleration allocation metric diagonal must be positive");
  }

  const Eigen::ArrayXd log_metric = metric_diagonal.array().log();
  if (!log_metric.allFinite()) {
    return fail(SolverStatus::kInvalidInput,
                "acceleration allocation metric is too ill-conditioned for "
                "the diagonal transform");
  }
  if (log_metric.maxCoeff() - log_metric.minCoeff() >
      kMaxReliableLogMetricSpread) {
    return fail(SolverStatus::kInvalidInput,
                "acceleration allocation metric is too ill-conditioned for "
                "the diagonal transform");
  }
  const double mean_log_metric = log_metric.mean();
  if (!std::isfinite(mean_log_metric)) {
    return fail(SolverStatus::kInvalidInput,
                "acceleration allocation metric is too ill-conditioned for "
                "the diagonal transform");
  }

  result.requested_metric_diagonal = metric_diagonal;
  result.reference_acceleration = reference_acceleration;
  if ((metric_diagonal.array() == metric_diagonal(0)).all()) {
    result.variable_scale = Eigen::VectorXd::Ones(variable_count);
  } else {
    result.variable_scale =
        (-0.5 * (log_metric - mean_log_metric)).exp().matrix();
  }
  result.inverse_variable_scale = result.variable_scale.cwiseInverse();
  if (!result.variable_scale.allFinite() ||
      !result.inverse_variable_scale.allFinite() ||
      (result.variable_scale.array() <= 0.0).any() ||
      (result.inverse_variable_scale.array() <= 0.0).any()) {
    return fail(SolverStatus::kInvalidInput,
                "acceleration allocation metric is too ill-conditioned for "
                "the diagonal transform");
  }
  return result;
}

} // namespace embodik::detail
