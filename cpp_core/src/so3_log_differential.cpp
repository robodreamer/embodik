#include "so3_log_differential.hpp"

#include <pinocchio/spatial/explog.hpp>
#include <pinocchio/spatial/skew.hpp>

#include <cmath>
#include <stdexcept>

namespace embodik::detail {
namespace {

constexpr double kSmallAngleThreshold = 0.1;

struct So3LogCoefficients {
  double quadratic = 0.0;
  double radial_derivative = 0.0;
};

So3LogCoefficients so3_log_coefficients(double angle) {
  const double angle_squared = angle * angle;
  if (angle < kSmallAngleThreshold) {
    const double angle_fourth = angle_squared * angle_squared;
    const double angle_sixth = angle_fourth * angle_squared;
    const double angle_eighth = angle_fourth * angle_fourth;
    return {1.0 / 12.0 + angle_squared / 720.0 +
                angle_fourth / 30240.0 + angle_sixth / 1209600.0 +
                angle_eighth / 47900160.0,
            1.0 / 360.0 + angle_squared / 7560.0 +
                angle_fourth / 201600.0 + angle_sixth / 5987520.0};
  }

  const double sine = std::sin(angle);
  const double cosine = std::cos(angle);
  const double one_minus_cosine = 1.0 - cosine;
  const double angle_cubed = angle_squared * angle;
  const double angle_fourth = angle_squared * angle_squared;
  return {1.0 / angle_squared -
              (1.0 + cosine) / (2.0 * angle * sine),
          -2.0 / angle_fourth +
              1.0 / (2.0 * angle_squared * one_minus_cosine) +
              (1.0 + cosine) / (2.0 * angle_cubed * sine)};
}

void validate_so3_log_inputs(
    const Eigen::Matrix3d &error_rotation,
    const Eigen::MatrixXd &right_angular_jacobian,
    const Eigen::Vector3d &right_angular_affine_bias,
    const Eigen::VectorXd &velocity) {
  if (!error_rotation.allFinite() || !right_angular_jacobian.allFinite() ||
      !right_angular_affine_bias.allFinite() || !velocity.allFinite()) {
    throw std::invalid_argument(
        "SO(3) log differential inputs must contain only finite values");
  }
  if (right_angular_jacobian.rows() != 3 ||
      right_angular_jacobian.cols() != velocity.size()) {
    throw std::invalid_argument(
        "SO(3) right angular Jacobian dimensions are inconsistent");
  }
  if (!is_valid_so3_rotation(error_rotation)) {
    throw std::invalid_argument(
        "SO(3) log differential requires a valid rotation matrix");
  }
}

} // namespace

bool is_valid_so3_rotation(const Eigen::Matrix3d &rotation) {
  if (!rotation.allFinite()) {
    return false;
  }
  const double orthogonality_error =
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm();
  return orthogonality_error <= kSo3RotationMatrixTolerance &&
         std::abs(rotation.determinant() - 1.0) <=
             kSo3RotationMatrixTolerance;
}

So3LogCoordinateDifferential evaluate_so3_log_differential(
    const Eigen::Matrix3d &error_rotation,
    const Eigen::MatrixXd &right_angular_jacobian,
    const Eigen::Vector3d &right_angular_affine_bias,
    const Eigen::VectorXd &velocity) {
  validate_so3_log_inputs(error_rotation, right_angular_jacobian,
                          right_angular_affine_bias, velocity);

  So3LogCoordinateDifferential result;
  result.value = pinocchio::log3(error_rotation);
  result.angle = result.value.norm();
  result.branch_margin = kSo3Pi - result.angle;
  if (!std::isfinite(result.angle) ||
      result.branch_margin <= kSo3LogMinimumBranchMargin) {
    throw std::domain_error(
        "SO(3) log differential is inside the near-pi branch guard");
  }

  const auto coefficients = so3_log_coefficients(result.angle);
  const Eigen::Matrix3d value_cross = pinocchio::skew(result.value);
  const Eigen::Matrix3d log_jacobian =
      Eigen::Matrix3d::Identity() + 0.5 * value_cross +
      coefficients.quadratic * value_cross * value_cross;
  const Eigen::Vector3d right_angular_rate =
      right_angular_jacobian * velocity;
  result.rate = log_jacobian * right_angular_rate;
  result.jacobian = log_jacobian * right_angular_jacobian;

  const Eigen::Vector3d contracted_jacobian_derivative =
      0.5 * result.rate.cross(right_angular_rate) +
      coefficients.quadratic *
          (result.rate.cross(result.value.cross(right_angular_rate)) +
           result.value.cross(result.rate.cross(right_angular_rate))) +
      coefficients.radial_derivative * result.value.dot(result.rate) *
          result.value.cross(result.value.cross(right_angular_rate));
  result.affine_bias = log_jacobian * right_angular_affine_bias +
                       contracted_jacobian_derivative;
  return result;
}

} // namespace embodik::detail
