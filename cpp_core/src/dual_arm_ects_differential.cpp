#include "dual_arm_ects_differential.hpp"

#include <embodik/dual_arm_ects.hpp>

#include <pinocchio/spatial/explog.hpp>
#include <pinocchio/spatial/skew.hpp>

#include <cmath>
#include <stdexcept>

namespace embodik::detail {
namespace {

constexpr double kSo3Pi = 3.14159265358979323846;
constexpr double kSo3LogMinimumBranchMargin = 1e-3;
constexpr double kSo3RotationMatrixTolerance = 1e-8;
constexpr double kSmallAngleThreshold = 0.1;

struct So3LogCoefficients {
  double quadratic = 0.0;
  double radial_derivative = 0.0;
};

struct So3LogCoordinateDifferential {
  Eigen::Vector3d value = Eigen::Vector3d::Zero();
  Eigen::Vector3d rate = Eigen::Vector3d::Zero();
  Eigen::MatrixXd jacobian;
  Eigen::Vector3d affine_bias = Eigen::Vector3d::Zero();
  double angle = 0.0;
  double branch_margin = 0.0;
};

struct So3LeftJacobianDifferential {
  Eigen::Matrix3d jacobian = Eigen::Matrix3d::Identity();
  Eigen::Vector3d contracted_derivative = Eigen::Vector3d::Zero();
};

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

So3LogCoordinateDifferential evaluate_so3_log_differential(
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

So3LeftJacobianDifferential so3_left_jacobian_differential(
    const Eigen::Vector3d &value, const Eigen::Vector3d &rate) {
  if (!value.allFinite() || !rate.allFinite()) {
    throw std::invalid_argument(
        "SO(3) exponential differential inputs must be finite");
  }

  const double theta_squared = value.squaredNorm();
  const double theta = std::sqrt(theta_squared);
  double linear = 0.0;
  double quadratic = 0.0;
  double linear_rate = 0.0;
  double quadratic_rate = 0.0;
  const double radial_rate_numerator = value.dot(rate);
  if (theta < 0.1) {
    const double theta_fourth = theta_squared * theta_squared;
    const double theta_sixth = theta_fourth * theta_squared;
    linear = 0.5 - theta_squared / 24.0 + theta_fourth / 720.0 -
             theta_sixth / 40320.0;
    quadratic = 1.0 / 6.0 - theta_squared / 120.0 +
                theta_fourth / 5040.0 - theta_sixth / 362880.0;
    linear_rate =
        (-1.0 / 12.0 + theta_squared / 180.0 -
         theta_fourth / 6720.0) *
        radial_rate_numerator;
    quadratic_rate =
        (-1.0 / 60.0 + theta_squared / 1260.0 -
         theta_fourth / 60480.0) *
        radial_rate_numerator;
  } else {
    const double sine = std::sin(theta);
    const double cosine = std::cos(theta);
    const double theta_cubed = theta_squared * theta;
    const double theta_fourth = theta_squared * theta_squared;
    const double theta_fifth = theta_fourth * theta;
    linear = (1.0 - cosine) / theta_squared;
    quadratic = (theta - sine) / theta_cubed;
    linear_rate =
        (theta * sine - 2.0 * (1.0 - cosine)) / theta_fourth *
        radial_rate_numerator;
    quadratic_rate =
        (3.0 * sine - theta * (2.0 + cosine)) / theta_fifth *
        radial_rate_numerator;
  }

  const Eigen::Matrix3d value_cross = pinocchio::skew(value);
  So3LeftJacobianDifferential result;
  result.jacobian = Eigen::Matrix3d::Identity() + linear * value_cross +
                    quadratic * value_cross * value_cross;
  const Eigen::Vector3d value_cross_rate = value.cross(rate);
  result.contracted_derivative =
      linear_rate * value_cross_rate +
      quadratic_rate * value.cross(value_cross_rate) +
      quadratic * rate.cross(value_cross_rate);
  return result;
}

EctsTaskDifferential relative_differential(
    const RelativeFrameKinematicDifferential &frames,
    const Eigen::VectorXd &velocity) {
  const Eigen::Matrix3d rotation_a_transpose =
      frames.frame_a.pose.rotation().transpose();
  const Eigen::Vector3d relative_translation =
      rotation_a_transpose *
      (frames.frame_b.pose.translation() -
       frames.frame_a.pose.translation());
  const Eigen::MatrixXd angular_jacobian_a =
      rotation_a_transpose * frames.frame_a.jacobian.bottomRows<3>();
  const Eigen::MatrixXd relative_linear_jacobian =
      rotation_a_transpose *
      (frames.frame_b.jacobian.topRows<3>() -
       frames.frame_a.jacobian.topRows<3>());

  EctsTaskDifferential result;
  result.pose = frames.frame_a.pose.inverse() * frames.frame_b.pose;
  result.spatial_jacobian.resize(6, velocity.size());
  result.spatial_jacobian.topRows<3>() =
      relative_linear_jacobian +
      pinocchio::skew(relative_translation) * angular_jacobian_a;
  result.spatial_jacobian.bottomRows<3>() =
      rotation_a_transpose *
      (frames.frame_b.jacobian.bottomRows<3>() -
       frames.frame_a.jacobian.bottomRows<3>());
  result.spatial_rate = result.spatial_jacobian * velocity;

  const Eigen::Vector3d angular_velocity_a = angular_jacobian_a * velocity;
  const Eigen::Vector3d relative_linear_velocity =
      relative_linear_jacobian * velocity;
  result.spatial_bias.head<3>() =
      rotation_a_transpose *
          (frames.frame_b.affine_bias.head<3>() -
           frames.frame_a.affine_bias.head<3>()) -
      angular_velocity_a.cross(relative_linear_velocity) -
      angular_velocity_a.cross(result.spatial_rate.head<3>()) +
      relative_translation.cross(rotation_a_transpose *
                                 frames.frame_a.affine_bias.tail<3>());
  result.spatial_bias.tail<3>() =
      rotation_a_transpose *
          (frames.frame_b.affine_bias.tail<3>() -
           frames.frame_a.affine_bias.tail<3>()) -
      angular_velocity_a.cross(result.spatial_rate.tail<3>());
  return result;
}

EctsTaskDifferential absolute_differential(
    const RelativeFrameKinematicDifferential &frames, double alpha,
    const Eigen::VectorXd &velocity) {
  if (alpha == 1.0) {
    return {frames.frame_a.pose, frames.frame_a.jacobian,
            frames.frame_a.jacobian * velocity, frames.frame_a.affine_bias};
  }
  if (alpha == 0.0) {
    return {frames.frame_b.pose, frames.frame_b.jacobian,
            frames.frame_b.jacobian * velocity, frames.frame_b.affine_bias};
  }

  const double beta = 1.0 - alpha;
  const Eigen::Matrix3d rotation_b_transpose =
      frames.frame_b.pose.rotation().transpose();
  const Eigen::MatrixXd right_relative_angular_jacobian =
      rotation_b_transpose *
      (frames.frame_b.jacobian.bottomRows<3>() -
       frames.frame_a.jacobian.bottomRows<3>());
  const Eigen::Vector3d right_relative_angular_rate =
      right_relative_angular_jacobian * velocity;
  const Eigen::Vector3d angular_velocity_b =
      rotation_b_transpose *
      (frames.frame_b.jacobian.bottomRows<3>() * velocity);
  const Eigen::Vector3d right_relative_angular_bias =
      rotation_b_transpose *
          (frames.frame_b.affine_bias.tail<3>() -
           frames.frame_a.affine_bias.tail<3>()) -
      angular_velocity_b.cross(right_relative_angular_rate);
  const Eigen::Matrix3d relative_rotation =
      frames.frame_a.pose.rotation().transpose() *
      frames.frame_b.pose.rotation();
  const auto relative_log = evaluate_so3_log_differential(
      relative_rotation, right_relative_angular_jacobian,
      right_relative_angular_bias, velocity);

  const Eigen::Vector3d exponential_value = beta * relative_log.value;
  const Eigen::Vector3d exponential_rate = beta * relative_log.rate;
  const auto exponential =
      so3_left_jacobian_differential(exponential_value, exponential_rate);
  const Eigen::MatrixXd exponential_angular_jacobian =
      exponential.jacobian * beta * relative_log.jacobian;
  const Eigen::Vector3d exponential_angular_rate =
      exponential.jacobian * exponential_rate;
  const Eigen::Vector3d exponential_angular_bias =
      exponential.jacobian * beta * relative_log.affine_bias +
      exponential.contracted_derivative;

  EctsTaskDifferential result;
  result.pose = compute_absolute_frame(frames.frame_a.pose,
                                       frames.frame_b.pose, alpha);
  result.spatial_jacobian.resize(6, velocity.size());
  result.spatial_jacobian.topRows<3>() =
      alpha * frames.frame_a.jacobian.topRows<3>() +
      beta * frames.frame_b.jacobian.topRows<3>();
  result.spatial_jacobian.bottomRows<3>() =
      frames.frame_a.jacobian.bottomRows<3>() +
      frames.frame_a.pose.rotation() * exponential_angular_jacobian;
  result.spatial_rate = result.spatial_jacobian * velocity;
  result.spatial_bias.head<3>() =
      alpha * frames.frame_a.affine_bias.head<3>() +
      beta * frames.frame_b.affine_bias.head<3>();
  const Eigen::Vector3d angular_velocity_a =
      frames.frame_a.jacobian.bottomRows<3>() * velocity;
  const Eigen::Vector3d world_exponential_rate =
      frames.frame_a.pose.rotation() * exponential_angular_rate;
  result.spatial_bias.tail<3>() =
      frames.frame_a.affine_bias.tail<3>() +
      angular_velocity_a.cross(world_exponential_rate) +
      frames.frame_a.pose.rotation() * exponential_angular_bias;
  return result;
}

} // namespace

EctsTaskDifferential evaluate_relative_ects_task_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b) {
  return relative_differential(
      evaluate_relative_frame_kinematic_differential(robot, frame_a, frame_b),
      robot.get_current_velocity());
}

EctsTaskDifferential evaluate_absolute_ects_task_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b, const pinocchio::SE3 &offset_a,
    const pinocchio::SE3 &offset_b, double alpha) {
  if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0) {
    throw std::invalid_argument("ECTS alpha must be finite and in [0, 1]");
  }
  const Eigen::VectorXd &velocity = robot.get_current_velocity();
  auto frames =
      evaluate_relative_frame_kinematic_differential(robot, frame_a, frame_b);
  frames.frame_a = apply_fixed_tcp_offset(frames.frame_a, offset_a, velocity);
  frames.frame_b = apply_fixed_tcp_offset(frames.frame_b, offset_b, velocity);
  return absolute_differential(frames, alpha, velocity);
}

EctsTaskDifferential evaluate_absolute_ects_translation_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b, const pinocchio::SE3 &offset_a,
    const pinocchio::SE3 &offset_b, double alpha) {
  if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0) {
    throw std::invalid_argument("ECTS alpha must be finite and in [0, 1]");
  }
  const Eigen::VectorXd &velocity = robot.get_current_velocity();
  auto frames =
      evaluate_relative_frame_kinematic_differential(robot, frame_a, frame_b);
  frames.frame_a = apply_fixed_tcp_offset(frames.frame_a, offset_a, velocity);
  frames.frame_b = apply_fixed_tcp_offset(frames.frame_b, offset_b, velocity);

  const double beta = 1.0 - alpha;
  EctsTaskDifferential result;
  result.pose = pinocchio::SE3(
      Eigen::Matrix3d::Identity(),
      alpha * frames.frame_a.pose.translation() +
          beta * frames.frame_b.pose.translation());
  result.spatial_jacobian = Eigen::MatrixXd::Zero(6, velocity.size());
  result.spatial_jacobian.topRows<3>() =
      alpha * frames.frame_a.jacobian.topRows<3>() +
      beta * frames.frame_b.jacobian.topRows<3>();
  result.spatial_rate = result.spatial_jacobian * velocity;
  result.spatial_bias.setZero();
  result.spatial_bias.head<3>() =
      alpha * frames.frame_a.affine_bias.head<3>() +
      beta * frames.frame_b.affine_bias.head<3>();
  return result;
}

} // namespace embodik::detail
