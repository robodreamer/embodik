#pragma once

#include <Eigen/Core>

namespace embodik::detail {

inline constexpr double kSo3Pi = 3.14159265358979323846;
inline constexpr double kSo3LogMinimumBranchMargin = 1e-3;
inline constexpr double kSo3RotationMatrixTolerance = 1e-8;

struct So3LogCoordinateDifferential {
  Eigen::Vector3d value = Eigen::Vector3d::Zero();
  Eigen::Vector3d rate = Eigen::Vector3d::Zero();
  Eigen::MatrixXd jacobian;
  Eigen::Vector3d affine_bias = Eigen::Vector3d::Zero();
  double angle = 0.0;
  double branch_margin = 0.0;
};

bool is_valid_so3_rotation(const Eigen::Matrix3d &rotation);

So3LogCoordinateDifferential evaluate_so3_log_differential(
    const Eigen::Matrix3d &error_rotation,
    const Eigen::MatrixXd &right_angular_jacobian,
    const Eigen::Vector3d &right_angular_affine_bias,
    const Eigen::VectorXd &velocity);

} // namespace embodik::detail
