/**
 * @file dual_arm_ects.cpp
 * @brief Implementation of ECTS utilities for dual-arm coordination
 *
 * Reference:
 *   H. A. Park, "Dual-arm coordinated-motion task specification and performance
 *   evaluation," 2016 IEEE/RSJ International Conference on Intelligent Robots
 *   and Systems (IROS), Daejeon, Korea (South), 2016, pp. 516-522,
 *   doi: 10.1109/IROS.2016.7759161.
 */

#include <embodik/dual_arm_ects.hpp>
#include <Eigen/Geometry>
#include <pinocchio/spatial/skew.hpp>
#include <stdexcept>

namespace embodik {

ECTSConfig map_ects_mode(const std::string &mode) {
  ECTSConfig cfg;
  if (mode == "orthogonal") {
    cfg.alpha = 0.5;
    cfg.coordinated = false;
  } else if (mode == "serial_left") {
    cfg.alpha = 1.0;
    cfg.coordinated = true;
  } else if (mode == "serial_right") {
    cfg.alpha = 0.0;
    cfg.coordinated = true;
  } else if (mode == "parallel") {
    cfg.alpha = 0.5;
    cfg.coordinated = true;
  } else if (mode == "blended") {
    cfg.alpha = 0.5;
    cfg.coordinated = true;
  } else {
    throw std::invalid_argument("Unknown ECTS mode: " + mode +
                                ". Supported: orthogonal, serial_left, "
                                "serial_right, parallel, blended");
  }
  return cfg;
}

ECTSConfig map_ects_mode_blended(double ratio) {
  ECTSConfig cfg;
  cfg.alpha = ratio;
  cfg.coordinated = true;
  return cfg;
}

pinocchio::SE3 compute_absolute_frame(const pinocchio::SE3 &T_1,
                                      const pinocchio::SE3 &T_2,
                                      double alpha) {
  // Position: weighted average
  Eigen::Vector3d p_abs =
      alpha * T_1.translation() + (1.0 - alpha) * T_2.translation();

  // Orientation: slerp from R_1 toward R_2 by (1 - alpha)
  Eigen::Quaterniond q1(T_1.rotation());
  Eigen::Quaterniond q2(T_2.rotation());
  q1.normalize();
  q2.normalize();
  // Handle quaternion hemisphere
  if (q1.dot(q2) < 0.0)
    q2.coeffs() *= -1.0;
  Eigen::Quaterniond q_abs = q1.slerp(1.0 - alpha, q2);

  return pinocchio::SE3(q_abs.toRotationMatrix(), p_abs);
}

pinocchio::SE3 compute_relative_frame(const pinocchio::SE3 &T_1,
                                      const pinocchio::SE3 &T_2) {
  return T_1.inverse() * T_2;
}

Eigen::MatrixXd compute_absolute_jacobian(const Eigen::MatrixXd &J_1,
                                          const Eigen::MatrixXd &J_2,
                                          double alpha) {
  return alpha * J_1 + (1.0 - alpha) * J_2;
}

Eigen::MatrixXd compute_relative_jacobian(const Eigen::MatrixXd &J_1,
                                          const Eigen::MatrixXd &J_2,
                                          const Eigen::Matrix3d &R_1,
                                          const Eigen::Vector3d &p_rel) {
  const int nv = static_cast<int>(J_1.cols());
  Eigen::MatrixXd J_rel(6, nv);

  Eigen::Matrix3d R1_T = R_1.transpose();

  // Position rows: R_1^T * (J_2_pos - J_1_pos) + skew(p_rel) * R_1^T * J_1_rot
  // This matches PLACO's relative_position_jacobian formula
  Eigen::MatrixXd J_1_pos = J_1.topRows<3>();
  Eigen::MatrixXd J_1_rot = J_1.bottomRows<3>();
  Eigen::MatrixXd J_2_pos = J_2.topRows<3>();

  J_rel.topRows<3>() =
      R1_T * (J_2_pos - J_1_pos) +
      pinocchio::skew(p_rel) * R1_T * J_1_rot;

  // Orientation rows: R_1^T * (J_2_rot - J_1_rot)
  Eigen::MatrixXd J_2_rot = J_2.bottomRows<3>();
  J_rel.bottomRows<3>() = R1_T * (J_2_rot - J_1_rot);

  return J_rel;
}

} // namespace embodik
