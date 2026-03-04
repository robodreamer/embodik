/**
 * @file dual_arm_ects.hpp
 * @brief Extended Cooperative Task Space (ECTS) utilities for dual-arm coordination
 *
 * Provides functions to compute absolute (object-centric) and relative (grasp)
 * frames and Jacobians for coordinated dual-arm manipulation.
 *
 * Reference:
 *   H. A. Park, "Dual-arm coordinated-motion task specification and performance
 *   evaluation," 2016 IEEE/RSJ International Conference on Intelligent Robots
 *   and Systems (IROS), Daejeon, Korea (South), 2016, pp. 516-522,
 *   doi: 10.1109/IROS.2016.7759161.
 */

#pragma once

#include <Eigen/Dense>
#include <pinocchio/spatial/se3.hpp>
#include <string>

namespace embodik {

/**
 * @brief Configuration for ECTS coordination mode
 */
struct ECTSConfig {
  double alpha = 0.5;
  bool coordinated = true;
};

/**
 * @brief Map a named coordination mode to ECTSConfig
 *
 * Supported modes:
 *   "orthogonal"   -> alpha=0.5, coordinated=false
 *   "serial_left"  -> alpha=1.0, coordinated=true
 *   "serial_right" -> alpha=0.0, coordinated=true
 *   "parallel"     -> alpha=0.5, coordinated=true
 *   "blended"      -> alpha=0.5, coordinated=true (use blended(ratio) for custom)
 */
ECTSConfig map_ects_mode(const std::string &mode);

/**
 * @brief Map a blended coordination mode with custom ratio
 */
ECTSConfig map_ects_mode_blended(double ratio);

/**
 * @brief Compute the absolute (object-centric) frame from two end-effector poses
 *
 * Position: p_abs = alpha * p_1 + (1 - alpha) * p_2
 * Orientation: Slerp between R_1 and R_2 with ratio (1 - alpha)
 */
pinocchio::SE3 compute_absolute_frame(const pinocchio::SE3 &T_1,
                                      const pinocchio::SE3 &T_2, double alpha);

/**
 * @brief Compute the relative frame: T_1^{-1} * T_2
 *
 * Position: R_1^T * (p_2 - p_1) (expressed in frame 1)
 * Orientation: R_1^T * R_2
 */
pinocchio::SE3 compute_relative_frame(const pinocchio::SE3 &T_1,
                                      const pinocchio::SE3 &T_2);

/**
 * @brief Compute the absolute Jacobian for a single-model robot
 *
 * J_abs = alpha * J_1 + (1 - alpha) * J_2
 * Both J_1 and J_2 are 6 x nv in world-aligned frame.
 */
Eigen::MatrixXd compute_absolute_jacobian(const Eigen::MatrixXd &J_1,
                                          const Eigen::MatrixXd &J_2,
                                          double alpha);

/**
 * @brief Compute the relative Jacobian for a single-model robot
 *
 * For position rows: R_1^T * (J_2_pos - J_1_pos) + skew(p_rel) * R_1^T * J_1_rot
 * For orientation rows: R_1^T * (J_2_rot - J_1_rot)
 *
 * This follows PLACO's relative_position_jacobian formula which accounts for
 * the rotating reference frame effect on relative position.
 */
Eigen::MatrixXd compute_relative_jacobian(const Eigen::MatrixXd &J_1,
                                          const Eigen::MatrixXd &J_2,
                                          const Eigen::Matrix3d &R_1,
                                          const Eigen::Vector3d &p_rel);

} // namespace embodik
