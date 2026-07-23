#pragma once

#include "so3_log_differential.hpp"

#include <embodik/robot_model.hpp>

#include <Eigen/Core>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <optional>
#include <string>

namespace embodik::detail {

struct GeometricCoordinateDifferential {
  Eigen::VectorXd value;
  Eigen::VectorXd rate;
  Eigen::MatrixXd jacobian;
  Eigen::VectorXd affine_bias;
  std::optional<Eigen::Matrix3d> so3_rotation;
};

GeometricCoordinateDifferential promote_translation_to_pose_differential(
    const GeometricCoordinateDifferential &translation,
    Eigen::Index variable_count);

GeometricCoordinateDifferential evaluate_fixed_frame_pose_differential(
    const RobotModel &robot, const std::string &frame_name,
    const pinocchio::SE3 &reference_pose);

GeometricCoordinateDifferential
evaluate_fixed_frame_pose_differential_at_state(
    const RobotModel &robot, const std::string &frame_name,
    const pinocchio::SE3 &reference_pose, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq);

GeometricCoordinateDifferential
evaluate_fixed_frame_pose_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_name, const pinocchio::SE3 &reference_pose,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq);

GeometricCoordinateDifferential evaluate_relative_pose_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b);

GeometricCoordinateDifferential evaluate_relative_pose_translation_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b);

GeometricCoordinateDifferential evaluate_relative_pose_differential_at_state(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq);

GeometricCoordinateDifferential
evaluate_relative_pose_translation_differential_at_state(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq);

GeometricCoordinateDifferential evaluate_relative_pose_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_a, const std::string &frame_b,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq);

GeometricCoordinateDifferential
evaluate_relative_pose_translation_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_a, const std::string &frame_b,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq);

} // namespace embodik::detail
