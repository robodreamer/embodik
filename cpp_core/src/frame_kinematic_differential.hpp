#pragma once

#include <embodik/robot_model.hpp>

#include <Eigen/Core>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <string>

namespace embodik::detail {

struct FrameKinematicDifferential {
  pinocchio::SE3 pose;
  Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian;
  Eigen::Matrix<double, 6, 1> affine_bias;
};

struct RelativeFrameKinematicDifferential {
  FrameKinematicDifferential frame_a;
  FrameKinematicDifferential frame_b;
};

FrameKinematicDifferential evaluate_frame_kinematic_differential(
    const RobotModel &robot, const std::string &frame_name);

FrameKinematicDifferential evaluate_frame_kinematic_differential_at_state(
    const RobotModel &robot, const std::string &frame_name,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq);

FrameKinematicDifferential evaluate_frame_kinematic_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_name, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq);

RelativeFrameKinematicDifferential
evaluate_relative_frame_kinematic_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b);

RelativeFrameKinematicDifferential
evaluate_relative_frame_kinematic_differential_at_state(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq);

RelativeFrameKinematicDifferential
evaluate_relative_frame_kinematic_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_a, const std::string &frame_b,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq);

FrameKinematicDifferential apply_fixed_tcp_offset(
    const FrameKinematicDifferential &frame, const pinocchio::SE3 &offset,
    const Eigen::VectorXd &velocity);

} // namespace embodik::detail
