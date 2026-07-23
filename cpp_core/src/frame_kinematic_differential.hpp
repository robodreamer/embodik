#pragma once

#include <embodik/robot_model.hpp>

#include <Eigen/Core>
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

RelativeFrameKinematicDifferential
evaluate_relative_frame_kinematic_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b);

FrameKinematicDifferential apply_fixed_tcp_offset(
    const FrameKinematicDifferential &frame, const pinocchio::SE3 &offset,
    const Eigen::VectorXd &velocity);

} // namespace embodik::detail
