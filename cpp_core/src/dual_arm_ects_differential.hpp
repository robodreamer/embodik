#pragma once

#include "frame_kinematic_differential.hpp"

#include <Eigen/Core>
#include <pinocchio/spatial/se3.hpp>

#include <string>

namespace embodik::detail {

struct EctsTaskDifferential {
  pinocchio::SE3 pose;
  Eigen::Matrix<double, 6, Eigen::Dynamic> spatial_jacobian;
  Eigen::Matrix<double, 6, 1> spatial_rate;
  Eigen::Matrix<double, 6, 1> spatial_bias;
};

EctsTaskDifferential evaluate_relative_ects_task_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b);

EctsTaskDifferential evaluate_absolute_ects_task_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b, const pinocchio::SE3 &offset_a,
    const pinocchio::SE3 &offset_b, double alpha);

EctsTaskDifferential evaluate_absolute_ects_translation_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b, const pinocchio::SE3 &offset_a,
    const pinocchio::SE3 &offset_b, double alpha);

} // namespace embodik::detail
