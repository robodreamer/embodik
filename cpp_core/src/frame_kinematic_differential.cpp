#include "frame_kinematic_differential.hpp"

#include <pinocchio/spatial/skew.hpp>

#include <stdexcept>

namespace embodik::detail {

FrameKinematicDifferential evaluate_frame_kinematic_differential(
    const RobotModel &robot, const std::string &frame_name) {
  if (!robot.has_frame(frame_name)) {
    throw std::invalid_argument(
        "frame differential requires an existing robot frame");
  }
  return {robot.get_frame_pose(frame_name),
          robot.get_frame_jacobian(frame_name),
          robot.get_frame_jacobian_bias(frame_name)};
}

RelativeFrameKinematicDifferential
evaluate_relative_frame_kinematic_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b) {
  return {evaluate_frame_kinematic_differential(robot, frame_a),
          evaluate_frame_kinematic_differential(robot, frame_b)};
}

FrameKinematicDifferential apply_fixed_tcp_offset(
    const FrameKinematicDifferential &frame, const pinocchio::SE3 &offset,
    const Eigen::VectorXd &velocity) {
  if (frame.jacobian.cols() != velocity.size() ||
      !offset.rotation().allFinite() || !offset.translation().allFinite()) {
    throw std::invalid_argument("TCP differential inputs are inconsistent");
  }

  FrameKinematicDifferential result = frame;
  result.pose = frame.pose * offset;
  const Eigen::Vector3d offset_world =
      frame.pose.rotation() * offset.translation();
  const Eigen::MatrixXd angular_jacobian = frame.jacobian.bottomRows<3>();
  result.jacobian.topRows<3>() =
      frame.jacobian.topRows<3>() -
      pinocchio::skew(offset_world) * angular_jacobian;

  const Eigen::Vector3d angular_velocity = angular_jacobian * velocity;
  result.affine_bias.head<3>() =
      frame.affine_bias.head<3>() +
      frame.affine_bias.tail<3>().cross(offset_world) +
      angular_velocity.cross(angular_velocity.cross(offset_world));
  return result;
}

} // namespace embodik::detail
