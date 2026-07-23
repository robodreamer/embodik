#include "frame_kinematic_differential.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/spatial/skew.hpp>

#include <stdexcept>

namespace embodik::detail {
namespace {

void validate_frame_name(const RobotModel &robot,
                         const std::string &frame_name) {
  if (!robot.has_frame(frame_name)) {
    throw std::invalid_argument(
        "frame differential requires an existing robot frame");
  }
}

void validate_frame_differential_state(const RobotModel &robot,
                                       const pinocchio::Data &scratch,
                                       const Eigen::VectorXd &q,
                                       const Eigen::VectorXd &dq) {
  const auto &model = robot.model();
  if (q.size() != model.nq || dq.size() != model.nv) {
    throw std::invalid_argument(
        "frame differential state dimensions are inconsistent");
  }
  if (!q.allFinite() || !dq.allFinite()) {
    throw std::invalid_argument(
        "frame differential state must contain only finite values");
  }
  if (scratch.oMi.size() != static_cast<std::size_t>(model.njoints) ||
      scratch.oMf.size() != model.frames.size()) {
    throw std::invalid_argument(
        "frame differential scratch data does not match the robot model");
  }
}

void update_frame_differential_state(const RobotModel &robot,
                                     pinocchio::Data &scratch,
                                     const Eigen::VectorXd &q,
                                     const Eigen::VectorXd &dq) {
  const auto &model = robot.model();
  pinocchio::forwardKinematics(model, scratch, q, dq);
  pinocchio::computeJointJacobiansTimeVariation(model, scratch, q, dq);
  pinocchio::updateFramePlacements(model, scratch);
}

FrameKinematicDifferential extract_frame_kinematic_differential(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_name, const Eigen::VectorXd &dq) {
  const auto &model = robot.model();
  const auto frame_id = model.getFrameId(frame_name);
  FrameKinematicDifferential result;
  result.pose = scratch.oMf[frame_id];
  result.jacobian.resize(6, model.nv);
  result.jacobian.setZero();
  Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian_derivative(6, model.nv);
  jacobian_derivative.setZero();
  pinocchio::getFrameJacobian(model, scratch, frame_id,
                              pinocchio::LOCAL_WORLD_ALIGNED,
                              result.jacobian);
  pinocchio::getFrameJacobianTimeVariation(
      model, scratch, frame_id, pinocchio::LOCAL_WORLD_ALIGNED,
      jacobian_derivative);
  result.affine_bias = jacobian_derivative * dq;
  return result;
}

} // namespace

FrameKinematicDifferential evaluate_frame_kinematic_differential(
    const RobotModel &robot, const std::string &frame_name) {
  validate_frame_name(robot, frame_name);
  return {robot.get_frame_pose(frame_name),
          robot.get_frame_jacobian(frame_name),
          robot.get_frame_jacobian_bias(frame_name)};
}

FrameKinematicDifferential evaluate_frame_kinematic_differential_at_state(
    const RobotModel &robot, const std::string &frame_name,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq) {
  pinocchio::Data scratch(robot.model());
  return evaluate_frame_kinematic_differential_at_state(robot, scratch,
                                                        frame_name, q, dq);
}

FrameKinematicDifferential evaluate_frame_kinematic_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_name, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq) {
  validate_frame_name(robot, frame_name);
  validate_frame_differential_state(robot, scratch, q, dq);
  update_frame_differential_state(robot, scratch, q, dq);
  return extract_frame_kinematic_differential(robot, scratch, frame_name, dq);
}

RelativeFrameKinematicDifferential
evaluate_relative_frame_kinematic_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b) {
  return {evaluate_frame_kinematic_differential(robot, frame_a),
          evaluate_frame_kinematic_differential(robot, frame_b)};
}

RelativeFrameKinematicDifferential
evaluate_relative_frame_kinematic_differential_at_state(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq) {
  pinocchio::Data scratch(robot.model());
  return evaluate_relative_frame_kinematic_differential_at_state(
      robot, scratch, frame_a, frame_b, q, dq);
}

RelativeFrameKinematicDifferential
evaluate_relative_frame_kinematic_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_a, const std::string &frame_b,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq) {
  validate_frame_name(robot, frame_a);
  validate_frame_name(robot, frame_b);
  validate_frame_differential_state(robot, scratch, q, dq);
  update_frame_differential_state(robot, scratch, q, dq);
  return {extract_frame_kinematic_differential(robot, scratch, frame_a, dq),
          extract_frame_kinematic_differential(robot, scratch, frame_b, dq)};
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
