#include "geometric_constraint_differential.hpp"

#include "frame_kinematic_differential.hpp"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/spatial/skew.hpp>

#include <stdexcept>

namespace embodik::detail {

GeometricCoordinateDifferential promote_translation_to_pose_differential(
    const GeometricCoordinateDifferential &translation,
    Eigen::Index variable_count) {
  if (translation.value.size() != 3 || translation.rate.size() != 3 ||
      translation.jacobian.rows() != 3 ||
      translation.jacobian.cols() != variable_count ||
      translation.affine_bias.size() != 3) {
    throw std::invalid_argument(
        "translation differential dimensions are inconsistent");
  }
  GeometricCoordinateDifferential result;
  result.value = Eigen::VectorXd::Zero(6);
  result.rate = Eigen::VectorXd::Zero(6);
  result.jacobian = Eigen::MatrixXd::Zero(6, variable_count);
  result.affine_bias = Eigen::VectorXd::Zero(6);
  result.value.head<3>() = translation.value;
  result.rate.head<3>() = translation.rate;
  result.jacobian.topRows<3>() = translation.jacobian;
  result.affine_bias.head<3>() = translation.affine_bias;
  return result;
}

namespace {

GeometricCoordinateDifferential
compose_moving_frame_point_translation_differential(
    const FrameKinematicDifferential &reference,
    const FrameKinematicDifferential &point_frame,
    const Eigen::VectorXd &velocity) {
  const Eigen::Matrix3d reference_rotation_transpose =
      reference.pose.rotation().transpose();
  const Eigen::Vector3d relative_translation =
      reference_rotation_transpose *
      (point_frame.pose.translation() - reference.pose.translation());
  const Eigen::MatrixXd reference_angular_jacobian =
      reference_rotation_transpose * reference.jacobian.bottomRows<3>();
  const Eigen::MatrixXd relative_linear_jacobian =
      reference_rotation_transpose *
      (point_frame.jacobian.topRows<3>() - reference.jacobian.topRows<3>());
  const Eigen::Vector3d reference_angular_velocity =
      reference_angular_jacobian * velocity;
  const Eigen::Vector3d relative_linear_velocity =
      relative_linear_jacobian * velocity;

  GeometricCoordinateDifferential result;
  result.value = relative_translation;
  result.jacobian = relative_linear_jacobian +
                    pinocchio::skew(relative_translation) *
                        reference_angular_jacobian;
  const Eigen::Vector3d point_rate = result.jacobian * velocity;
  result.rate = point_rate;
  result.affine_bias =
      reference_rotation_transpose *
          (point_frame.affine_bias.head<3>() -
           reference.affine_bias.head<3>()) -
      reference_angular_velocity.cross(relative_linear_velocity) -
      reference_angular_velocity.cross(point_rate) +
      relative_translation.cross(reference_rotation_transpose *
                                 reference.affine_bias.tail<3>());
  return result;
}

GeometricCoordinateDifferential compose_fixed_frame_pose_differential(
    const FrameKinematicDifferential &frame,
    const pinocchio::SE3 &reference_pose, const Eigen::VectorXd &velocity) {
  if (!reference_pose.translation().allFinite() ||
      !reference_pose.rotation().allFinite()) {
    throw std::invalid_argument("fixed frame pose reference must be finite");
  }

  const Eigen::Matrix3d body_rotation = frame.pose.rotation().transpose();
  const Eigen::MatrixXd right_angular_jacobian =
      body_rotation * frame.jacobian.bottomRows<3>();
  const Eigen::Vector3d right_angular_bias =
      body_rotation * frame.affine_bias.tail<3>();
  const auto angular = evaluate_so3_log_differential(
      reference_pose.rotation().transpose() * frame.pose.rotation(),
      right_angular_jacobian, right_angular_bias, velocity);

  GeometricCoordinateDifferential result;
  result.value.resize(6);
  result.rate.resize(6);
  result.jacobian.resize(6, velocity.size());
  result.affine_bias.resize(6);
  result.value.head<3>() =
      frame.pose.translation() - reference_pose.translation();
  result.value.tail<3>() = angular.value;
  result.so3_rotation = frame.pose.rotation();
  result.jacobian.topRows<3>() = frame.jacobian.topRows<3>();
  result.jacobian.bottomRows<3>() = angular.jacobian;
  result.rate.head<3>() = result.jacobian.topRows<3>() * velocity;
  result.rate.tail<3>() = angular.rate;
  result.affine_bias.head<3>() = frame.affine_bias.head<3>();
  result.affine_bias.tail<3>() = angular.affine_bias;
  return result;
}

GeometricCoordinateDifferential compose_relative_pose_differential(
    const RelativeFrameKinematicDifferential &frames,
    const Eigen::VectorXd &velocity) {
  GeometricCoordinateDifferential result =
      promote_translation_to_pose_differential(
          compose_moving_frame_point_translation_differential(
              frames.frame_a, frames.frame_b, velocity),
          velocity.size());

  const Eigen::Matrix3d rotation_b_transpose =
      frames.frame_b.pose.rotation().transpose();
  const Eigen::MatrixXd right_angular_jacobian =
      rotation_b_transpose *
      (frames.frame_b.jacobian.bottomRows<3>() -
       frames.frame_a.jacobian.bottomRows<3>());
  const Eigen::Vector3d right_angular_rate =
      right_angular_jacobian * velocity;
  const Eigen::Vector3d frame_b_angular_velocity =
      rotation_b_transpose *
      (frames.frame_b.jacobian.bottomRows<3>() * velocity);
  const Eigen::Vector3d right_angular_affine_bias =
      rotation_b_transpose *
          (frames.frame_b.affine_bias.tail<3>() -
           frames.frame_a.affine_bias.tail<3>()) -
      frame_b_angular_velocity.cross(right_angular_rate);
  const auto angular = evaluate_so3_log_differential(
      frames.frame_a.pose.rotation().transpose() *
          frames.frame_b.pose.rotation(),
      right_angular_jacobian, right_angular_affine_bias, velocity);
  result.value.tail<3>() = angular.value;
  result.so3_rotation =
      frames.frame_a.pose.rotation().transpose() *
      frames.frame_b.pose.rotation();
  result.rate.tail<3>() = angular.rate;
  result.jacobian.bottomRows<3>() = angular.jacobian;
  result.affine_bias.tail<3>() = angular.affine_bias;
  return result;
}

} // namespace

GeometricCoordinateDifferential evaluate_fixed_frame_pose_differential(
    const RobotModel &robot, const std::string &frame_name,
    const pinocchio::SE3 &reference_pose) {
  return compose_fixed_frame_pose_differential(
      evaluate_frame_kinematic_differential(robot, frame_name), reference_pose,
      robot.get_current_velocity());
}

GeometricCoordinateDifferential
evaluate_fixed_frame_pose_differential_at_state(
    const RobotModel &robot, const std::string &frame_name,
    const pinocchio::SE3 &reference_pose, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq) {
  pinocchio::Data scratch(robot.model());
  return evaluate_fixed_frame_pose_differential_at_state(
      robot, scratch, frame_name, reference_pose, q, dq);
}

GeometricCoordinateDifferential
evaluate_fixed_frame_pose_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_name, const pinocchio::SE3 &reference_pose,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq) {
  return compose_fixed_frame_pose_differential(
      evaluate_frame_kinematic_differential_at_state(robot, scratch,
                                                      frame_name, q, dq),
      reference_pose, dq);
}

GeometricCoordinateDifferential evaluate_relative_pose_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b) {
  return compose_relative_pose_differential(
      evaluate_relative_frame_kinematic_differential(robot, frame_a, frame_b),
      robot.get_current_velocity());
}

GeometricCoordinateDifferential evaluate_relative_pose_translation_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b) {
  const auto frames =
      evaluate_relative_frame_kinematic_differential(robot, frame_a, frame_b);
  return compose_moving_frame_point_translation_differential(
      frames.frame_a, frames.frame_b, robot.get_current_velocity());
}

GeometricCoordinateDifferential evaluate_relative_pose_differential_at_state(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq) {
  pinocchio::Data scratch(robot.model());
  return evaluate_relative_pose_differential_at_state(
      robot, scratch, frame_a, frame_b, q, dq);
}

GeometricCoordinateDifferential
evaluate_relative_pose_translation_differential_at_state(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq) {
  pinocchio::Data scratch(robot.model());
  return evaluate_relative_pose_translation_differential_at_state(
      robot, scratch, frame_a, frame_b, q, dq);
}

GeometricCoordinateDifferential evaluate_relative_pose_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_a, const std::string &frame_b,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq) {
  return compose_relative_pose_differential(
      evaluate_relative_frame_kinematic_differential_at_state(
          robot, scratch, frame_a, frame_b, q, dq),
      dq);
}

GeometricCoordinateDifferential
evaluate_relative_pose_translation_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_a, const std::string &frame_b,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq) {
  const auto frames = evaluate_relative_frame_kinematic_differential_at_state(
      robot, scratch, frame_a, frame_b, q, dq);
  return compose_moving_frame_point_translation_differential(
      frames.frame_a, frames.frame_b, dq);
}

GeometricCoordinateDifferential
compose_com_in_frame_differential(const Eigen::Vector3d &com_world,
                                  const Eigen::Vector3d &com_rate_world,
                                  const Eigen::MatrixXd &com_jacobian_world,
                                  const Eigen::Vector3d &com_bias_world,
                                  const pinocchio::SE3 &frame_pose,
                                  Eigen::Index variable_count) {
  if (com_jacobian_world.rows() != 3 ||
      com_jacobian_world.cols() != variable_count ||
      !com_world.allFinite() || !com_rate_world.allFinite() ||
      !com_jacobian_world.allFinite() || !com_bias_world.allFinite() ||
      !frame_pose.translation().allFinite() ||
      !frame_pose.rotation().allFinite()) {
    throw std::invalid_argument("CoM differential inputs are inconsistent");
  }
  const Eigen::Matrix3d R_world_frame = frame_pose.rotation().transpose();
  GeometricCoordinateDifferential result;
  result.value = R_world_frame * (com_world - frame_pose.translation());
  result.rate = R_world_frame * com_rate_world;
  result.jacobian = R_world_frame * com_jacobian_world;
  result.affine_bias = R_world_frame * com_bias_world;
  return result;
}

GeometricCoordinateDifferential
evaluate_com_in_frame_differential(const RobotModel &robot,
                                   const std::string &frame_name) {
  pinocchio::SE3 frame_pose = pinocchio::SE3::Identity();
  if (frame_name != "world") {
    frame_pose = robot.get_frame_pose(frame_name);
  }
  return compose_com_in_frame_differential(
      robot.get_com_position(), robot.get_com_velocity(),
      robot.get_com_jacobian(), robot.get_com_jacobian_bias(), frame_pose,
      robot.nv());
}

GeometricCoordinateDifferential
evaluate_com_in_frame_differential_at_state(
    const RobotModel &robot, const std::string &frame_name,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq) {
  pinocchio::Data scratch(robot.model());
  return evaluate_com_in_frame_differential_at_state(robot, scratch, frame_name,
                                                     q, dq);
}

GeometricCoordinateDifferential
evaluate_com_in_frame_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_name, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq) {
  if (q.size() != robot.nq() || dq.size() != robot.nv() || !q.allFinite() ||
      !dq.allFinite()) {
    throw std::invalid_argument("CoM explicit-state inputs are inconsistent");
  }
  const Eigen::VectorXd zero_acceleration = Eigen::VectorXd::Zero(robot.nv());
  pinocchio::centerOfMass(robot.model(), scratch, q, dq, zero_acceleration,
                          false);
  pinocchio::jacobianCenterOfMass(robot.model(), scratch, q, false);

  pinocchio::SE3 frame_pose = pinocchio::SE3::Identity();
  if (frame_name != "world") {
    pinocchio::forwardKinematics(robot.model(), scratch, q, dq,
                                 zero_acceleration);
    pinocchio::updateFramePlacements(robot.model(), scratch);
    const auto frame_id = robot.model().getFrameId(frame_name);
    if (frame_id >= robot.model().frames.size()) {
      throw std::invalid_argument("CoM support frame not found");
    }
    frame_pose = scratch.oMf[frame_id];
  }

  return compose_com_in_frame_differential(
      scratch.com[0], scratch.vcom[0], scratch.Jcom, scratch.acom[0],
      frame_pose, robot.nv());
}

} // namespace embodik::detail
