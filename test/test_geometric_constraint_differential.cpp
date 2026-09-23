#include "geometric_constraint_differential.hpp"

#include <embodik/robot_model.hpp>

#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <pinocchio/spatial/explog.hpp>

namespace embodik::test {
namespace {

constexpr double kPi = 3.14159265358979323846;

void create_test_urdf(const std::string &filename) {
  std::ofstream file(filename);
  file << R"(<?xml version="1.0"?>
<robot name="geometric_differential_test_robot">
  <link name="base_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.3 0 0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="4.0" effort="20.0"/>
  </joint>

  <link name="tip">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0.2 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" velocity="4.0" effort="20.0"/>
  </joint>
</robot>)";
}

class GeometricConstraintDifferentialTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_path_ = "/tmp/embodik_geometric_differential_test.urdf";
    create_test_urdf(urdf_path_);
    robot_ = std::make_shared<RobotModel>(urdf_path_, false);
    q_.resize(2);
    q_ << 0.35, -0.25;
    dq_.resize(2);
    dq_ << 0.45, -0.2;
    robot_->update_kinematics(q_, dq_);
  }

  void TearDown() override { std::remove(urdf_path_.c_str()); }

  std::string urdf_path_;
  std::shared_ptr<RobotModel> robot_;
  Eigen::VectorXd q_;
  Eigen::VectorXd dq_;
};

Eigen::Matrix3d right_log_jacobian(const Eigen::Matrix3d &rotation) {
  Eigen::Matrix3d result;
  pinocchio::Jlog3(rotation, result);
  return result;
}

Eigen::Vector3d finite_difference_log_bias(
    const Eigen::Matrix3d &error_rotation,
    const Eigen::Vector3d &tangent_velocity,
    const Eigen::Vector3d &tangent_bias, double step) {
  const Eigen::Matrix3d rotation_plus =
      error_rotation * pinocchio::exp3(step * tangent_velocity);
  const Eigen::Matrix3d rotation_minus =
      error_rotation * pinocchio::exp3(-step * tangent_velocity);
  const Eigen::Vector3d rate_plus =
      right_log_jacobian(rotation_plus) *
      (tangent_velocity + step * tangent_bias);
  const Eigen::Vector3d rate_minus =
      right_log_jacobian(rotation_minus) *
      (tangent_velocity - step * tangent_bias);
  return (rate_plus - rate_minus) / (2.0 * step);
}

Eigen::VectorXd relative_pose_value_at_state(const RobotModel &robot,
                                             const Eigen::VectorXd &q) {
  const Eigen::VectorXd zero_velocity = Eigen::VectorXd::Zero(robot.nv());
  return detail::evaluate_relative_pose_differential_at_state(
             robot, "link1", "tip", q, zero_velocity)
      .value;
}

Eigen::VectorXd relative_pose_rate_at_state(const RobotModel &robot,
                                            const Eigen::VectorXd &q,
                                            const Eigen::VectorXd &dq) {
  return detail::evaluate_relative_pose_differential_at_state(
             robot, "link1", "tip", q, dq)
      .rate;
}

TEST_F(GeometricConstraintDifferentialTest,
       So3LogDifferentialMatchesFiniteDifferences) {
  const Eigen::Vector3d tangent_velocity(0.3, -0.4, 0.2);
  const Eigen::Vector3d tangent_bias(-0.5, 0.1, 0.25);
  const Eigen::Matrix3d right_jacobian = Eigen::Matrix3d::Identity();
  constexpr double kStep = 1e-5;

  for (const Eigen::Vector3d &rotation_vector :
       {Eigen::Vector3d(0.04, -0.02, 0.01),
        Eigen::Vector3d(0.4, -0.2, 0.1),
        Eigen::Vector3d(2.0, -0.5, 0.2),
        Eigen::Vector3d((kPi - 2e-3) *
                        Eigen::Vector3d(1.0, -2.0, 3.0).normalized())}) {
    const Eigen::Matrix3d error_rotation = pinocchio::exp3(rotation_vector);
    const auto differential = detail::evaluate_so3_log_differential(
        error_rotation, right_jacobian, tangent_bias, tangent_velocity);

    Eigen::Matrix3d finite_difference_jacobian;
    for (int column = 0; column < 3; ++column) {
      const Eigen::Vector3d tangent = right_jacobian.col(column);
      const Eigen::Vector3d value_plus = pinocchio::log3(
          error_rotation * pinocchio::exp3(kStep * tangent));
      const Eigen::Vector3d value_minus = pinocchio::log3(
          error_rotation * pinocchio::exp3(-kStep * tangent));
      finite_difference_jacobian.col(column) =
          (value_plus - value_minus) / (2.0 * kStep);
    }

    EXPECT_TRUE(differential.jacobian.isApprox(
        finite_difference_jacobian, 2e-8));
    EXPECT_TRUE(differential.rate.isApprox(
        differential.jacobian * tangent_velocity, 1e-12));
    EXPECT_TRUE(differential.affine_bias.isApprox(
        finite_difference_log_bias(error_rotation, tangent_velocity,
                                   tangent_bias, kStep),
        2e-8));
  }
}

TEST_F(GeometricConstraintDifferentialTest,
       So3LogDifferentialMatchesNonIdentityRightJacobianFiniteDifferences) {
  const Eigen::Matrix3d error_rotation =
      pinocchio::exp3(Eigen::Vector3d(0.35, -0.28, 0.17));
  Eigen::Matrix<double, 3, 4> right_jacobian;
  right_jacobian << 0.7, -0.2, 0.15, 0.05, 0.1, 0.45, -0.3, 0.2, -0.25,
      0.05, 0.55, -0.1;
  Eigen::Vector4d velocity;
  velocity << 0.4, -0.3, 0.2, -0.1;
  const Eigen::Vector3d tangent_bias(-0.12, 0.08, 0.05);
  constexpr double kStep = 1e-6;

  const auto differential = detail::evaluate_so3_log_differential(
      error_rotation, right_jacobian, tangent_bias, velocity);

  Eigen::Matrix<double, 3, 4> finite_difference_jacobian;
  for (int column = 0; column < right_jacobian.cols(); ++column) {
    const Eigen::Vector3d tangent = right_jacobian.col(column);
    const Eigen::Vector3d value_plus = pinocchio::log3(
        error_rotation * pinocchio::exp3(kStep * tangent));
    const Eigen::Vector3d value_minus = pinocchio::log3(
        error_rotation * pinocchio::exp3(-kStep * tangent));
    finite_difference_jacobian.col(column) =
        (value_plus - value_minus) / (2.0 * kStep);
  }

  const Eigen::Vector3d tangent_velocity = right_jacobian * velocity;
  EXPECT_TRUE(
      differential.jacobian.isApprox(finite_difference_jacobian, 3e-10));
  EXPECT_TRUE(differential.rate.isApprox(
      finite_difference_jacobian * velocity, 3e-10));
  EXPECT_TRUE(differential.affine_bias.isApprox(
      finite_difference_log_bias(error_rotation, tangent_velocity,
                                 tangent_bias, kStep),
      5e-9));
}

TEST_F(GeometricConstraintDifferentialTest,
       So3LogDifferentialRejectsNearPiAndMalformedInputs) {
  const Eigen::Vector3d axis =
      Eigen::Vector3d(1.0, -2.0, 3.0).normalized();
  const Eigen::Matrix3d right_jacobian = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

  EXPECT_NO_THROW(detail::evaluate_so3_log_differential(
      pinocchio::exp3((kPi - 2e-3) * axis), right_jacobian, zero, zero));
  EXPECT_THROW(detail::evaluate_so3_log_differential(
                   pinocchio::exp3((kPi - 5e-4) * axis), right_jacobian,
                   zero, zero),
               std::domain_error);
  Eigen::Matrix3d malformed = Eigen::Matrix3d::Identity();
  malformed(0, 0) = 2.0;
  EXPECT_THROW(detail::evaluate_so3_log_differential(
                   malformed, right_jacobian, zero, zero),
               std::invalid_argument);
}

TEST_F(GeometricConstraintDifferentialTest,
       FixedFramePoseDifferentialMatchesRobotModelAndScratchPath) {
  const pinocchio::SE3 pose = robot_->get_frame_pose("tip");
  const Eigen::Vector3d desired_log(0.25, -0.15, 0.1);
  const pinocchio::SE3 reference_pose(
      pose.rotation() * pinocchio::exp3(-desired_log),
      pose.translation() + Eigen::Vector3d(0.01, -0.02, 0.03));

  const auto differential = detail::evaluate_fixed_frame_pose_differential(
      *robot_, "tip", reference_pose);
  ASSERT_EQ(differential.value.size(), 6);
  ASSERT_EQ(differential.jacobian.rows(), 6);
  ASSERT_EQ(differential.jacobian.cols(), robot_->nv());
  EXPECT_TRUE(differential.value.head<3>().isApprox(
      pose.translation() - reference_pose.translation(), 1e-12));
  EXPECT_TRUE(differential.value.tail<3>().isApprox(desired_log, 1e-12));
  EXPECT_TRUE(differential.rate.isApprox(differential.jacobian * dq_, 1e-12));
  EXPECT_TRUE(differential.affine_bias.head<3>().isApprox(
      robot_->get_frame_jacobian_bias("tip").head<3>(), 1e-12));

  pinocchio::Data scratch(robot_->model());
  const auto scratch_result =
      detail::evaluate_fixed_frame_pose_differential_at_state(
          *robot_, scratch, "tip", reference_pose, q_, dq_);
  EXPECT_TRUE(scratch_result.value.isApprox(differential.value, 1e-12));
  EXPECT_TRUE(scratch_result.rate.isApprox(differential.rate, 1e-12));
  EXPECT_TRUE(scratch_result.jacobian.isApprox(differential.jacobian, 1e-12));
  EXPECT_TRUE(
      scratch_result.affine_bias.isApprox(differential.affine_bias, 1e-12));
}

TEST_F(GeometricConstraintDifferentialTest,
       FixedFramePoseDifferentialRejectsBadStateAndFrame) {
  const pinocchio::SE3 pose = robot_->get_frame_pose("tip");
  Eigen::VectorXd bad_q = q_;
  bad_q(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(detail::evaluate_fixed_frame_pose_differential_at_state(
                   *robot_, "tip", pose, bad_q, dq_),
               std::invalid_argument);
  EXPECT_THROW(detail::evaluate_fixed_frame_pose_differential(
                   *robot_, "missing_frame", pose),
               std::invalid_argument);
}

TEST_F(GeometricConstraintDifferentialTest,
       RelativePoseDifferentialMatchesTwoMovingFrameFiniteDifferences) {
  const auto differential =
      detail::evaluate_relative_pose_differential(*robot_, "link1", "tip");
  ASSERT_EQ(differential.value.size(), 6);
  ASSERT_EQ(differential.jacobian.rows(), 6);
  ASSERT_EQ(differential.jacobian.cols(), robot_->nv());

  constexpr double kStep = 1e-6;
  Eigen::MatrixXd finite_difference_jacobian(6, robot_->nv());
  for (Eigen::Index column = 0; column < robot_->nv(); ++column) {
    Eigen::VectorXd direction = Eigen::VectorXd::Zero(robot_->nv());
    direction(column) = 1.0;
    const Eigen::VectorXd value_plus =
        relative_pose_value_at_state(*robot_, q_ + kStep * direction);
    const Eigen::VectorXd value_minus =
        relative_pose_value_at_state(*robot_, q_ - kStep * direction);
    finite_difference_jacobian.col(column) =
        (value_plus - value_minus) / (2.0 * kStep);
  }

  EXPECT_TRUE(differential.value.isApprox(
      relative_pose_value_at_state(*robot_, q_), 1e-12));
  EXPECT_TRUE(
      differential.jacobian.isApprox(finite_difference_jacobian, 2e-8));

  const Eigen::VectorXd value_plus =
      relative_pose_value_at_state(*robot_, q_ + kStep * dq_);
  const Eigen::VectorXd value_minus =
      relative_pose_value_at_state(*robot_, q_ - kStep * dq_);
  const Eigen::VectorXd finite_difference_rate =
      (value_plus - value_minus) / (2.0 * kStep);
  EXPECT_TRUE(differential.rate.isApprox(finite_difference_rate, 2e-8));

  const Eigen::VectorXd rate_plus =
      relative_pose_rate_at_state(*robot_, q_ + kStep * dq_, dq_);
  const Eigen::VectorXd rate_minus =
      relative_pose_rate_at_state(*robot_, q_ - kStep * dq_, dq_);
  const Eigen::VectorXd finite_difference_bias =
      (rate_plus - rate_minus) / (2.0 * kStep);
  EXPECT_LT((differential.affine_bias - finite_difference_bias).norm(), 5e-7)
      << "analytical: " << differential.affine_bias.transpose()
      << "\nfinite difference: " << finite_difference_bias.transpose();

  pinocchio::Data scratch(robot_->model());
  const auto scratch_result = detail::evaluate_relative_pose_differential_at_state(
      *robot_, scratch, "link1", "tip", q_, dq_);
  EXPECT_TRUE(scratch_result.value.isApprox(differential.value, 1e-12));
  EXPECT_TRUE(scratch_result.rate.isApprox(differential.rate, 1e-12));
  EXPECT_TRUE(scratch_result.jacobian.isApprox(differential.jacobian, 1e-12));
  EXPECT_TRUE(
      scratch_result.affine_bias.isApprox(differential.affine_bias, 1e-12));
}

} // namespace
} // namespace embodik::test
