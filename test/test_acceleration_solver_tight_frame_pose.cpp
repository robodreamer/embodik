#include "acceleration_fixed_frame_pose_constraint.hpp"
#include "geometric_constraint_differential.hpp"

#include <embodik/acceleration_solver.hpp>

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

namespace embodik::test {
namespace {

constexpr double kTolerance = 1e-7;

Eigen::VectorXd vector(std::initializer_list<double> values) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(values.size()));
  Eigen::Index index = 0;
  for (double value : values) {
    out(index++) = value;
  }
  return out;
}

void expect_no_motion_outputs(const AccelerationSolverResult &result) {
  EXPECT_TRUE(result.solution.empty());
  EXPECT_EQ(result.joint_accelerations.size(), 0);
  EXPECT_EQ(result.joint_velocities_next.size(), 0);
  EXPECT_EQ(result.q_solution.size(), 0);
  EXPECT_EQ(result.predicted_torques.size(), 0);
}

class ScopedUrdf {
public:
  ScopedUrdf(std::string filename, const std::string &contents)
      : path_("/tmp/" + std::move(filename)) {
    std::ofstream file(path_);
    file << contents;
  }
  ~ScopedUrdf() { std::remove(path_.c_str()); }

  const std::string &path() const { return path_; }

private:
  std::string path_;
};

std::string two_link_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_tight_frame_pose_test_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.2" upper="3.2" velocity="500.0" effort="100.0"/>
  </joint>
  <link name="tip"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.2" upper="3.2" velocity="500.0" effort="100.0"/>
  </joint>
  <link name="tool"/>
  <joint name="tool_fixed" type="fixed">
    <parent link="tip"/>
    <child link="tool"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
</robot>)";
}

Eigen::Matrix4d matrix_from_pose(const pinocchio::SE3 &pose) {
  Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
  matrix.topLeftCorner<3, 3>() = pose.rotation();
  matrix.topRightCorner<3, 1>() = pose.translation();
  return matrix;
}

Eigen::Matrix3d z_rotation(double angle) {
  return Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

GeometricConstraintAccelerationPolicy policy6(double rate,
                                              double acceleration,
                                              double braking) {
  GeometricConstraintAccelerationPolicy policy;
  policy.rate_limits = Eigen::VectorXd::Constant(6, rate);
  policy.acceleration_limits = Eigen::VectorXd::Constant(6, acceleration);
  policy.lower_braking_accelerations =
      Eigen::VectorXd::Constant(6, braking);
  policy.upper_braking_accelerations =
      Eigen::VectorXd::Constant(6, braking);
  return policy;
}

TightFramePoseAccelerationConstraint tight_frame(
    const std::string &source_id, const std::string &frame_name,
    const Eigen::Matrix4d &target_pose,
    const Eigen::Matrix<double, 6, 1> &axis_mask, double position_epsilon,
    double orientation_epsilon,
    const GeometricConstraintAccelerationPolicy &policy) {
  TightFramePoseAccelerationConstraint constraint;
  constraint.source_id = source_id;
  constraint.definition.frame_name = frame_name;
  constraint.definition.target_pose = target_pose;
  constraint.definition.axis_mask = axis_mask;
  constraint.definition.position_epsilon = position_epsilon;
  constraint.definition.orientation_epsilon = orientation_epsilon;
  constraint.policy = policy;
  return constraint;
}

AccelerationSolveOptions options_with_limits(double limit) {
  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({limit, limit});
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  return options;
}

detail::ScalarGeometricPathSegment so3_segment() {
  detail::ScalarGeometricPathSegment segment;
  segment.previous.tangent = Eigen::Vector2d::Zero();
  segment.current.tangent = Eigen::Vector2d::Constant(1e-3);
  segment.previous.coordinates.value = Eigen::VectorXd::Zero(6);
  segment.current.coordinates.value = Eigen::VectorXd::Zero(6);
  segment.previous.coordinates.so3_rotation = Eigen::Matrix3d::Identity();
  segment.current.coordinates.so3_rotation = Eigen::Matrix3d::Identity();
  return segment;
}

class AccelerationSolverTightFramePoseTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_tight_frame_pose_test.urdf", two_link_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
    robot_->set_gravity(Eigen::Vector3d::Zero());
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(AccelerationSolverTightFramePoseTest,
       CapabilitiesExposeFixedFramePoseFamilies) {
  const auto capabilities = AccelerationSolver::capabilities();
  EXPECT_TRUE(capabilities.supports_tight_frame_pose_constraints);
  EXPECT_TRUE(capabilities.supports_torso_pose_bound_constraints);
}

TEST_F(AccelerationSolverTightFramePoseTest,
       LoweredRowsUseExactWorldTranslationAndSo3LogDifferential) {
  const Eigen::VectorXd q = vector({0.45, -0.2});
  const Eigen::VectorXd dq = vector({0.7, -0.15});
  robot_->update_kinematics(q, dq);
  auto reference_pose = robot_->get_frame_pose("tool");
  reference_pose.translation().x() -= 0.05;
  reference_pose.rotation() = reference_pose.rotation() * z_rotation(-0.1);
  const auto exact = detail::evaluate_fixed_frame_pose_differential(
      *robot_, "tool", reference_pose);

  Eigen::Matrix<double, 6, 1> mask =
      Eigen::Matrix<double, 6, 1>::Zero();
  mask(0) = 1.0;
  mask(5) = 1.0;
  const auto record = detail::make_tight_frame_pose_record(tight_frame(
      "tool_x_yaw", "tool", matrix_from_pose(reference_pose), mask, 0.5, 0.5,
      policy6(100.0, 100.0, 100.0)));

  const auto prepared = detail::prepare_fixed_frame_pose_constraint(
      record, *robot_, 0.01, Eigen::Vector2d::Constant(-1000.0),
      Eigen::Vector2d::Constant(1000.0));

  ASSERT_TRUE(prepared.satisfied()) << prepared.message;
  ASSERT_GT(exact.value.head<3>().norm(), 1e-3);
  ASSERT_GT(exact.value.tail<3>().norm(), 1e-3);
  const auto &physical =
      prepared.prepared->scalar.state_box.physical_constraint;
  ASSERT_EQ(physical.coefficient_matrix.rows(), 2);
  EXPECT_TRUE(physical.coefficient_matrix.row(0).isApprox(
      exact.jacobian.row(0), kTolerance));
  EXPECT_TRUE(physical.coefficient_matrix.row(1).isApprox(
      exact.jacobian.row(5), kTolerance));
  EXPECT_NEAR(physical.affine_bias(0), exact.affine_bias(0), kTolerance);
  EXPECT_NEAR(physical.affine_bias(1), exact.affine_bias(5), kTolerance);
}

TEST_F(AccelerationSolverTightFramePoseTest,
       TranslationOnlyBypassesCurrentNearPiOrientation) {
  AccelerationSolver solver(robot_);
  const double near_pi = detail::kSo3Pi - 5e-4;
  const Eigen::VectorXd q = vector({near_pi, 0.0});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  robot_->update_kinematics(q, dq);

  Eigen::Matrix<double, 6, 1> translation_mask =
      Eigen::Matrix<double, 6, 1>::Zero();
  translation_mask(1) = 1.0;
  auto options = options_with_limits(100.0);
  options.tight_frame_pose_constraints.push_back(tight_frame(
      "translation_only", "tool", Eigen::Matrix4d::Identity(),
      translation_mask, 0.1, 0.1, policy6(100.0, 100.0, 100.0)));

  const auto result = solver.solve(q, dq, 0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
}

TEST_F(AccelerationSolverTightFramePoseTest,
       RotationActiveCurrentNearPiFailsAsInvalidInputAndClearsOutputs) {
  AccelerationSolver solver(robot_);
  Eigen::Matrix<double, 6, 1> yaw_mask =
      Eigen::Matrix<double, 6, 1>::Zero();
  yaw_mask(5) = 1.0;
  auto options = options_with_limits(100.0);
  options.tight_frame_pose_constraints.push_back(tight_frame(
      "yaw_near_pi", "tool", Eigen::Matrix4d::Identity(), yaw_mask, 4.0, 4.0,
      policy6(100.0, 100.0, 100.0)));

  const auto result =
      solver.solve(vector({detail::kSo3Pi - 5e-4, 0.0}),
                   Eigen::Vector2d::Zero(), 0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  EXPECT_NE(result.status_message.find("near-pi"), std::string::npos);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTightFramePoseTest,
       RotationActiveBetweenSampleNearPiFailsAsNumericalAndClearsOutputs) {
  AccelerationSolver solver(robot_);
  Eigen::Matrix<double, 6, 1> yaw_mask =
      Eigen::Matrix<double, 6, 1>::Zero();
  yaw_mask(5) = 1.0;
  auto options = options_with_limits(100.0);
  options.tight_frame_pose_constraints.push_back(tight_frame(
      "yaw_segment_guard", "tool", Eigen::Matrix4d::Identity(), yaw_mask, 4.0,
      4.0, policy6(1000.0, 100.0, 100.0)));

  const auto result =
      solver.solve(vector({detail::kSo3Pi - 2e-3, 0.0}), vector({0.8, 0.0}),
                   0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kNumericalError)
      << result.status_message;
  EXPECT_NE(result.status_message.find("SO(3)"), std::string::npos);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTightFramePoseTest,
       PredictedStateSupportRejectionFailsClosed) {
  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");

  const Eigen::VectorXd q = vector({0.2, -0.2});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  constexpr double kDt = 0.1;
  AccelerationTaskReference reference;
  reference.desired_acceleration = -2.0 * q / (kDt * kDt);
  solver.set_task_reference("posture", reference);

  robot_->update_kinematics(q, dq);
  Eigen::Matrix<double, 6, 1> x_mask = Eigen::Matrix<double, 6, 1>::Zero();
  x_mask(0) = 1.0;
  auto options = options_with_limits(1e6);
  options.tight_frame_pose_constraints.push_back(tight_frame(
      "predicted_singular_x", "tool",
      matrix_from_pose(robot_->get_frame_pose("tool")), x_mask, 0.1, 0.1,
      policy6(100.0, 1e5, 1000.0)));

  const auto result = solver.solve(q, dq, kDt, options);

  EXPECT_EQ(result.status, SolverStatus::kNumericalError)
      << result.status_message;
  EXPECT_NE(result.status_message.find("predicted next-state"),
            std::string::npos);
  EXPECT_NE(result.status_message.find("braking support"), std::string::npos);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTightFramePoseTest,
       RejectsMalformedTightFrameInputAndDuplicateSourceIds) {
  AccelerationSolver solver(robot_);
  Eigen::Matrix<double, 6, 1> x_mask = Eigen::Matrix<double, 6, 1>::Zero();
  x_mask(0) = 1.0;
  auto options = options_with_limits(100.0);
  options.tight_frame_pose_constraints.push_back(tight_frame(
      "bad_eps", "tool", Eigen::Matrix4d::Identity(), x_mask, 0.0, 0.1,
      policy6(10.0, 10.0, 10.0)));
  auto result = solver.solve(vector({0.0, 0.0}), Eigen::Vector2d::Zero(), 0.01,
                             options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  auto duplicate = options_with_limits(100.0);
  AffineAccelerationConstraint affine;
  affine.source_id = "dup";
  affine.coefficient_matrix = Eigen::MatrixXd::Identity(1, 2);
  affine.affine_bias = Eigen::VectorXd::Zero(1);
  affine.lower_bounds = vector({-1.0});
  affine.upper_bounds = vector({1.0});
  duplicate.affine_constraints.push_back(affine);
  duplicate.tight_frame_pose_constraints.push_back(tight_frame(
      "dup", "tool", Eigen::Matrix4d::Identity(), x_mask, 0.1, 0.1,
      policy6(10.0, 10.0, 10.0)));
  result = solver.solve(vector({0.0, 0.0}), Eigen::Vector2d::Zero(), 0.01,
                        duplicate);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);
}

TEST(AccelerationSo3SegmentGuardTest, RejectsMalformedSegmentInputs) {
  auto segment = so3_segment();
  segment.previous.coordinates.value = Eigen::VectorXd::Zero(2);
  auto result = detail::validate_so3_log_rotation_segment(segment);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError);

  segment = so3_segment();
  segment.current.tangent = Eigen::VectorXd::Zero(3);
  result = detail::validate_so3_log_rotation_segment(segment);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError);

  segment = so3_segment();
  segment.previous.tangent(0) = std::numeric_limits<double>::quiet_NaN();
  result = detail::validate_so3_log_rotation_segment(segment);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError);

  segment = so3_segment();
  segment.current.coordinates.so3_rotation.reset();
  result = detail::validate_so3_log_rotation_segment(segment);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError);

  segment = so3_segment();
  segment.current.coordinates.so3_rotation = 2.0 * Eigen::Matrix3d::Identity();
  result = detail::validate_so3_log_rotation_segment(segment);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError);
}

} // namespace
} // namespace embodik::test
