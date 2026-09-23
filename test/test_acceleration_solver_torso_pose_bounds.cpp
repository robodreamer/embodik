#include "geometric_constraint_differential.hpp"

#include <embodik/acceleration_solver.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
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

std::string two_link_contact_urdf(const std::string &effort1 = "100.0",
                                  const std::string &effort2 = "100.0") {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_torso_pose_bound_test_robot">
  <link name="base_link"/>
  <link name="link1">
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.2" upper="3.2" velocity="500.0" effort=")" +
         effort1 + R"("/>
  </joint>
  <link name="tip">
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.2" upper="3.2" velocity="500.0" effort=")" +
         effort2 + R"("/>
  </joint>
  <link name="torso"/>
  <joint name="torso_fixed" type="fixed">
    <parent link="tip"/>
    <child link="torso"/>
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

AccelerationSolveOptions options_with_limits(double limit) {
  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({limit, limit});
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  return options;
}

AccelerationTaskReference reference(const Eigen::VectorXd &acceleration) {
  AccelerationTaskReference out;
  out.desired_acceleration = acceleration;
  return out;
}

TorsoPoseBoundAccelerationConstraint torso_bound(
    const std::string &source_id, const std::string &frame_name,
    const Eigen::Matrix4d &reference_pose,
    const Eigen::Matrix<double, 6, 1> &lower,
    const Eigen::Matrix<double, 6, 1> &upper,
    const Eigen::Matrix<double, 6, 1> &axis_mask,
    const GeometricConstraintAccelerationPolicy &policy) {
  TorsoPoseBoundAccelerationConstraint constraint;
  constraint.source_id = source_id;
  constraint.definition.frame_name = frame_name;
  constraint.definition.reference_pose = reference_pose;
  constraint.definition.lower_bounds = lower;
  constraint.definition.upper_bounds = upper;
  constraint.definition.axis_mask = axis_mask;
  constraint.policy = policy;
  return constraint;
}

double yaw_physical_acceleration(RobotModel &robot, const Eigen::VectorXd &q,
                                 const Eigen::VectorXd &dq,
                                 const Eigen::VectorXd &ddq,
                                 const Eigen::Matrix4d &reference_pose) {
  robot.update_kinematics(q, dq);
  const auto differential = detail::evaluate_fixed_frame_pose_differential(
      robot, "torso", pinocchio::SE3(reference_pose.topLeftCorner<3, 3>(),
                                     reference_pose.topRightCorner<3, 1>()));
  return (differential.jacobian * ddq + differential.affine_bias)(5);
}

class AccelerationSolverTorsoPoseBoundsTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_torso_pose_bound_test.urdf",
        two_link_contact_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
    robot_->set_gravity(Eigen::Vector3d::Zero());
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(AccelerationSolverTorsoPoseBoundsTest,
       AsymmetricBoundsUseExplicitReferenceWithoutRecentering) {
  const Eigen::VectorXd reference_q = vector({0.0, 0.0});
  const Eigen::VectorXd q = vector({0.12, 0.0});
  const Eigen::VectorXd q_farther_from_reference = vector({0.13, 0.0});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  robot_->update_kinematics(reference_q, dq);
  const Eigen::Matrix4d reference_pose =
      matrix_from_pose(robot_->get_frame_pose("torso"));

  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");
  solver.set_task_reference("posture", reference(vector({100.0, 0.0})));

  Eigen::Matrix<double, 6, 1> lower =
      Eigen::Matrix<double, 6, 1>::Constant(-10.0);
  Eigen::Matrix<double, 6, 1> upper =
      Eigen::Matrix<double, 6, 1>::Constant(10.0);
  lower(5) = -0.20;
  upper(5) = 0.15;
  Eigen::Matrix<double, 6, 1> mask = Eigen::Matrix<double, 6, 1>::Zero();
  mask(5) = 1.0;
  auto options = options_with_limits(1000.0);
  options.torso_pose_bound_constraints.push_back(torso_bound(
      "torso_yaw_bounds", "torso", reference_pose, lower, upper, mask,
      policy6(100.0, 1000.0, 1000.0)));

  const auto result = solver.solve(q, dq, 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  const double yaw_acceleration = yaw_physical_acceleration(
      *robot_, q, dq, result.joint_accelerations, reference_pose);
  EXPECT_LE(yaw_acceleration, 6.0 + kTolerance);
  EXPECT_GT(yaw_acceleration, 0.1);

  const auto reused_options_result =
      solver.solve(q_farther_from_reference, dq, 0.1, options);

  ASSERT_EQ(reused_options_result.status, SolverStatus::kSuccess)
      << reused_options_result.status_message;
  const double reused_yaw_acceleration = yaw_physical_acceleration(
      *robot_, q_farther_from_reference, dq,
      reused_options_result.joint_accelerations, reference_pose);
  EXPECT_LE(reused_yaw_acceleration, 4.0 + kTolerance);
  EXPECT_GT(reused_yaw_acceleration, 0.1);
}

TEST_F(AccelerationSolverTorsoPoseBoundsTest,
       ComposesWithAllocationEffortLocksContactAndTaskExclusions) {
  urdf_ = std::make_unique<ScopedUrdf>(
      "embodik_acceleration_torso_pose_bound_effort_test.urdf",
      two_link_contact_urdf("1e9", "1e9"));
  robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
  robot_->set_gravity(Eigen::Vector3d::Zero());

  const Eigen::VectorXd q = vector({0.25, -0.1});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  robot_->update_kinematics(q, dq);
  const Eigen::Matrix4d reference_pose =
      matrix_from_pose(robot_->get_frame_pose("torso"));

  AccelerationSolver solver(robot_);
  auto task = solver.add_frame_task("excluded_task", "torso",
                                    TaskType::FRAME_POSITION);
  task->set_excluded_joint_indices({0});
  task->setTargetPose(robot_->get_frame_pose("torso").translation() +
                          Eigen::Vector3d(10.0, 0.0, 0.0),
                      Eigen::Matrix3d::Identity());

  Eigen::Matrix<double, 6, 1> lower =
      Eigen::Matrix<double, 6, 1>::Constant(-10.0);
  Eigen::Matrix<double, 6, 1> upper =
      Eigen::Matrix<double, 6, 1>::Constant(10.0);
  lower(5) = 0.0;
  upper(5) = 0.0;
  Eigen::Matrix<double, 6, 1> mask = Eigen::Matrix<double, 6, 1>::Zero();
  mask(5) = 1.0;

  auto options = options_with_limits(100.0);
  GeneralizedAccelerationAllocation allocation;
  allocation.metric_diagonal = vector({2.0, 0.5});
  allocation.reference_acceleration = vector({4.0, -3.0});
  options.generalized_acceleration_allocation = allocation;
  options.effort_constraints = EffortConstraintOptions{};
  options.effort_constraints->limits_override = vector({1e9, 1e9});
  options.zero_acceleration_joint_indices = {0};
  options.contact_acceleration_constraints.push_back(
      ContactAccelerationConstraint{"torso_point_contact", "torso",
                                    ContactType::kPointContact});
  options.torso_pose_bound_constraints.push_back(torso_bound(
      "torso_yaw_lock", "torso", reference_pose, lower, upper, mask,
      policy6(100.0, 100.0, 100.0)));

  const auto result = solver.solve(q, dq, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.allocation_diagnostics.applied);
  EXPECT_EQ(result.predicted_torques.size(), robot_->nv());
  EXPECT_NEAR(result.joint_accelerations(0), 0.0, kTolerance);
  EXPECT_NEAR(yaw_physical_acceleration(*robot_, q, dq,
                                        result.joint_accelerations,
                                        reference_pose),
              0.0, kTolerance);
}

TEST_F(AccelerationSolverTorsoPoseBoundsTest,
       RejectsMalformedInputsDuplicateIdsAndCurrentSupportGaps) {
  AccelerationSolver solver(robot_);
  const Eigen::VectorXd q = vector({0.0, 0.0});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  robot_->update_kinematics(q, dq);
  const Eigen::Matrix4d reference_pose =
      matrix_from_pose(robot_->get_frame_pose("torso"));

  Eigen::Matrix<double, 6, 1> lower =
      Eigen::Matrix<double, 6, 1>::Constant(-1.0);
  Eigen::Matrix<double, 6, 1> upper =
      Eigen::Matrix<double, 6, 1>::Constant(1.0);
  Eigen::Matrix<double, 6, 1> mask = Eigen::Matrix<double, 6, 1>::Zero();
  mask(5) = 1.0;

  auto missing_reference = options_with_limits(100.0);
  TorsoPoseBoundAccelerationConstraint missing;
  missing.source_id = "missing_reference";
  missing.definition.frame_name = "torso";
  missing.definition.lower_bounds = lower;
  missing.definition.upper_bounds = upper;
  missing.definition.axis_mask = mask;
  missing.policy = policy6(10.0, 10.0, 10.0);
  missing_reference.torso_pose_bound_constraints.push_back(missing);
  auto result = solver.solve(q, dq, 0.01, missing_reference);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  auto duplicate = options_with_limits(100.0);
  duplicate.affine_constraints.push_back(AffineAccelerationConstraint{
      "dup", Eigen::MatrixXd::Identity(1, 2), Eigen::VectorXd::Zero(1),
      vector({-1.0}), vector({1.0}), {}, {}});
  duplicate.torso_pose_bound_constraints.push_back(torso_bound(
      "dup", "torso", reference_pose, lower, upper, mask,
      policy6(10.0, 10.0, 10.0)));
  result = solver.solve(q, dq, 0.01, duplicate);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  auto unsupported_z = options_with_limits(100.0);
  Eigen::Matrix<double, 6, 1> z_mask = Eigen::Matrix<double, 6, 1>::Zero();
  z_mask(2) = 1.0;
  unsupported_z.torso_pose_bound_constraints.push_back(torso_bound(
      "unsupported_z", "torso", reference_pose, lower, upper, z_mask,
      policy6(10.0, 10.0, 0.1)));
  result = solver.solve(q, dq, 0.01, unsupported_z);
  EXPECT_EQ(result.status, SolverStatus::kInfeasible);
  EXPECT_NE(result.status_message.find("current-state"), std::string::npos);
  expect_no_motion_outputs(result);
}

} // namespace
} // namespace embodik::test
