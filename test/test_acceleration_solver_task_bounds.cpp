#include <embodik/acceleration_solver.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

namespace embodik::test {
namespace {

constexpr double kTolerance = 1e-8;

Eigen::VectorXd vector(std::initializer_list<double> values) {
  Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
  Eigen::Index index = 0;
  for (double value : values) {
    result(index++) = value;
  }
  return result;
}

Eigen::MatrixXd matrix(Eigen::Index rows, Eigen::Index cols,
                       std::initializer_list<double> values) {
  Eigen::MatrixXd result(rows, cols);
  Eigen::Index index = 0;
  for (double value : values) {
    result(index / cols, index % cols) = value;
    ++index;
  }
  return result;
}

void expect_no_motion_outputs(const AccelerationSolverResult &result) {
  EXPECT_TRUE(result.solution.empty());
  EXPECT_EQ(result.joint_accelerations.size(), 0);
  EXPECT_EQ(result.joint_velocities_next.size(), 0);
  EXPECT_EQ(result.q_solution.size(), 0);
}

std::string three_joint_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_solver_task_bounds_test_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="2.0" velocity="20.0" effort="20.0"/>
  </joint>
  <link name="link2"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="2.0" velocity="20.0" effort="20.0"/>
  </joint>
  <link name="tip"/>
  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="2.0" velocity="20.0" effort="20.0"/>
  </joint>
</robot>)";
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

class AccelerationSolverTaskBoundsTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_solver_task_bounds_test.urdf",
        three_joint_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
  }

  AccelerationSolveOptions unconstrained_options() {
    AccelerationSolveOptions options;
    options.acceleration_limits_override = vector({100.0, 100.0, 100.0});
    options.apply_position_limits = false;
    options.apply_velocity_limits = false;
    return options;
  }

  TaskAccelerationBounds joint1_bounds(double lower, double upper) {
    TaskAccelerationBounds bounds;
    bounds.source_id = "joint1_task_bounds";
    bounds.task_name = "joint1_task";
    bounds.lower_bounds = vector({lower});
    bounds.upper_bounds = vector({upper});
    return bounds;
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(AccelerationSolverTaskBoundsTest,
       ReferenceOutsideTaskBoundSaturatesButRemainsTarget) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({5.0});
  solver.set_task_reference("joint1_task", reference);

  auto options = unconstrained_options();
  options.task_acceleration_bounds.push_back(joint1_bounds(-10.0, 1.0));
  const auto result = solver.solve(vector({0.0, 0.0, 0.0}),
                                   vector({0.0, 0.0, 0.0}), 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 1.0, kTolerance);
  ASSERT_EQ(result.task_diagnostics.size(), 1);
  EXPECT_NEAR(result.task_diagnostics[0].reference_acceleration(0), 5.0,
              kTolerance);
  EXPECT_NEAR(result.task_diagnostics[0].achieved_acceleration(0) +
                  result.task_diagnostics[0].jacobian_bias(0),
              1.0, kTolerance);
}

TEST_F(AccelerationSolverTaskBoundsTest, OneSidedBoundsUseFiniteSentinels) {
  AccelerationSolver upper_solver(robot_);
  upper_solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference upper_reference;
  upper_reference.desired_acceleration = vector({5.0});
  upper_solver.set_task_reference("joint1_task", upper_reference);
  auto upper_options = unconstrained_options();
  auto upper = joint1_bounds(-123.0, 1.0);
  upper.lower_bound_active = {false};
  upper.upper_bound_active = {true};
  upper_options.task_acceleration_bounds.push_back(upper);

  const auto upper_result =
      upper_solver.solve(vector({0.0, 0.0, 0.0}),
                         vector({0.0, 0.0, 0.0}), 0.1, upper_options);
  ASSERT_EQ(upper_result.status, SolverStatus::kSuccess)
      << upper_result.status_message;
  EXPECT_NEAR(upper_result.joint_accelerations(0), 1.0, kTolerance);

  AccelerationSolver lower_solver(robot_);
  auto lower_task = lower_solver.add_joint_task("joint1_task", "joint1", 0.0);
  lower_task->setSolveMode(TaskSolveMode::kMinError);
  AccelerationTaskReference lower_reference;
  lower_reference.desired_acceleration = vector({-5.0});
  lower_solver.set_task_reference("joint1_task", lower_reference);
  auto lower_options = unconstrained_options();
  auto lower = joint1_bounds(1.0, 123.0);
  lower.lower_bound_active = {true};
  lower.upper_bound_active = {false};
  lower_options.task_acceleration_bounds.push_back(lower);

  const auto lower_result =
      lower_solver.solve(vector({0.0, 0.0, 0.0}),
                         vector({0.0, 0.0, 0.0}), 0.1, lower_options);
  ASSERT_EQ(lower_result.status, SolverStatus::kSuccess)
      << lower_result.status_message;
  EXPECT_NEAR(lower_result.joint_accelerations(0), 1.0, kTolerance);
}

TEST_F(AccelerationSolverTaskBoundsTest,
       FrameBoundsUsePhysicalJdotBiasSign) {
  const auto q = vector({0.4, -0.3, 0.2});
  const auto dq = vector({1.2, -0.7, 0.4});
  robot_->update_kinematics(q, dq);
  const Eigen::Vector3d bias =
      robot_->get_frame_jacobian_bias("tip").head<3>();
  Eigen::Index active_row = 0;
  for (Eigen::Index row = 0; row < bias.size(); ++row) {
    if (std::abs(bias(row)) > std::abs(bias(active_row))) {
      active_row = row;
    }
  }
  ASSERT_GT(std::abs(bias(active_row)), 1e-10);

  AccelerationSolver solver(robot_);
  auto frame = solver.add_frame_task("tip_position", "tip",
                                     TaskType::FRAME_POSITION);
  frame->setTargetPose(robot_->get_frame_pose("tip").translation(),
                       Eigen::Matrix3d::Identity());
  AccelerationTaskReference reference;
  reference.desired_acceleration = Eigen::Vector3d::Zero();
  solver.set_task_reference("tip_position", reference);

  TaskAccelerationBounds bounds;
  bounds.source_id = "tip_physical_zero";
  bounds.task_name = "tip_position";
  bounds.lower_bounds = Eigen::Vector3d::Constant(-100.0);
  bounds.upper_bounds = Eigen::Vector3d::Constant(100.0);
  bounds.lower_bounds(active_row) = 0.0;
  bounds.upper_bounds(active_row) = 0.0;

  auto options = unconstrained_options();
  options.task_acceleration_bounds.push_back(bounds);
  const auto result = solver.solve(q, dq, 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.task_diagnostics.size(), 1);
  const auto physical = result.task_diagnostics[0].achieved_acceleration +
                        result.task_diagnostics[0].jacobian_bias;
  EXPECT_NEAR(physical(active_row), 0.0, kTolerance);
}

TEST_F(AccelerationSolverTaskBoundsTest,
       HardTaskBoundOverridesPriorityHierarchy) {
  AccelerationSolver solver(robot_);
  auto high = solver.add_joint_task("joint1_task", "joint1", 0.0);
  high->setPriority(0);
  auto low = solver.add_joint_task("joint2_task", "joint2", 0.0);
  low->setPriority(1);
  AccelerationTaskReference high_reference;
  high_reference.desired_acceleration = vector({5.0});
  solver.set_task_reference("joint1_task", high_reference);
  AccelerationTaskReference low_reference;
  low_reference.desired_acceleration = vector({-3.0});
  solver.set_task_reference("joint2_task", low_reference);

  auto options = unconstrained_options();
  options.task_acceleration_bounds.push_back(joint1_bounds(-10.0, 1.0));
  const auto result = solver.solve(vector({0.0, 0.0, 0.0}),
                                   vector({0.0, 0.0, 0.0}), 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 1.0, kTolerance);
  EXPECT_NEAR(result.joint_accelerations(1), -3.0, kTolerance);
}

TEST_F(AccelerationSolverTaskBoundsTest,
       TaskExclusionsDoNotMaskPhysicalBoundRows) {
  AccelerationSolver solver(robot_);
  auto task = solver.add_joint_task("joint1_task", "joint1", 0.0);
  task->set_excluded_joint_indices({0});
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({0.0});
  solver.set_task_reference("joint1_task", reference);

  auto options = unconstrained_options();
  options.task_acceleration_bounds.push_back(joint1_bounds(1.0, 1.0));
  const auto result = solver.solve(vector({0.0, 0.0, 0.0}),
                                   vector({0.0, 0.0, 0.0}), 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 1.0, kTolerance);
}

TEST_F(AccelerationSolverTaskBoundsTest,
       ValidationFailuresAreTaxonomizedAndFailClosed) {
  AccelerationSolver solver(robot_);
  auto inactive = solver.add_joint_task("inactive_task", "joint1", 0.0);
  inactive->setActive(false);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  const auto q = vector({0.0, 0.0, 0.0});
  const auto dq = vector({0.0, 0.0, 0.0});

  auto options = unconstrained_options();
  auto bounds = joint1_bounds(-1.0, 1.0);
  bounds.task_name = "missing_task";
  options.task_acceleration_bounds = {bounds};
  auto result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  bounds = joint1_bounds(-1.0, 1.0);
  bounds.task_name = "inactive_task";
  options.task_acceleration_bounds = {bounds};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  bounds = joint1_bounds(-1.0, 1.0);
  options.task_acceleration_bounds = {bounds, bounds};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  AffineAccelerationConstraint affine;
  affine.source_id = "shared_source";
  affine.coefficient_matrix = matrix(1, 3, {1.0, 0.0, 0.0});
  affine.affine_bias = vector({0.0});
  affine.lower_bounds = vector({-1.0});
  affine.upper_bounds = vector({1.0});
  bounds = joint1_bounds(-1.0, 1.0);
  bounds.source_id = "shared_source";
  options.affine_constraints = {affine};
  options.task_acceleration_bounds = {bounds};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  bounds = joint1_bounds(-1.0, 1.0);
  bounds.lower_bounds = vector({-1.0, -1.0});
  options.task_acceleration_bounds = {bounds};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kConstraintBoundsMismatch);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  bounds = joint1_bounds(-1.0, 1.0);
  bounds.lower_bound_active = {true, false};
  options.task_acceleration_bounds = {bounds};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kShapeMismatch);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  bounds = joint1_bounds(-1.0, 1.0);
  bounds.lower_bounds(0) = std::numeric_limits<double>::quiet_NaN();
  options.task_acceleration_bounds = {bounds};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kNonFiniteInput);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  bounds = joint1_bounds(2.0, 1.0);
  options.task_acceleration_bounds = {bounds};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  bounds = joint1_bounds(-1.0, 1.0);
  bounds.lower_bound_active = {false};
  bounds.upper_bound_active = {false};
  options.task_acceleration_bounds = {bounds};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTaskBoundsTest,
       ZeroPhysicalRowsCanBeFeasibleOrContradictory) {
  AccelerationSolver feasible_solver(robot_);
  feasible_solver.add_frame_task("tip_position", "tip",
                                 TaskType::FRAME_POSITION);
  TaskAccelerationBounds feasible;
  feasible.source_id = "zero_z_feasible";
  feasible.task_name = "tip_position";
  feasible.lower_bounds = vector({-100.0, -100.0, 0.0});
  feasible.upper_bounds = vector({100.0, 100.0, 0.0});
  auto options = unconstrained_options();
  options.task_acceleration_bounds = {feasible};
  auto result = feasible_solver.solve(vector({0.0, 0.0, 0.0}),
                                      vector({0.0, 0.0, 0.0}), 0.1,
                                      options);
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;

  AccelerationSolver contradictory_solver(robot_);
  contradictory_solver.add_frame_task("tip_position", "tip",
                                      TaskType::FRAME_POSITION);
  TaskAccelerationBounds contradictory = feasible;
  contradictory.source_id = "zero_z_contradictory";
  contradictory.lower_bounds(2) = 1.0;
  contradictory.upper_bounds(2) = 1.0;
  options = unconstrained_options();
  options.task_acceleration_bounds = {contradictory};
  result = contradictory_solver.solve(vector({0.0, 0.0, 0.0}),
                                      vector({0.0, 0.0, 0.0}), 0.1,
                                      options);
  EXPECT_NE(result.status, SolverStatus::kSuccess);
  expect_no_motion_outputs(result);
}

} // namespace
} // namespace embodik::test
