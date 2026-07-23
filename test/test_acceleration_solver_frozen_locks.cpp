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

constexpr double kTolerance = 1e-9;

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
<robot name="acceleration_solver_frozen_locks_test_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" velocity="10.0" effort="20.0"/>
  </joint>
  <link name="link2"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" velocity="10.0" effort="20.0"/>
  </joint>
  <link name="tip"/>
  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" velocity="10.0" effort="20.0"/>
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

class AccelerationSolverFrozenLocksTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_solver_frozen_locks_test.urdf",
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

  void add_posture_reference(AccelerationSolver &solver,
                             const Eigen::VectorXd &ddq) {
    solver.add_posture_task("posture");
    AccelerationTaskReference reference;
    reference.desired_acceleration = ddq;
    solver.set_task_reference("posture", reference);
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(AccelerationSolverFrozenLocksTest, PublishedFrozenFixtureSaturatesSum) {
  AccelerationSolver solver(robot_);
  add_posture_reference(solver, vector({10.0, 10.0, 0.0}));

  FrozenNextVelocityConstraint frozen;
  frozen.source_id = "published_sum_fixture";
  frozen.coefficient_matrix = matrix(1, 3, {1.0, 1.0, 0.0});
  frozen.affine_bias = vector({0.5});
  frozen.lower_bounds = vector({-10.0});
  frozen.upper_bounds = vector({0.4});

  auto options = unconstrained_options();
  options.frozen_next_velocity_constraints.push_back(frozen);
  const auto result =
      solver.solve(vector({0.0, 0.0, 0.0}), vector({0.2, -0.1, 0.0}), 0.1,
                   options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0) + result.joint_accelerations(1),
              2.5, kTolerance);
  EXPECT_NEAR(result.joint_velocities_next(0) +
                  result.joint_velocities_next(1) + 0.1 * 0.5,
              0.4, kTolerance);
}

TEST_F(AccelerationSolverFrozenLocksTest, FrozenRowsMatchEquivalentDirectRows) {
  const auto q = vector({0.0, 0.0, 0.0});
  const auto dq = vector({0.4, -0.2, 0.1});
  constexpr double dt = 0.05;

  FrozenNextVelocityConstraint frozen;
  frozen.source_id = "same_physics";
  frozen.coefficient_matrix = matrix(1, 3, {2.0, -1.0, 0.0});
  frozen.affine_bias = vector({0.3});
  frozen.lower_bounds = vector({-0.5});
  frozen.upper_bounds = vector({0.7});

  AffineAccelerationConstraint direct;
  direct.source_id = "same_physics_direct";
  direct.coefficient_matrix = dt * frozen.coefficient_matrix;
  direct.affine_bias =
      frozen.coefficient_matrix * dq + dt * frozen.affine_bias;
  direct.lower_bounds = frozen.lower_bounds;
  direct.upper_bounds = frozen.upper_bounds;

  AccelerationSolver frozen_solver(robot_);
  auto frozen_posture = frozen_solver.add_posture_task("posture");
  frozen_posture->setSolveMode(TaskSolveMode::kMinError);
  AccelerationTaskReference frozen_reference;
  frozen_reference.desired_acceleration = vector({10.0, -10.0, 0.0});
  frozen_solver.set_task_reference("posture", frozen_reference);
  auto frozen_options = unconstrained_options();
  frozen_options.frozen_next_velocity_constraints.push_back(frozen);
  const auto frozen_result = frozen_solver.solve(q, dq, dt, frozen_options);

  AccelerationSolver direct_solver(robot_);
  auto direct_posture = direct_solver.add_posture_task("posture");
  direct_posture->setSolveMode(TaskSolveMode::kMinError);
  AccelerationTaskReference direct_reference;
  direct_reference.desired_acceleration = vector({10.0, -10.0, 0.0});
  direct_solver.set_task_reference("posture", direct_reference);
  auto direct_options = unconstrained_options();
  direct_options.affine_constraints.push_back(direct);
  const auto direct_result = direct_solver.solve(q, dq, dt, direct_options);

  ASSERT_EQ(frozen_result.status, SolverStatus::kSuccess)
      << frozen_result.status_message;
  ASSERT_EQ(direct_result.status, SolverStatus::kSuccess)
      << direct_result.status_message;
  EXPECT_TRUE(frozen_result.joint_accelerations.isApprox(
      direct_result.joint_accelerations, kTolerance));
}

TEST_F(AccelerationSolverFrozenLocksTest,
       OneSidedFrozenRowsUseRowScaledFiniteSentinel) {
  AccelerationSolver solver(robot_);
  add_posture_reference(solver, vector({-1.0, 0.0, 0.0}));

  FrozenNextVelocityConstraint scaled;
  scaled.source_id = "scaled_one_sided";
  scaled.coefficient_matrix = matrix(1, 3, {1.0e12, 0.0, 0.0});
  scaled.affine_bias = vector({0.0});
  scaled.lower_bounds = vector({123.0});
  scaled.upper_bounds = vector({0.0});
  scaled.lower_bound_active = {false};
  scaled.upper_bound_active = {true};

  auto options = unconstrained_options();
  options.frozen_next_velocity_constraints.push_back(scaled);
  const auto result = solver.solve(vector({0.0, 0.0, 0.0}),
                                   vector({0.0, 0.0, 0.0}), 1.0, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), -1.0, kTolerance);
}

TEST_F(AccelerationSolverFrozenLocksTest,
       FrozenRepresentabilityIsCheckedAfterTimestepScaling) {
  AccelerationSolver solver(robot_);
  add_posture_reference(solver, vector({-1.0, 0.0, 0.0}));

  FrozenNextVelocityConstraint scaled;
  scaled.source_id = "timestep_scaled";
  scaled.coefficient_matrix = matrix(1, 3, {1.0e300, 0.0, 0.0});
  scaled.affine_bias = vector({0.0});
  scaled.lower_bounds = vector({123.0});
  scaled.upper_bounds = vector({0.0});
  scaled.lower_bound_active = {false};
  scaled.upper_bound_active = {true};

  auto options = unconstrained_options();
  options.frozen_next_velocity_constraints.push_back(scaled);
  const auto result = solver.solve(vector({0.0, 0.0, 0.0}),
                                   vector({0.0, 0.0, 0.0}), 1.0e-300,
                                   options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), -1.0, kTolerance);
}

TEST_F(AccelerationSolverFrozenLocksTest,
       ReusedFrozenOptionsRecomputeFromCurrentDq) {
  AccelerationSolver solver(robot_);
  FrozenNextVelocityConstraint stop_joint1;
  stop_joint1.source_id = "stop_joint1";
  stop_joint1.coefficient_matrix = matrix(1, 3, {1.0, 0.0, 0.0});
  stop_joint1.affine_bias = vector({0.0});
  stop_joint1.lower_bounds = vector({0.0});
  stop_joint1.upper_bounds = vector({0.0});

  auto options = unconstrained_options();
  options.frozen_next_velocity_constraints.push_back(stop_joint1);
  const auto q = vector({0.0, 0.0, 0.0});
  const auto first =
      solver.solve(q, vector({0.4, 0.0, 0.0}), 0.1, options);
  const auto second =
      solver.solve(q, vector({-0.2, 0.0, 0.0}), 0.1, options);

  ASSERT_EQ(first.status, SolverStatus::kSuccess) << first.status_message;
  ASSERT_EQ(second.status, SolverStatus::kSuccess) << second.status_message;
  EXPECT_NEAR(first.joint_accelerations(0), -4.0, kTolerance);
  EXPECT_NEAR(second.joint_accelerations(0), 2.0, kTolerance);
  EXPECT_NEAR(first.joint_velocities_next(0), 0.0, kTolerance);
  EXPECT_NEAR(second.joint_velocities_next(0), 0.0, kTolerance);
}

TEST_F(AccelerationSolverFrozenLocksTest, PhysicalAcceptanceUsesVelocityUnits) {
  AccelerationSolver solver(robot_);

  FrozenNextVelocityConstraint next_velocity;
  next_velocity.source_id = "velocity_units";
  next_velocity.coefficient_matrix = matrix(1, 3, {1.0, 0.0, 0.0});
  next_velocity.affine_bias = vector({2.0});
  next_velocity.lower_bounds = vector({0.55});
  next_velocity.upper_bounds = vector({0.55});

  auto options = unconstrained_options();
  options.frozen_next_velocity_constraints.push_back(next_velocity);
  const auto result =
      solver.solve(vector({0.0, 0.0, 0.0}), vector({0.25, 0.0, 0.0}), 0.1,
                   options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_velocities_next(0) + 0.1 * 2.0, 0.55,
              kTolerance);
}

TEST_F(AccelerationSolverFrozenLocksTest,
       MalformedFrozenRowsAndArithmeticFailClosed) {
  AccelerationSolver solver(robot_);
  const auto q = vector({0.0, 0.0, 0.0});
  const auto dq = vector({0.0, 0.0, 0.0});

  FrozenNextVelocityConstraint row;
  row.source_id = "row";
  row.coefficient_matrix = matrix(1, 3, {1.0, 0.0, 0.0});
  row.affine_bias = vector({0.0});
  row.lower_bounds = vector({0.0});
  row.upper_bounds = vector({1.0});

  auto empty_id = row;
  empty_id.source_id.clear();
  auto options = unconstrained_options();
  options.frozen_next_velocity_constraints = {empty_id};
  auto result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  auto inactive = row;
  inactive.lower_bound_active = {false};
  inactive.upper_bound_active = {false};
  options = unconstrained_options();
  options.frozen_next_velocity_constraints = {inactive};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  auto wrong_columns = row;
  wrong_columns.coefficient_matrix = matrix(1, 2, {1.0, 0.0});
  options = unconstrained_options();
  options.frozen_next_velocity_constraints = {wrong_columns};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kShapeMismatch);
  expect_no_motion_outputs(result);

  auto wrong_flags = row;
  wrong_flags.lower_bound_active = {true, false};
  options = unconstrained_options();
  options.frozen_next_velocity_constraints = {wrong_flags};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kShapeMismatch);
  expect_no_motion_outputs(result);

  auto nonfinite = row;
  nonfinite.affine_bias(0) = std::numeric_limits<double>::quiet_NaN();
  options = unconstrained_options();
  options.frozen_next_velocity_constraints = {nonfinite};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kNonFiniteInput);
  expect_no_motion_outputs(result);

  auto inverted = row;
  inverted.lower_bounds = vector({2.0});
  inverted.upper_bounds = vector({1.0});
  options = unconstrained_options();
  options.frozen_next_velocity_constraints = {inverted};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  auto overflow = row;
  overflow.coefficient_matrix = matrix(1, 3, {1.0e150, 0.0, 0.0});
  options = unconstrained_options();
  options.frozen_next_velocity_constraints = {overflow};
  result = solver.solve(q, vector({1.0e160, 0.0, 0.0}), 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError);
  expect_no_motion_outputs(result);

  auto underflow = row;
  underflow.coefficient_matrix = matrix(1, 3, {1.0e-320, 0.0, 0.0});
  options = unconstrained_options();
  options.frozen_next_velocity_constraints = {underflow};
  result = solver.solve(q, dq, 1.0e-10, options);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverFrozenLocksTest, CrossFamilyDuplicateIdsFailClosed) {
  AccelerationSolver solver(robot_);
  AffineAccelerationConstraint direct;
  direct.source_id = "duplicate";
  direct.coefficient_matrix = matrix(1, 3, {1.0, 0.0, 0.0});
  direct.affine_bias = vector({0.0});
  direct.lower_bounds = vector({0.0});
  direct.upper_bounds = vector({1.0});

  FrozenNextVelocityConstraint frozen;
  frozen.source_id = "duplicate";
  frozen.coefficient_matrix = matrix(1, 3, {1.0, 0.0, 0.0});
  frozen.affine_bias = vector({0.0});
  frozen.lower_bounds = vector({0.0});
  frozen.upper_bounds = vector({1.0});

  auto options = unconstrained_options();
  options.affine_constraints = {direct};
  options.frozen_next_velocity_constraints = {frozen};
  const auto result = solver.solve(vector({0.0, 0.0, 0.0}),
                                   vector({0.0, 0.0, 0.0}), 0.1, options);

  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverFrozenLocksTest, ThreeLockPoliciesRemainDistinct) {
  AccelerationSolver solver(robot_);

  auto options = unconstrained_options();
  options.zero_acceleration_joint_indices = {0};
  options.zero_next_velocity_joint_indices = {1};
  options.fixed_current_position_joint_indices = {2};
  const auto dq = vector({0.3, 0.4, -0.5});
  const auto result =
      solver.solve(vector({0.1, 0.2, 0.3}), dq, 0.2, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 0.0, kTolerance);
  EXPECT_NEAR(result.joint_accelerations(1), -2.0, kTolerance);
  EXPECT_NEAR(result.joint_accelerations(2), 5.0, kTolerance);
  EXPECT_NEAR(result.joint_velocities_next(1), 0.0, kTolerance);
  EXPECT_NEAR(result.q_solution(2), 0.3, kTolerance);
}

TEST_F(AccelerationSolverFrozenLocksTest,
       RestingLocksRemainFeasibleAtTinyPositiveTimestep) {
  AccelerationSolver solver(robot_);
  auto options = unconstrained_options();
  options.zero_next_velocity_joint_indices = {0};
  options.fixed_current_position_joint_indices = {1};

  const auto result =
      solver.solve(vector({0.0, 0.0, 0.0}), vector({0.0, 0.0, 0.0}),
                   1.0e-308, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.joint_accelerations.isZero(kTolerance));
  EXPECT_TRUE(result.joint_velocities_next.isZero(kTolerance));
  EXPECT_TRUE(result.q_solution.isZero(kTolerance));
}

TEST_F(AccelerationSolverFrozenLocksTest,
       DuplicateOverlapAndOutOfRangeLocksFailClosed) {
  AccelerationSolver solver(robot_);
  const auto q = vector({0.0, 0.0, 0.0});
  const auto dq = vector({0.0, 0.0, 0.0});

  auto options = unconstrained_options();
  options.zero_acceleration_joint_indices = {0, 0};
  auto result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  options.zero_acceleration_joint_indices = {0};
  options.zero_next_velocity_joint_indices = {0};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  options.fixed_current_position_joint_indices = {3};
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverFrozenLocksTest,
       LockConflictWithJointStateBoxFailsClosed) {
  AccelerationSolver solver(robot_);
  auto options = unconstrained_options();
  options.apply_position_limits = true;
  options.zero_acceleration_joint_indices = {0};
  const auto result =
      solver.solve(vector({0.99, 0.0, 0.0}), vector({3.0, 0.0, 0.0}), 0.1,
                   options);

  EXPECT_NE(result.status, SolverStatus::kSuccess);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverFrozenLocksTest, NoTaskFrozenRowsStillApply) {
  AccelerationSolver solver(robot_);
  FrozenNextVelocityConstraint stop_joint1;
  stop_joint1.source_id = "no_task_stop";
  stop_joint1.coefficient_matrix = matrix(1, 3, {1.0, 0.0, 0.0});
  stop_joint1.affine_bias = vector({0.0});
  stop_joint1.lower_bounds = vector({0.0});
  stop_joint1.upper_bounds = vector({0.0});

  auto options = unconstrained_options();
  options.frozen_next_velocity_constraints.push_back(stop_joint1);
  const auto result =
      solver.solve(vector({0.0, 0.0, 0.0}), vector({0.3, 0.0, 0.0}), 0.1,
                   options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.task_diagnostics.empty());
  EXPECT_NEAR(result.joint_accelerations(0), -3.0, kTolerance);
  EXPECT_NEAR(result.joint_velocities_next(0), 0.0, kTolerance);
}

} // namespace
} // namespace embodik::test
