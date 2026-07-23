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

Eigen::VectorXd solution_vector(const AccelerationSolverResult &result) {
  if (result.solution.empty()) {
    return {};
  }
  return Eigen::Map<const Eigen::VectorXd>(result.solution.data(),
                                           result.solution.size());
}

std::string three_joint_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_solver_allocation_test_robot">
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

class AccelerationSolverAllocationTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_solver_allocation_test.urdf",
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

  GeneralizedAccelerationAllocation allocation(Eigen::VectorXd metric,
                                               Eigen::VectorXd reference) {
    GeneralizedAccelerationAllocation out;
    out.metric_diagonal = std::move(metric);
    out.reference_acceleration = std::move(reference);
    return out;
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

TEST_F(AccelerationSolverAllocationTest, NoTaskReferenceInsideStateBoxIsHeld) {
  AccelerationSolver solver(robot_);

  auto options = unconstrained_options();
  options.generalized_acceleration_allocation =
      allocation(vector({4.0, 1.0, 9.0}), vector({1.0, -2.0, 0.5}));
  const auto result =
      solver.solve(vector({0.0, 0.0, 0.0}), vector({0.0, 0.0, 0.0}), 0.1,
                   options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.joint_accelerations.isApprox(vector({1.0, -2.0, 0.5}),
                                                  kTolerance));
  EXPECT_TRUE(solution_vector(result).isApprox(result.joint_accelerations, 0.0));
  EXPECT_TRUE(result.allocation_diagnostics.applied);
  EXPECT_TRUE(result.allocation_diagnostics.weighted_physical_residual
                  .isZero(kTolerance));
  EXPECT_NEAR(result.allocation_diagnostics.objective_value, 0.0, kTolerance);
}

TEST_F(AccelerationSolverAllocationTest, NoTaskReferenceOutsideStateBoxClamps) {
  AccelerationSolver solver(robot_);

  auto options = unconstrained_options();
  options.acceleration_limits_override = vector({1.0, 2.0, 3.0});
  options.generalized_acceleration_allocation =
      allocation(vector({1.0, 9.0, 4.0}), vector({5.0, -5.0, 0.5}));
  const auto result =
      solver.solve(vector({0.0, 0.0, 0.0}), vector({0.0, 0.0, 0.0}), 0.1,
                   options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.joint_accelerations.isApprox(vector({1.0, -2.0, 0.5}),
                                                  kTolerance));
  EXPECT_NEAR(result.allocation_diagnostics.weighted_physical_residual(0),
              -4.0, kTolerance);
  EXPECT_NEAR(result.allocation_diagnostics.weighted_physical_residual(1),
              9.0, kTolerance);
}

TEST_F(AccelerationSolverAllocationTest,
       UniformZeroReferenceMatchesBaselineExactly) {
  AccelerationSolver baseline(robot_);
  add_posture_reference(baseline, vector({2.0, -1.0, 0.5}));
  auto options = unconstrained_options();
  const auto baseline_result =
      baseline.solve(vector({0.0, 0.0, 0.0}), vector({0.0, 0.0, 0.0}), 0.1,
                     options);

  AccelerationSolver allocated(robot_);
  add_posture_reference(allocated, vector({2.0, -1.0, 0.5}));
  options.generalized_acceleration_allocation =
      allocation(Eigen::Vector3d::Constant(3.0), Eigen::Vector3d::Zero());
  const auto allocated_result =
      allocated.solve(vector({0.0, 0.0, 0.0}), vector({0.0, 0.0, 0.0}), 0.1,
                      options);

  ASSERT_EQ(baseline_result.status, SolverStatus::kSuccess)
      << baseline_result.status_message;
  ASSERT_EQ(allocated_result.status, SolverStatus::kSuccess)
      << allocated_result.status_message;
  EXPECT_TRUE(allocated_result.joint_accelerations.isApprox(
      baseline_result.joint_accelerations, 0.0));
  EXPECT_TRUE(solution_vector(allocated_result)
                  .isApprox(allocated_result.joint_accelerations, 0.0));
}

TEST_F(AccelerationSolverAllocationTest,
       PhysicalRowsAndDiagnosticsUseReconstructedAcceleration) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference high;
  high.desired_acceleration = vector({10.0});
  solver.set_task_reference("joint1_task", high);

  TaskAccelerationBounds task_bound;
  task_bound.source_id = "joint1_bound";
  task_bound.task_name = "joint1_task";
  task_bound.lower_bounds = vector({0.4});
  task_bound.upper_bounds = vector({0.4});

  FrozenNextVelocityConstraint frozen;
  frozen.source_id = "joint3_stop";
  frozen.coefficient_matrix = matrix(1, 3, {0.0, 0.0, 1.0});
  frozen.affine_bias = vector({0.0});
  frozen.lower_bounds = vector({0.0});
  frozen.upper_bounds = vector({0.0});

  AffineAccelerationConstraint affine;
  affine.source_id = "sum_bound";
  affine.coefficient_matrix = matrix(1, 3, {1.0, 1.0, 0.0});
  affine.affine_bias = vector({0.1});
  affine.lower_bounds = vector({-10.0});
  affine.upper_bounds = vector({0.7});

  auto options = unconstrained_options();
  options.generalized_acceleration_allocation =
      allocation(Eigen::Vector3d::Ones(), vector({1.0, -1.0, 2.0}));
  options.task_acceleration_bounds.push_back(task_bound);
  options.frozen_next_velocity_constraints.push_back(frozen);
  options.affine_constraints.push_back(affine);
  options.zero_next_velocity_joint_indices = {1};
  const auto result =
      solver.solve(vector({0.0, 0.0, 0.0}), vector({0.0, -0.02, 0.3}), 0.1,
                   options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 0.4, kTolerance);
  EXPECT_NEAR(result.joint_accelerations(1), 0.2, kTolerance);
  EXPECT_NEAR(result.joint_accelerations(2), -3.0, kTolerance);
  EXPECT_NEAR(result.joint_accelerations(0) + result.joint_accelerations(1) +
                  0.1,
              0.7, kTolerance);
  ASSERT_EQ(result.task_diagnostics.size(), 1);
  EXPECT_NEAR(result.task_diagnostics[0].achieved_acceleration(0), 0.4,
              kTolerance);
}

TEST_F(AccelerationSolverAllocationTest,
       RedundantFrameTaskFollowsPhysicalMetric) {
  const auto q = vector({0.0, 0.0, 0.0});
  const auto dq = vector({0.0, 0.0, 0.0});
  robot_->update_kinematics(q, dq);

  AccelerationSolver solver(robot_);
  auto frame = solver.add_frame_task("tip_position", "tip",
                                     TaskType::FRAME_POSITION);
  frame->setTargetPose(robot_->get_frame_pose("tip").translation(),
                       Eigen::Matrix3d::Identity());
  frame->setSolveMode(TaskSolveMode::kMinError);
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({0.0, 1.0, 0.0});
  reference.proportional_gain = 0.0;
  reference.derivative_gain = 0.0;
  solver.set_task_reference("tip_position", reference);

  auto options = unconstrained_options();
  options.generalized_acceleration_allocation =
      allocation(vector({4.0, 1.0, 1.0}), Eigen::Vector3d::Zero());
  const auto result = solver.solve(q, dq, 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 0.25, kTolerance);
  EXPECT_NEAR(result.joint_accelerations(1), 0.5, kTolerance);
  EXPECT_NEAR(result.joint_accelerations(2), 0.0, kTolerance);
  ASSERT_EQ(result.task_diagnostics.size(), 1);
  EXPECT_NEAR(result.task_diagnostics[0].achieved_acceleration(1), 1.0,
              kTolerance);
}

TEST_F(AccelerationSolverAllocationTest,
       InvalidAllocationFailsClosedAndClearsOutputs) {
  AccelerationSolver solver(robot_);
  add_posture_reference(solver, vector({1.0, 1.0, 1.0}));

  auto options = unconstrained_options();
  options.generalized_acceleration_allocation =
      allocation(vector({1.0, 1.0}), Eigen::Vector3d::Zero());
  auto result =
      solver.solve(vector({0.0, 0.0, 0.0}), vector({0.0, 0.0, 0.0}), 0.1,
                   options);
  EXPECT_EQ(result.status, SolverStatus::kShapeMismatch);
  expect_no_motion_outputs(result);

  options.generalized_acceleration_allocation =
      allocation(vector({1.0, 0.0, 1.0}), Eigen::Vector3d::Zero());
  result = solver.solve(vector({0.0, 0.0, 0.0}),
                        vector({0.0, 0.0, 0.0}), 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  options.generalized_acceleration_allocation = allocation(
      vector({1.0, 1.0, 1.0}),
      vector({0.0, std::numeric_limits<double>::quiet_NaN(), 0.0}));
  result = solver.solve(vector({0.0, 0.0, 0.0}),
                        vector({0.0, 0.0, 0.0}), 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kNonFiniteInput);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverAllocationTest,
       NonfiniteAllocationDiagnosticsFailClosed) {
  AccelerationSolver solver(robot_);
  add_posture_reference(solver, vector({1.0, 1.0, 1.0}));

  auto options = unconstrained_options();
  options.generalized_acceleration_allocation = allocation(
      Eigen::Vector3d::Constant(1e308), Eigen::Vector3d::Zero());
  const auto result =
      solver.solve(vector({0.0, 0.0, 0.0}), vector({0.0, 0.0, 0.0}), 0.1,
                   options);

  EXPECT_EQ(result.status, SolverStatus::kNumericalError);
  expect_no_motion_outputs(result);
  EXPECT_FALSE(result.allocation_diagnostics.applied);
  EXPECT_EQ(result.allocation_diagnostics.weighted_physical_residual.size(), 0);
}

TEST_F(AccelerationSolverAllocationTest, RepeatedAllocatedSolveIsDeterministic) {
  AccelerationSolver solver(robot_);
  add_posture_reference(solver, vector({3.0, -4.0, 5.0}));

  auto options = unconstrained_options();
  options.generalized_acceleration_allocation =
      allocation(vector({8.0, 2.0, 5.0}), vector({0.5, -0.25, 0.75}));
  const auto q = vector({0.1, -0.2, 0.3});
  const auto dq = vector({0.4, -0.5, 0.6});
  const auto first = solver.solve(q, dq, 0.02, options);
  const auto second = solver.solve(q, dq, 0.02, options);

  ASSERT_EQ(first.status, SolverStatus::kSuccess) << first.status_message;
  ASSERT_EQ(second.status, SolverStatus::kSuccess) << second.status_message;
  EXPECT_TRUE(first.joint_accelerations.isApprox(second.joint_accelerations,
                                                 kTolerance));
  EXPECT_TRUE(first.joint_velocities_next.isApprox(second.joint_velocities_next,
                                                   kTolerance));
  EXPECT_TRUE(first.q_solution.isApprox(second.q_solution, kTolerance));
}

} // namespace
} // namespace embodik::test
