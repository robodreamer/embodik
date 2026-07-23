#include <embodik/acceleration_solver.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

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

bool contains_index(const std::vector<int> &indices, int index) {
  return std::find(indices.begin(), indices.end(), index) != indices.end();
}

void expect_no_motion_outputs(const AccelerationSolverResult &result) {
  EXPECT_TRUE(result.solution.empty());
  EXPECT_EQ(result.joint_accelerations.size(), 0);
  EXPECT_EQ(result.joint_velocities_next.size(), 0);
  EXPECT_EQ(result.q_solution.size(), 0);
}

std::string two_joint_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_solver_test_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" velocity="10.0" effort="20.0"/>
  </joint>
  <link name="tip"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" velocity="10.0" effort="20.0"/>
  </joint>
</robot>)";
}

std::string planar_joint_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="planar_acceleration_solver_test_robot">
  <link name="base_link"/>
  <link name="planar_link"/>
  <joint name="planar_joint" type="planar">
    <parent link="base_link"/>
    <child link="planar_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <limit effort="10.0" velocity="10.0"/>
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

class AccelerationSolverTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_solver_test.urdf", two_joint_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
  }

  AccelerationSolveOptions options(Eigen::VectorXd limits) {
    AccelerationSolveOptions out;
    out.acceleration_limits_override = std::move(limits);
    return out;
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

static_assert(std::is_base_of_v<SolverResult, AccelerationSolverResult>);
static_assert(!std::is_copy_constructible_v<AccelerationSolver>);
static_assert(!std::is_copy_assignable_v<AccelerationSolver>);
static_assert(std::is_nothrow_move_constructible_v<AccelerationSolver>);
static_assert(std::is_nothrow_move_assignable_v<AccelerationSolver>);

TEST_F(AccelerationSolverTest, CapabilitiesExposeMinimalR04Scope) {
  const auto capabilities = AccelerationSolver::capabilities();
  EXPECT_TRUE(capabilities.supports_fixed_base_scalar_joints);
  EXPECT_FALSE(capabilities.supports_floating_base);
  EXPECT_FALSE(capabilities.supports_scale_elastic);
  EXPECT_FALSE(capabilities.supports_collision_constraints);
  EXPECT_TRUE(capabilities.supports_effort_constraints);
}

TEST_F(AccelerationSolverTest, RejectsUnsupportedModelsAtConstruction) {
  EXPECT_THROW(AccelerationSolver(nullptr), std::invalid_argument);

  auto floating = std::make_shared<RobotModel>(urdf_->path(), true);
  EXPECT_THROW({ AccelerationSolver unsupported(floating); },
               std::invalid_argument);

  ScopedUrdf planar_urdf("embodik_acceleration_planar_test.urdf",
                         planar_joint_urdf());
  auto planar = std::make_shared<RobotModel>(planar_urdf.path(), false);
  EXPECT_THROW({ AccelerationSolver unsupported(planar); },
               std::invalid_argument);
}

TEST_F(AccelerationSolverTest, RequiresExplicitAccelerationLimits) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  const auto result = solver.solve(vector({0.0, 0.0}), vector({0.0, 0.0}),
                                   0.01);

  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTest, UsesValidatedModelAccelerationLimits) {
  robot_->set_acceleration_limits(vector({0.3, 0.4}));
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({1.0});
  solver.set_task_reference("joint1_task", reference);

  AccelerationSolveOptions solve_options;
  solve_options.apply_position_limits = false;
  solve_options.apply_velocity_limits = false;
  const auto result = solver.solve(vector({0.0, 0.0}),
                                   vector({0.0, 0.0}), 0.01,
                                   solve_options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 0.3, kTolerance);
  EXPECT_TRUE(result.acceleration_limits_applied);
  EXPECT_TRUE(contains_index(result.saturated_acceleration_indices, 0));
}

TEST_F(AccelerationSolverTest, RejectsInvalidExplicitAccelerationLimits) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  const auto q = vector({0.0, 0.0});
  const auto dq = vector({0.0, 0.0});

  auto wrong_size = options(vector({1.0}));
  const auto wrong_size_result = solver.solve(q, dq, 0.01, wrong_size);
  EXPECT_EQ(wrong_size_result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(wrong_size_result);

  auto zero = options(vector({1.0, 0.0}));
  const auto zero_result = solver.solve(q, dq, 0.01, zero);
  EXPECT_EQ(zero_result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(zero_result);

  auto nonfinite_limits = vector({1.0, 1.0});
  nonfinite_limits(1) = std::numeric_limits<double>::quiet_NaN();
  const auto nonfinite_result =
      solver.solve(q, dq, 0.01, options(nonfinite_limits));
  EXPECT_EQ(nonfinite_result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(nonfinite_result);
}

TEST_F(AccelerationSolverTest, SolvesJointTaskWithExplicitStateOnly) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({3.0});
  solver.set_task_reference("joint1_task", reference);

  auto result = solver.solve(vector({0.0, 0.0}), vector({0.0, 0.0}), 0.1,
                             options(vector({10.0, 10.0})));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.joint_accelerations.size(), 2);
  EXPECT_NEAR(result.joint_accelerations(0), 3.0, kTolerance);
  EXPECT_NEAR(result.joint_accelerations(1), 0.0, kTolerance);
  EXPECT_NEAR(result.joint_velocities_next(0), 0.3, kTolerance);
  EXPECT_NEAR(result.q_solution(0), 0.015, kTolerance);
}

TEST_F(AccelerationSolverTest, CombinedPositionRateBoxLimitsAcceleration) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({100.0});
  solver.set_task_reference("joint1_task", reference);

  auto result = solver.solve(vector({0.9, 0.0}), vector({0.0, 0.0}), 0.2,
                             options(vector({100.0, 100.0})));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 2.5, kTolerance);
  ASSERT_FALSE(result.saturated_position_indices.empty());
}

TEST_F(AccelerationSolverTest,
       CombinedLimitShapingPreservesNextStateBrakingViability) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({2.0});
  solver.set_task_reference("joint1_task", reference);

  constexpr double dt = 0.1;
  constexpr double acceleration_limit = 2.0;
  constexpr double expected_ddq = -1.6148351928654925;
  const auto result =
      solver.solve(vector({0.9, 0.0}), vector({0.6, 0.0}), dt,
                   options(vector({acceleration_limit, 10.0})));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), expected_ddq, kTolerance);
  EXPECT_TRUE(contains_index(result.saturated_position_indices, 0));
  EXPECT_NEAR(result.q_solution(0),
              0.9 + 0.6 * dt + 0.5 * expected_ddq * dt * dt, kTolerance);
  const double next_margin = 1.0 - result.q_solution(0);
  EXPECT_LE(result.joint_velocities_next(0) *
                result.joint_velocities_next(0),
            2.0 * acceleration_limit * next_margin + kTolerance);
}

TEST_F(AccelerationSolverTest, CombinedLimitShapingMirrorsAtLowerBoundary) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({-2.0});
  solver.set_task_reference("joint1_task", reference);

  const auto result =
      solver.solve(vector({-0.9, 0.0}), vector({-0.6, 0.0}), 0.1,
                   options(vector({2.0, 10.0})));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 1.6148351928654925, kTolerance);
  EXPECT_TRUE(contains_index(result.saturated_position_indices, 0));
}

TEST_F(AccelerationSolverTest, RawAccelerationLimitSaturatesFirstSolve) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({5.0});
  solver.set_task_reference("joint1_task", reference);

  auto solve_options = options(vector({0.4, 10.0}));
  solve_options.apply_position_limits = false;
  solve_options.apply_velocity_limits = false;
  const auto result = solver.solve(vector({0.0, 0.0}),
                                   vector({0.0, 0.0}), 0.01,
                                   solve_options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.acceleration_limits_applied);
  EXPECT_NEAR(result.joint_accelerations(0), 0.4, kTolerance);
  ASSERT_EQ(result.task_scales.size(), 1U);
  EXPECT_NEAR(result.task_scales[0], 0.08, kTolerance);
  EXPECT_TRUE(contains_index(result.saturated_acceleration_indices, 0));
}

TEST_F(AccelerationSolverTest, AccelerationAndVelocityLimitsAreNative) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference reference;
  reference.desired_velocity = vector({100.0});
  solver.set_task_reference("joint1_task", reference);

  auto solve_options = options(vector({5.0, 5.0}));
  solve_options.apply_position_limits = false;
  auto result = solver.solve(vector({0.0, 0.0}), vector({9.8, 0.0}), 0.1,
                             solve_options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 2.0, kTolerance);
  EXPECT_NEAR(result.joint_velocities_next(0), 10.0, kTolerance);
  ASSERT_FALSE(result.saturated_velocity_indices.empty());
}

TEST_F(AccelerationSolverTest, InfeasibleBrakingEnvelopeFailsClosed) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);

  auto result = solver.solve(vector({0.99, 0.0}), vector({3.0, 0.0}), 0.1,
                             options(vector({1.0, 1.0})));

  EXPECT_EQ(result.status, SolverStatus::kInfeasible);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTest, OutwardVelocityAtBoundaryFailsClosed) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);

  const auto result =
      solver.solve(vector({1.0, 0.0}), vector({1.0, 0.0}), 0.1,
                   options(vector({20.0, 20.0})));

  EXPECT_EQ(result.status, SolverStatus::kInfeasible);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTest, NoActiveTaskStillAppliesMandatoryBraking) {
  AccelerationSolver solver(robot_);

  const auto result =
      solver.solve(vector({0.9, 0.0}), vector({2.0, 0.0}), 0.1,
                   options(vector({20.0, 100.0})));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), -20.0, kTolerance);
  EXPECT_NEAR(result.q_solution(0), 1.0, kTolerance);
  EXPECT_TRUE(result.task_scales.empty());
  EXPECT_TRUE(result.task_diagnostics.empty());
  EXPECT_TRUE(contains_index(result.saturated_acceleration_indices, 0));
  EXPECT_TRUE(contains_index(result.saturated_position_indices, 0));
}

TEST_F(AccelerationSolverTest, RejectsNonfiniteInputsAndScaleElasticFailClosed) {
  AccelerationSolver solver(robot_);
  auto task = solver.add_joint_task("joint1_task", "joint1", 0.0);

  auto nonfinite_q = vector({0.0, 0.0});
  nonfinite_q(0) = std::numeric_limits<double>::quiet_NaN();
  auto nonfinite = solver.solve(nonfinite_q, vector({0.0, 0.0}), 0.1,
                                options(vector({10.0, 10.0})));
  EXPECT_EQ(nonfinite.status, SolverStatus::kNonFiniteInput);
  expect_no_motion_outputs(nonfinite);

  task->setSolveMode(TaskSolveMode::kScaleElastic);
  auto elastic = solver.solve(vector({0.0, 0.0}), vector({0.0, 0.0}), 0.1,
                              options(vector({10.0, 10.0})));
  EXPECT_EQ(elastic.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(elastic);
}

TEST_F(AccelerationSolverTest, InvalidDimensionsAndTimestepsFailClosed) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  const auto solve_options = options(vector({10.0, 10.0}));

  const auto wrong_q = solver.solve(vector({0.0}), vector({0.0, 0.0}), 0.1,
                                    solve_options);
  EXPECT_EQ(wrong_q.status, SolverStatus::kShapeMismatch);
  expect_no_motion_outputs(wrong_q);

  const auto wrong_dq = solver.solve(vector({0.0, 0.0}), vector({0.0}), 0.1,
                                     solve_options);
  EXPECT_EQ(wrong_dq.status, SolverStatus::kShapeMismatch);
  expect_no_motion_outputs(wrong_dq);

  const auto zero_dt = solver.solve(vector({0.0, 0.0}),
                                    vector({0.0, 0.0}), 0.0, solve_options);
  EXPECT_EQ(zero_dt.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(zero_dt);

  const auto nan_dt = solver.solve(
      vector({0.0, 0.0}), vector({0.0, 0.0}),
      std::numeric_limits<double>::quiet_NaN(), solve_options);
  EXPECT_EQ(nan_dt.status, SolverStatus::kNonFiniteInput);
  expect_no_motion_outputs(nan_dt);
}

TEST_F(AccelerationSolverTest, RejectsInvalidTaskReferencesAtSetTime) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);

  AccelerationTaskReference wrong_dimension;
  wrong_dimension.desired_acceleration = vector({1.0, 2.0});
  EXPECT_THROW(solver.set_task_reference("joint1_task", wrong_dimension),
               std::invalid_argument);

  AccelerationTaskReference nonfinite;
  nonfinite.desired_velocity =
      vector({std::numeric_limits<double>::quiet_NaN()});
  EXPECT_THROW(solver.set_task_reference("joint1_task", nonfinite),
               std::invalid_argument);
  EXPECT_THROW(solver.set_task_reference("unknown", {}),
               std::invalid_argument);
}

TEST_F(AccelerationSolverTest, HigherPriorityObjectiveIsPreserved) {
  AccelerationSolver solver(robot_);
  auto primary = solver.add_joint_task("primary", "joint1", 0.0);
  primary->setPriority(0);
  auto secondary = solver.add_joint_task("secondary", "joint1", 0.0);
  secondary->setPriority(1);
  secondary->setSolveMode(TaskSolveMode::kMinError);
  AccelerationTaskReference primary_reference;
  primary_reference.desired_acceleration = vector({2.0});
  AccelerationTaskReference secondary_reference;
  secondary_reference.desired_acceleration = vector({-2.0});
  solver.set_task_reference("primary", primary_reference);
  solver.set_task_reference("secondary", secondary_reference);

  auto solve_options = options(vector({10.0, 10.0}));
  solve_options.apply_position_limits = false;
  solve_options.apply_velocity_limits = false;
  const auto result = solver.solve(vector({0.0, 0.0}), vector({0.0, 0.0}),
                                   0.1, solve_options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 2.0, kTolerance);
}

TEST_F(AccelerationSolverTest, TaskWeightAppliesOnceToPositionError) {
  AccelerationSolver solver(robot_);
  auto task = solver.add_joint_task("joint1_task", "joint1", 0.4);
  task->setWeight(0.25);
  AccelerationTaskReference reference;
  reference.proportional_gain = 4.0;
  solver.set_task_reference("joint1_task", reference);

  auto solve_options = options(vector({10.0, 10.0}));
  solve_options.apply_position_limits = false;
  solve_options.apply_velocity_limits = false;
  const auto result = solver.solve(vector({0.0, 0.0}), vector({0.0, 0.0}),
                                   0.1, solve_options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 0.4, kTolerance);
}

TEST_F(AccelerationSolverTest,
       UnconstrainedAccelerationMatchesVelocityEquivalentReference) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference reference;
  reference.desired_velocity = vector({0.4});
  reference.derivative_gain = 10.0;
  reference.proportional_gain = 0.0;
  solver.set_task_reference("joint1_task", reference);

  auto solve_options = options(vector({100.0, 100.0}));
  solve_options.apply_position_limits = false;
  solve_options.apply_velocity_limits = false;
  const auto result = solver.solve(vector({0.0, 0.0}),
                                   vector({0.0, 0.0}), 0.1,
                                   solve_options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 4.0, kTolerance);
  EXPECT_NEAR(result.joint_velocities_next(0), 0.4, kTolerance);
}

TEST_F(AccelerationSolverTest, RepeatedExplicitStateHasNoHiddenCascade) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({1.5});
  solver.set_task_reference("joint1_task", reference);
  auto solve_options = options(vector({10.0, 10.0}));
  solve_options.apply_position_limits = false;
  solve_options.apply_velocity_limits = false;
  const auto q = vector({0.1, -0.2});
  const auto dq = vector({0.3, -0.4});

  const auto first = solver.solve(q, dq, 0.02, solve_options);
  const auto second = solver.solve(q, dq, 0.02, solve_options);

  ASSERT_EQ(first.status, SolverStatus::kSuccess) << first.status_message;
  ASSERT_EQ(second.status, SolverStatus::kSuccess) << second.status_message;
  EXPECT_TRUE(first.joint_accelerations.isApprox(
      second.joint_accelerations, kTolerance));
  EXPECT_TRUE(first.joint_velocities_next.isApprox(
      second.joint_velocities_next, kTolerance));
  EXPECT_TRUE(first.q_solution.isApprox(second.q_solution, kTolerance));
}

} // namespace
} // namespace embodik::test
