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

std::string two_joint_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_solver_affine_test_robot">
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

class AccelerationSolverAffineConstraintTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_solver_affine_test.urdf", two_joint_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
  }

  AccelerationSolveOptions unconstrained_options() {
    AccelerationSolveOptions options;
    options.acceleration_limits_override = vector({20.0, 20.0});
    options.apply_position_limits = false;
    options.apply_velocity_limits = false;
    return options;
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(AccelerationSolverAffineConstraintTest,
       AffineBiasIsAppliedOnceAndOneSidedRowsStayPhysical) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({5.0});
  solver.set_task_reference("joint1_task", reference);

  AffineAccelerationConstraint upper;
  upper.source_id = "upper_bias_once";
  upper.coefficient_matrix = matrix(1, 2, {1.0, 0.0});
  upper.affine_bias = vector({2.0});
  upper.lower_bounds = vector({0.0});
  upper.upper_bounds = vector({3.0});
  upper.lower_bound_active = {false};
  upper.upper_bound_active = {true};

  auto options = unconstrained_options();
  options.affine_constraints.push_back(upper);
  const auto result = solver.solve(vector({0.0, 0.0}),
                                   vector({0.0, 0.0}), 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 1.0, kTolerance);
}

TEST_F(AccelerationSolverAffineConstraintTest,
       LowerAffineBoundUsesThePhysicalBiasSign) {
  AccelerationSolver solver(robot_);
  auto task = solver.add_joint_task("joint1_task", "joint1", 0.0);
  task->setSolveMode(TaskSolveMode::kMinError);
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({-5.0});
  solver.set_task_reference("joint1_task", reference);

  AffineAccelerationConstraint lower;
  lower.source_id = "lower_bias_once";
  lower.coefficient_matrix = matrix(1, 2, {1.0, 0.0});
  lower.affine_bias = vector({2.0});
  lower.lower_bounds = vector({3.0});
  lower.upper_bounds = vector({0.0});
  lower.lower_bound_active = {true};
  lower.upper_bound_active = {false};

  auto options = unconstrained_options();
  options.affine_constraints.push_back(lower);
  const auto result = solver.solve(vector({0.0, 0.0}),
                                   vector({0.0, 0.0}), 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 1.0, kTolerance);
}

TEST_F(AccelerationSolverAffineConstraintTest,
       InactiveSidesRemainInactiveAcrossRowScalesAndLargeBias) {
  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({-1.0, 1.0});
  solver.set_task_reference("posture", reference);

  AffineAccelerationConstraint scaled;
  scaled.source_id = "scaled_one_sided";
  scaled.coefficient_matrix =
      matrix(2, 2, {1.0e12, 0.0, 0.0, 1.0e-12});
  scaled.affine_bias = vector({-1.0e10, 0.0});
  scaled.lower_bounds = vector({123.0, 0.0});
  scaled.upper_bounds = vector({0.0, -123.0});
  scaled.lower_bound_active = {false, true};
  scaled.upper_bound_active = {true, false};

  auto options = unconstrained_options();
  options.affine_constraints.push_back(scaled);
  const auto result = solver.solve(vector({0.0, 0.0}),
                                   vector({0.0, 0.0}), 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.joint_accelerations.isApprox(
      vector({-1.0, 1.0}), kTolerance));
}

TEST_F(AccelerationSolverAffineConstraintTest,
       CoupledAffineIntervalLimitsCombinedTaskScaling) {
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint1_task", "joint1", 0.0);
  solver.add_joint_task("joint2_task", "joint2", 0.0);
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({1.0});
  solver.set_task_reference("joint1_task", reference);
  solver.set_task_reference("joint2_task", reference);

  AffineAccelerationConstraint coupled;
  coupled.source_id = "sum_limit";
  coupled.coefficient_matrix = matrix(1, 2, {1.0, 1.0});
  coupled.affine_bias = vector({0.0});
  coupled.lower_bounds = vector({-10.0});
  coupled.upper_bounds = vector({0.5});

  auto options = unconstrained_options();
  options.affine_constraints.push_back(coupled);
  const auto result = solver.solve(vector({0.0, 0.0}),
                                   vector({0.0, 0.0}), 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations.sum(), 0.5, kTolerance);
  EXPECT_NEAR(result.joint_accelerations(0), 0.25, kTolerance);
  EXPECT_NEAR(result.joint_accelerations(1), 0.25, kTolerance);
}

TEST_F(AccelerationSolverAffineConstraintTest,
       EqualityRankDeficientAndZeroCoefficientRowsAreHandled) {
  AccelerationSolver solver(robot_);

  AffineAccelerationConstraint equality;
  equality.source_id = "rank_deficient_equalities";
  equality.coefficient_matrix =
      matrix(3, 2, {1.0, 0.0, 1.0, 0.0, 0.0, 0.0});
  equality.affine_bias = vector({0.0, 0.0, 0.0});
  equality.lower_bounds = vector({0.5, 0.5, 0.0});
  equality.upper_bounds = vector({0.5, 0.5, 0.0});

  auto options = unconstrained_options();
  options.affine_constraints.push_back(equality);
  const auto result = solver.solve(vector({0.0, 0.0}),
                                   vector({0.0, 0.0}), 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(0), 0.5, kTolerance);
}

TEST_F(AccelerationSolverAffineConstraintTest,
       MalformedAffineRowsFailClosedBeforeBackend) {
  AccelerationSolver solver(robot_);
  const auto q = vector({0.0, 0.0});
  const auto dq = vector({0.0, 0.0});

  AffineAccelerationConstraint empty_id;
  empty_id.coefficient_matrix = matrix(1, 2, {1.0, 0.0});
  empty_id.affine_bias = vector({0.0});
  empty_id.lower_bounds = vector({0.0});
  empty_id.upper_bounds = vector({1.0});
  auto options = unconstrained_options();
  options.affine_constraints.push_back(empty_id);
  auto result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  AffineAccelerationConstraint inactive = empty_id;
  inactive.source_id = "inactive";
  inactive.lower_bound_active = {false};
  inactive.upper_bound_active = {false};
  options = unconstrained_options();
  options.affine_constraints.push_back(inactive);
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  AffineAccelerationConstraint nonfinite = empty_id;
  nonfinite.source_id = "nonfinite";
  nonfinite.coefficient_matrix(0, 0) =
      std::numeric_limits<double>::quiet_NaN();
  options = unconstrained_options();
  options.affine_constraints.push_back(nonfinite);
  result = solver.solve(q, dq, 0.1, options);
  EXPECT_EQ(result.status, SolverStatus::kNonFiniteInput);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverAffineConstraintTest,
       DuplicateAndInfeasibleAffineRowsClearExecutableOutputs) {
  AccelerationSolver solver(robot_);
  AffineAccelerationConstraint row;
  row.source_id = "duplicate";
  row.coefficient_matrix = matrix(1, 2, {1.0, 0.0});
  row.affine_bias = vector({0.0});
  row.lower_bounds = vector({0.0});
  row.upper_bounds = vector({1.0});

  auto options = unconstrained_options();
  options.affine_constraints = {row, row};
  auto result = solver.solve(vector({0.0, 0.0}), vector({0.0, 0.0}), 0.1,
                             options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  AffineAccelerationConstraint infeasible = row;
  infeasible.source_id = "infeasible_zero_row";
  infeasible.coefficient_matrix = matrix(1, 2, {0.0, 0.0});
  infeasible.lower_bounds = vector({1.0});
  infeasible.upper_bounds = vector({1.0});
  options = unconstrained_options();
  options.affine_constraints = {infeasible};
  result = solver.solve(vector({0.0, 0.0}), vector({0.0, 0.0}), 0.1,
                        options);
  EXPECT_NE(result.status, SolverStatus::kSuccess);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverAffineConstraintTest,
       TaskExclusionsDoNotMaskPhysicalAffineRows) {
  AccelerationSolver solver(robot_);
  auto posture = solver.add_posture_task("posture");
  posture->set_excluded_joint_indices({1});
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({0.0, 10.0});
  solver.set_task_reference("posture", reference);

  AffineAccelerationConstraint physical_joint2;
  physical_joint2.source_id = "joint2_physical";
  physical_joint2.coefficient_matrix = matrix(1, 2, {0.0, 1.0});
  physical_joint2.affine_bias = vector({0.0});
  physical_joint2.lower_bounds = vector({1.0});
  physical_joint2.upper_bounds = vector({1.0});

  auto options = unconstrained_options();
  options.affine_constraints.push_back(physical_joint2);
  const auto result = solver.solve(vector({0.0, 0.0}),
                                   vector({0.0, 0.0}), 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(1), 1.0, kTolerance);
}

TEST_F(AccelerationSolverAffineConstraintTest,
       UnrepresentableAffineRowScaleFailsClosed) {
  AccelerationSolver solver(robot_);
  AffineAccelerationConstraint huge;
  huge.source_id = "huge";
  huge.coefficient_matrix =
      matrix(1, 2, {std::numeric_limits<double>::max(), 0.0});
  huge.affine_bias = vector({0.0});
  huge.lower_bounds = vector({0.0});
  huge.upper_bounds = vector({1.0});

  auto options = unconstrained_options();
  options.affine_constraints.push_back(huge);
  const auto result = solver.solve(vector({0.0, 0.0}),
                                   vector({0.0, 0.0}), 0.1, options);

  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);
}

} // namespace
} // namespace embodik::test
