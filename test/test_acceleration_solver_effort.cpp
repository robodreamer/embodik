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
constexpr double kEffortTolerance = 1e-7;

Eigen::VectorXd vector(std::initializer_list<double> values) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(values.size()));
  Eigen::Index index = 0;
  for (double value : values) {
    out(index++) = value;
  }
  return out;
}

bool contains_index(const std::vector<int> &indices, int index) {
  return std::find(indices.begin(), indices.end(), index) != indices.end();
}

void expect_no_motion_outputs(const AccelerationSolverResult &result) {
  EXPECT_TRUE(result.solution.empty());
  EXPECT_EQ(result.joint_accelerations.size(), 0);
  EXPECT_EQ(result.joint_velocities_next.size(), 0);
  EXPECT_EQ(result.q_solution.size(), 0);
  EXPECT_EQ(result.predicted_torques.size(), 0);
  EXPECT_TRUE(result.saturated_effort_indices.empty());
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

std::string two_link_dynamics_urdf(const std::string &effort1 = "100.0",
                                   const std::string &effort2 = "100.0") {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_effort_test_robot">
  <link name="base_link"/>
  <link name="link1">
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="5.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.2"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="20.0" effort=")" +
         effort1 + R"("/>
  </joint>
  <link name="link2">
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="3.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="20.0" effort=")" +
         effort2 + R"("/>
  </joint>
</robot>)";
}

AccelerationSolveOptions limit_options(const Eigen::VectorXd &limits) {
  AccelerationSolveOptions options;
  options.acceleration_limits_override = limits;
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  return options;
}

AccelerationTaskReference reference(const Eigen::VectorXd &acceleration) {
  AccelerationTaskReference out;
  out.desired_acceleration = acceleration;
  return out;
}

class AccelerationSolverEffortTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_effort_test.urdf", two_link_dynamics_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(AccelerationSolverEffortTest,
       InactiveEffortEnvelopeLeavesKinematicSolutionUnchanged) {
  robot_->set_gravity(Eigen::Vector3d::Zero());
  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");
  solver.set_task_reference("posture", reference(vector({1.0, -2.0})));

  const Eigen::VectorXd q = vector({0.3, -0.4});
  const Eigen::VectorXd dq = vector({0.2, -0.1});
  const auto disabled = solver.solve(q, dq, 0.01,
                                     limit_options(vector({100.0, 100.0})));

  auto enabled_options = limit_options(vector({100.0, 100.0}));
  EffortConstraintOptions effort;
  effort.limits_override = vector({1e6, 1e6});
  enabled_options.effort_constraints = effort;
  const auto enabled = solver.solve(q, dq, 0.01, enabled_options);

  ASSERT_EQ(disabled.status, SolverStatus::kSuccess)
      << disabled.status_message;
  ASSERT_EQ(enabled.status, SolverStatus::kSuccess) << enabled.status_message;
  EXPECT_TRUE(enabled.joint_accelerations.isApprox(
      disabled.joint_accelerations, kTolerance));
  EXPECT_TRUE(enabled.effort_limits_applied);
  EXPECT_EQ(disabled.predicted_torques.size(), 0);
  EXPECT_EQ(enabled.predicted_torques.size(), robot_->nv());
  EXPECT_TRUE(enabled.saturated_effort_indices.empty());
}

TEST_F(AccelerationSolverEffortTest,
       FullMassMatrixEffortLimitCouplesJointAccelerations) {
  robot_->set_gravity(Eigen::Vector3d::Zero());
  const Eigen::VectorXd q = Eigen::Vector2d::Zero();
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  const Eigen::MatrixXd mass_matrix = robot_->compute_mass_matrix(q);
  ASSERT_GT(std::abs(mass_matrix(0, 1)), 0.1);

  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint2_task", "joint2", 0.0);
  solver.set_task_reference("joint2_task", reference(vector({10.0})));

  auto options = limit_options(vector({100.0, 100.0}));
  EffortConstraintOptions effort;
  effort.limits_override = vector({5.0, 100.0});
  options.effort_constraints = effort;
  const auto result = solver.solve(q, dq, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.predicted_torques.size(), robot_->nv());
  EXPECT_TRUE(result.predicted_torques.isApprox(
      robot_->rnea(q, dq, result.joint_accelerations), kEffortTolerance));
  EXPECT_LE(std::abs(result.predicted_torques(0)), 5.0 + kEffortTolerance);
  EXPECT_NEAR(result.joint_accelerations(1), 10.0, kTolerance);
  EXPECT_GT(std::abs(result.joint_accelerations(0)), 0.1);
  EXPECT_TRUE(contains_index(result.saturated_effort_indices, 0));
}

TEST_F(AccelerationSolverEffortTest,
       NegativeCoupledTorqueBoundReportsMirroredSaturation) {
  robot_->set_gravity(Eigen::Vector3d::Zero());
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint2_task", "joint2", 0.0);
  solver.set_task_reference("joint2_task", reference(vector({-10.0})));

  auto options = limit_options(vector({100.0, 100.0}));
  EffortConstraintOptions effort;
  effort.limits_override = vector({5.0, 100.0});
  options.effort_constraints = effort;
  const auto result =
      solver.solve(Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), 0.01,
                   options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.predicted_torques(0), -5.0, kEffortTolerance);
  EXPECT_LT(result.joint_accelerations(1), -0.1);
  EXPECT_TRUE(contains_index(result.saturated_effort_indices, 0));
}

TEST_F(AccelerationSolverEffortTest,
       ModelEffortLimitsAreUsedWhenNoOverrideIsProvided) {
  ScopedUrdf bounded_urdf("embodik_acceleration_effort_model_limits.urdf",
                          two_link_dynamics_urdf("5.0", "100.0"));
  auto bounded_robot =
      std::make_shared<RobotModel>(bounded_urdf.path(), false);
  bounded_robot->set_gravity(Eigen::Vector3d::Zero());
  AccelerationSolver solver(bounded_robot);
  solver.add_joint_task("joint2_task", "joint2", 0.0);
  solver.set_task_reference("joint2_task", reference(vector({10.0})));

  auto options = limit_options(vector({100.0, 100.0}));
  options.effort_constraints = EffortConstraintOptions{};
  const Eigen::VectorXd q = Eigen::Vector2d::Zero();
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  const auto result = solver.solve(q, dq, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.effort_limits_applied);
  ASSERT_EQ(result.predicted_torques.size(), bounded_robot->nv());
  EXPECT_TRUE(result.predicted_torques.isApprox(
      bounded_robot->rnea(q, dq, result.joint_accelerations),
      kEffortTolerance));
  EXPECT_LE(std::abs(result.predicted_torques(0)),
            5.0 + kEffortTolerance);
  EXPECT_TRUE(contains_index(result.saturated_effort_indices, 0));
}

TEST_F(AccelerationSolverEffortTest,
       StaticGravityInsideEffortEnvelopeRemainsFeasible) {
  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");
  solver.set_task_reference("posture", reference(vector({0.0, 0.0})));

  const Eigen::VectorXd q = Eigen::Vector2d::Zero();
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  const Eigen::VectorXd gravity =
      robot_->rnea(q, dq, Eigen::Vector2d::Zero());
  ASSERT_GT(gravity.cwiseAbs().maxCoeff(), 1.0);

  auto options = limit_options(vector({0.1, 0.1}));
  EffortConstraintOptions effort;
  effort.limits_override = gravity.cwiseAbs() + Eigen::Vector2d::Ones();
  options.effort_constraints = effort;
  const auto result = solver.solve(q, dq, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.joint_accelerations.isZero(kTolerance));
  EXPECT_TRUE(result.predicted_torques.isApprox(gravity, kEffortTolerance));
  EXPECT_TRUE(result.saturated_effort_indices.empty());
}

TEST_F(AccelerationSolverEffortTest,
       GravityOutsideEffortEnvelopeReturnsInfeasibleAndClearsOutputs) {
  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");
  solver.set_task_reference("posture", reference(vector({0.0, 0.0})));

  auto options = limit_options(vector({0.1, 0.1}));
  EffortConstraintOptions effort;
  effort.limits_override = vector({1.0, 1.0});
  options.effort_constraints = effort;
  const auto result = solver.solve(Eigen::Vector2d::Zero(),
                                   Eigen::Vector2d::Zero(), 0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kInfeasible);
  EXPECT_TRUE(result.effort_limits_applied);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverEffortTest,
       EffortSafetyMarginTightensCoupledTorqueEnvelope) {
  robot_->set_gravity(Eigen::Vector3d::Zero());
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint2_task", "joint2", 0.0);
  solver.set_task_reference("joint2_task", reference(vector({10.0})));

  auto nominal_options = limit_options(vector({100.0, 100.0}));
  EffortConstraintOptions nominal_effort;
  nominal_effort.limits_override = vector({5.0, 100.0});
  nominal_options.effort_constraints = nominal_effort;

  auto margin_options = nominal_options;
  margin_options.effort_constraints->margin_fraction = 0.2;

  const auto nominal = solver.solve(Eigen::Vector2d::Zero(),
                                    Eigen::Vector2d::Zero(), 0.01,
                                    nominal_options);
  const auto margin = solver.solve(Eigen::Vector2d::Zero(),
                                   Eigen::Vector2d::Zero(), 0.01,
                                   margin_options);

  ASSERT_EQ(nominal.status, SolverStatus::kSuccess)
      << nominal.status_message;
  ASSERT_EQ(margin.status, SolverStatus::kSuccess) << margin.status_message;
  EXPECT_LE(std::abs(nominal.predicted_torques(0)), 5.0 + kEffortTolerance);
  EXPECT_LE(std::abs(margin.predicted_torques(0)), 4.0 + kEffortTolerance);
  EXPECT_GT(std::abs(margin.joint_accelerations(0)),
            std::abs(nominal.joint_accelerations(0)) + kTolerance);
  EXPECT_TRUE(contains_index(margin.saturated_effort_indices, 0));
}

TEST_F(AccelerationSolverEffortTest,
       NoActiveTasksCompensateNonzeroVelocityEffortBias) {
  robot_->set_gravity(Eigen::Vector3d::Zero());
  const Eigen::VectorXd q = vector({0.7, -0.9});
  const Eigen::VectorXd dq = vector({2.0, -1.5});
  const Eigen::MatrixXd mass_matrix = robot_->compute_mass_matrix(q);
  const Eigen::VectorXd bias =
      robot_->rnea(q, dq, Eigen::Vector2d::Zero());
  ASSERT_GT(bias.norm(), 0.1);
  const Eigen::VectorXd cancellation =
      -mass_matrix.ldlt().solve(bias);

  AccelerationSolver solver(robot_);
  auto options =
      limit_options(2.0 * cancellation.cwiseAbs() + Eigen::Vector2d::Ones());
  EffortConstraintOptions effort;
  effort.limits_override = vector({1e-8, 1e-8});
  options.effort_constraints = effort;
  const auto result = solver.solve(q, dq, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.effort_limits_applied);
  EXPECT_TRUE(result.joint_accelerations.isApprox(cancellation, 1e-6));
  EXPECT_LE(result.predicted_torques.cwiseAbs().maxCoeff(),
            1e-8 + kEffortTolerance);
  EXPECT_TRUE(result.task_diagnostics.empty());
}

TEST_F(AccelerationSolverEffortTest,
       AllocationAndEffortComposeInPhysicalCoordinates) {
  robot_->set_gravity(Eigen::Vector3d::Zero());
  AccelerationSolver solver(robot_);
  solver.add_joint_task("joint2_task", "joint2", 0.0);
  solver.set_task_reference("joint2_task", reference(vector({10.0})));

  auto options = limit_options(vector({100.0, 100.0}));
  GeneralizedAccelerationAllocation allocation;
  allocation.metric_diagonal = vector({100.0, 0.01});
  allocation.reference_acceleration = vector({0.5, 0.0});
  options.generalized_acceleration_allocation = allocation;
  EffortConstraintOptions effort;
  effort.limits_override = vector({5.0, 100.0});
  options.effort_constraints = effort;
  const auto result = solver.solve(Eigen::Vector2d::Zero(),
                                   Eigen::Vector2d::Zero(), 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  const Eigen::MatrixXd mass_matrix =
      robot_->compute_mass_matrix(Eigen::Vector2d::Zero());
  const double expected_joint1_acceleration =
      (5.0 - mass_matrix(0, 1) * 10.0) / mass_matrix(0, 0);
  EXPECT_TRUE(result.allocation_diagnostics.applied);
  EXPECT_TRUE(result.effort_limits_applied);
  EXPECT_NEAR(result.joint_accelerations(0), expected_joint1_acceleration,
              kTolerance);
  EXPECT_NEAR(result.joint_accelerations(1), 10.0, kTolerance);
  EXPECT_TRUE(result.predicted_torques.isApprox(
      robot_->rnea(Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(),
                   result.joint_accelerations),
      kEffortTolerance));
  EXPECT_LE(std::abs(result.predicted_torques(0)), 5.0 + kEffortTolerance);
  EXPECT_TRUE(contains_index(result.saturated_effort_indices, 0));
}

TEST_F(AccelerationSolverEffortTest, RejectsInvalidEffortInputs) {
  AccelerationSolver solver(robot_);
  const Eigen::VectorXd q = Eigen::Vector2d::Zero();
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();

  auto wrong_size = limit_options(vector({10.0, 10.0}));
  EffortConstraintOptions wrong_size_effort;
  wrong_size_effort.limits_override = vector({1.0});
  wrong_size.effort_constraints = wrong_size_effort;
  auto result = solver.solve(q, dq, 0.01, wrong_size);
  EXPECT_EQ(result.status, SolverStatus::kShapeMismatch);
  expect_no_motion_outputs(result);

  auto nonfinite = limit_options(vector({10.0, 10.0}));
  EffortConstraintOptions nonfinite_effort;
  nonfinite_effort.limits_override = vector({1.0, 2.0});
  nonfinite_effort.limits_override->operator()(0) =
      std::numeric_limits<double>::quiet_NaN();
  nonfinite.effort_constraints = nonfinite_effort;
  result = solver.solve(q, dq, 0.01, nonfinite);
  EXPECT_EQ(result.status, SolverStatus::kNonFiniteInput);
  expect_no_motion_outputs(result);

  auto bad_margin = limit_options(vector({10.0, 10.0}));
  EffortConstraintOptions bad_margin_effort;
  bad_margin_effort.limits_override = vector({1.0, 2.0});
  bad_margin_effort.margin_fraction = 1.0;
  bad_margin.effort_constraints = bad_margin_effort;
  result = solver.solve(q, dq, 0.01, bad_margin);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  auto nan_margin = limit_options(vector({10.0, 10.0}));
  EffortConstraintOptions nan_margin_effort;
  nan_margin_effort.limits_override = vector({1.0, 2.0});
  nan_margin_effort.margin_fraction =
      std::numeric_limits<double>::quiet_NaN();
  nan_margin.effort_constraints = nan_margin_effort;
  result = solver.solve(q, dq, 0.01, nan_margin);
  EXPECT_EQ(result.status, SolverStatus::kNonFiniteInput);
  expect_no_motion_outputs(result);

  auto nonpositive = limit_options(vector({10.0, 10.0}));
  EffortConstraintOptions nonpositive_effort;
  nonpositive_effort.limits_override = vector({1.0, 0.0});
  nonpositive.effort_constraints = nonpositive_effort;
  result = solver.solve(q, dq, 0.01, nonpositive);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);
}

TEST(AccelerationSolverEffortMetadataTest,
     RejectsNonPositiveModelEffortMetadataWithoutOverride) {
  ScopedUrdf urdf("embodik_acceleration_effort_bad_metadata.urdf",
                  two_link_dynamics_urdf("0.0", "100.0"));
  auto robot = std::make_shared<RobotModel>(urdf.path(), false);
  AccelerationSolver solver(robot);

  auto options = limit_options(vector({10.0, 10.0}));
  options.effort_constraints = EffortConstraintOptions{};
  const auto result = solver.solve(Eigen::Vector2d::Zero(),
                                   Eigen::Vector2d::Zero(), 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  EXPECT_FALSE(result.effort_limits_applied);
  expect_no_motion_outputs(result);

  auto override_options = limit_options(vector({10.0, 10.0}));
  EffortConstraintOptions effort;
  effort.limits_override = vector({100.0, 100.0});
  override_options.effort_constraints = effort;
  const auto override_result =
      solver.solve(Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), 0.01,
                   override_options);
  ASSERT_EQ(override_result.status, SolverStatus::kSuccess)
      << override_result.status_message;
  EXPECT_TRUE(override_result.effort_limits_applied);
}

TEST_F(AccelerationSolverEffortTest, DeterministicAcrossRepeatedSolves) {
  robot_->set_gravity(Eigen::Vector3d::Zero());
  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");
  solver.set_task_reference("posture", reference(vector({3.0, -2.0})));

  auto options = limit_options(vector({100.0, 100.0}));
  EffortConstraintOptions effort;
  effort.limits_override = vector({20.0, 20.0});
  options.effort_constraints = effort;
  const Eigen::VectorXd q = vector({0.1, -0.2});
  const Eigen::VectorXd dq = vector({0.3, -0.1});

  const auto first = solver.solve(q, dq, 0.01, options);
  const auto second = solver.solve(q, dq, 0.01, options);
  ASSERT_EQ(first.status, SolverStatus::kSuccess) << first.status_message;
  ASSERT_EQ(second.status, SolverStatus::kSuccess) << second.status_message;
  EXPECT_TRUE(first.joint_accelerations.isApprox(second.joint_accelerations,
                                                 0.0));
  EXPECT_TRUE(first.predicted_torques.isApprox(second.predicted_torques, 0.0));
  EXPECT_EQ(first.saturated_effort_indices, second.saturated_effort_indices);
}

} // namespace
} // namespace embodik::test
