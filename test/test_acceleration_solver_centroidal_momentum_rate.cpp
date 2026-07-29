#include <embodik/acceleration_solver.hpp>

#include <gtest/gtest.h>

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

void expect_no_motion_outputs(const AccelerationSolverResult &result) {
  EXPECT_TRUE(result.solution.empty());
  EXPECT_EQ(result.joint_accelerations.size(), 0);
  EXPECT_EQ(result.joint_velocities_next.size(), 0);
  EXPECT_EQ(result.q_solution.size(), 0);
}

std::string asymmetric_centroidal_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_centroidal_momentum_rate_test_robot">
  <link name="base_link">
    <inertial>
      <origin xyz="0.07 -0.03 0.11" rpy="0.05 -0.02 0.04"/>
      <mass value="3.25"/>
      <inertia ixx="0.31" ixy="0.012" ixz="-0.017" iyy="0.43" iyz="0.023" izz="0.52"/>
    </inertial>
  </link>
  <link name="shoulder">
    <inertial>
      <origin xyz="0.34 0.06 -0.02" rpy="-0.03 0.04 0.07"/>
      <mass value="1.75"/>
      <inertia ixx="0.08" ixy="-0.006" ixz="0.004" iyy="0.11" iyz="-0.003" izz="0.13"/>
    </inertial>
  </link>
  <joint name="yaw" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder"/>
    <origin xyz="0.13 -0.08 0.22" rpy="0.01 0.03 -0.02"/>
    <axis xyz="0.2 0.1 0.97"/>
    <limit lower="-2.4" upper="2.3" velocity="70.0" effort="400.0"/>
  </joint>
  <link name="forearm">
    <inertial>
      <origin xyz="-0.08 0.27 0.05" rpy="0.09 -0.06 0.02"/>
      <mass value="0.95"/>
      <inertia ixx="0.044" ixy="0.005" ixz="-0.002" iyy="0.061" iyz="0.007" izz="0.073"/>
    </inertial>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="shoulder"/>
    <child link="forearm"/>
    <origin xyz="0.41 0.18 -0.09" rpy="-0.04 0.08 0.03"/>
    <axis xyz="-0.1 0.95 0.2"/>
    <limit lower="-1.35" upper="1.45" velocity="30.0" effort="350.0"/>
  </joint>
  <link name="tool">
    <inertial>
      <origin xyz="0.12 -0.16 0.19" rpy="-0.02 0.11 -0.05"/>
      <mass value="0.55"/>
      <inertia ixx="0.019" ixy="-0.002" ixz="0.001" iyy="0.026" iyz="-0.003" izz="0.031"/>
    </inertial>
  </link>
  <joint name="pitch" type="revolute">
    <parent link="forearm"/>
    <child link="tool"/>
    <origin xyz="-0.23 0.36 0.14" rpy="0.06 -0.03 0.05"/>
    <axis xyz="0.25 -0.4 0.88"/>
    <limit lower="-1.7" upper="1.8" velocity="50.0" effort="200.0"/>
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

class AccelerationSolverCentroidalMomentumRateTest
    : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_centroidal_momentum_rate.urdf",
        asymmetric_centroidal_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
    robot_->set_gravity(Eigen::Vector3d::Zero());
  }

  AccelerationSolveOptions unconstrained_options() const {
    AccelerationSolveOptions options;
    options.acceleration_limits_override = vector({100.0, 100.0, 100.0});
    options.apply_position_limits = false;
    options.apply_velocity_limits = false;
    return options;
  }

  Eigen::VectorXd q() const { return vector({0.37, -0.11, -0.42}); }
  Eigen::VectorXd dq() const { return vector({-0.23, 0.31, 0.17}); }
  Eigen::VectorXd feasible_ddq() const { return vector({0.41, -0.19, 0.29}); }

  Eigen::VectorXd hdot_for(const Eigen::VectorXd &q_value,
                           const Eigen::VectorXd &dq_value,
                           const Eigen::VectorXd &ddq_value) const {
    return robot_->compute_centroidal_momentum_matrix(q_value, dq_value) *
               ddq_value +
           robot_->compute_centroidal_momentum_matrix_bias(q_value, dq_value);
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(AccelerationSolverCentroidalMomentumRateTest,
       ObjectiveTracksPhysicalMomentumRateAndReportsDiagnostics) {
  const Eigen::VectorXd q_value = q();
  const Eigen::VectorXd dq_value = dq();
  const Eigen::VectorXd current_h =
      robot_->compute_centroidal_momentum(q_value, dq_value);
  const Eigen::VectorXd hdot_target =
      hdot_for(q_value, dq_value, feasible_ddq());

  CentroidalMomentumRateObjective objective;
  objective.source_id = "centroidal_rate";
  objective.h_target = current_h;
  objective.hdot_feedforward = hdot_target;
  objective.proportional_gain = 0.0;
  objective.priority = 0;
  objective.solve_mode = TaskSolveMode::kScale;

  auto options = unconstrained_options();
  options.centroidal_momentum_rate_objectives.push_back(objective);

  AccelerationSolver solver(robot_);
  const auto result = solver.solve(q_value, dq_value, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  const Eigen::VectorXd achieved = hdot_for(q_value, dq_value,
                                            result.joint_accelerations);
  EXPECT_TRUE(achieved.isApprox(hdot_target, kTolerance));
  ASSERT_EQ(result.centroidal_momentum_rate_diagnostics.size(), 1);
  const auto &diagnostic = result.centroidal_momentum_rate_diagnostics[0];
  EXPECT_EQ(diagnostic.source_id, "centroidal_rate");
  EXPECT_TRUE(diagnostic.target_momentum.isApprox(current_h, kTolerance));
  EXPECT_TRUE(diagnostic.reference_momentum_rate.isApprox(hdot_target,
                                                          kTolerance));
  EXPECT_TRUE(diagnostic.current_momentum.isApprox(current_h, kTolerance));
  EXPECT_TRUE(diagnostic.bias_momentum_rate.isApprox(
      robot_->compute_centroidal_momentum_matrix_bias(q_value, dq_value),
      kTolerance));
  EXPECT_TRUE(diagnostic.achieved_momentum_rate.isApprox(achieved,
                                                         kTolerance));
  EXPECT_TRUE(diagnostic.residual.isApprox(achieved - hdot_target,
                                           kTolerance));
  EXPECT_EQ(diagnostic.selected_axes,
            std::vector<bool>({true, true, true, true, true, true}));
  EXPECT_NEAR(diagnostic.scale, 1.0, kTolerance);
  EXPECT_EQ(diagnostic.effective_mode, TaskSolveMode::kScale);
}

TEST_F(AccelerationSolverCentroidalMomentumRateTest,
       ProportionalTermAndNonzeroBiasDefineReference) {
  const Eigen::VectorXd q_value = q();
  const Eigen::VectorXd dq_value = dq();
  const Eigen::VectorXd current_h =
      robot_->compute_centroidal_momentum(q_value, dq_value);
  const Eigen::VectorXd desired_h =
      current_h + vector({0.12, -0.07, 0.03, 0.05, -0.02, 0.04});
  const Eigen::VectorXd feedforward =
      hdot_for(q_value, dq_value, feasible_ddq());

  CentroidalMomentumRateObjective objective;
  objective.source_id = "centroidal_pd";
  objective.h_target = desired_h;
  objective.hdot_feedforward = feedforward;
  objective.proportional_gain = 0.25;
  objective.priority = 0;
  objective.solve_mode = TaskSolveMode::kMinError;

  auto options = unconstrained_options();
  options.centroidal_momentum_rate_objectives.push_back(objective);

  AccelerationSolver solver(robot_);
  const auto result = solver.solve(q_value, dq_value, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.centroidal_momentum_rate_diagnostics.size(), 1);
  const Eigen::VectorXd expected_reference =
      feedforward + 0.25 * (desired_h - current_h);
  EXPECT_TRUE(result.centroidal_momentum_rate_diagnostics[0]
                  .reference_momentum_rate.isApprox(expected_reference,
                                                     kTolerance));
  EXPECT_GT(robot_->compute_centroidal_momentum_matrix_bias(q_value, dq_value)
                .norm(),
            1e-10);
}

TEST_F(AccelerationSolverCentroidalMomentumRateTest,
       SelectedAxisHardBoundsClampPhysicalMomentumRate) {
  const Eigen::VectorXd q_value = q();
  const Eigen::VectorXd dq_value = dq();
  const Eigen::VectorXd target_hdot =
      hdot_for(q_value, dq_value, feasible_ddq());

  CentroidalMomentumRateBounds bounds;
  bounds.source_id = "angular_z_rate_bound";
  bounds.lower_bounds = Eigen::VectorXd::Constant(6, -100.0);
  bounds.upper_bounds = Eigen::VectorXd::Constant(6, 100.0);
  bounds.lower_bounds(2) = target_hdot(2);
  bounds.upper_bounds(2) = target_hdot(2);
  bounds.axis_mask = {false, false, true, false, false, false};

  auto options = unconstrained_options();
  options.centroidal_momentum_rate_bounds.push_back(bounds);

  AccelerationSolver solver(robot_);
  const auto result = solver.solve(q_value, dq_value, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  const Eigen::VectorXd achieved = hdot_for(q_value, dq_value,
                                            result.joint_accelerations);
  EXPECT_NEAR(achieved(2), target_hdot(2), 1e-7);
}

TEST_F(AccelerationSolverCentroidalMomentumRateTest,
       ObjectiveComposesWithHardBoundsAndExistingFamilies) {
  const Eigen::VectorXd q_value = q();
  const Eigen::VectorXd dq_value = dq();
  const Eigen::VectorXd objective_hdot =
      hdot_for(q_value, dq_value, feasible_ddq());
  const Eigen::VectorXd bounded_hdot =
      hdot_for(q_value, dq_value, vector({-0.2, 0.15, -0.05}));

  CentroidalMomentumRateObjective objective;
  objective.source_id = "centroidal_soft";
  objective.h_target = robot_->compute_centroidal_momentum(q_value, dq_value);
  objective.hdot_feedforward = objective_hdot;
  objective.proportional_gain = 0.0;
  objective.solve_mode = TaskSolveMode::kMinError;

  CentroidalMomentumRateBounds bounds;
  bounds.source_id = "linear_y_bound";
  bounds.lower_bounds = Eigen::VectorXd::Constant(6, -100.0);
  bounds.upper_bounds = Eigen::VectorXd::Constant(6, 100.0);
  bounds.lower_bounds(4) = bounded_hdot(4);
  bounds.upper_bounds(4) = bounded_hdot(4);
  bounds.axis_mask = {false, false, false, false, true, false};

  auto options = unconstrained_options();
  options.centroidal_momentum_rate_objectives.push_back(objective);
  options.centroidal_momentum_rate_bounds.push_back(bounds);
  options.effort_constraints = EffortConstraintOptions{};

  AccelerationSolver solver(robot_);
  const auto result = solver.solve(q_value, dq_value, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.effort_limits_applied);
  const Eigen::VectorXd achieved = hdot_for(q_value, dq_value,
                                            result.joint_accelerations);
  EXPECT_NEAR(achieved(4), bounded_hdot(4), 1e-7);
}

TEST_F(AccelerationSolverCentroidalMomentumRateTest,
       DisabledOptionsPreserveBaselineOutputs) {
  AccelerationSolver baseline_solver(robot_);
  AccelerationSolver explicit_empty_solver(robot_);
  const Eigen::VectorXd q_value = q();
  const Eigen::VectorXd dq_value = dq();

  auto baseline_options = unconstrained_options();
  auto explicit_options = unconstrained_options();
  ASSERT_TRUE(explicit_options.centroidal_momentum_rate_objectives.empty());
  ASSERT_TRUE(explicit_options.centroidal_momentum_rate_bounds.empty());

  const auto baseline =
      baseline_solver.solve(q_value, dq_value, 0.01, baseline_options);
  const auto explicit_empty =
      explicit_empty_solver.solve(q_value, dq_value, 0.01, explicit_options);

  ASSERT_EQ(baseline.status, SolverStatus::kSuccess)
      << baseline.status_message;
  ASSERT_EQ(explicit_empty.status, SolverStatus::kSuccess)
      << explicit_empty.status_message;
  EXPECT_TRUE(baseline.joint_accelerations.isApprox(
      explicit_empty.joint_accelerations, 1e-12));
  EXPECT_TRUE(baseline.joint_velocities_next.isApprox(
      explicit_empty.joint_velocities_next, 1e-12));
  EXPECT_TRUE(baseline.q_solution.isApprox(explicit_empty.q_solution, 1e-12));
  EXPECT_TRUE(explicit_empty.centroidal_momentum_rate_diagnostics.empty());
}

TEST_F(AccelerationSolverCentroidalMomentumRateTest,
       InvalidInputsAndDuplicateSourceIdsFailClosed) {
  const Eigen::VectorXd q_value = q();
  const Eigen::VectorXd dq_value = dq();

  auto options = unconstrained_options();
  CentroidalMomentumRateObjective objective;
  objective.source_id = "bad";
  objective.h_target = Eigen::VectorXd::Zero(5);
  objective.hdot_feedforward = Eigen::VectorXd::Zero(6);
  options.centroidal_momentum_rate_objectives.push_back(objective);
  AccelerationSolver solver(robot_);
  auto result = solver.solve(q_value, dq_value, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kConstraintBoundsMismatch);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  objective.h_target = Eigen::VectorXd::Zero(6);
  objective.hdot_feedforward =
      Eigen::VectorXd::Constant(6, std::numeric_limits<double>::quiet_NaN());
  options.centroidal_momentum_rate_objectives.push_back(objective);
  result = solver.solve(q_value, dq_value, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kNonFiniteInput);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  objective.hdot_feedforward = Eigen::VectorXd::Zero(6);
  objective.axis_mask = {false, false, false, false, false, false};
  options.centroidal_momentum_rate_objectives.push_back(objective);
  result = solver.solve(q_value, dq_value, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  objective.axis_mask.clear();
  objective.source_id = "same";
  CentroidalMomentumRateBounds bounds;
  bounds.source_id = "same";
  bounds.lower_bounds = Eigen::VectorXd::Constant(6, -1.0);
  bounds.upper_bounds = Eigen::VectorXd::Constant(6, 1.0);
  options.centroidal_momentum_rate_objectives.push_back(objective);
  options.centroidal_momentum_rate_bounds.push_back(bounds);
  result = solver.solve(q_value, dq_value, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  options = unconstrained_options();
  bounds.source_id = "bad_bounds";
  bounds.lower_bounds = Eigen::VectorXd::Constant(6, 1.0);
  bounds.upper_bounds = Eigen::VectorXd::Constant(6, -1.0);
  options.centroidal_momentum_rate_bounds.push_back(bounds);
  result = solver.solve(q_value, dq_value, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverCentroidalMomentumRateTest,
       CapabilitiesAdvertiseFixedBaseCentroidalRateOnly) {
  const auto capabilities = AccelerationSolver::capabilities();
  EXPECT_TRUE(capabilities.supports_fixed_base_centroidal_momentum_rate_objective);
  EXPECT_TRUE(capabilities.supports_fixed_base_centroidal_momentum_rate_bounds);
  EXPECT_FALSE(capabilities.supports_floating_base_centroidal_momentum_rate);
  EXPECT_FALSE(capabilities.supports_dynamic_balance);
}

} // namespace
} // namespace embodik::test
