#include "acceleration_geometric_constraint_policy.hpp"
#include "acceleration_scalar_geometric_constraint.hpp"
#include "frame_kinematic_differential.hpp"

#include <embodik/acceleration_solver.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

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

std::string two_link_urdf(const std::string &effort1 = "100.0",
                          const std::string &effort2 = "100.0") {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_tight_point_test_robot">
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
    <limit lower="-3.14" upper="3.14" velocity="500.0" effort=")" +
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
    <limit lower="-3.14" upper="3.14" velocity="500.0" effort=")" +
         effort2 + R"("/>
  </joint>
  <link name="tool"/>
  <joint name="tool_fixed" type="fixed">
    <parent link="tip"/>
    <child link="tool"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
</robot>)";
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

GeometricConstraintAccelerationPolicy policy(double rate_limit,
                                             double acceleration_limit,
                                             double lower_braking,
                                             double upper_braking) {
  GeometricConstraintAccelerationPolicy out;
  out.rate_limits = Eigen::VectorXd::Constant(3, rate_limit);
  out.acceleration_limits =
      Eigen::VectorXd::Constant(3, acceleration_limit);
  out.lower_braking_accelerations =
      Eigen::VectorXd::Constant(3, lower_braking);
  out.upper_braking_accelerations =
      Eigen::VectorXd::Constant(3, upper_braking);
  return out;
}

TightPointAccelerationConstraint tight_point(
    const std::string &source_id, const std::string &frame_name,
    const Eigen::Vector3d &target, const Eigen::Vector3d &mask,
    double epsilon, const GeometricConstraintAccelerationPolicy &policy) {
  TightPointAccelerationConstraint constraint;
  constraint.source_id = source_id;
  constraint.definition.frame_name = frame_name;
  constraint.definition.target_point = target;
  constraint.definition.position_epsilon = epsilon;
  constraint.definition.axis_mask = mask;
  constraint.policy = policy;
  return constraint;
}

Eigen::Vector3d frame_translation(RobotModel &robot, const Eigen::VectorXd &q,
                                  const Eigen::VectorXd &dq) {
  robot.update_kinematics(q, dq);
  return robot.get_frame_pose("tool").translation();
}

Eigen::Vector3d frame_acceleration(RobotModel &robot, const Eigen::VectorXd &q,
                                   const Eigen::VectorXd &dq,
                                   const Eigen::VectorXd &ddq) {
  robot.update_kinematics(q, dq);
  return robot.get_frame_jacobian("tool").topRows<3>() * ddq +
         robot.get_frame_jacobian_bias("tool").head<3>();
}

class AccelerationSolverTightPointTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_tight_point_test.urdf", two_link_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
    robot_->set_gravity(Eigen::Vector3d::Zero());
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST(AccelerationGeometricConstraintPolicyTest,
     ReusesPolicyAndAxisValidationAtSixDimensions) {
  GeometricConstraintAccelerationPolicy policy;
  policy.rate_limits = Eigen::VectorXd::Ones(6);
  policy.acceleration_limits = Eigen::VectorXd::Constant(6, 2.0);
  policy.lower_braking_accelerations = Eigen::VectorXd::Constant(6, 0.5);
  policy.upper_braking_accelerations = Eigen::VectorXd::Constant(6, 0.75);

  const auto policy_result = detail::validate_geometric_constraint_policy(
      "six-dimensional test", "policy", policy, 6);
  EXPECT_TRUE(policy_result.satisfied()) << policy_result.message;

  Eigen::VectorXd mask = Eigen::VectorXd::Zero(6);
  mask(0) = 1.0;
  mask(5) = 1.0;
  const auto axis_result = detail::validate_geometric_constraint_axis_mask(
      "six-dimensional test", "axes", mask, 6);
  ASSERT_TRUE(axis_result.satisfied()) << axis_result.message;
  EXPECT_EQ(axis_result.active_axes, std::vector<int>({0, 5}));
}

TEST(AccelerationScalarGeometricConstraintTest,
     GenericEntryPointsRejectMalformedSpecifications) {
  detail::ScalarGeometricConstraintSpecification specification;
  specification.family_label = "test scalar constraint";
  specification.source_id = "test";
  specification.dimension = 2;
  specification.active_axes = {0, 1};
  specification.state_lower_bounds = vector({-0.1, -0.2});
  specification.state_upper_bounds = vector({0.3, 0.4});
  specification.policy.rate_limits = Eigen::Vector2d::Ones();
  specification.policy.acceleration_limits =
      Eigen::Vector2d::Constant(2.0);
  specification.policy.lower_braking_accelerations =
      Eigen::Vector2d::Constant(0.5);
  specification.policy.upper_braking_accelerations =
      Eigen::Vector2d::Constant(0.75);

  detail::ScalarGeometricCoordinateSample sample;
  sample.value = Eigen::Vector2d::Zero();
  sample.rate = Eigen::Vector2d::Zero();
  sample.acceleration = Eigen::Vector2d::Zero();
  const auto asymmetric =
      detail::validate_scalar_geometric_sample_acceptance(specification, sample,
                                                          false);
  EXPECT_TRUE(asymmetric.satisfied()) << asymmetric.message;

  auto invalid_axis = specification;
  invalid_axis.active_axes = {0, 2};
  const auto invalid_axis_result =
      detail::validate_scalar_geometric_sample_acceptance(invalid_axis, sample,
                                                          false);
  EXPECT_EQ(invalid_axis_result.status, SolverStatus::kShapeMismatch);

  detail::ScalarGeometricDifferential differential;
  differential.coefficient_matrix = Eigen::Matrix2d::Identity();
  differential.affine_bias = Eigen::Vector2d::Zero();
  differential.coordinates = sample;
  const auto invalid_support =
      detail::validate_scalar_geometric_joint_box_support(
          invalid_axis, differential, Eigen::Vector2d::Constant(-1.0),
          Eigen::Vector2d::Constant(1.0), "test-state");
  EXPECT_EQ(invalid_support.status, SolverStatus::kShapeMismatch);

  const detail::ScalarGeometricPathValidationOptions path_options;
  const auto invalid_path = detail::validate_scalar_geometric_path(
      invalid_axis, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), 0.01,
      path_options,
      [](const detail::ScalarGeometricPathPoint &,
         detail::ScalarGeometricCoordinateSample *) {
        return detail::ScalarGeometricConstraintResult{};
      });
  EXPECT_EQ(invalid_path.status, SolverStatus::kShapeMismatch);

  auto nonfinite_policy = specification;
  nonfinite_policy.policy.rate_limits(0) =
      std::numeric_limits<double>::quiet_NaN();
  const auto nonfinite_policy_result =
      detail::validate_scalar_geometric_sample_acceptance(
          nonfinite_policy, sample, false);
  EXPECT_EQ(nonfinite_policy_result.status, SolverStatus::kNonFiniteInput);

  auto malformed_sample = sample;
  malformed_sample.acceleration = Eigen::VectorXd::Zero(1);
  const auto malformed_sample_result =
      detail::validate_scalar_geometric_sample_acceptance(
          specification, malformed_sample, false);
  EXPECT_EQ(malformed_sample_result.status, SolverStatus::kShapeMismatch);
}

TEST_F(AccelerationSolverTightPointTest, CapabilitiesExposeTightPointSupport) {
  const auto capabilities = AccelerationSolver::capabilities();
  EXPECT_TRUE(capabilities.supports_tight_point_constraints);
  EXPECT_TRUE(capabilities.supports_fixed_base_contact_kinematics);
}

TEST_F(AccelerationSolverTightPointTest,
       MultiAxisMultiRecordBoundsPhysicalAcceleration) {
  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");
  solver.set_task_reference("posture", reference(vector({60.0, -40.0})));

  const Eigen::VectorXd q = vector({0.35, -0.25});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  const Eigen::Vector3d target = frame_translation(*robot_, q, dq);
  auto options = options_with_limits(200.0);
  options.tight_point_constraints.push_back(tight_point(
      "tool_xy_box", "tool", target, Eigen::Vector3d(1.0, 1.0, 0.0),
      0.02, policy(10.0, 0.2, 0.2, 0.2)));
  options.tight_point_constraints.push_back(tight_point(
      "tool_x_box", "tool", target, Eigen::Vector3d(1.0, 0.0, 0.0),
      0.02, policy(10.0, 0.3, 0.3, 0.3)));

  const auto result = solver.solve(q, dq, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  const Eigen::Vector3d physical =
      frame_acceleration(*robot_, q, dq, result.joint_accelerations);
  EXPECT_LE(std::abs(physical.x()), 0.2 + kTolerance);
  EXPECT_LE(std::abs(physical.y()), 0.2 + kTolerance);
  const Eigen::Vector3d next_translation =
      frame_translation(*robot_, result.q_solution,
                        result.joint_velocities_next);
  EXPECT_LE(std::abs(next_translation.x() - target.x()), 0.02 + kTolerance);
  EXPECT_LE(std::abs(next_translation.y() - target.y()), 0.02 + kTolerance);
}

TEST_F(AccelerationSolverTightPointTest,
       FrameJdotBiasMatchesFiniteDifferencePhysicalAcceleration) {
  const Eigen::VectorXd q = vector({0.4, -0.35});
  const Eigen::VectorXd dq = vector({0.8, -0.25});
  const Eigen::VectorXd ddq = vector({0.7, -0.2});
  constexpr double kStep = 1e-6;

  const auto differential =
      detail::evaluate_frame_kinematic_differential_at_state(
          *robot_, "tool", q, dq);
  const Eigen::Vector3d physical =
      differential.jacobian.topRows<3>() * ddq +
      differential.affine_bias.head<3>();

  const Eigen::VectorXd q_plus =
      robot_->integrate(q, kStep * dq + 0.5 * kStep * kStep * ddq);
  const Eigen::VectorXd q_minus =
      robot_->integrate(q, -kStep * dq + 0.5 * kStep * kStep * ddq);
  const Eigen::VectorXd dq_plus = dq + kStep * ddq;
  const Eigen::VectorXd dq_minus = dq - kStep * ddq;
  const auto plus = detail::evaluate_frame_kinematic_differential_at_state(
      *robot_, "tool", q_plus, dq_plus);
  const auto minus = detail::evaluate_frame_kinematic_differential_at_state(
      *robot_, "tool", q_minus, dq_minus);
  const Eigen::Vector3d finite_difference =
      (plus.jacobian.topRows<3>() * dq_plus -
       minus.jacobian.topRows<3>() * dq_minus) /
      (2.0 * kStep);

  EXPECT_TRUE(physical.isApprox(finite_difference, 1e-6))
      << "physical: " << physical.transpose()
      << " fd: " << finite_difference.transpose();
}

TEST_F(AccelerationSolverTightPointTest,
       OutsideStateAndMalformedRecordsFailClosed) {
  AccelerationSolver solver(robot_);
  const Eigen::VectorXd q = vector({0.2, -0.1});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  const Eigen::Vector3d target = frame_translation(*robot_, q, dq);

  auto outside = options_with_limits(100.0);
  outside.tight_point_constraints.push_back(tight_point(
      "outside", "tool", target + Eigen::Vector3d(0.1, 0.0, 0.0),
      Eigen::Vector3d(1.0, 0.0, 0.0), 1e-3,
      policy(10.0, 10.0, 10.0, 10.0)));
  const auto outside_result = solver.solve(q, dq, 0.01, outside);
  EXPECT_EQ(outside_result.status, SolverStatus::kInfeasible);
  expect_no_motion_outputs(outside_result);

  auto malformed_mask = options_with_limits(100.0);
  malformed_mask.tight_point_constraints.push_back(tight_point(
      "bad_mask", "tool", target, Eigen::Vector3d(0.5, 0.0, 0.0),
      0.01, policy(10.0, 10.0, 10.0, 10.0)));
  const auto mask_result = solver.solve(q, dq, 0.01, malformed_mask);
  EXPECT_EQ(mask_result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(mask_result);

  auto bad_policy = options_with_limits(100.0);
  auto invalid_policy = policy(10.0, 1.0, 2.0, 1.0);
  bad_policy.tight_point_constraints.push_back(tight_point(
      "bad_policy", "tool", target, Eigen::Vector3d(1.0, 0.0, 0.0),
      0.01, invalid_policy));
  const auto policy_result = solver.solve(q, dq, 0.01, bad_policy);
  EXPECT_EQ(policy_result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(policy_result);

  auto nonfinite = options_with_limits(100.0);
  Eigen::Vector3d nonfinite_target = target;
  nonfinite_target.x() = std::numeric_limits<double>::quiet_NaN();
  nonfinite.tight_point_constraints.push_back(tight_point(
      "nonfinite", "tool", nonfinite_target, Eigen::Vector3d(1.0, 0.0, 0.0),
      0.01, policy(10.0, 10.0, 10.0, 10.0)));
  const auto nonfinite_result = solver.solve(q, dq, 0.01, nonfinite);
  EXPECT_EQ(nonfinite_result.status, SolverStatus::kNonFiniteInput);
  expect_no_motion_outputs(nonfinite_result);

  auto unknown_frame = options_with_limits(100.0);
  unknown_frame.tight_point_constraints.push_back(tight_point(
      "unknown", "missing", target, Eigen::Vector3d(1.0, 0.0, 0.0),
      0.01, policy(10.0, 10.0, 10.0, 10.0)));
  const auto unknown_result = solver.solve(q, dq, 0.01, unknown_frame);
  EXPECT_EQ(unknown_result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(unknown_result);

  auto duplicate = options_with_limits(100.0);
  AffineAccelerationConstraint affine;
  affine.source_id = "dup";
  affine.coefficient_matrix = Eigen::MatrixXd::Identity(1, 2);
  affine.affine_bias = Eigen::VectorXd::Zero(1);
  affine.lower_bounds = vector({-1.0});
  affine.upper_bounds = vector({1.0});
  duplicate.affine_constraints.push_back(affine);
  duplicate.tight_point_constraints.push_back(tight_point(
      "dup", "tool", target, Eigen::Vector3d(1.0, 0.0, 0.0), 0.01,
      policy(10.0, 10.0, 10.0, 10.0)));
  const auto duplicate_result = solver.solve(q, dq, 0.01, duplicate);
  EXPECT_EQ(duplicate_result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(duplicate_result);
}

TEST_F(AccelerationSolverTightPointTest,
       NonlinearPathTrustStepRejectionClearsMotionAndTorqueOutputs) {
  AccelerationSolver solver(robot_);
  const Eigen::VectorXd q = vector({0.0, 0.0});
  const Eigen::VectorXd dq = vector({200.0, 0.0});
  const Eigen::Vector3d target = frame_translation(*robot_, q, dq);
  auto options = options_with_limits(1e8);
  options.effort_constraints = EffortConstraintOptions{};
  options.effort_constraints->limits_override = vector({1e12, 1e12});
  options.tight_point_constraints.push_back(tight_point(
      "fast_y", "tool", target, Eigen::Vector3d(0.0, 1.0, 0.0),
      10.0, policy(1000.0, 1e8, 1e8, 1e8)));

  const auto result = solver.solve(q, dq, 0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kNumericalError);
  EXPECT_NE(result.status_message.find("trust step"), std::string::npos);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTightPointTest,
       CurrentStateSupportRejectionFailsClosed) {
  AccelerationSolver solver(robot_);
  const Eigen::VectorXd q = vector({0.3, -0.2});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  const Eigen::Vector3d target = frame_translation(*robot_, q, dq);
  auto options = options_with_limits(100.0);
  options.tight_point_constraints.push_back(tight_point(
      "unactuated_z", "tool", target, Eigen::Vector3d(0.0, 0.0, 1.0),
      0.01, policy(10.0, 1.0, 0.1, 0.1)));

  const auto result = solver.solve(q, dq, 0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kInfeasible);
  EXPECT_NE(result.status_message.find("current-state"),
            std::string::npos);
  EXPECT_NE(result.status_message.find("braking support"),
            std::string::npos);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTightPointTest,
       PredictedStateSupportRejectionFailsClosed) {
  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");

  const Eigen::VectorXd q = vector({0.2, -0.2});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  constexpr double kDt = 0.1;
  const Eigen::VectorXd acceleration_to_singularity =
      -2.0 * q / (kDt * kDt);
  solver.set_task_reference(
      "posture", reference(acceleration_to_singularity));

  const Eigen::Vector3d target = frame_translation(*robot_, q, dq);
  auto options = options_with_limits(1e6);
  options.tight_point_constraints.push_back(tight_point(
      "predicted_singular_x", "tool", target,
      Eigen::Vector3d(1.0, 0.0, 0.0), 0.1,
      policy(100.0, 1e5, 1000.0, 1000.0)));

  const auto result = solver.solve(q, dq, kDt, options);

  EXPECT_EQ(result.status, SolverStatus::kNumericalError)
      << result.status_message << " ddq="
      << result.joint_accelerations.transpose() << " q_next="
      << result.q_solution.transpose() << " dq_next="
      << result.joint_velocities_next.transpose();
  EXPECT_NE(result.status_message.find("predicted next-state"),
            std::string::npos);
  EXPECT_NE(result.status_message.find("braking support"),
            std::string::npos);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTightPointTest,
       ComposesWithAllocationEffortAndLocksInPhysicalAcceleration) {
  urdf_ = std::make_unique<ScopedUrdf>(
      "embodik_acceleration_tight_point_effort_test.urdf",
      two_link_urdf("1e9", "1e9"));
  robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
  robot_->set_gravity(Eigen::Vector3d::Zero());
  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");
  solver.set_task_reference("posture", reference(vector({100.0, -60.0})));

  const Eigen::VectorXd q = vector({0.35, -0.2});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  const Eigen::Vector3d target = frame_translation(*robot_, q, dq);
  auto options = options_with_limits(200.0);
  GeneralizedAccelerationAllocation allocation;
  allocation.metric_diagonal = vector({2.0, 0.5});
  allocation.reference_acceleration = vector({3.0, -3.0});
  options.generalized_acceleration_allocation = allocation;
  options.effort_constraints = EffortConstraintOptions{};
  options.effort_constraints->limits_override = vector({1e9, 1e9});
  options.zero_acceleration_joint_indices = {0};
  options.tight_point_constraints.push_back(tight_point(
      "allocated_y", "tool", target, Eigen::Vector3d(0.0, 1.0, 0.0),
      0.02, policy(10.0, 0.25, 0.25, 0.25)));

  const auto result = solver.solve(q, dq, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.allocation_diagnostics.applied);
  EXPECT_NEAR(result.joint_accelerations(0), 0.0, kTolerance);
  EXPECT_EQ(result.predicted_torques.size(), robot_->nv());
  const Eigen::Vector3d physical =
      frame_acceleration(*robot_, q, dq, result.joint_accelerations);
  EXPECT_LE(std::abs(physical.y()), 0.25 + kTolerance);
}

} // namespace
} // namespace embodik::test
