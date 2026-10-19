#include "acceleration_braking_witness.hpp"
#include "acceleration_com_support_polygon_constraint.hpp"
#include "acceleration_scalar_geometric_constraint.hpp"
#include "geometric_constraint_differential.hpp"
#include "support_polygon_geometry.hpp"

#include <embodik/acceleration_solver.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>

namespace embodik::test {
namespace {

constexpr double kTolerance = 1e-7;

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

std::string two_link_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_com_support_polygon_test_robot">
  <link name="base_link"/>
  <link name="support"/>
  <joint name="support_fixed" type="fixed">
    <parent link="base_link"/>
    <child link="support"/>
    <origin xyz="0.1 -0.2 0" rpy="0 0 0"/>
  </joint>
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
    <limit lower="-3.14" upper="3.14" velocity="500.0" effort="100.0"/>
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
    <limit lower="-3.14" upper="3.14" velocity="500.0" effort="100.0"/>
  </joint>
</robot>)";
}

Eigen::MatrixXd square(const Eigen::Vector2d &center, double half_span) {
  Eigen::MatrixXd vertices(4, 2);
  vertices << center.x() - half_span, center.y() - half_span,
      center.x() + half_span, center.y() - half_span, center.x() + half_span,
      center.y() + half_span, center.x() - half_span,
      center.y() + half_span;
  return vertices;
}

AccelerationSolveOptions options_with_limits(double limit) {
  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({limit, limit});
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  return options;
}

ComSupportPolygonAccelerationConstraint polygon_constraint(
    const std::string &source_id, const Eigen::MatrixXd &vertices,
    double rate = 0.4, double acceleration = 1.0) {
  ComSupportPolygonAccelerationConstraint constraint;
  constraint.source_id = source_id;
  constraint.definition.support_polygon = vertices;
  constraint.policy.rate_limit = rate;
  constraint.policy.acceleration_limit = acceleration;
  constraint.policy.braking_acceleration = acceleration;
  return constraint;
}

class AccelerationSolverComSupportPolygonTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_com_support_polygon_test.urdf",
        two_link_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
    robot_->set_gravity(Eigen::Vector3d::Zero());
    q_ = vector({0.25, -0.4});
    dq_ = vector({0.2, -0.1});
    robot_->update_kinematics(q_, dq_);
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
  Eigen::VectorXd q_;
  Eigen::VectorXd dq_;
};

TEST(ComSupportPolygonGeometryTest,
     CanonicalHullCharSizeMarginAndProximityAreStrict) {
  Eigen::MatrixXd unordered(6, 2);
  unordered << 1, 1, -1, -1, 1, -1, 0, 0, -1, 1, 1, 1;
  ComSupportPolygonConstraintDefinition definition;
  definition.support_polygon = unordered;
  definition.margin = 0.1;
  definition.proximity_fraction = 0.5;
  const auto prepared = detail::prepare_support_polygon_geometry(
      definition, "test polygon", "square");
  ASSERT_TRUE(prepared.satisfied()) << prepared.message;
  EXPECT_EQ(prepared.geometry.hull.size(), 4U);
  EXPECT_NEAR(prepared.geometry.char_size, std::sqrt(2.0), kTolerance);
  EXPECT_NEAR(prepared.geometry.inradius, 1.0 - 0.1, 2e-2);
  EXPECT_NEAR(prepared.geometry.proximity_threshold,
              0.5 * prepared.geometry.inradius, kTolerance);
  EXPECT_LE((prepared.geometry.halfspace_normals *
                 Eigen::Vector2d::Zero() -
             prepared.geometry.halfspace_offsets)
                .maxCoeff(),
            0.0);

  definition.support_polygon = Eigen::MatrixXd::Zero(2, 2);
  EXPECT_EQ(detail::prepare_support_polygon_geometry(
                definition, "test polygon", "two_points")
                .status,
            SolverStatus::kInvalidInput);
  definition.support_polygon.resize(3, 2);
  definition.support_polygon << 0, 0, 1, 0, 2, 0;
  EXPECT_EQ(detail::prepare_support_polygon_geometry(
                definition, "test polygon", "collinear")
                .status,
            SolverStatus::kInvalidInput);
  definition.support_polygon = square(Eigen::Vector2d::Zero(), 1.0);
  definition.margin = 0.0;
  definition.proximity_fraction = 0.0;
  const auto always_active = detail::prepare_support_polygon_geometry(
      definition, "test polygon", "always_active");
  ASSERT_TRUE(always_active.satisfied()) << always_active.message;
  EXPECT_TRUE(std::isinf(always_active.geometry.proximity_threshold));

  definition.margin = 1.01;
  EXPECT_EQ(detail::prepare_support_polygon_geometry(
                definition, "test polygon", "margin")
                .status,
            SolverStatus::kInvalidInput);
  definition.margin = 0.0;
  definition.proximity_fraction = -0.01;
  EXPECT_EQ(detail::prepare_support_polygon_geometry(
                definition, "test polygon", "proximity")
                .status,
            SolverStatus::kInvalidInput);
  definition.proximity_fraction = 0.0;
  definition.support_polygon(0, 0) =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(detail::prepare_support_polygon_geometry(
                definition, "test polygon", "nan")
                .status,
            SolverStatus::kNonFiniteInput);
}

TEST(ComSupportPolygonApiTest, DefaultsAndCapabilitiesAreTruthful) {
  const ComSupportPolygonAccelerationPolicy policy;
  EXPECT_DOUBLE_EQ(policy.rate_limit, 0.4);
  EXPECT_DOUBLE_EQ(policy.acceleration_limit, 0.1);
  EXPECT_DOUBLE_EQ(policy.braking_acceleration, 0.1);
  EXPECT_EQ(policy.outside_policy,
            ComSupportPolygonOutsidePolicy::kRecoverNonWorsening);
  EXPECT_TRUE(AccelerationSolveOptions{}.com_support_polygon_constraints
                  .empty());
  EXPECT_TRUE(AccelerationSolver::capabilities()
                  .supports_com_support_polygon_constraints);
  EXPECT_FALSE(AccelerationSolver::capabilities().supports_dynamic_contact);
}

TEST_F(AccelerationSolverComSupportPolygonTest,
       ExactComDifferentialMatchesFiniteDifferencesAndProjectedBias) {
  const auto differential =
      detail::evaluate_com_in_frame_differential(*robot_, "world");
  EXPECT_TRUE(differential.rate.isApprox(differential.jacobian * dq_,
                                         1e-12));
  EXPECT_TRUE(differential.affine_bias.isApprox(
      robot_->get_com_jacobian_bias(), 1e-12));

  constexpr double step = 1e-6;
  const Eigen::VectorXd q_plus = robot_->integrate(q_, step * dq_);
  const Eigen::VectorXd q_minus = robot_->integrate(q_, -step * dq_);
  const auto rate_plus =
      detail::evaluate_com_in_frame_differential_at_state(
          *robot_, "world", q_plus, dq_)
          .rate;
  const auto rate_minus =
      detail::evaluate_com_in_frame_differential_at_state(
          *robot_, "world", q_minus, dq_)
          .rate;
  const Eigen::Vector3d finite_difference_bias =
      (rate_plus - rate_minus) / (2.0 * step);
  EXPECT_TRUE(differential.affine_bias.isApprox(finite_difference_bias,
                                                2e-5));

  ComSupportPolygonConstraintDefinition definition;
  definition.support_polygon =
      square(robot_->get_com_position().head<2>(), 0.5);
  const auto geometry = detail::prepare_support_polygon_geometry(
      definition, "test polygon", "projection");
  ASSERT_TRUE(geometry.satisfied()) << geometry.message;
  const auto prepared = detail::prepare_com_support_polygon_constraint(
      polygon_constraint("projection", definition.support_polygon), *robot_,
      0.01, Eigen::Vector2d::Constant(-10.0),
      Eigen::Vector2d::Constant(10.0));
  ASSERT_TRUE(prepared.satisfied()) << prepared.message;
  const Eigen::VectorXd projected_bias =
      geometry.geometry.halfspace_normals * differential.affine_bias.head<2>();
  EXPECT_TRUE(prepared.prepared->state_box.physical_constraint.affine_bias
                  .isApprox(projected_bias, 1e-12));
}

TEST_F(AccelerationSolverComSupportPolygonTest,
       AcceptsWorldAndRootFixedFramesButRejectsMovingFrames) {
  auto world = polygon_constraint(
      "world", square(robot_->get_com_position().head<2>(), 0.5));
  EXPECT_TRUE(detail::validate_com_support_polygon_constraint(world, *robot_)
                  .satisfied());

  const auto support_pose = robot_->get_frame_pose("support");
  const Eigen::Vector2d com_support =
      (support_pose.rotation().transpose() *
       (robot_->get_com_position() - support_pose.translation()))
          .head<2>();
  auto root_fixed = polygon_constraint("root_fixed", square(com_support, 0.5));
  root_fixed.definition.frame_name = "support";
  EXPECT_TRUE(detail::validate_com_support_polygon_constraint(root_fixed,
                                                              *robot_)
                  .satisfied());

  auto moving = root_fixed;
  moving.source_id = "moving";
  moving.definition.frame_name = "tip";
  const auto rejected =
      detail::validate_com_support_polygon_constraint(moving, *robot_);
  EXPECT_EQ(rejected.status, SolverStatus::kInvalidInput);
}

TEST_F(AccelerationSolverComSupportPolygonTest,
       BoundaryDeadZoneIsConsistentAcrossPreparationAndNonlinearAcceptance) {
  constexpr double kBoundaryEpsilon = 1e-4;
  const Eigen::VectorXd zero_dq = Eigen::Vector2d::Zero();
  robot_->update_kinematics(q_, zero_dq);
  const Eigen::Vector2d com = robot_->get_com_position().head<2>();
  constexpr double kHalfSpan = 0.5;
  const Eigen::Vector2d center =
      com + Eigen::Vector2d(-kHalfSpan - 0.5 * kBoundaryEpsilon, 0.0);
  auto constraint =
      polygon_constraint("boundary_dead_zone", square(center, kHalfSpan));
  constraint.policy.boundary_epsilon = kBoundaryEpsilon;

  AccelerationSolver solver(robot_);
  auto options = options_with_limits(10.0);
  options.com_support_polygon_constraints = {constraint};
  const auto result = solver.solve(q_, zero_dq, 0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
}

TEST_F(AccelerationSolverComSupportPolygonTest,
       NonfiniteAndMalformedPoliciesHaveDistinctFailureTaxonomy) {
  auto constraint = polygon_constraint(
      "bad_policy", square(robot_->get_com_position().head<2>(), 0.5));
  constraint.policy.rate_limit =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
      detail::validate_com_support_polygon_constraint(constraint, *robot_)
          .status,
      SolverStatus::kNonFiniteInput);

  constraint.policy = ComSupportPolygonAccelerationPolicy{};
  constraint.policy.acceleration_limit = 0.0;
  EXPECT_EQ(
      detail::validate_com_support_polygon_constraint(constraint, *robot_)
          .status,
      SolverStatus::kInvalidInput);
}

TEST_F(AccelerationSolverComSupportPolygonTest,
       InsideBoundaryAndProximityRowsSolveAndStayContained) {
  AccelerationSolver solver(robot_);
  auto com_task = solver.add_com_task("com");
  com_task->setTargetPosition(robot_->get_com_position() +
                              Eigen::Vector3d(1.0, 0.0, 0.0));
  AccelerationTaskReference reference;
  reference.desired_acceleration = Eigen::Vector3d(10.0, 0.0, 0.0);
  solver.set_task_reference("com", reference);

  auto options = options_with_limits(2.0);
  auto polygon =
      polygon_constraint("support", square(robot_->get_com_position().head<2>(),
                                           0.5),
                         0.2, 0.5);
  polygon.definition.proximity_fraction = 0.0;
  options.com_support_polygon_constraints = {polygon};

  const Eigen::VectorXd zero_dq = Eigen::Vector2d::Zero();
  const auto result = solver.solve(q_, zero_dq, 0.01, options);
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.q_solution.size(), robot_->nq());
  robot_->update_kinematics(result.q_solution, result.joint_velocities_next);
  const auto geometry = detail::prepare_support_polygon_geometry(
      polygon.definition, "test polygon", "support");
  ASSERT_TRUE(geometry.satisfied()) << geometry.message;
  const Eigen::VectorXd signed_dist =
      geometry.geometry.halfspace_normals *
          robot_->get_com_position().head<2>() -
      geometry.geometry.halfspace_offsets;
  EXPECT_LE(signed_dist.maxCoeff(), 1e-7);
}

TEST_F(AccelerationSolverComSupportPolygonTest,
       OutsideRejectRecoverAndOutwardRateSemanticsAreDiscriminated) {
  AccelerationSolver solver(robot_);
  auto options = options_with_limits(20.0);
  const Eigen::VectorXd q_recover = vector({1.57079632679, 0.0});
  robot_->update_kinematics(q_recover, Eigen::Vector2d::Zero());
  Eigen::Vector2d shifted = robot_->get_com_position().head<2>();
  shifted.x() -= 0.04;
  auto reject = polygon_constraint("reject", square(shifted, 0.02), 2.0, 20.0);
  reject.policy.outside_policy = ComSupportPolygonOutsidePolicy::kReject;
  options.com_support_polygon_constraints = {reject};
  const auto rejected =
      solver.solve(q_recover, Eigen::Vector2d::Zero(), 0.01, options);
  EXPECT_EQ(rejected.status, SolverStatus::kInfeasible);
  expect_no_motion_outputs(rejected);

  auto recover = reject;
  recover.source_id = "recover";
  recover.policy.outside_policy =
      ComSupportPolygonOutsidePolicy::kRecoverNonWorsening;
  recover.policy.outside_recovery_scale = 0.01;
  recover.policy.outside_min_recovery_speed = 1e-5;
  options.com_support_polygon_constraints = {recover};
  const auto recovered =
      solver.solve(q_recover, Eigen::Vector2d::Constant(0.0), 0.01, options);
  ASSERT_EQ(recovered.status, SolverStatus::kSuccess)
      << recovered.status_message;
  const auto geometry = detail::prepare_support_polygon_geometry(
      recover.definition, "test polygon", "recover");
  ASSERT_TRUE(geometry.satisfied()) << geometry.message;
  const auto initial =
      detail::evaluate_com_in_frame_differential_at_state(
          *robot_, recover.definition.frame_name, q_recover,
          Eigen::Vector2d::Zero())
          .value.head<2>();
  const auto next =
      detail::evaluate_com_in_frame_differential_at_state(
          *robot_, recover.definition.frame_name, recovered.q_solution,
          recovered.joint_velocities_next)
          .value.head<2>();
  const double initial_violation =
      (geometry.geometry.halfspace_normals * initial -
       geometry.geometry.halfspace_offsets)
          .maxCoeff();
  const double next_violation =
      (geometry.geometry.halfspace_normals * next -
       geometry.geometry.halfspace_offsets)
          .maxCoeff();
  EXPECT_LT(next_violation, initial_violation);

  robot_->update_kinematics(q_recover, vector({-0.3, 0.0}));
  options.com_support_polygon_constraints = {recover};
  const auto outward = solver.solve(q_recover, vector({-0.3, 0.0}), 0.01,
                                    options);
  EXPECT_EQ(outward.status, SolverStatus::kInfeasible);
  expect_no_motion_outputs(outward);
}

TEST(AccelerationBrakingWitnessTest,
     CoupledRowsRejectAnIndividuallyBrakeableButJointlyImpossibleCorner) {
  AffineAccelerationConstraint physical;
  physical.source_id = "corner";
  physical.coefficient_matrix.resize(2, 2);
  physical.coefficient_matrix << 1.0, 1.0, 1.0, -1.0;
  physical.affine_bias = Eigen::Vector2d::Zero();
  physical.lower_bounds = Eigen::Vector2d::Constant(-2.0);
  physical.upper_bounds = Eigen::Vector2d::Constant(2.0);

  const auto impossible =
      detail::validate_common_acceleration_braking_witness(
          physical, Eigen::Vector2d::Constant(-0.5),
          Eigen::Vector2d::Constant(0.5),
          {{0, 0.9, detail::AccelerationBrakingDirection::kDecreaseCoordinate},
           {1, 0.9,
            detail::AccelerationBrakingDirection::kDecreaseCoordinate}},
          "current-state");
  EXPECT_EQ(impossible.status, SolverStatus::kInfeasible)
      << impossible.message;
  EXPECT_NE(impossible.message.find("common braking witness"),
            std::string::npos);

  const auto feasible = detail::validate_common_acceleration_braking_witness(
      physical, Eigen::Vector2d::Constant(-0.5),
      Eigen::Vector2d::Constant(0.5),
      {{0, 0.4, detail::AccelerationBrakingDirection::kDecreaseCoordinate},
       {1, 0.4, detail::AccelerationBrakingDirection::kDecreaseCoordinate}},
      "current-state");
  EXPECT_TRUE(feasible.satisfied()) << feasible.message;
}

TEST_F(AccelerationSolverComSupportPolygonTest,
       PredictedJointBoxLossFailsClosedAfterCurrentStatePreparation) {
  constexpr double kDt = 0.01;
  const Eigen::VectorXd zero_dq = Eigen::Vector2d::Zero();
  robot_->update_kinematics(q_, zero_dq);
  const auto differential =
      detail::evaluate_com_in_frame_differential(*robot_, "world");
  const Eigen::Vector2d center = differential.value.head<2>();
  auto constraint = polygon_constraint(
      "predicted_support",
      square(center + Eigen::Vector2d(-0.49, 0.0), 0.5), 1.0, 1.0);
  constraint.definition.proximity_fraction = 1.0;
  constraint.policy.braking_acceleration = 0.1;

  const auto prepared = detail::prepare_com_support_polygon_constraint(
      constraint, *robot_, kDt, Eigen::Vector2d::Constant(-30.0),
      Eigen::Vector2d::Constant(30.0));
  ASSERT_TRUE(prepared.satisfied()) << prepared.message;
  ASSERT_TRUE(prepared.prepared.has_value());

  const Eigen::VectorXd signed_values =
      prepared.prepared->geometry.halfspace_normals * center -
      prepared.prepared->geometry.halfspace_offsets;
  Eigen::Index outward_edge = 0;
  signed_values.maxCoeff(&outward_edge);
  const Eigen::RowVectorXd outward_row =
      prepared.prepared->geometry.halfspace_normals.row(outward_edge) *
      differential.jacobian.topRows<2>();
  ASSERT_GT(outward_row.squaredNorm(), 1e-12);
  const Eigen::VectorXd accepted_acceleration =
      0.1 * outward_row.transpose() / outward_row.squaredNorm();

  const auto acceptance =
      detail::validate_com_support_polygon_constraint_acceptance(
          *prepared.prepared, *robot_, q_, zero_dq, accepted_acceleration,
          kDt, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
  EXPECT_EQ(acceptance.status, SolverStatus::kNumericalError)
      << acceptance.message;
  EXPECT_NE(acceptance.message.find("predicted support"), std::string::npos)
      << acceptance.message;
}

TEST_F(AccelerationSolverComSupportPolygonTest,
       MultiplePolygonsComposeWithOtherHardFamiliesAndDuplicateIdsAreGlobal) {
  AccelerationSolver solver(robot_);
  auto frame_task =
      solver.add_frame_task("tip_task", "tip", TaskType::FRAME_POSITION);
  frame_task->set_excluded_joint_indices({1});
  frame_task->setSolveMode(TaskSolveMode::kMinError);
  frame_task->setTargetPosition(robot_->get_frame_pose("tip").translation());
  auto options = options_with_limits(5.0);
  GeneralizedAccelerationAllocation allocation;
  allocation.metric_diagonal = vector({2.0, 0.5});
  allocation.reference_acceleration = vector({0.3, -0.2});
  options.generalized_acceleration_allocation = allocation;
  options.effort_constraints = EffortConstraintOptions{};
  options.zero_acceleration_joint_indices = {1};
  AffineAccelerationConstraint affine;
  affine.source_id = "affine";
  affine.coefficient_matrix = Eigen::RowVector2d(1.0, 0.0);
  affine.affine_bias = Eigen::VectorXd::Zero(1);
  affine.lower_bounds = vector({-1.0});
  affine.upper_bounds = vector({1.0});
  options.affine_constraints = {affine};
  options.contact_acceleration_constraints.push_back(
      {"contact", "support", ContactType::kRigidContact});
  options.com_support_polygon_constraints = {
      polygon_constraint("poly_a",
                         square(robot_->get_com_position().head<2>(), 0.5)),
      polygon_constraint("poly_b",
                         square(robot_->get_com_position().head<2>(), 0.7))};

  const auto composed = solver.solve(q_, dq_, 0.01, options);
  ASSERT_EQ(composed.status, SolverStatus::kSuccess)
      << composed.status_message;
  EXPECT_TRUE(composed.allocation_diagnostics.applied);
  EXPECT_EQ(composed.predicted_torques.size(), robot_->nv());
  EXPECT_NEAR(composed.joint_accelerations(1), 0.0, kTolerance);

  options.com_support_polygon_constraints[1].source_id = "affine";
  const auto duplicate = solver.solve(q_, dq_, 0.01, options);
  EXPECT_EQ(duplicate.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(duplicate);
}

// Independent physical formulas below intentionally do not use the production
// CoM differential. Only the path sampling policy is shared.
detail::PreparedComSupportPolygonConstraint derivative_fixture(int nv) {
  detail::PreparedComSupportPolygonConstraint prepared;
  prepared.specification = polygon_constraint(
      "path_derivative", square(Eigen::Vector2d::Zero(), 3.0), 10.0, 0.01);
  prepared.specification.definition.frame_name = "world";
  const auto geometry = detail::prepare_support_polygon_geometry(
      prepared.specification.definition, "derivative test", "path_derivative");
  EXPECT_TRUE(geometry.satisfied()) << geometry.message;
  prepared.geometry = geometry.geometry;
  prepared.state_box.physical_constraint.coefficient_matrix =
      Eigen::MatrixXd::Zero(prepared.geometry.halfspace_normals.rows(), nv);
  return prepared;
}

using PhysicalAcceleration =
    std::function<Eigen::Vector2d(double, const Eigen::VectorXd &)>;

std::vector<double> derivative_worst_times(
    const detail::PreparedComSupportPolygonConstraint &prepared,
    const Eigen::VectorXd &dq, const Eigen::VectorXd &a, double dt,
    const PhysicalAcceleration &physical) {
  const detail::ScalarGeometricPathValidationOptions policy;
  const int segments = std::max(policy.minimum_segments, static_cast<int>(std::ceil(
      dt * dq.cwiseAbs().cwiseMax((dq + dt * a).cwiseAbs()).maxCoeff() /
      policy.maximum_joint_tangent_step)));
  std::vector<double> times;
  for (Eigen::Index edge = 0; edge < prepared.geometry.halfspace_normals.rows(); ++edge) {
    double worst = 0.0, time = -1.0;
    for (double t : detail::scalar_geometric_path_sample_times(segments, dt)) {
      const double excess = std::abs(prepared.geometry.halfspace_normals.row(edge).dot(
          physical(t, a))) - prepared.specification.policy.acceleration_limit;
      if (excess > 1e-8 && excess > worst) {
        worst = excess;
        time = t;
      }
    }
    // All fixture edges violate; output rows therefore retain normal order.
    EXPECT_GE(time, 0.0);
    times.push_back(time);
  }
  return times;
}

TEST(ComPathDerivativeTest, PendulumAnalyticDerivativeAndAnchorForBothSigns) {
  ScopedUrdf urdf("embodik_com_path_derivative_pendulum.urdf", R"(
<robot name="pendulum"><link name="base"/>
<link name="body"><inertial><origin xyz="0 1 0"/><mass value="1"/>
<inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".1"/>
</inertial></link><joint name="pivot" type="revolute"><parent link="base"/>
<child link="body"/><axis xyz="0 0 1"/>
<limit lower="-3" upper="3" velocity="10" effort="100"/></joint></robot>)");
  RobotModel robot(urdf.path(), false);
  const auto prepared = derivative_fixture(1);
  constexpr double q = -0.518, dq = 0.333, dt = 0.283;
  for (double acceleration : {-2.0, 2.0}) {
    SCOPED_TRACE(acceleration);
    const Eigen::VectorXd candidate = vector({acceleration});
    const PhysicalAcceleration physical = [=](double t, const Eigen::VectorXd &a) {
      const double angle = q + t * dq + 0.5 * t * t * a(0);
      const double rate = dq + t * a(0);
      return Eigen::Vector2d(-std::cos(angle) * a(0) + std::sin(angle) * rate * rate,
                            -std::sin(angle) * a(0) - std::cos(angle) * rate * rate);
    };
    const auto times = derivative_worst_times(
        prepared, vector({dq}), candidate, dt, physical);
    ASSERT_TRUE(std::any_of(times.begin(), times.end(), [](double t) { return t > 0.0; }));
    AffineAccelerationConstraint row;
    const auto result = detail::linearize_com_support_polygon_path_acceleration(
        prepared, robot, vector({q}), vector({dq}), candidate, dt, &row);
    ASSERT_TRUE(result.satisfied()) << result.message;
    ASSERT_EQ(row.coefficient_matrix.rows(), static_cast<Eigen::Index>(times.size()));
    for (Eigen::Index edge = 0; edge < row.coefficient_matrix.rows(); ++edge) {
      const double t = times[edge], a = acceleration;
      const double angle = q + t * dq + 0.5 * t * t * a;
      const double rate = dq + t * a;
      const Eigen::Vector2d derivative(
          -std::cos(angle) + 0.5 * t * t * std::sin(angle) * a +
              0.5 * t * t * std::cos(angle) * rate * rate + 2 * t * std::sin(angle) * rate,
          -std::sin(angle) - 0.5 * t * t * std::cos(angle) * a +
              0.5 * t * t * std::sin(angle) * rate * rate - 2 * t * std::cos(angle) * rate);
      const auto normal = prepared.geometry.halfspace_normals.row(edge);
      EXPECT_NEAR(row.coefficient_matrix(edge, 0), normal.dot(derivative), 2e-8);
      EXPECT_NEAR(row.coefficient_matrix.row(edge).dot(candidate) + row.affine_bias(edge),
                  normal.dot(physical(t, candidate)), 1e-12);
    }
  }
}

TEST(ComPathDerivativeTest, RotatedRootFrameProjectsTheCompleteWorldGradient) {
  ScopedUrdf urdf("embodik_com_rotated_root_derivative.urdf", R"(
<robot name="rotated"><link name="base"/><link name="support"/>
<joint name="fixed_support" type="fixed"><parent link="base"/><child link="support"/>
<origin xyz=".1 -.2 .3" rpy=".2 -.1 .3"/></joint>
<link name="body"><inertial><origin xyz="0 0 1"/><mass value="1"/>
<inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".1"/></inertial></link>
<joint name="pivot" type="revolute"><parent link="base"/><child link="body"/>
<axis xyz="0 1 0"/><limit lower="-3" upper="3" velocity="10" effort="100"/>
</joint></robot>)");
  RobotModel robot(urdf.path(), false);
  auto prepared = derivative_fixture(1);
  prepared.specification.definition.frame_name = "support";
  const Eigen::Matrix3d rotation =
      (Eigen::AngleAxisd(.3, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(-.1, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(.2, Eigen::Vector3d::UnitX())).toRotationMatrix();
  constexpr double q = .3, dq = .4, dt = .15;
  for (double a : {-3., 3.}) {
    const Eigen::VectorXd candidate = vector({a});
    const PhysicalAcceleration physical = [=](double t, const Eigen::VectorXd &command) {
      const double x = q + t * dq + .5 * t * t * command(0), v = dq + t * command(0);
      const Eigen::Vector3d world(command(0) * std::cos(x) - v * v * std::sin(x), 0.,
                                  -command(0) * std::sin(x) - v * v * std::cos(x));
      return Eigen::Vector2d((rotation.transpose() * world).head<2>());
    };
    const auto times = derivative_worst_times(
        prepared, vector({dq}), candidate, dt, physical);
    ASSERT_EQ(times.size(), 4U);
    AffineAccelerationConstraint row;
    const auto result = detail::linearize_com_support_polygon_path_acceleration(
        prepared, robot, vector({q}), vector({dq}), candidate, dt, &row);
    ASSERT_TRUE(result.satisfied()) << result.message;
    ASSERT_EQ(row.coefficient_matrix.rows(), static_cast<Eigen::Index>(times.size()));
    for (Eigen::Index edge = 0; edge < row.coefficient_matrix.rows(); ++edge) {
      const double t = times[edge], x = q + t * dq + .5 * t * t * a, v = dq + t * a;
      const Eigen::Vector3d world(
          std::cos(x) - .5 * t * t * a * std::sin(x) -
              2 * t * v * std::sin(x) - .5 * t * t * v * v * std::cos(x),
          0., -std::sin(x) - .5 * t * t * a * std::cos(x) -
                  2 * t * v * std::cos(x) + .5 * t * t * v * v * std::sin(x));
      const Eigen::Vector2d derivative = (rotation.transpose() * world).head<2>();
      const auto normal = prepared.geometry.halfspace_normals.row(edge);
      EXPECT_NEAR(row.coefficient_matrix(edge, 0), normal.dot(derivative), 2e-8);
      EXPECT_NEAR(row.coefficient_matrix.row(edge).dot(candidate) + row.affine_bias(edge),
                  normal.dot(physical(t, candidate)), 1e-12);
    }
  }
}

TEST_F(AccelerationSolverComSupportPolygonTest,
       CoupledPathDerivativeMatchesIndependentDirectionalDifferences) {
  const auto prepared = derivative_fixture(2);
  const Eigen::VectorXd q = vector({-0.3, 0.7}), dq = vector({0.4, -0.2});
  const Eigen::VectorXd acceleration = vector({2.0, -1.3});
  constexpr double dt = 0.23;
  const PhysicalAcceleration physical = [&](double t, const Eigen::VectorXd &a) {
    const Eigen::VectorXd angles = q + t * dq + 0.5 * t * t * a;
    const Eigen::VectorXd rates = dq + t * a;
    const auto rotating = [](double angle, double rate, double accel) {
      return Eigen::Vector2d(-std::cos(angle) * rate * rate - std::sin(angle) * accel,
                            -std::sin(angle) * rate * rate + std::cos(angle) * accel);
    };
    // Fixture CoM = (2/3)e(q1) + (1/6)e(q1+q2).
    return Eigen::Vector2d((2.0 / 3.0) * rotating(angles(0), rates(0), a(0)) +
        (1.0 / 6.0) * rotating(angles.sum(), rates.sum(), a.sum()));
  };
  const auto times = derivative_worst_times(prepared, dq, acceleration, dt, physical);
  ASSERT_TRUE(std::any_of(times.begin(), times.end(), [](double t) { return t > 0.0; }));
  AffineAccelerationConstraint row;
  const auto result = detail::linearize_com_support_polygon_path_acceleration(
      prepared, *robot_, q, dq, acceleration, dt, &row);
  ASSERT_TRUE(result.satisfied()) << result.message;
  ASSERT_EQ(row.coefficient_matrix.rows(), static_cast<Eigen::Index>(times.size()));
  for (const Eigen::Vector2d direction : {Eigen::Vector2d(0.6, 0.8),
                                         Eigen::Vector2d(-0.8, 0.6)}) {
    for (double h : {1e-4, 5e-5}) {
      for (Eigen::Index edge = 0; edge < row.coefficient_matrix.rows(); ++edge) {
        const auto normal = prepared.geometry.halfspace_normals.row(edge);
        const Eigen::Vector2d difference =
            (physical(times[edge], acceleration + h * direction) -
             physical(times[edge], acceleration - h * direction)) / (2 * h);
        EXPECT_NEAR(row.coefficient_matrix.row(edge).dot(direction), normal.dot(difference), 2e-8);
        EXPECT_NEAR(row.coefficient_matrix.row(edge).dot(acceleration) + row.affine_bias(edge),
                    normal.dot(physical(times[edge], acceleration)), 1e-12);
      }
    }
  }
}

TEST_F(AccelerationSolverComSupportPolygonTest,
       PathRefinementPreservesModelStateOnSuccessAndEvaluationFailure) {
  // Materialize acceleration-level CoM data before taking any snapshot.
  // A position-only CoM getter can mark com_updated_ without populating vcom;
  // reading the bias later would then change the snapshotted velocity even
  // without refinement. Cache readiness is outside this isolation contract.
  (void)robot_->get_com_jacobian_bias();
  const Eigen::VectorXd cached_q = robot_->get_current_configuration();
  const Eigen::VectorXd cached_dq = robot_->get_current_velocity();
  const Eigen::Matrix4d pose = robot_->get_frame_pose("tip").toHomogeneousMatrix();
  const Eigen::MatrixXd frame_jacobian = robot_->get_frame_jacobian("tip");
  const Eigen::Vector3d com = robot_->get_com_position();
  const Eigen::Vector3d com_velocity = robot_->get_com_velocity();
  const Eigen::MatrixXd com_jacobian = robot_->get_com_jacobian();
  const Eigen::Vector3d com_bias = robot_->get_com_jacobian_bias();
  const auto expect_unchanged = [&] {
    EXPECT_TRUE(robot_->get_current_configuration().isApprox(cached_q, 0.0));
    EXPECT_TRUE(robot_->get_current_velocity().isApprox(cached_dq, 0.0));
    EXPECT_TRUE(robot_->get_frame_pose("tip").toHomogeneousMatrix().isApprox(pose, 0.0));
    EXPECT_TRUE(robot_->get_frame_jacobian("tip").isApprox(frame_jacobian, 0.0));
    EXPECT_TRUE(robot_->get_com_position().isApprox(com, 0.0));
    EXPECT_TRUE(robot_->get_com_velocity().isApprox(com_velocity, 0.0));
    EXPECT_TRUE(robot_->get_com_jacobian().isApprox(com_jacobian, 0.0));
    EXPECT_TRUE(robot_->get_com_jacobian_bias().isApprox(com_bias, 0.0));
  };
  // Control: the same getter sequence must be stable without any refinement.
  expect_unchanged();
  auto prepared = derivative_fixture(2);
  const Eigen::VectorXd q = vector({-0.3, 0.7}), dq = vector({0.4, -0.2});
  const Eigen::VectorXd acceleration = vector({2.0, -1.3});
  AffineAccelerationConstraint first, repeated;
  const auto success = detail::linearize_com_support_polygon_path_acceleration(
      prepared, *robot_, q, dq, acceleration, 0.23, &first);
  ASSERT_TRUE(success.satisfied()) << success.message;
  expect_unchanged();
  // Frame lookup fails after scratch CoM evaluation has already run.
  prepared.specification.definition.frame_name = "missing_support_frame";
  const auto failed = detail::linearize_com_support_polygon_path_acceleration(
      prepared, *robot_, q, dq, acceleration, 0.23, &repeated);
  EXPECT_EQ(failed.status, SolverStatus::kNumericalError) << failed.message;
  expect_unchanged();
  prepared.specification.definition.frame_name = "world";
  const auto again = detail::linearize_com_support_polygon_path_acceleration(
      prepared, *robot_, q, dq, acceleration, 0.23, &repeated);
  ASSERT_TRUE(again.satisfied()) << again.message;
  EXPECT_TRUE(first.coefficient_matrix.isApprox(repeated.coefficient_matrix, 0.0));
  EXPECT_TRUE(first.affine_bias.isApprox(repeated.affine_bias, 0.0));
  expect_unchanged();
}

TEST(ComPathSamplingTest, AnalyticPendulumQuarterPeakCannotBeAccepted) {
  ScopedUrdf urdf("embodik_com_quarter_peak.urdf", R"(
<robot name="quarter_peak">
  <link name="base"/>
  <link name="pendulum"><inertial><origin xyz="0 0 1"/>
    <mass value="1"/><inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".1"/>
  </inertial></link>
  <joint name="joint" type="revolute"><parent link="base"/><child link="pendulum"/>
    <origin xyz="0 0 2"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" velocity="20" effort="100"/>
  </joint>
</robot>)");
  auto robot = std::make_shared<RobotModel>(urdf.path(), false);
  constexpr double dt = 0.01, acceleration = 10.0, peak_rate = 3.6;
  constexpr double peak_time = 0.25 * dt;
  const double initial_rate = peak_rate - acceleration * peak_time;
  // x = sin(q). Its acceleration peaks where
  // x''' = -3*a*v*sin(q) - v^3*cos(q) = 0.
  const double peak_q = -std::atan(peak_rate * peak_rate / (3 * acceleration));
  const double initial_q = peak_q - peak_time * initial_rate -
                           0.5 * peak_time * peak_time * acceleration;
  const auto physical_acceleration = [&](double t) {
    const double q = initial_q + t * initial_rate + 0.5 * t * t * acceleration;
    const double v = initial_rate + t * acceleration;
    return acceleration * std::cos(q) - v * v * std::sin(q);
  };
  const int segments = std::max(16, static_cast<int>(std::ceil(
      dt * std::max(initial_rate, initial_rate + dt * acceleration) / 0.002)));
  ASSERT_EQ(segments, 19);
  double grid_maximum = 0;
  for (int i = 0; i <= segments; ++i)
    grid_maximum = std::max(grid_maximum,
                           std::abs(physical_acceleration(dt * i / segments)));
  const double peak = physical_acceleration(peak_time);
  ASSERT_GT(peak - grid_maximum, 1e-6);
  const double limit = 0.5 * (grid_maximum + peak);
  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({100.0});
  auto constraint = polygon_constraint("quarter", square(Eigen::Vector2d::Zero(), 5.0),
                                       20.0, limit);
  constraint.definition.margin = 0.0;
  options.com_support_polygon_constraints = {constraint};
  AffineAccelerationConstraint lock;
  lock.source_id = "analytic-acceleration-lock";
  lock.coefficient_matrix = Eigen::MatrixXd::Identity(1, 1);
  lock.affine_bias = vector({0.0});
  lock.lower_bounds = lock.upper_bounds = vector({acceleration});
  options.affine_constraints = {lock};
  AccelerationSolver solver(robot);
  const auto result = solver.solve(vector({initial_q}), vector({initial_rate}), dt, options);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError) << result.status_message;
  EXPECT_NE(result.status_message.find("CoM"), std::string::npos);
  EXPECT_EQ(result.q_solution.size(), 0);
  EXPECT_EQ(result.joint_accelerations.size(), 0);
}

} // namespace
} // namespace embodik::test
