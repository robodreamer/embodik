#include "acceleration_capture_point_constraint.hpp"
#include "acceleration_zmp_constraint.hpp"
#include "geometric_constraint_differential.hpp"

#include <embodik/acceleration_solver.hpp>

#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>

namespace embodik::test {
namespace {

constexpr double kTolerance = 1e-8;

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

Eigen::MatrixXd square(const Eigen::Vector2d &center, double half_span) {
  Eigen::MatrixXd vertices(4, 2);
  vertices << center.x() - half_span, center.y() - half_span,
      center.x() + half_span, center.y() - half_span, center.x() + half_span,
      center.y() + half_span, center.x() - half_span,
      center.y() + half_span;
  return vertices;
}

void expect_no_motion_outputs(const AccelerationSolverResult &result) {
  EXPECT_TRUE(result.solution.empty());
  EXPECT_EQ(result.joint_accelerations.size(), 0);
  EXPECT_EQ(result.joint_velocities_next.size(), 0);
  EXPECT_EQ(result.q_solution.size(), 0);
}

std::string dynamic_support_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_dynamic_support_test_robot">
  <link name="base_link">
    <inertial>
      <origin xyz="0.0 0.0 0.1" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>
  </link>
  <link name="support"/>
  <joint name="support_fixed" type="fixed">
    <parent link="base_link"/>
    <child link="support"/>
    <origin xyz="0.1 -0.2 0" rpy="0 0 0"/>
  </joint>
  <link name="link1">
    <inertial>
      <origin xyz="0.5 0.0 0.2" rpy="0.02 0.01 -0.03"/>
      <mass value="1.5"/>
      <inertia ixx="0.08" ixy="0.004" ixz="-0.002" iyy="0.12" iyz="0.003" izz="0.10"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="500.0" effort="1000.0"/>
  </joint>
  <link name="tip">
    <inertial>
      <origin xyz="0.35 0.05 0.1" rpy="-0.03 0.02 0.01"/>
      <mass value="0.8"/>
      <inertia ixx="0.04" ixy="-0.002" ixz="0.001" iyy="0.06" iyz="0.003" izz="0.05"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="tip"/>
    <origin xyz="1 0 0" rpy="0.04 -0.02 0.01"/>
    <axis xyz="0.1 0.0 0.995"/>
    <limit lower="-3.14" upper="3.14" velocity="500.0" effort="1000.0"/>
  </joint>
</robot>)";
}

AccelerationSolveOptions options_with_limits(double limit, Eigen::Index nv) {
  AccelerationSolveOptions options;
  options.acceleration_limits_override = Eigen::VectorXd::Constant(nv, limit);
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  return options;
}

CapturePointAccelerationConstraint capture_constraint(
    const std::string &source_id, const Eigen::MatrixXd &polygon) {
  CapturePointAccelerationConstraint constraint;
  constraint.source_id = source_id;
  constraint.definition.support_polygon = polygon;
  constraint.omega = 3.0;
  return constraint;
}

ZmpAccelerationConstraint zmp_constraint(const std::string &source_id,
                                         const Eigen::MatrixXd &polygon) {
  ZmpAccelerationConstraint constraint;
  constraint.source_id = source_id;
  constraint.definition.support_polygon = polygon;
  constraint.fz_min = 1.0;
  return constraint;
}

class AccelerationSolverDynamicSupportTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_dynamic_support.urdf",
        dynamic_support_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
    robot_->set_gravity(Eigen::Vector3d(0.0, 0.0, -9.81));
    q_ = vector({0.35, -0.25});
    dq_ = vector({0.12, -0.08});
    robot_->update_kinematics(q_, dq_);
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
  Eigen::VectorXd q_;
  Eigen::VectorXd dq_;
};

TEST_F(AccelerationSolverDynamicSupportTest,
       CapabilitiesAdvertiseNarrowFixedBaseFamiliesOnly) {
  const auto capabilities = AccelerationSolver::capabilities();
  EXPECT_TRUE(capabilities.supports_fixed_base_capture_point_constraints);
  EXPECT_TRUE(capabilities.supports_fixed_base_zmp_constraints);
  EXPECT_FALSE(capabilities.supports_dynamic_contact);
  EXPECT_FALSE(capabilities.supports_dynamic_balance);
}

TEST_F(AccelerationSolverDynamicSupportTest,
       CapturePointAffineRowsMatchFrozenFormulaAndDerivedOmega) {
  const double dt = 0.02;
  const auto com = detail::evaluate_com_in_frame_differential(*robot_, "world");
  auto constraint = capture_constraint(
      "cp_rows",
      square(com.value.head<2>() + com.rate.head<2>() / 3.0, 1.0));
  std::unordered_set<std::string> source_ids;
  std::vector<CapturePointAccelerationConstraint> cp_constraints;
  cp_constraints.push_back(constraint);
  const auto assembly =
      detail::make_capture_point_constraints(*robot_, cp_constraints, dt,
                                             &source_ids);
  ASSERT_TRUE(assembly.satisfied()) << assembly.message;
  ASSERT_EQ(assembly.constraints.size(), 1U);

  const Eigen::VectorXd ddq = vector({0.4, -0.2});
  const Eigen::Vector2d expected_point =
      com.value.head<2>() + dt * com.rate.head<2>() +
      com.rate.head<2>() / 3.0 +
      (0.5 * dt * dt + dt / 3.0) *
          (com.jacobian.topRows<2>() * ddq + com.affine_bias.head<2>());
  const auto &affine = assembly.constraints[0];
  const Eigen::VectorXd physical =
      affine.coefficient_matrix * ddq + affine.affine_bias;
  const auto geometry = detail::prepare_support_polygon_geometry(
      constraint.definition, "test", "cp_rows");
  ASSERT_TRUE(geometry.satisfied()) << geometry.message;
  EXPECT_TRUE(physical.isApprox(
      geometry.geometry.halfspace_normals * expected_point, kTolerance));

  constraint.source_id = "derived";
  constraint.omega.reset();
  constraint.definition.support_polygon =
      square(com.value.head<2>() + com.rate.head<2>() / 3.0, 2.0);
  source_ids.clear();
  cp_constraints.clear();
  cp_constraints.push_back(constraint);
  const auto derived =
      detail::make_capture_point_constraints(*robot_, cp_constraints, dt,
                                             &source_ids);
  ASSERT_TRUE(derived.satisfied()) << derived.message;
  EXPECT_GT(derived.prepared_constraints[0].frozen_omega, 0.0);
}

TEST_F(AccelerationSolverDynamicSupportTest,
       ZmpRowsMatchCrossMultipliedFormulaAndPositiveForceGate) {
  const auto com = detail::evaluate_com_in_frame_differential(*robot_, "world");
  auto constraint = zmp_constraint("zmp_rows", square(com.value.head<2>(), 1.0));
  std::unordered_set<std::string> source_ids;
  std::vector<ZmpAccelerationConstraint> zmp_constraints;
  zmp_constraints.push_back(constraint);
  const auto assembly =
      detail::make_zmp_constraints(*robot_, zmp_constraints, &source_ids);
  ASSERT_TRUE(assembly.satisfied()) << assembly.message;
  ASSERT_EQ(assembly.constraints.size(), 1U);

  const Eigen::VectorXd ddq = vector({0.3, -0.1});
  const Eigen::Matrix<double, 6, 1> hdot =
      robot_->get_centroidal_momentum_matrix() * ddq +
      robot_->get_centroidal_momentum_matrix_bias();
  const Eigen::Vector3d force =
      hdot.head<3>() - robot_->get_total_mass() * robot_->get_gravity();
  const double fz = force.z();
  const auto geometry = detail::prepare_support_polygon_geometry(
      constraint.definition, "test", "zmp_rows");
  ASSERT_TRUE(geometry.satisfied()) << geometry.message;
  const auto &affine = assembly.constraints[0];
  const Eigen::VectorXd physical =
      affine.coefficient_matrix * ddq + affine.affine_bias;
  EXPECT_NEAR(physical(0), fz, kTolerance);
  for (Eigen::Index edge = 0; edge < geometry.geometry.halfspace_offsets.size();
       ++edge) {
    const double nx = geometry.geometry.halfspace_normals(edge, 0);
    const double ny = geometry.geometry.halfspace_normals(edge, 1);
    const double expected =
        (nx * com.value.x() + ny * com.value.y() -
         geometry.geometry.halfspace_offsets(edge)) *
            fz +
        ny * (hdot(3) - com.value.z() * force.y()) -
        nx * (hdot(4) + com.value.z() * force.x());
    EXPECT_NEAR(physical(edge + 1), expected, kTolerance);
  }

  AccelerationSolver solver(robot_);
  auto options = options_with_limits(10.0, robot_->nv());
  constraint.source_id = "force_gate";
  constraint.fz_min = 1e3;
  options.zmp_constraints.clear();
  options.zmp_constraints.push_back(constraint);
  const auto gated = solver.solve(q_, dq_, 0.01, options);
  EXPECT_TRUE(gated.status == SolverStatus::kInfeasible ||
              gated.status == SolverStatus::kNoProgress)
      << gated.status_message;
  expect_no_motion_outputs(gated);
}

TEST_F(AccelerationSolverDynamicSupportTest,
       RootFixedFramesPassButMovingFramesAndBadGeometryFailClosed) {
  const auto support_pose = robot_->get_frame_pose("support");
  const Eigen::Vector2d com_support =
      (support_pose.rotation().transpose() *
       (robot_->get_com_position() - support_pose.translation()))
          .head<2>();
  auto cp = capture_constraint("cp_support", square(com_support, 0.5));
  cp.definition.frame_name = "support";
  std::unordered_set<std::string> source_ids;
  std::vector<CapturePointAccelerationConstraint> cp_constraints;
  cp_constraints.push_back(cp);
  EXPECT_TRUE(detail::make_capture_point_constraints(*robot_, cp_constraints, 0.01,
                                                     &source_ids)
                  .satisfied());

  cp.source_id = "cp_moving";
  cp.definition.frame_name = "tip";
  source_ids.clear();
  cp_constraints.clear();
  cp_constraints.push_back(cp);
  EXPECT_EQ(detail::make_capture_point_constraints(*robot_, cp_constraints, 0.01,
                                                   &source_ids)
                .status,
            SolverStatus::kInvalidInput);

  auto zmp = zmp_constraint("zmp_bad", Eigen::MatrixXd::Zero(2, 2));
  source_ids.clear();
  std::vector<ZmpAccelerationConstraint> zmp_constraints;
  zmp_constraints.push_back(zmp);
  EXPECT_EQ(detail::make_zmp_constraints(*robot_, zmp_constraints, &source_ids).status,
            SolverStatus::kInvalidInput);

  cp = capture_constraint("cp_nan", square(robot_->get_com_position().head<2>(), 0.5));
  cp.omega = std::numeric_limits<double>::quiet_NaN();
  source_ids.clear();
  cp_constraints.clear();
  cp_constraints.push_back(cp);
  EXPECT_EQ(detail::make_capture_point_constraints(*robot_, cp_constraints, 0.01,
                                                   &source_ids)
                .status,
            SolverStatus::kInvalidInput);
}

TEST_F(AccelerationSolverDynamicSupportTest,
       StationaryContinuityAndDisabledInvarianceHold) {
  const Eigen::VectorXd zero_dq = Eigen::VectorXd::Zero(robot_->nv());
  robot_->update_kinematics(q_, zero_dq);
  const Eigen::Vector2d com = robot_->get_com_position().head<2>();

  AccelerationSolver baseline_solver(robot_);
  auto baseline_options = options_with_limits(20.0, robot_->nv());
  const auto baseline = baseline_solver.solve(q_, zero_dq, 0.01,
                                              baseline_options);
  ASSERT_EQ(baseline.status, SolverStatus::kSuccess)
      << baseline.status_message;

  AccelerationSolver constrained_solver(robot_);
  auto constrained_options = baseline_options;
  constrained_options.capture_point_constraints.push_back(
      capture_constraint("cp", square(com, 0.5)));
  constrained_options.zmp_constraints.push_back(
      zmp_constraint("zmp", square(com, 0.5)));
  const auto constrained =
      constrained_solver.solve(q_, zero_dq, 0.01, constrained_options);
  ASSERT_EQ(constrained.status, SolverStatus::kSuccess)
      << constrained.status_message;
  EXPECT_TRUE(constrained.joint_accelerations.isApprox(
      baseline.joint_accelerations, 1e-12));
  ASSERT_EQ(constrained.capture_point_diagnostics.size(), 1U);
  ASSERT_EQ(constrained.zmp_diagnostics.size(), 1U);
  const auto cp_diff = detail::evaluate_com_in_frame_differential_at_state(
      *robot_, "world", constrained.q_solution,
      constrained.joint_velocities_next);
  const Eigen::Vector2d expected_cp =
      cp_diff.value.head<2>() + cp_diff.rate.head<2>() / 3.0;
  const auto &cp_diag = constrained.capture_point_diagnostics[0];
  EXPECT_EQ(cp_diag.source_id, "cp");
  EXPECT_TRUE(cp_diag.postvalidated);
  EXPECT_NEAR(cp_diag.frozen_omega, 3.0, kTolerance);
  EXPECT_TRUE(cp_diag.predicted_point_xy.isApprox(expected_cp, kTolerance));
  EXPECT_EQ(cp_diag.half_plane_slacks.size(), 4);
  EXPECT_NEAR(cp_diag.min_slack, cp_diag.half_plane_slacks.minCoeff(),
              kTolerance);
  EXPECT_GT(cp_diag.min_slack, 0.0);

  const Eigen::Matrix<double, 6, 1> hdot_next =
      robot_->compute_centroidal_momentum_matrix(
          constrained.q_solution, constrained.joint_velocities_next) *
          constrained.joint_accelerations +
      robot_->compute_centroidal_momentum_matrix_bias(
          constrained.q_solution, constrained.joint_velocities_next);
  const Eigen::Vector3d force_next =
      hdot_next.head<3>() - robot_->get_total_mass() * robot_->get_gravity();
  const double expected_force_z = force_next.z();
  const Eigen::Vector2d expected_zmp(
      cp_diff.value.x() -
          (hdot_next(4) + cp_diff.value.z() * force_next.x()) /
              expected_force_z,
      cp_diff.value.y() +
          (hdot_next(3) - cp_diff.value.z() * force_next.y()) /
              expected_force_z);
  const auto &zmp_diag = constrained.zmp_diagnostics[0];
  EXPECT_EQ(zmp_diag.source_id, "zmp");
  EXPECT_TRUE(zmp_diag.postvalidated);
  EXPECT_NEAR(zmp_diag.force_z, expected_force_z, kTolerance);
  EXPECT_TRUE(zmp_diag.predicted_point_xy.isApprox(expected_zmp, kTolerance));
  EXPECT_EQ(zmp_diag.half_plane_slacks.size(), 4);
  EXPECT_NEAR(zmp_diag.min_slack, zmp_diag.half_plane_slacks.minCoeff(),
              kTolerance);
  EXPECT_GT(zmp_diag.min_slack, 0.0);

  AccelerationSolver explicit_empty_solver(robot_);
  auto explicit_empty_options = baseline_options;
  ASSERT_TRUE(explicit_empty_options.capture_point_constraints.empty());
  ASSERT_TRUE(explicit_empty_options.zmp_constraints.empty());
  const auto explicit_empty =
      explicit_empty_solver.solve(q_, zero_dq, 0.01, explicit_empty_options);
  ASSERT_EQ(explicit_empty.status, SolverStatus::kSuccess)
      << explicit_empty.status_message;
  EXPECT_TRUE(explicit_empty.q_solution.isApprox(baseline.q_solution, 1e-12));
}

TEST_F(AccelerationSolverDynamicSupportTest,
       StationaryBoundaryDoesNotChatter) {
  const Eigen::VectorXd zero_dq = Eigen::VectorXd::Zero(robot_->nv());
  robot_->update_kinematics(q_, zero_dq);
  const Eigen::Vector2d com = robot_->get_com_position().head<2>();
  const Eigen::MatrixXd boundary_polygon =
      square(com + Eigen::Vector2d(0.5, 0.0), 0.5);

  AccelerationSolver solver(robot_);
  auto options = options_with_limits(20.0, robot_->nv());
  options.capture_point_constraints.push_back(
      capture_constraint("cp_boundary", boundary_polygon));
  options.zmp_constraints.push_back(
      zmp_constraint("zmp_boundary", boundary_polygon));

  for (int step = 0; step < 50; ++step) {
    const auto result = solver.solve(q_, zero_dq, 0.01, options);
    ASSERT_EQ(result.status, SolverStatus::kSuccess)
        << "step " << step << ": " << result.status_message;
    EXPECT_LE(result.joint_accelerations.norm(), 1e-12);
    EXPECT_LE(result.joint_velocities_next.norm(), 1e-12);
    EXPECT_TRUE(result.q_solution.isApprox(q_, 1e-12));
    ASSERT_EQ(result.capture_point_diagnostics.size(), 1U);
    ASSERT_EQ(result.zmp_diagnostics.size(), 1U);
    EXPECT_GE(result.capture_point_diagnostics[0].min_slack, -kTolerance);
    EXPECT_GE(result.zmp_diagnostics[0].min_slack, -kTolerance);
  }
}

TEST_F(AccelerationSolverDynamicSupportTest,
       DuplicateIdsAreGlobalAndCombinedFamiliesCompose) {
  AccelerationSolver solver(robot_);
  auto options = options_with_limits(50.0, robot_->nv());
  const Eigen::Vector2d com = robot_->get_com_position().head<2>();
  options.capture_point_constraints.push_back(
      capture_constraint("dynamic_support", square(com, 2.0)));
  options.zmp_constraints.push_back(zmp_constraint("zmp", square(com, 2.0)));
  options.contact_acceleration_constraints.push_back(
      {"contact", "support", ContactType::kRigidContact});
  CentroidalMomentumRateBounds hdot_bounds;
  hdot_bounds.source_id = "hdot_box";
  hdot_bounds.lower_bounds = Eigen::VectorXd::Constant(6, -1e4);
  hdot_bounds.upper_bounds = Eigen::VectorXd::Constant(6, 1e4);
  options.centroidal_momentum_rate_bounds = {hdot_bounds};
  GeneralizedAccelerationAllocation allocation;
  allocation.metric_diagonal = vector({2.0, 0.5});
  allocation.reference_acceleration = vector({0.1, -0.1});
  options.generalized_acceleration_allocation = allocation;

  const auto composed = solver.solve(q_, dq_, 0.01, options);
  ASSERT_EQ(composed.status, SolverStatus::kSuccess)
      << composed.status_message;
  EXPECT_TRUE(composed.allocation_diagnostics.applied);

  options.zmp_constraints[0].source_id = "dynamic_support";
  const auto duplicate = solver.solve(q_, dq_, 0.01, options);
  EXPECT_EQ(duplicate.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(duplicate);
}

} // namespace
} // namespace embodik::test
