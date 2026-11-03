#include "acceleration_fixed_frame_pose_constraint.hpp"
#include "geometric_constraint_differential.hpp"

#include <embodik/acceleration_solver.hpp>

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

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

std::string two_link_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_tight_frame_pose_test_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.2" upper="3.2" velocity="500.0" effort="100.0"/>
  </joint>
  <link name="tip"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.2" upper="3.2" velocity="500.0" effort="100.0"/>
  </joint>
  <link name="tool"/>
  <joint name="tool_fixed" type="fixed">
    <parent link="tip"/>
    <child link="tool"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
</robot>)";
}

Eigen::Matrix4d matrix_from_pose(const pinocchio::SE3 &pose) {
  Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
  matrix.topLeftCorner<3, 3>() = pose.rotation();
  matrix.topRightCorner<3, 1>() = pose.translation();
  return matrix;
}

Eigen::Matrix3d z_rotation(double angle) {
  return Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

GeometricConstraintAccelerationPolicy policy6(double rate,
                                              double acceleration,
                                              double braking) {
  GeometricConstraintAccelerationPolicy policy;
  policy.rate_limits = Eigen::VectorXd::Constant(6, rate);
  policy.acceleration_limits = Eigen::VectorXd::Constant(6, acceleration);
  policy.lower_braking_accelerations =
      Eigen::VectorXd::Constant(6, braking);
  policy.upper_braking_accelerations =
      Eigen::VectorXd::Constant(6, braking);
  return policy;
}

TightFramePoseAccelerationConstraint tight_frame(
    const std::string &source_id, const std::string &frame_name,
    const Eigen::Matrix4d &target_pose,
    const Eigen::Matrix<double, 6, 1> &axis_mask, double position_epsilon,
    double orientation_epsilon,
    const GeometricConstraintAccelerationPolicy &policy) {
  TightFramePoseAccelerationConstraint constraint;
  constraint.source_id = source_id;
  constraint.definition.frame_name = frame_name;
  constraint.definition.target_pose = target_pose;
  constraint.definition.axis_mask = axis_mask;
  constraint.definition.position_epsilon = position_epsilon;
  constraint.definition.orientation_epsilon = orientation_epsilon;
  constraint.policy = policy;
  return constraint;
}

AccelerationSolveOptions options_with_limits(double limit) {
  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({limit, limit});
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  return options;
}

detail::ScalarGeometricPathSegment so3_segment() {
  detail::ScalarGeometricPathSegment segment;
  segment.previous.tangent = Eigen::Vector2d::Zero();
  segment.current.tangent = Eigen::Vector2d::Constant(1e-3);
  segment.previous.coordinates.value = Eigen::VectorXd::Zero(6);
  segment.current.coordinates.value = Eigen::VectorXd::Zero(6);
  segment.previous.coordinates.so3_rotation = Eigen::Matrix3d::Identity();
  segment.current.coordinates.so3_rotation = Eigen::Matrix3d::Identity();
  return segment;
}

class AccelerationSolverTightFramePoseTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_tight_frame_pose_test.urdf", two_link_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
    robot_->set_gravity(Eigen::Vector3d::Zero());
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(AccelerationSolverTightFramePoseTest,
       TranslationBrakingViolationCarriesContextAndExactGradient) {
  const Eigen::VectorXd q = vector({-1.2, 0.0});
  const Eigen::VectorXd dq = vector({0.2, 0.0});
  const Eigen::VectorXd acceleration = vector({-0.5, 0.0});
  robot_->update_kinematics(q, dq);
  Eigen::Matrix<double, 6, 1> mask = Eigen::Matrix<double, 6, 1>::Zero();
  mask(0) = 1.0;
  const auto reference = robot_->get_frame_pose("tool");
  const auto record = detail::make_tight_frame_pose_record(tight_frame(
      "translation_braking", "tool", matrix_from_pose(reference), mask,
      0.010276011537982174, 0.1, policy6(1.0, 10.0, 10.0)));
  const auto prepared = detail::prepare_fixed_frame_pose_constraint(
      record, *robot_, 0.01, Eigen::Vector2d::Constant(-15.0),
      Eigen::Vector2d::Constant(15.0));
  ASSERT_TRUE(prepared.satisfied()) << prepared.message;
  const auto rejected = detail::validate_fixed_frame_pose_constraint_acceptance(
      *prepared.prepared, *robot_, q, dq, acceleration, 0.01,
      Eigen::Vector2d::Constant(-15.0), Eigen::Vector2d::Constant(15.0));
  ASSERT_FALSE(rejected.satisfied());
  ASSERT_TRUE(rejected.violation.has_value()) << rejected.message;
  EXPECT_EQ(rejected.violation->kind,
            detail::ScalarGeometricViolationKind::kBrakingViability);
  EXPECT_EQ(rejected.violation->axis, 0);
  EXPECT_TRUE(rejected.violation->upper);
  EXPECT_DOUBLE_EQ(rejected.violation->time, 0.01);
  AffineAccelerationConstraint refinement;
  const auto linearized = detail::linearize_fixed_frame_pose_violation(
      *prepared.prepared, *robot_, q, dq, acceleration, *rejected.violation,
      &refinement);
  ASSERT_TRUE(linearized.satisfied()) << linearized.message;
  const auto residual = [&](const Eigen::VectorXd &a) {
    const Eigen::VectorXd qn = q + 0.01 * dq + 0.00005 * a;
    const Eigen::VectorXd vn = dq + 0.01 * a;
    const double x = std::cos(qn(0)) + std::cos(qn(0) + qn(1));
    const double rate = -std::sin(qn(0)) * vn(0) -
                       std::sin(qn(0) + qn(1)) * (vn(0) + vn(1));
    return rate * rate - 20.0 * (record.upper_bounds(0) +
                                reference.translation().x() - x);
  };
  EXPECT_NEAR(refinement.coefficient_matrix.row(0).dot(acceleration) +
                  refinement.affine_bias(0), residual(acceleration), 1e-12);
  for (int joint = 0; joint < 2; ++joint) {
    Eigen::VectorXd plus = acceleration, minus = acceleration;
    plus(joint) += 1e-5;
    minus(joint) -= 1e-5;
    EXPECT_NEAR(refinement.coefficient_matrix(0, joint),
                (residual(plus) - residual(minus)) / 2e-5, 1e-9);
  }
  EXPECT_FALSE(refinement.lower_bound_active[0]);
  EXPECT_TRUE(refinement.upper_bound_active[0]);
}

TEST_F(AccelerationSolverTightFramePoseTest,
       TranslationGradientMatchesIndependentPlanarCoordinatesOnBothSides) {
  for (bool upper : {false, true}) {
    const double sign = upper ? -1.0 : 1.0;
    const Eigen::VectorXd q = vector({sign * 0.7, -sign * 0.35});
    const Eigen::VectorXd dq = vector({0.2, -0.12});
    const Eigen::VectorXd a = vector({-0.4, 0.15});
    constexpr double time = 0.017;
    const auto sample = [&](const Eigen::VectorXd &acceleration) {
      const Eigen::VectorXd qn = q + time * dq + 0.5 * time * time * acceleration;
      const Eigen::VectorXd vn = dq + time * acceleration;
      Eigen::Vector2d result;
      result(0) = std::cos(qn(0)) + std::cos(qn(0) + qn(1));
      result(1) = -std::sin(qn(0)) * vn(0) -
                  std::sin(qn(0) + qn(1)) * (vn(0) + vn(1));
      return result;
    };
    const auto center = sample(a);
    const double direction = upper ? 1.0 : -1.0;
    ASSERT_GT(direction * center(1), 0.0);
    const double bound = center(0) + direction * center(1) * center(1) / 20.0 -
                         direction * 1e-6;
    detail::PreparedFixedFramePoseConstraint prepared;
    prepared.specification.frame_name = "tool";
    prepared.specification.policy = policy6(1.0, 10.0, 10.0);
    prepared.specification.lower_bounds.setConstant(bound);
    prepared.specification.upper_bounds.setConstant(bound);
    detail::ScalarGeometricViolation violation{
        detail::ScalarGeometricViolationKind::kBrakingViability, 0, upper, time};
    AffineAccelerationConstraint refinement;
    const auto result = detail::linearize_fixed_frame_pose_violation(
        prepared, *robot_, q, dq, a, violation, &refinement);
    ASSERT_TRUE(result.satisfied()) << result.message;
    const auto residual = [&](const Eigen::VectorXd &acceleration) {
      const auto values = sample(acceleration);
      return values(1) * values(1) + 20.0 * direction * (values(0) - bound);
    };
    EXPECT_NEAR(refinement.coefficient_matrix.row(0).dot(a) + refinement.affine_bias(0),
                residual(a), 1e-12);
    for (int joint = 0; joint < 2; ++joint) {
      Eigen::VectorXd plus = a, minus = a;
      plus(joint) += 1e-5;
      minus(joint) -= 1e-5;
      EXPECT_NEAR(refinement.coefficient_matrix(0, joint),
                  (residual(plus) - residual(minus)) / 2e-5, 1e-9);
    }
    violation.kind = detail::ScalarGeometricViolationKind::kStateBound;
    if (upper) prepared.specification.upper_bounds(0) = center(0) - 1e-6;
    else prepared.specification.lower_bounds(0) = center(0) + 1e-6;
    ASSERT_TRUE(detail::linearize_fixed_frame_pose_violation(
        prepared, *robot_, q, dq, a, violation, &refinement).satisfied());
    for (int joint = 0; joint < 2; ++joint) {
      Eigen::VectorXd plus = a, minus = a;
      plus(joint) += 1e-5;
      minus(joint) -= 1e-5;
      EXPECT_NEAR(refinement.coefficient_matrix(0, joint),
                  direction * (sample(plus)(0) - sample(minus)(0)) / 2e-5, 1e-9);
    }
  }
}

TEST_F(AccelerationSolverTightFramePoseTest,
       TranslationGradientMatchesIndependentSpatialChain) {
  auto urdf = two_link_urdf();
  const auto axis = urdf.find("<axis xyz=\"0 0 1\"/>");
  ASSERT_NE(axis, std::string::npos);
  urdf.replace(axis, std::string("<axis xyz=\"0 0 1\"/>").size(),
               "<axis xyz=\"0 1 0\"/>");
  ScopedUrdf file("embodik_spatial_translation_gradient.urdf", urdf);
  RobotModel robot(file.path(), false);
  const Eigen::VectorXd q = vector({0.4, -0.3});
  const Eigen::VectorXd dq = vector({0.2, 0.1});
  const Eigen::VectorXd a = vector({-0.4, 0.15});
  constexpr double time = 0.017;
  const auto sample = [&](const Eigen::VectorXd &acceleration) {
    const Eigen::VectorXd qn = q + time * dq + 0.5 * time * time * acceleration;
    const Eigen::VectorXd vn = dq + time * acceleration;
    Eigen::Matrix<double, 3, 2> values;
    values.col(0) << std::cos(qn(0)) * (1.0 + std::cos(qn(1))),
                     std::sin(qn(1)),
                    -std::sin(qn(0)) * (1.0 + std::cos(qn(1)));
    values.col(1) << -std::sin(qn(0)) * (1.0 + std::cos(qn(1))) * vn(0) -
                         std::cos(qn(0)) * std::sin(qn(1)) * vn(1),
                     std::cos(qn(1)) * vn(1),
                    -std::cos(qn(0)) * (1.0 + std::cos(qn(1))) * vn(0) +
                         std::sin(qn(0)) * std::sin(qn(1)) * vn(1);
    return values;
  };
  const auto center = sample(a);
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    const bool upper = center(coordinate, 1) > 0.0;
    const double direction = upper ? 1.0 : -1.0;
    const double bound = center(coordinate, 0) +
        direction * center(coordinate, 1) * center(coordinate, 1) / 20.0 - direction * 1e-6;
    detail::PreparedFixedFramePoseConstraint prepared;
    prepared.specification.frame_name = "tool";
    prepared.specification.policy = policy6(1.0, 10.0, 10.0);
    prepared.specification.lower_bounds.setConstant(bound);
    prepared.specification.upper_bounds.setConstant(bound);
    const detail::ScalarGeometricViolation violation{
        detail::ScalarGeometricViolationKind::kBrakingViability, coordinate, upper, time};
    AffineAccelerationConstraint refinement;
    const auto linearized = detail::linearize_fixed_frame_pose_violation(
        prepared, robot, q, dq, a, violation, &refinement);
    ASSERT_TRUE(linearized.satisfied()) << linearized.message;
    const auto residual = [&](const Eigen::VectorXd &acceleration) {
      const auto values = sample(acceleration);
      return values(coordinate, 1) * values(coordinate, 1) +
             20.0 * direction * (values(coordinate, 0) - bound);
    };
    EXPECT_NEAR(refinement.coefficient_matrix.row(0).dot(a) + refinement.affine_bias(0),
                residual(a), 1e-12);
    for (int joint = 0; joint < 2; ++joint) {
      Eigen::VectorXd plus = a, minus = a;
      plus(joint) += 1e-5;
      minus(joint) -= 1e-5;
      EXPECT_NEAR(refinement.coefficient_matrix(0, joint),
                  (residual(plus) - residual(minus)) / 2e-5, 1e-9);
    }
  }
}

TEST_F(AccelerationSolverTightFramePoseTest,
       RotationGradientMatchesIndependentSpatialCoordinatesOnBothSides) {
  auto urdf = two_link_urdf();
  const std::string old_axis = "<axis xyz=\"0 0 1\"/>";
  urdf.replace(urdf.find(old_axis), old_axis.size(), "<axis xyz=\"0 1 0\"/>");
  ScopedUrdf file("embodik_spatial_rotation_gradient.urdf", urdf);
  RobotModel robot(file.path(), false);
  const Eigen::VectorXd q = vector({0.4, -0.3});
  constexpr double time = 0.017;
  const auto rotation = [](const Eigen::VectorXd &state) -> Eigen::Matrix3d {
    return (Eigen::AngleAxisd(state(0), Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(state(1), Eigen::Vector3d::UnitZ())).toRotationMatrix();
  };
  for (double offset : {0.0, 0.7}) {
    for (double sign : {-1.0, 1.0}) {
      const Eigen::VectorXd dq = sign * vector({0.2, 0.1});
      const Eigen::VectorXd a = sign * vector({-0.4, 0.15});
      const Eigen::VectorXd qn = q + time * dq + 0.5 * time * time * a;
      const Eigen::Matrix3d reference = rotation(qn) *
          Eigen::AngleAxisd(offset, Eigen::Vector3d(1.0, 2.0, -1.0).normalized())
              .toRotationMatrix();
      const auto coordinate = [&](const Eigen::VectorXd &state) -> Eigen::Vector3d {
        const Eigen::AngleAxisd log(reference.transpose() * rotation(state));
        return log.angle() * log.axis();
      };
      // Differentiate the independent closed-form rotation chain in time;
      // this does not use the solver's differential or model Jacobian.
      const auto sample = [&](const Eigen::VectorXd &acceleration) {
        const Eigen::VectorXd state = q + time * dq + 0.5 * time * time * acceleration;
        const Eigen::VectorXd velocity = dq + time * acceleration;
        constexpr double h = 1e-4;
        Eigen::Matrix<double, 3, 2> values;
        values.col(0) = coordinate(state);
        values.col(1) = (-coordinate(state + 2.0 * h * velocity) +
                         8.0 * coordinate(state + h * velocity) -
                         8.0 * coordinate(state - h * velocity) +
                         coordinate(state - 2.0 * h * velocity)) / (12.0 * h);
        return values;
      };
      const auto center = sample(a);
      for (int axis = 0; axis < 3; ++axis) {
        const bool upper = center(axis, 1) > 0.0;
        const double direction = upper ? 1.0 : -1.0;
        const double bound = center(axis, 0) +
            direction * center(axis, 1) * center(axis, 1) / 20.0 - direction * 1e-6;
        detail::PreparedFixedFramePoseConstraint prepared;
        prepared.specification.frame_name = "tool";
        prepared.specification.reference_pose.topLeftCorner<3, 3>() = reference;
        prepared.specification.policy = policy6(1.0, 10.0, 10.0);
        prepared.specification.lower_bounds.setConstant(bound);
        prepared.specification.upper_bounds.setConstant(bound);
        detail::ScalarGeometricViolation violation{
            detail::ScalarGeometricViolationKind::kBrakingViability,
            axis + 3, upper, time};
        AffineAccelerationConstraint refinement;
        const auto result = detail::linearize_fixed_frame_pose_violation(
            prepared, robot, q, dq, a, violation, &refinement);
        ASSERT_TRUE(result.satisfied()) << result.message;
        const auto residual = [&](const Eigen::VectorXd &acceleration) {
          const auto values = sample(acceleration);
          return values(axis, 1) * values(axis, 1) +
                 20.0 * direction * (values(axis, 0) - bound);
        };
        EXPECT_NEAR(refinement.coefficient_matrix.row(0).dot(a) +
                        refinement.affine_bias(0), residual(a), 1e-10);
        for (int joint = 0; joint < 2; ++joint) {
          Eigen::VectorXd plus = a, minus = a;
          plus(joint) += 1e-3;
          minus(joint) -= 1e-3;
          EXPECT_NEAR(refinement.coefficient_matrix(0, joint),
                      (residual(plus) - residual(minus)) / 2e-3, 1e-8);
        }
        violation.kind = detail::ScalarGeometricViolationKind::kStateBound;
        if (upper) prepared.specification.upper_bounds(axis + 3) = center(axis, 0) - 1e-6;
        else prepared.specification.lower_bounds(axis + 3) = center(axis, 0) + 1e-6;
        ASSERT_TRUE(detail::linearize_fixed_frame_pose_violation(
            prepared, robot, q, dq, a, violation, &refinement).satisfied());
        for (int joint = 0; joint < 2; ++joint) {
          Eigen::VectorXd plus = a, minus = a;
          plus(joint) += 1e-3;
          minus(joint) -= 1e-3;
          EXPECT_NEAR(refinement.coefficient_matrix(0, joint),
                      direction * (sample(plus)(axis, 0) - sample(minus)(axis, 0)) /
                          2e-3, 1e-9);
        }
      }
    }
  }
}

TEST_F(AccelerationSolverTightFramePoseTest,
       BatchCollectsActiveEndpointFacesWithOneTrustBox) {
  auto urdf = two_link_urdf();
  const std::string old_axis = "<axis xyz=\"0 0 1\"/>";
  urdf.replace(urdf.find(old_axis), old_axis.size(), "<axis xyz=\"0 1 0\"/>");
  ScopedUrdf file("embodik_spatial_rotation_batch.urdf", urdf);
  RobotModel robot(file.path(), false);
  const Eigen::VectorXd q = vector({0.4, -0.3});
  const Eigen::VectorXd dq = vector({0.2, 0.1});
  const Eigen::VectorXd a = vector({-0.4, 0.15});
  constexpr double dt = 0.01;
  robot.update_kinematics(q, dq);
  const auto reference = robot.get_frame_pose("tool");
  const auto sample = detail::evaluate_fixed_frame_pose_differential_at_state(
      robot, "tool", reference, q + dt * dq + 0.5 * dt * dt * a, dq + dt * a);
  detail::PreparedFixedFramePoseConstraint prepared;
  auto &record = prepared.specification;
  record.family_label = "test";
  record.source_id = "batch";
  record.frame_name = "tool";
  record.reference_pose = matrix_from_pose(reference);
  record.policy = policy6(1.0, 20.0, 4.0);
  record.lower_bounds.setConstant(-100.0);
  record.upper_bounds.setConstant(100.0);
  prepared.rotation_active = true;
  prepared.active_axes = {3, 4};
  for (int axis : prepared.active_axes) {
    const double reserve = sample.rate(axis) * sample.rate(axis) / 8.0 - 1e-6;
    if (sample.rate(axis) > 0.0) record.upper_bounds(axis) = sample.value(axis) + reserve;
    else record.lower_bounds(axis) = sample.value(axis) - reserve;
  }
  // A deliberately violated inactive coordinate must not create an extra cut.
  record.lower_bounds(5) = record.upper_bounds(5) = -1.0;
  auto &specification = prepared.scalar.specification;
  specification.family_label = record.family_label;
  specification.source_id = record.source_id;
  specification.dimension = 6;
  specification.active_axes = prepared.active_axes;
  specification.state_lower_bounds = record.lower_bounds;
  specification.state_upper_bounds = record.upper_bounds;
  specification.policy = record.policy;
  for (double trigger_time : {dt, dt * 57.0 / 57.0}) {
    const detail::ScalarGeometricViolation trigger{
        detail::ScalarGeometricViolationKind::kBrakingViability,
        3, sample.rate(3) > 0.0, trigger_time};
    AffineAccelerationConstraint single, batch;
    ASSERT_TRUE(detail::linearize_fixed_frame_pose_violation(
        prepared, robot, q, dq, a, trigger, &single).satisfied());
    const auto result = detail::linearize_fixed_frame_pose_violations(
        prepared, robot, q, dq, a, dt, trigger, &batch);
    ASSERT_TRUE(result.satisfied()) << result.message;
    ASSERT_EQ(batch.coefficient_matrix.rows(), 2 + robot.nv());
    EXPECT_TRUE(batch.coefficient_matrix.row(0).isApprox(single.coefficient_matrix.row(0), 1e-14));
    EXPECT_DOUBLE_EQ(batch.affine_bias(0), single.affine_bias(0));
    const detail::ScalarGeometricViolation other{
        detail::ScalarGeometricViolationKind::kBrakingViability, 4,
        sample.rate(4) > 0.0, dt};
    AffineAccelerationConstraint other_row;
    ASSERT_TRUE(detail::linearize_fixed_frame_pose_violation(
        prepared, robot, q, dq, a, other, &other_row).satisfied());
    EXPECT_TRUE(batch.coefficient_matrix.row(1).isApprox(other_row.coefficient_matrix.row(0), 1e-14));
    EXPECT_DOUBLE_EQ(batch.affine_bias(1), other_row.affine_bias(0));
    EXPECT_TRUE(batch.coefficient_matrix.bottomRows(robot.nv()).isIdentity());
    EXPECT_TRUE(batch.lower_bounds.tail(robot.nv()).isApprox(single.lower_bounds.tail(robot.nv()), 0.0));
    EXPECT_TRUE(batch.upper_bounds.tail(robot.nv()).isApprox(single.upper_bounds.tail(robot.nv()), 0.0));
    EXPECT_FALSE(batch.lower_bound_active[0]);
    EXPECT_FALSE(batch.lower_bound_active[1]);
    EXPECT_TRUE(batch.upper_bound_active[0]);
    EXPECT_TRUE(batch.upper_bound_active[1]);
  }

  // Interior sampling checks state/rate/acceleration, with braking reserved
  // for the endpoint. A violated interior braking reserve must not add a cut.
  const double interior_time = 0.5 * dt;
  const auto interior = detail::evaluate_fixed_frame_pose_differential_at_state(
      robot, "tool", reference,
      q + interior_time * dq + 0.5 * interior_time * interior_time * a,
      dq + interior_time * a);
  record.lower_bounds(3) = interior.value(3) + 1e-6;
  record.upper_bounds(4) = interior.value(4) +
      interior.rate(4) * interior.rate(4) / 8.0 - 1e-6;
  specification.state_lower_bounds = record.lower_bounds;
  specification.state_upper_bounds = record.upper_bounds;
  const detail::ScalarGeometricViolation trigger{
      detail::ScalarGeometricViolationKind::kStateBound, 3, false, interior_time};
  AffineAccelerationConstraint batch;
  const auto result = detail::linearize_fixed_frame_pose_violations(
      prepared, robot, q, dq, a, dt, trigger, &batch);
  ASSERT_TRUE(result.satisfied()) << result.message;
  // Trigger state, endpoint state on the same axis, endpoint braking on the
  // other axis, followed by one trust box. No interior braking row is added.
  EXPECT_EQ(batch.coefficient_matrix.rows(), 3 + robot.nv());
}

TEST_F(AccelerationSolverTightFramePoseTest,
       NonlinearRotationRefinementKeepsOriginalHardRows) {
  auto urdf = two_link_urdf();
  const std::string old_axis = "<axis xyz=\"0 0 1\"/>";
  urdf.replace(urdf.find(old_axis), old_axis.size(), "<axis xyz=\"0 1 0\"/>");
  ScopedUrdf file("embodik_spatial_rotation_recovery.urdf", urdf);
  auto robot = std::make_shared<RobotModel>(file.path(), false);
  AccelerationSolver solver(robot);
  const Eigen::VectorXd q = vector({0.4, -0.3});
  const Eigen::VectorXd dq = vector({0.2, 0.1});
  auto task = solver.add_posture_task("braking_target");
  task->setSolveMode(TaskSolveMode::kMinError);
  AccelerationTaskReference reference;
  reference.proportional_gain = reference.derivative_gain = 0.0;
  reference.desired_acceleration = vector({-0.4, 0.15});
  solver.set_task_reference("braking_target", reference);
  robot->update_kinematics(q, dq);
  Eigen::Matrix<double, 6, 1> mask = Eigen::Matrix<double, 6, 1>::Zero();
  mask(3) = 1.0;
  auto options = options_with_limits(15.0);
  options.tight_frame_pose_constraints.push_back(tight_frame(
      "rotation_braking", "tool", matrix_from_pose(robot->get_frame_pose("tool")),
      mask, 0.1, 0.0010007881529456087, policy6(1.0, 20.0, 4.0)));
  AffineAccelerationConstraint protected_second;
  protected_second.source_id = "protected_second_joint";
  protected_second.coefficient_matrix = Eigen::Matrix<double, 1, 2>(0.0, 1.0);
  protected_second.affine_bias = Eigen::VectorXd::Zero(1);
  protected_second.lower_bounds = protected_second.upper_bounds = vector({0.15});
  options.affine_constraints.push_back(protected_second);
  const auto result = solver.solve(q, dq, 0.01, options);
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_GT(result.nonlinear_refinement_iterations, 0);
  EXPECT_NEAR(result.joint_accelerations(1), 0.15, 1e-10);
  EXPECT_LE(result.joint_accelerations.cwiseAbs().maxCoeff(), 15.0);
  EXPECT_TRUE(result.q_solution.isApprox(
      q + 0.01 * dq + 0.00005 * result.joint_accelerations, 1e-12));
  const auto record = detail::make_tight_frame_pose_record(
      options.tight_frame_pose_constraints.front());
  robot->update_kinematics(q, dq);
  const auto prepared = detail::prepare_fixed_frame_pose_constraint(
      record, *robot, 0.01, Eigen::Vector2d::Constant(-15.0),
      Eigen::Vector2d::Constant(15.0));
  ASSERT_TRUE(prepared.satisfied()) << prepared.message;
  const auto acceptance = detail::validate_fixed_frame_pose_constraint_acceptance(
      *prepared.prepared, *robot, q, dq, result.joint_accelerations, 0.01,
      Eigen::Vector2d::Constant(-15.0), Eigen::Vector2d::Constant(15.0));
  EXPECT_TRUE(acceptance.satisfied()) << acceptance.message;
}

TEST_F(AccelerationSolverTightFramePoseTest,
       NonlinearTranslationRefinementKeepsOriginalHardRows) {
  AccelerationSolver solver(robot_);
  const Eigen::VectorXd q = vector({-1.2, 0.0});
  const Eigen::VectorXd dq = vector({0.2, 0.0});
  auto task = solver.add_posture_task("braking_target");
  task->setSolveMode(TaskSolveMode::kMinError);
  AccelerationTaskReference reference;
  reference.proportional_gain = reference.derivative_gain = 0.0;
  reference.desired_acceleration = vector({-0.5, 0.0});
  solver.set_task_reference("braking_target", reference);
  robot_->update_kinematics(q, dq);
  Eigen::Matrix<double, 6, 1> mask = Eigen::Matrix<double, 6, 1>::Zero();
  mask(0) = 1.0;
  auto options = options_with_limits(15.0);
  options.tight_frame_pose_constraints.push_back(tight_frame(
      "translation_braking", "tool", matrix_from_pose(robot_->get_frame_pose("tool")),
      mask, 0.010276011537982174, 0.1, policy6(1.0, 10.0, 10.0)));
  AffineAccelerationConstraint protected_second;
  protected_second.source_id = "protected_second_joint";
  protected_second.coefficient_matrix = Eigen::Matrix<double, 1, 2>(0.0, 1.0);
  protected_second.affine_bias = Eigen::VectorXd::Zero(1);
  protected_second.lower_bounds = protected_second.upper_bounds = Eigen::VectorXd::Zero(1);
  options.affine_constraints.push_back(protected_second);
  const auto result = solver.solve(q, dq, 0.01, options);
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_GT(result.nonlinear_refinement_iterations, 0);
  EXPECT_LT(result.joint_accelerations(0), -0.5);
  EXPECT_NEAR(result.joint_accelerations(1), 0.0, 1e-10);
  EXPECT_LE(result.joint_accelerations.cwiseAbs().maxCoeff(), 15.0);
  const auto record = detail::make_tight_frame_pose_record(options.tight_frame_pose_constraints[0]);
  robot_->update_kinematics(q, dq);
  const auto prepared = detail::prepare_fixed_frame_pose_constraint(
      record, *robot_, 0.01, Eigen::Vector2d::Constant(-15.0), Eigen::Vector2d::Constant(15.0));
  ASSERT_TRUE(prepared.satisfied()) << prepared.message;
  const auto accepted = detail::validate_fixed_frame_pose_constraint_acceptance(
      *prepared.prepared, *robot_, q, dq, result.joint_accelerations, 0.01,
      Eigen::Vector2d::Constant(-15.0), Eigen::Vector2d::Constant(15.0));
  EXPECT_TRUE(accepted.satisfied()) << accepted.message;
}

class FrameTrustRecoveryTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_frame_trust_recovery.urdf", R"(<robot name="frame_trust">
      <link name="base"/><link name="carriage"/>
      <joint name="slide" type="prismatic"><parent link="base"/>
      <child link="carriage"/><axis xyz="1 0 0"/>
      <limit lower="-1" upper="1" velocity="10" effort="100"/></joint>
      <link name="rotor"/><joint name="angle" type="revolute">
      <parent link="carriage"/><child link="rotor"/><axis xyz="0 0 1"/>
      <limit lower="-3.2" upper="3.2" velocity="10" effort="100"/></joint>
      <link name="tool"/><joint name="tool_fixed" type="fixed">
      <parent link="rotor"/><child link="tool"/><origin xyz="1 0 0"/>
      </joint></robot>)");
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
    robot_->update_kinematics(q_, dq_);
    solver_ = std::make_unique<AccelerationSolver>(robot_);
    auto task = solver_->add_posture_task("request");
    task->setSolveMode(TaskSolveMode::kMinError);
    set_request(4.0);
    options_ = options_with_limits(15.0);
    options_.prefer_direct_min_error_qp = true;
    options_.record_linear_problems = true;
    Eigen::Matrix<double, 6, 1> mask =
        Eigen::Matrix<double, 6, 1>::Zero();
    mask(0) = 1.0;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose(0, 3) = 1.0019; // x >= .9999; 0.1 mm initial clearance.
    options_.tight_frame_pose_constraints.push_back(tight_frame(
        "held_x", "tool", pose, mask, .002, .1, policy6(1., 20., 2.)));
    // This caller row deliberately collides with the internal source-ID stem.
    AffineAccelerationConstraint locked_slide;
    locked_slide.source_id = "__frame_pose_refinement";
    locked_slide.coefficient_matrix = Eigen::Matrix<double, 1, 2>(1., 0.);
    locked_slide.affine_bias = Eigen::VectorXd::Zero(1);
    locked_slide.lower_bounds = locked_slide.upper_bounds = vector({0.});
    options_.affine_constraints.push_back(locked_slide);
    AffineAccelerationConstraint collision_cut = locked_slide;
    collision_cut.source_id = "__collision_sample_refinement";
    collision_cut.coefficient_matrix = Eigen::Matrix<double, 1, 2>(0., 1.);
    collision_cut.upper_bounds = vector({4.});
    options_.affine_constraints.push_back(collision_cut);
  }

  void set_request(double value) {
    AccelerationTaskReference reference;
    reference.proportional_gain = reference.derivative_gain = 0.;
    reference.desired_acceleration = vector({0., value});
    solver_->set_task_reference("request", reference);
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
  std::unique_ptr<AccelerationSolver> solver_;
  AccelerationSolveOptions options_;
  Eigen::Vector2d q_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d dq_ = Eigen::Vector2d::Zero();
};

TEST_F(FrameTrustRecoveryTest, ReleasesOnlyOwnedBoxAndRetainsCallerRowsAndCuts) {
  // The exact nonlinear feasible set contains a useful nonzero command.
  ASSERT_EQ(solver_->validate_candidate(q_, dq_, .1, vector({0., 2.}), options_).status,
            SolverStatus::kSuccess);
  const auto result = solver_->solve(q_, dq_, .1, options_);
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_GT(result.nonlinear_refinement_iterations, 1);
  EXPECT_LE(result.nonlinear_refinement_iterations, 9);
  EXPECT_LE(result.linear_problems.size(), 10u);
  EXPECT_NEAR(result.joint_accelerations(0), 0., 1e-12);
  EXPECT_GT(result.joint_accelerations(1), 0.);
  EXPECT_LT(result.joint_accelerations(1), 4.);
  ASSERT_EQ(options_.affine_constraints.size(), 2u);
  for (const auto &problem : result.linear_problems) {
    for (const auto &caller : options_.affine_constraints) {
      int found = 0;
      for (Eigen::Index row = 0; row < problem.constraint_matrix.rows(); ++row) {
        if (problem.constraint_row_sources[row] != caller.source_id) continue;
        ++found;
        EXPECT_TRUE(problem.constraint_matrix.row(row).isApprox(
            caller.coefficient_matrix.row(0), 0.));
        EXPECT_DOUBLE_EQ(problem.lower_bounds(row), caller.lower_bounds(0));
        EXPECT_DOUBLE_EQ(problem.upper_bounds(row), caller.upper_bounds(0));
      }
      EXPECT_EQ(found, 1);
    }
  }
  const auto checked = solver_->validate_candidate(
      q_, dq_, .1, result.joint_accelerations, options_);
  EXPECT_EQ(checked.status, SolverStatus::kSuccess) << checked.status_message;
}

TEST_F(FrameTrustRecoveryTest, IncompatiblePhysicalCommandStillFailsClosed) {
  AffineAccelerationConstraint forced = options_.affine_constraints.back();
  forced.source_id = "forced_unsafe_rotation";
  forced.lower_bounds = forced.upper_bounds = vector({4.});
  options_.affine_constraints.push_back(forced);
  const auto result = solver_->solve(q_, dq_, .1, options_);
  EXPECT_NE(result.status, SolverStatus::kSuccess);
  expect_no_motion_outputs(result);
  EXPECT_LE(result.nonlinear_refinement_iterations, 9);
  EXPECT_LE(result.linear_problems.size(), 10u);
  const auto prescribed = solver_->validate_candidate(
      q_, dq_, .1, vector({0., 4.}), options_);
  EXPECT_NE(prescribed.status, SolverStatus::kSuccess);
  expect_no_motion_outputs(prescribed);
}

TEST_F(FrameTrustRecoveryTest, ExhaustionNeverRestartsBudgetOrDiscardsDescendantCuts) {
  auto constraint = options_.tight_frame_pose_constraints.front();
  auto definition = constraint.definition;
  definition.target_pose(0, 3) = 1.002; // Tangent boundary: only rest is exact.
  constraint.definition = definition;
  options_.tight_frame_pose_constraints = {constraint};
  options_.affine_constraints.back().upper_bounds = vector({15.});
  set_request(15.);
  ASSERT_EQ(solver_->validate_candidate(q_, dq_, .1, vector({0., 0.}), options_).status,
            SolverStatus::kSuccess);
  const auto result = solver_->solve(q_, dq_, .1, options_);
  EXPECT_NE(result.status, SolverStatus::kSuccess);
  expect_no_motion_outputs(result);
  EXPECT_EQ(result.nonlinear_refinement_iterations, 9);
  EXPECT_EQ(result.linear_problems.size(), 10u);
  std::set<std::string> prior_sources;
  for (const auto &problem : result.linear_problems) {
    const std::set<std::string> sources(problem.constraint_row_sources.begin(),
                                         problem.constraint_row_sources.end());
    for (const auto &source : prior_sources)
      EXPECT_EQ(sources.count(source), 1u) << source;
    prior_sources = sources;
  }
}

TEST(FrameTrustRowOwnershipTest, ExplicitIndicesPreserveOtherRowsAndErrorsAreAtomic) {
  AffineAccelerationConstraint model;
  model.source_id = "arbitrary_caller_visible_name";
  model.coefficient_matrix.resize(4, 2);
  model.coefficient_matrix << 1., 0., 1., 0., 0., 1., 0., 1.;
  model.affine_bias = vector({1., 2., 3., 4.});
  model.lower_bounds = vector({-1., -2., -3., -4.});
  model.upper_bounds = vector({1., 2., 3., 4.});
  model.lower_bound_active = {false, true, true, false};
  model.upper_bound_active = {true, true, true, true};
  auto invalid = model;
  invalid.affine_bias.conservativeResize(3);
  EXPECT_THROW(detail::remove_fixed_frame_pose_trust_rows(&invalid, {1, 2}),
               std::logic_error);
  EXPECT_TRUE(invalid.coefficient_matrix.isApprox(model.coefficient_matrix, 0.));
  detail::remove_fixed_frame_pose_trust_rows(&model, {1, 2});
  ASSERT_EQ(model.coefficient_matrix.rows(), 2);
  EXPECT_EQ(model.source_id, "arbitrary_caller_visible_name");
  EXPECT_TRUE(model.coefficient_matrix.isIdentity());
  EXPECT_TRUE(model.affine_bias.isApprox(vector({1., 4.}), 0.));
  EXPECT_TRUE(model.lower_bounds.isApprox(vector({-1., -4.}), 0.));
  EXPECT_TRUE(model.upper_bounds.isApprox(vector({1., 4.}), 0.));
  EXPECT_EQ(model.lower_bound_active, (std::vector<bool>{false, false}));
  EXPECT_EQ(model.upper_bound_active, (std::vector<bool>{true, true}));
}

TEST_F(AccelerationSolverTightFramePoseTest,
       CapabilitiesExposeFixedFramePoseFamilies) {
  const auto capabilities = AccelerationSolver::capabilities();
  EXPECT_TRUE(capabilities.supports_tight_frame_pose_constraints);
  EXPECT_TRUE(capabilities.supports_torso_pose_bound_constraints);
}

TEST_F(AccelerationSolverTightFramePoseTest,
       LoweredRowsUseExactWorldTranslationAndSo3LogDifferential) {
  const Eigen::VectorXd q = vector({0.45, -0.2});
  const Eigen::VectorXd dq = vector({0.7, -0.15});
  robot_->update_kinematics(q, dq);
  auto reference_pose = robot_->get_frame_pose("tool");
  reference_pose.translation().x() -= 0.05;
  reference_pose.rotation() = reference_pose.rotation() * z_rotation(-0.1);
  const auto exact = detail::evaluate_fixed_frame_pose_differential(
      *robot_, "tool", reference_pose);

  Eigen::Matrix<double, 6, 1> mask =
      Eigen::Matrix<double, 6, 1>::Zero();
  mask(0) = 1.0;
  mask(5) = 1.0;
  const auto record = detail::make_tight_frame_pose_record(tight_frame(
      "tool_x_yaw", "tool", matrix_from_pose(reference_pose), mask, 0.5, 0.5,
      policy6(100.0, 100.0, 100.0)));

  const auto prepared = detail::prepare_fixed_frame_pose_constraint(
      record, *robot_, 0.01, Eigen::Vector2d::Constant(-1000.0),
      Eigen::Vector2d::Constant(1000.0));

  ASSERT_TRUE(prepared.satisfied()) << prepared.message;
  ASSERT_GT(exact.value.head<3>().norm(), 1e-3);
  ASSERT_GT(exact.value.tail<3>().norm(), 1e-3);
  const auto &physical =
      prepared.prepared->scalar.state_box.physical_constraint;
  ASSERT_EQ(physical.coefficient_matrix.rows(), 2);
  EXPECT_TRUE(physical.coefficient_matrix.row(0).isApprox(
      exact.jacobian.row(0), kTolerance));
  EXPECT_TRUE(physical.coefficient_matrix.row(1).isApprox(
      exact.jacobian.row(5), kTolerance));
  EXPECT_NEAR(physical.affine_bias(0), exact.affine_bias(0), kTolerance);
  EXPECT_NEAR(physical.affine_bias(1), exact.affine_bias(5), kTolerance);
}

TEST_F(AccelerationSolverTightFramePoseTest,
       TranslationOnlyBypassesCurrentNearPiOrientation) {
  AccelerationSolver solver(robot_);
  const double near_pi = detail::kSo3Pi - 5e-4;
  const Eigen::VectorXd q = vector({near_pi, 0.0});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  robot_->update_kinematics(q, dq);

  Eigen::Matrix<double, 6, 1> translation_mask =
      Eigen::Matrix<double, 6, 1>::Zero();
  translation_mask(1) = 1.0;
  auto options = options_with_limits(100.0);
  options.tight_frame_pose_constraints.push_back(tight_frame(
      "translation_only", "tool", Eigen::Matrix4d::Identity(),
      translation_mask, 0.1, 0.1, policy6(100.0, 100.0, 100.0)));

  const auto result = solver.solve(q, dq, 0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
}

TEST_F(AccelerationSolverTightFramePoseTest,
       RotationActiveCurrentNearPiFailsAsInvalidInputAndClearsOutputs) {
  AccelerationSolver solver(robot_);
  Eigen::Matrix<double, 6, 1> yaw_mask =
      Eigen::Matrix<double, 6, 1>::Zero();
  yaw_mask(5) = 1.0;
  auto options = options_with_limits(100.0);
  options.tight_frame_pose_constraints.push_back(tight_frame(
      "yaw_near_pi", "tool", Eigen::Matrix4d::Identity(), yaw_mask, 4.0, 4.0,
      policy6(100.0, 100.0, 100.0)));

  const auto result =
      solver.solve(vector({detail::kSo3Pi - 5e-4, 0.0}),
                   Eigen::Vector2d::Zero(), 0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  EXPECT_NE(result.status_message.find("near-pi"), std::string::npos);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTightFramePoseTest,
       RotationActiveBetweenSampleNearPiFailsAsNumericalAndClearsOutputs) {
  AccelerationSolver solver(robot_);
  Eigen::Matrix<double, 6, 1> yaw_mask =
      Eigen::Matrix<double, 6, 1>::Zero();
  yaw_mask(5) = 1.0;
  auto options = options_with_limits(100.0);
  options.tight_frame_pose_constraints.push_back(tight_frame(
      "yaw_segment_guard", "tool", Eigen::Matrix4d::Identity(), yaw_mask, 4.0,
      4.0, policy6(1000.0, 100.0, 100.0)));

  const auto result =
      solver.solve(vector({detail::kSo3Pi - 2e-3, 0.0}), vector({0.8, 0.0}),
                   0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kNumericalError)
      << result.status_message;
  EXPECT_NE(result.status_message.find("SO(3)"), std::string::npos);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTightFramePoseTest,
       PredictedStateSupportRejectionFailsClosed) {
  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");

  const Eigen::VectorXd q = vector({0.2, -0.2});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  constexpr double kDt = 0.1;
  AccelerationTaskReference reference;
  reference.desired_acceleration = -2.0 * q / (kDt * kDt);
  solver.set_task_reference("posture", reference);

  robot_->update_kinematics(q, dq);
  Eigen::Matrix<double, 6, 1> x_mask = Eigen::Matrix<double, 6, 1>::Zero();
  x_mask(0) = 1.0;
  auto options = options_with_limits(1e6);
  options.tight_frame_pose_constraints.push_back(tight_frame(
      "predicted_singular_x", "tool",
      matrix_from_pose(robot_->get_frame_pose("tool")), x_mask, 0.1, 0.1,
      policy6(100.0, 1e5, 1000.0)));

  const auto result = solver.solve(q, dq, kDt, options);

  EXPECT_EQ(result.status, SolverStatus::kNumericalError)
      << result.status_message;
  EXPECT_NE(result.status_message.find("predicted next-state"),
            std::string::npos);
  EXPECT_NE(result.status_message.find("braking support"), std::string::npos);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverTightFramePoseTest,
       RejectsMalformedTightFrameInputAndDuplicateSourceIds) {
  AccelerationSolver solver(robot_);
  Eigen::Matrix<double, 6, 1> x_mask = Eigen::Matrix<double, 6, 1>::Zero();
  x_mask(0) = 1.0;
  auto options = options_with_limits(100.0);
  options.tight_frame_pose_constraints.push_back(tight_frame(
      "bad_eps", "tool", Eigen::Matrix4d::Identity(), x_mask, 0.0, 0.1,
      policy6(10.0, 10.0, 10.0)));
  auto result = solver.solve(vector({0.0, 0.0}), Eigen::Vector2d::Zero(), 0.01,
                             options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  auto duplicate = options_with_limits(100.0);
  AffineAccelerationConstraint affine;
  affine.source_id = "dup";
  affine.coefficient_matrix = Eigen::MatrixXd::Identity(1, 2);
  affine.affine_bias = Eigen::VectorXd::Zero(1);
  affine.lower_bounds = vector({-1.0});
  affine.upper_bounds = vector({1.0});
  duplicate.affine_constraints.push_back(affine);
  duplicate.tight_frame_pose_constraints.push_back(tight_frame(
      "dup", "tool", Eigen::Matrix4d::Identity(), x_mask, 0.1, 0.1,
      policy6(10.0, 10.0, 10.0)));
  result = solver.solve(vector({0.0, 0.0}), Eigen::Vector2d::Zero(), 0.01,
                        duplicate);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);
}

TEST(AccelerationSo3SegmentGuardTest, RejectsMalformedSegmentInputs) {
  auto segment = so3_segment();
  segment.previous.coordinates.value = Eigen::VectorXd::Zero(2);
  auto result = detail::validate_so3_log_rotation_segment(segment);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError);

  segment = so3_segment();
  segment.current.tangent = Eigen::VectorXd::Zero(3);
  result = detail::validate_so3_log_rotation_segment(segment);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError);

  segment = so3_segment();
  segment.previous.tangent(0) = std::numeric_limits<double>::quiet_NaN();
  result = detail::validate_so3_log_rotation_segment(segment);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError);

  segment = so3_segment();
  segment.current.coordinates.so3_rotation.reset();
  result = detail::validate_so3_log_rotation_segment(segment);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError);

  segment = so3_segment();
  segment.current.coordinates.so3_rotation = 2.0 * Eigen::Matrix3d::Identity();
  result = detail::validate_so3_log_rotation_segment(segment);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError);
}

TEST(ScalarGeometricEndpointContract, BrakingCertificateMatchesNextStateEntry) {
  detail::ScalarGeometricConstraintSpecification specification;
  specification.family_label = "test coordinate";
  specification.source_id = "endpoint-entry";
  specification.dimension = 1;
  specification.active_axes = {0};
  specification.state_lower_bounds = vector({-1.0});
  specification.state_upper_bounds = vector({1.0});
  specification.policy.rate_limits = vector({10.0});
  specification.policy.acceleration_limits = vector({10.0});
  specification.policy.lower_braking_accelerations = vector({2.0});
  specification.policy.upper_braking_accelerations = vector({2.0});
  for (const double direction : {-1.0, 1.0}) {
    for (const double residual : {-1e-9, 0.0, 5e-10}) {
      detail::ScalarGeometricCoordinateSample sample;
      sample.value = vector({direction * .75});
      sample.rate = vector({direction * std::sqrt(1.0 + residual)});
      sample.acceleration = vector({0.0});
      detail::StateBoxRowInput input;
      input.value = sample.value(0);
      input.rate = sample.rate(0);
      input.state_bounds = {-1.0, 1.0, true, true};
      input.rate_bounds = {-10.0, 10.0, true, true};
      input.acceleration_bounds = {-10.0, 10.0, true, true};
      input.lower_braking_acceleration = input.upper_braking_acceleration = 2.0;
      const auto next = detail::shape_state_box_row(input, .01);
      const auto endpoint = detail::validate_scalar_geometric_sample_acceptance(
          specification, sample, true);
      EXPECT_EQ(endpoint.satisfied(), next.status == SolverStatus::kSuccess);
      if (residual > 0.0) {
        ASSERT_TRUE(endpoint.violation.has_value());
        EXPECT_EQ(endpoint.violation->kind, detail::ScalarGeometricViolationKind::kBrakingViability);
        EXPECT_EQ(endpoint.violation->upper, direction > 0.0);
      }
    }
    detail::ScalarGeometricCoordinateSample sample;
    sample.value = vector({direction * (1.0 + 5e-10)});
    sample.rate = vector({0.0});
    sample.acceleration = vector({0.0});
    EXPECT_FALSE(detail::validate_scalar_geometric_sample_acceptance(specification, sample, true).satisfied());
    // Interior sampling keeps its existing geometric comparison tolerance.
    EXPECT_TRUE(detail::validate_scalar_geometric_sample_acceptance(specification, sample, false).satisfied());
    sample.value(0) = direction;
    sample.rate(0) = direction * 1e-8;
    EXPECT_FALSE(detail::validate_scalar_geometric_sample_acceptance(specification, sample, true).satisfied());
    sample.rate(0) = -direction * 1e-8;
    EXPECT_TRUE(detail::validate_scalar_geometric_sample_acceptance(specification, sample, true).satisfied());
  }
}

} // namespace
} // namespace embodik::test
