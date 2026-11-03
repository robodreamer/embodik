#include "acceleration_affine_state_constraint.hpp"

#include <gtest/gtest.h>
#include <cmath>

namespace embodik::detail {
namespace {

PreparedAffineStateConstraint prepared(const Eigen::MatrixXd &matrix,
                                      double value, double rate,
                                      double lower, double upper) {
  AffineStateConstraint definition;
  definition.source_id = "test-affine";
  definition.coefficient_matrix = matrix;
  definition.offset = Eigen::VectorXd::Constant(1, value);
  definition.lower_bounds = Eigen::VectorXd::Constant(1, lower);
  definition.upper_bounds = Eigen::VectorXd::Constant(1, upper);
  definition.policy.rate_limits = Eigen::VectorXd::Constant(1, 2.0);
  definition.policy.acceleration_limits = Eigen::VectorXd::Constant(1, 10.0);
  definition.policy.lower_braking_accelerations = Eigen::VectorXd::Constant(1, 4.0);
  definition.policy.upper_braking_accelerations = Eigen::VectorXd::Constant(1, 4.0);
  ScalarGeometricConstraintSpecification specification;
  specification.family_label = "affine state constraint";
  specification.source_id = definition.source_id;
  specification.dimension = 1;
  specification.active_axes = {0};
  specification.state_lower_bounds = definition.lower_bounds;
  specification.state_upper_bounds = definition.upper_bounds;
  specification.policy = definition.policy;
  ScalarGeometricDifferential differential;
  differential.coefficient_matrix = matrix;
  differential.affine_bias = Eigen::VectorXd::Zero(1);
  differential.coordinates.value = Eigen::VectorXd::Constant(1, value);
  differential.coordinates.rate = Eigen::VectorXd::Constant(1, rate);
  differential.coordinates.acceleration = Eigen::VectorXd::Zero(1);
  const auto shaped = prepare_scalar_geometric_constraint(
      specification, differential, 1.0, matrix.cols(),
      Eigen::VectorXd::Constant(matrix.cols(), -10.0),
      Eigen::VectorXd::Constant(matrix.cols(), 10.0));
  EXPECT_TRUE(shaped.satisfied()) << shaped.message;
  return {definition, differential, shaped.prepared.value()};
}

TEST(AffineStateAcceptance, RejectsActualEndpointLostByDistributingMatrix) {
  Eigen::MatrixXd matrix(1, 2);
  matrix << std::ldexp(1.0, 54), -std::ldexp(1.0, 54);
  const auto record = prepared(matrix, 0.25, 0.0, 0.0, 0.5);
  const double base = std::ldexp(1.0, -52);
  Eigen::Vector2d acceleration(std::nextafter(base, INFINITY),
                               std::nextafter(base, 0.0));
  const Eigen::Vector2d next_q = Eigen::Vector2d::Ones() + 0.5 * acceleration;
  EXPECT_GT((matrix * next_q)(0) + 0.25, 0.5);
  EXPECT_TRUE(validate_scalar_geometric_linearized_acceptance(
      record.scalar, acceleration, 1.0).satisfied());
  const auto checked = validate_affine_state_constraint_acceptance(
      record, acceleration, 1.0, next_q, acceleration,
      Eigen::Vector2d::Constant(-10.0), Eigen::Vector2d::Constant(10.0));
  EXPECT_FALSE(checked.satisfied());
}

TEST(AffineStateAcceptance, RejectsInteriorExcursionWithValidEndpoints) {
  const Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(1, 1);
  auto record = prepared(matrix, 0.0, 1.0, -1.0, 1.0);
  // Exercise the acceptance guard independently of conservative preparation.
  record.scalar.specification.state_upper_bounds(0) = 0.2;
  record.scalar.state_box.state_rows[0].state_bounds.upper = 0.2;
  const Eigen::VectorXd acceleration = Eigen::VectorXd::Constant(1, -2.0);
  EXPECT_TRUE(validate_scalar_geometric_linearized_acceptance(
      record.scalar, acceleration, 1.0).satisfied());
  const auto checked = validate_affine_state_constraint_acceptance(
      record, acceleration, 1.0, Eigen::VectorXd::Zero(1),
      Eigen::VectorXd::Constant(1, -1.0),
      Eigen::VectorXd::Constant(1, -10.0), Eigen::VectorXd::Constant(1, 10.0));
  EXPECT_FALSE(checked.satisfied());
}

TEST(AffineStateAcceptance, PredictedSupportFailureIsNotGlobalInfeasibility) {
  const Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(1, 1);
  const auto record = prepared(matrix, 0.0, 0.1, -1.0, 1.0);
  const auto zero = Eigen::VectorXd::Zero(1).eval();
  const auto rate = Eigen::VectorXd::Constant(1, 0.1).eval();
  const auto checked = validate_affine_state_constraint_acceptance(
      record, zero, 1.0, rate, rate, zero, zero);
  EXPECT_EQ(checked.status, SolverStatus::kNoProgress);
}

TEST(AffineStateAcceptance, OnlyDirectionalModeOmitsUnusedBrakingDirection) {
  const Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(1, 1);
  auto record = prepared(matrix, 0.0, 1.0, -1.0, 1.0);
  const auto lower = Eigen::VectorXd::Constant(1, -10.0).eval();
  const auto upper = Eigen::VectorXd::Zero(1).eval();
  const auto &specification = record.scalar.specification;
  EXPECT_FALSE(validate_scalar_geometric_joint_box_support(
      specification, record.current, lower, upper, "test").satisfied());
  EXPECT_TRUE(validate_scalar_geometric_joint_box_support(
      specification, record.current, lower, upper, "test",
      BrakingSupportRequirement::kReachableBoundary, 1.0).satisfied());
  record.current.coordinates.rate(0) = -1.0;
  EXPECT_FALSE(validate_scalar_geometric_joint_box_support(
      specification, record.current, lower, upper, "test",
      BrakingSupportRequirement::kReachableBoundary, 1.0).satisfied());
  record.current.coordinates.rate(0) = 0.0;
  EXPECT_TRUE(validate_scalar_geometric_joint_box_support(
      specification, record.current, upper, upper, "test",
      BrakingSupportRequirement::kReachableBoundary, 1.0).satisfied());
}

TEST(AffineStateAcceptance, ReachableBoundaryRequiresAuthorityBeforeDanger) {
  const Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(1, 1);
  auto record = prepared(matrix, 0.0, 1.0, -1.0, 1.0);
  const auto zero = Eigen::VectorXd::Zero(1).eval();
  EXPECT_TRUE(validate_scalar_geometric_joint_box_support(
      record.scalar.specification, record.current, zero, zero, "test",
      BrakingSupportRequirement::kReachableBoundary, 0.01).satisfied());
  record.current.coordinates.value(0) = 0.95;
  EXPECT_FALSE(validate_scalar_geometric_joint_box_support(
      record.scalar.specification, record.current, zero, zero, "test",
      BrakingSupportRequirement::kReachableBoundary, 0.01).satisfied());
}

TEST(AffineStateAcceptance, OverflowCannotSuppressBoundarySupport) {
  const Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(1, 1);
  auto record = prepared(matrix, 0.0, 1.0, -1.0, 1.0);
  record.current.coordinates.rate(0) = 1e300;
  const auto zero = Eigen::VectorXd::Zero(1).eval();
  const auto checked = validate_scalar_geometric_joint_box_support(
      record.scalar.specification, record.current, zero, zero, "test",
      BrakingSupportRequirement::kReachableBoundary, 0.01);
  EXPECT_EQ(checked.status, SolverStatus::kNumericalError);
}

TEST(AffineStateAcceptance, BrakingDenominatorOverflowCannotLookLikeZeroDistance) {
  const Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(1, 1);
  auto record = prepared(matrix, 0.0, 1.0, -1.0, 1.0);
  record.current.coordinates.rate(0) = 1e154;
  auto &specification = record.scalar.specification;
  specification.state_upper_bounds(0) = 0.55;
  specification.policy.rate_limits(0) = 1e155;
  specification.policy.acceleration_limits(0) = 1e308;
  specification.policy.lower_braking_accelerations(0) = 1e308;
  specification.policy.upper_braking_accelerations(0) = 1e308;
  const auto zero = Eigen::VectorXd::Zero(1).eval();
  const auto checked = validate_scalar_geometric_joint_box_support(
      specification, record.current, zero, zero, "test",
      BrakingSupportRequirement::kReachableBoundary, 1e-155);
  EXPECT_EQ(checked.status, SolverStatus::kNumericalError);
}

} // namespace
} // namespace embodik::detail
