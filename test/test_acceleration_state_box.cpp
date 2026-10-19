#include "acceleration_state_box.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace embodik::detail {
namespace {

constexpr double kTolerance = 1e-12;

ScalarBounds active(double lower, double upper) {
  return {lower, upper, true, true};
}

StateBoxRowInput fixture(double sign = 1.0) {
  StateBoxRowInput input;
  input.value = sign * 0.04;
  input.rate = sign * 0.12;
  input.state_bounds = active(-0.05, 0.05);
  input.acceleration_bounds = active(-1.0, 1.0);
  input.lower_braking_acceleration = 1.0;
  input.upper_braking_acceleration = 1.0;
  return input;
}

TEST(AccelerationStateBoxTest, CanonicalizesOnlyRoundoffOutsideEitherFace) {
  const double epsilon = std::numeric_limits<double>::epsilon();
  EXPECT_DOUBLE_EQ(
      canonicalize_state_box_acceleration_roundoff(-epsilon * epsilon,
                                                  0.0, 15.0),
      0.0);
  EXPECT_DOUBLE_EQ(
      canonicalize_state_box_acceleration_roundoff(epsilon * epsilon,
                                                  -15.0, 0.0),
      0.0);
  EXPECT_DOUBLE_EQ(canonicalize_state_box_acceleration_roundoff(
                       std::nextafter(2.0, 0.0), 2.0, 4.0),
                   2.0);
  EXPECT_DOUBLE_EQ(canonicalize_state_box_acceleration_roundoff(
                       std::nextafter(-2.0, 0.0), -4.0, -2.0),
                   -2.0);
  EXPECT_DOUBLE_EQ(
      canonicalize_state_box_acceleration_roundoff(1.0, 0.0, 2.0), 1.0);
}

TEST(AccelerationStateBoxTest, DoesNotRepairLargerStateBoxViolations) {
  EXPECT_DOUBLE_EQ(
      canonicalize_state_box_acceleration_roundoff(-1e-9, 0.0, 15.0),
      -1e-9);
  EXPECT_DOUBLE_EQ(
      canonicalize_state_box_acceleration_roundoff(1e-9, -15.0, 0.0),
      1e-9);
}

TEST(AccelerationStateBoxTest, OppositeFaceDoesNotInflateRoundoffAllowance) {
  const double infinity = std::numeric_limits<double>::infinity();
  for (double opposite_face : {1e300, infinity}) {
    EXPECT_DOUBLE_EQ(canonicalize_state_box_acceleration_roundoff(
                         -1e-9, 0.0, opposite_face),
                     -1e-9);
    EXPECT_DOUBLE_EQ(canonicalize_state_box_acceleration_roundoff(
                         1e-9, -opposite_face, 0.0),
                     1e-9);
    EXPECT_DOUBLE_EQ(canonicalize_state_box_acceleration_roundoff(
                         -std::numeric_limits<double>::epsilon(), 0.0,
                         opposite_face),
                     0.0);
  }
}

TEST(AccelerationStateBoxTest, InvalidCanonicalizationInputsRemainUnchanged) {
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_DOUBLE_EQ(
      canonicalize_state_box_acceleration_roundoff(-1.0, 2.0, 1.0), -1.0);
  EXPECT_DOUBLE_EQ(
      canonicalize_state_box_acceleration_roundoff(-1.0, nan, 1.0), -1.0);
  EXPECT_DOUBLE_EQ(
      canonicalize_state_box_acceleration_roundoff(-1.0, 0.0, nan), -1.0);
  EXPECT_EQ(canonicalize_state_box_acceleration_roundoff(infinity, 0.0, 1.0),
            infinity);
  EXPECT_TRUE(std::isnan(
      canonicalize_state_box_acceleration_roundoff(nan, 0.0, 1.0)));
}

TEST(AccelerationStateBoxTest,
     BoundaryRoundoffCanBrakeWithoutChangingTheSuppliedState) {
  constexpr double dt = 0.01;
  const double epsilon = std::numeric_limits<double>::epsilon();
  for (double sign : {-1.0, 1.0}) {
    StateBoxRowInput input;
    input.value = 0.0;
    input.rate = sign * epsilon * epsilon * dt;
    input.state_bounds = sign < 0.0 ? active(0.0, 1.0)
                                    : active(-1.0, 0.0);
    input.acceleration_bounds = active(-15.0, 15.0);
    input.lower_braking_acceleration = 15.0;
    input.upper_braking_acceleration = 15.0;
    const auto braking = shape_state_box_row(input, dt);
    ASSERT_EQ(braking.status, SolverStatus::kSuccess) << braking.message;
    const double expected_acceleration = -sign * std::min(
        15.0, std::abs(input.rate) / (dt * std::sqrt(64.0 * epsilon)));
    EXPECT_DOUBLE_EQ(braking.lower_acceleration, expected_acceleration);
    EXPECT_DOUBLE_EQ(braking.upper_acceleration, expected_acceleration);
    EXPECT_GE(braking.lower_acceleration, input.acceleration_bounds.lower);
    EXPECT_LE(braking.upper_acceleration, input.acceleration_bounds.upper);
    EXPECT_DOUBLE_EQ(input.value, 0.0);
    EXPECT_DOUBLE_EQ(input.rate, sign * epsilon * epsilon * dt);
    auto predicted = input;
    predicted.value = input.value +
                      (dt * input.rate +
                       0.5 * dt * dt * braking.lower_acceleration);
    predicted.rate = input.rate + dt * braking.lower_acceleration;
    EXPECT_GE(predicted.value, input.state_bounds.lower);
    EXPECT_LE(predicted.value, input.state_bounds.upper);
    EXPECT_LE(sign * predicted.rate, 0.0);
    const auto next_box = shape_state_box_row(predicted, dt);
    ASSERT_EQ(next_box.status, SolverStatus::kSuccess) << next_box.message;
    EXPECT_LE(std::max(next_box.lower_acceleration,
                       input.acceleration_bounds.lower),
              std::min(next_box.upper_acceleration,
                       input.acceleration_bounds.upper));
    const double lower = sign < 0.0 ? 0.0 : -15.0;
    const double upper = sign < 0.0 ? 15.0 : 0.0;
    const double repaired = canonicalize_state_box_acceleration_roundoff(
        sign * epsilon * epsilon, lower, upper);
    input.rate = dt * repaired;
    EXPECT_EQ(shape_state_box_row(input, dt).status, SolverStatus::kSuccess);
  }
}

TEST(AccelerationStateBoxTest, CompatibleBoxIsSignSymmetric) {
  constexpr double dt = 0.1;
  const auto upper = shape_state_box_row(fixture(), dt);
  const auto lower = shape_state_box_row(fixture(-1.0), dt);

  ASSERT_EQ(upper.status, SolverStatus::kSuccess) << upper.message;
  ASSERT_EQ(lower.status, SolverStatus::kSuccess) << lower.message;
  EXPECT_NEAR(upper.lower_acceleration, -1.0, kTolerance);
  EXPECT_NEAR(upper.upper_acceleration, -0.6753049234040402, kTolerance);
  EXPECT_NEAR(lower.lower_acceleration, 0.6753049234040402, kTolerance);
  EXPECT_NEAR(lower.upper_acceleration, 1.0, kTolerance);
  EXPECT_TRUE(has_state_box_cause(upper.upper_causes,
                                  StateBoxBoundCause::kNextStateViability));
}

TEST(AccelerationStateBoxTest,
     PositionRateShapingUsesTimestepBeforeAccelerationConversion) {
  StateBoxRowInput input;
  input.value = 0.9;
  input.rate = 0.0;
  input.state_bounds = active(-10.0, 1.0);
  input.acceleration_bounds = active(-100.0, 100.0);
  input.lower_braking_acceleration = 100.0;
  input.upper_braking_acceleration = 100.0;

  const auto shaped = shape_state_box_row(input, 0.2);

  ASSERT_EQ(shaped.status, SolverStatus::kSuccess) << shaped.message;
  EXPECT_NEAR(shaped.upper_acceleration, 2.5, kTolerance);
  EXPECT_TRUE(has_state_box_cause(shaped.upper_causes,
                                  StateBoxBoundCause::kStateRate));
}

TEST(AccelerationStateBoxTest, RejectsBoundaryAndBrakingViolations) {
  auto outside = fixture();
  outside.value = 0.06;
  auto outward = fixture();
  outward.value = 0.05;
  outward.rate = 0.01;
  auto outside_envelope = fixture();
  outside_envelope.rate = 1.0;

  EXPECT_EQ(shape_state_box_row(outside, 0.1).failure,
            StateBoxFailure::kCurrentStateOutside);
  EXPECT_EQ(shape_state_box_row(outward, 0.1).failure,
            StateBoxFailure::kOutwardRateAtBoundary);
  EXPECT_EQ(shape_state_box_row(outside_envelope, 0.1).failure,
            StateBoxFailure::kOutsideBrakingEnvelope);
}

TEST(AccelerationStateBoxTest, EmptyIntersectionAndOverflowFailClosed) {
  StateBoxRowInput empty;
  empty.value = 0.0;
  empty.rate = 1.0;
  empty.rate_bounds = {0.0, 0.0, false, true};
  empty.acceleration_bounds = active(-1.0, 1.0);

  StateBoxRowInput overflow;
  overflow.value = -1e308;
  overflow.rate = 1e308;
  overflow.state_bounds = active(-1e308, 1e308);
  overflow.acceleration_bounds = active(-1e308, 1e308);
  overflow.lower_braking_acceleration = 1e308;
  overflow.upper_braking_acceleration = 1e308;

  EXPECT_EQ(shape_state_box_row(empty, 0.1).failure,
            StateBoxFailure::kEmptyInterval);
  EXPECT_EQ(shape_state_box_row(overflow, 2.0).status,
            SolverStatus::kNumericalError);
}

TEST(AccelerationStateBoxTest,
     LinearizedPreparationPreservesMatrixBiasAndShapedRows) {
  LinearizedStateBoxInput input;
  input.source_id = "linearized_endpoint_rows";
  input.coefficient_matrix.resize(1, 2);
  input.coefficient_matrix << 2.0, -1.0;
  input.affine_bias = Eigen::VectorXd::Constant(1, 0.3);
  input.rows = {fixture()};

  const auto prepared = prepare_linearized_state_box(input, 0.1, 2);

  ASSERT_EQ(prepared.status, SolverStatus::kSuccess) << prepared.message;
  const auto &constraint = prepared.physical_constraint;
  EXPECT_EQ(constraint.source_id, input.source_id);
  EXPECT_TRUE(constraint.coefficient_matrix.isApprox(input.coefficient_matrix,
                                                     0.0));
  EXPECT_TRUE(constraint.affine_bias.isApprox(input.affine_bias, 0.0));
  ASSERT_EQ(prepared.state_rows.size(), 1U);
  ASSERT_EQ(prepared.shaped_rows.size(), 1U);
  EXPECT_EQ(prepared.state_rows[0].value, input.rows[0].value);
  EXPECT_EQ(prepared.state_rows[0].rate, input.rows[0].rate);
  EXPECT_NEAR(constraint.lower_bounds(0), -1.0, kTolerance);
  EXPECT_NEAR(constraint.upper_bounds(0), -0.6753049234040402, kTolerance);
  ASSERT_EQ(constraint.lower_bound_active.size(), 1U);
  ASSERT_EQ(constraint.upper_bound_active.size(), 1U);
  EXPECT_TRUE(constraint.lower_bound_active[0]);
  EXPECT_TRUE(constraint.upper_bound_active[0]);
}

TEST(AccelerationStateBoxTest,
     LinearizedPreparationRejectsShapeAndNonfiniteInputsAtomically) {
  LinearizedStateBoxInput wrong_shape;
  wrong_shape.source_id = "wrong_shape";
  wrong_shape.coefficient_matrix = Eigen::MatrixXd::Zero(2, 2);
  wrong_shape.affine_bias = Eigen::VectorXd::Zero(1);
  wrong_shape.rows = {fixture()};
  auto nonfinite = wrong_shape;
  nonfinite.coefficient_matrix.resize(1, 2);
  nonfinite.affine_bias(0) = std::numeric_limits<double>::quiet_NaN();

  const auto shape_result =
      prepare_linearized_state_box(wrong_shape, 0.1, 2);
  const auto nonfinite_result =
      prepare_linearized_state_box(nonfinite, 0.1, 2);

  EXPECT_EQ(shape_result.status, SolverStatus::kShapeMismatch);
  EXPECT_TRUE(shape_result.physical_constraint.coefficient_matrix.size() == 0);
  EXPECT_TRUE(shape_result.state_rows.empty());
  EXPECT_TRUE(shape_result.shaped_rows.empty());
  EXPECT_EQ(nonfinite_result.status, SolverStatus::kNonFiniteInput);
  EXPECT_TRUE(nonfinite_result.physical_constraint.coefficient_matrix.size() ==
              0);
  EXPECT_TRUE(nonfinite_result.state_rows.empty());
  EXPECT_TRUE(nonfinite_result.shaped_rows.empty());
}

TEST(AccelerationStateBoxTest,
     LinearizedPreparationEvaluatesZeroRowsInPhysicalCoordinates) {
  LinearizedStateBoxInput admissible;
  admissible.source_id = "constant_physical_acceleration";
  admissible.coefficient_matrix = Eigen::MatrixXd::Zero(1, 2);
  admissible.affine_bias = Eigen::VectorXd::Constant(1, -0.8);
  admissible.rows = {fixture()};
  auto infeasible = admissible;
  infeasible.affine_bias(0) = 0.0;

  const auto admissible_result =
      prepare_linearized_state_box(admissible, 0.1, 2);
  const auto infeasible_result =
      prepare_linearized_state_box(infeasible, 0.1, 2);

  ASSERT_EQ(admissible_result.status, SolverStatus::kSuccess)
      << admissible_result.message;
  EXPECT_TRUE(
      admissible_result.physical_constraint.coefficient_matrix.isZero(0.0));
  ASSERT_EQ(admissible_result.state_rows.size(), 1U);
  ASSERT_EQ(admissible_result.shaped_rows.size(), 1U);
  EXPECT_EQ(infeasible_result.status, SolverStatus::kInfeasible);
  EXPECT_TRUE(
      infeasible_result.physical_constraint.coefficient_matrix.size() == 0);
  EXPECT_TRUE(infeasible_result.state_rows.empty());
  EXPECT_TRUE(infeasible_result.shaped_rows.empty());
}

struct AccelerationPolishProblem {
  Eigen::VectorXd lower = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd upper = Eigen::VectorXd::Ones(2);
  Eigen::MatrixXd hard = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd hard_lower = lower;
  Eigen::VectorXd hard_upper = upper;
  std::vector<Eigen::MatrixXd> objectives{Eigen::MatrixXd::Identity(2, 2)};
  Eigen::VectorXd value = (Eigen::Vector2d() << -1e-13, 0.25).finished();
  bool changed = true;

  bool apply() {
    return polish_state_box_acceleration(
        lower, upper, hard, hard_lower, hard_upper, objectives,
        1e-8, 1e-10, 1e-8, &value, &changed);
  }
};

TEST(AccelerationStateBoxPolish, InBoxSolutionIsBitwiseUnchanged) {
  AccelerationPolishProblem problem;
  problem.value(0) = -0.0;
  const Eigen::VectorXd original = problem.value;
  ASSERT_TRUE(problem.apply());
  EXPECT_FALSE(problem.changed);
  EXPECT_EQ(std::memcmp(problem.value.data(), original.data(),
                        sizeof(double) * original.size()), 0);
}

TEST(AccelerationStateBoxPolish, PreservesUnboundedAxesAndOneSidedHardRows) {
  const double infinity = std::numeric_limits<double>::infinity();
  AccelerationPolishProblem problem;
  problem.upper(0) = infinity;
  problem.lower(1) = -infinity;
  problem.upper(1) = infinity;
  problem.hard = Eigen::MatrixXd::Zero(1, 2);
  problem.hard(0, 0) = 1.0;
  problem.hard_lower = Eigen::VectorXd::Zero(1);
  problem.hard_upper = Eigen::VectorXd::Constant(1, infinity);
  ASSERT_TRUE(problem.apply());
  EXPECT_TRUE(problem.changed);
  EXPECT_DOUBLE_EQ(problem.value(0), 0.0);
  EXPECT_DOUBLE_EQ(problem.value(1), 0.25);
  problem.lower.setConstant(-infinity);
  problem.upper.setConstant(infinity);
  problem.value(0) = -0.0;
  const Eigen::VectorXd original = problem.value;
  ASSERT_TRUE(problem.apply());
  EXPECT_FALSE(problem.changed);
  EXPECT_EQ(std::memcmp(problem.value.data(), original.data(),
                        sizeof(double) * original.size()), 0);
}

TEST(AccelerationStateBoxPolish, RepairsResidualBeforeNearZeroIntegration) {
  constexpr double dt = 0.01;
  for (double sign : {-1.0, 1.0}) {
    StateBoxRowInput input;
    input.value = -sign * 1e-20;
    input.rate = -sign * 3e-17;
    input.state_bounds = sign < 0.0 ? active(0.0, 1.0) : active(-1.0, 0.0);
    input.rate_bounds = active(-1.0, 1.0);
    input.acceleration_bounds = active(-15.0, 15.0);
    input.lower_braking_acceleration = 15.0;
    input.upper_braking_acceleration = 15.0;
    const auto box = shape_state_box_row(input, dt);
    ASSERT_EQ(box.status, SolverStatus::kSuccess) << box.message;
    const double raw = sign * 1e-13;
    const double old = canonicalize_state_box_acceleration_roundoff(
        raw, box.lower_acceleration, box.upper_acceleration);
    EXPECT_DOUBLE_EQ(old, raw);
    const double old_next = input.value +
        (dt * input.rate + 0.5 * dt * dt * old);
    EXPECT_TRUE(sign < 0.0 ? old_next < input.state_bounds.lower
                           : old_next > input.state_bounds.upper);
    Eigen::VectorXd value = Eigen::VectorXd::Constant(1, raw);
    const Eigen::VectorXd lower =
        Eigen::VectorXd::Constant(1, box.lower_acceleration);
    const Eigen::VectorXd upper =
        Eigen::VectorXd::Constant(1, box.upper_acceleration);
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(1, 1);
    bool changed = false;
    ASSERT_TRUE(polish_state_box_acceleration(
        lower, upper, identity, lower, upper, {identity},
        1e-8, 1e-10, 1e-8, &value, &changed));
    ASSERT_TRUE(changed);
    EXPECT_DOUBLE_EQ(value(0), sign < 0.0 ? box.lower_acceleration
                                         : box.upper_acceleration);
    auto next = input;
    next.value = input.value +
        (dt * input.rate + 0.5 * dt * dt * value(0));
    next.rate = input.rate + dt * value(0);
    EXPECT_GE(next.value, input.state_bounds.lower);
    EXPECT_LE(next.value, input.state_bounds.upper);
    EXPECT_EQ(shape_state_box_row(next, dt).status, SolverStatus::kSuccess);
  }
}

TEST(AccelerationStateBoxPolish, PreservesEachAchievedHierarchyImage) {
  AccelerationPolishProblem problem;
  problem.objectives = {Eigen::MatrixXd::Zero(1, 2),
                        Eigen::MatrixXd::Zero(1, 2)};
  problem.objectives[0](0, 1) = 1.0;
  problem.objectives[1](0, 0) = 1.0;
  const Eigen::VectorXd original = problem.value;
  ASSERT_TRUE(problem.apply());
  EXPECT_TRUE(problem.changed);
  EXPECT_DOUBLE_EQ(problem.value(0), 0.0);
  for (const auto &matrix : problem.objectives) {
    EXPECT_LE((matrix * problem.value - matrix * original).norm(), 1e-8);
  }
}

TEST(AccelerationStateBoxPolish, RejectsPhysicalCoupledRowDamage) {
  AccelerationPolishProblem problem;
  problem.hard = Eigen::MatrixXd::Zero(1, 2);
  problem.hard(0, 0) = 1e8;
  problem.hard_lower = Eigen::VectorXd::Constant(1, -1e-5);
  problem.hard_upper = problem.hard_lower;
  const Eigen::VectorXd original = problem.value;
  EXPECT_FALSE(problem.apply());
  EXPECT_FALSE(problem.changed);
  EXPECT_TRUE(problem.value == original);
}

TEST(AccelerationStateBoxPolish, RetainsNormalizedCertificateUnderRowRescaling) {
  for (double scale : {1e-9, 1.0}) {
    AccelerationPolishProblem problem;
    problem.value(0) = -5e-9;
    problem.hard = Eigen::MatrixXd::Zero(1, 2);
    problem.hard(0, 0) = scale;
    problem.hard_lower = Eigen::VectorXd::Constant(1, -5e-9 * scale);
    problem.hard_upper = problem.hard_lower;
    const Eigen::VectorXd original = problem.value;
    EXPECT_FALSE(problem.apply());
    EXPECT_FALSE(problem.changed);
    EXPECT_TRUE(problem.value == original);
  }
}

TEST(AccelerationStateBoxPolish, RejectsCompletedImageDamage) {
  AccelerationPolishProblem problem;
  problem.objectives = {Eigen::MatrixXd::Zero(1, 2)};
  problem.objectives[0](0, 0) = 1e8;
  const Eigen::VectorXd original = problem.value;
  EXPECT_FALSE(problem.apply());
  EXPECT_FALSE(problem.changed);
  EXPECT_TRUE(problem.value == original);
}

TEST(AccelerationStateBoxPolish, CertifiesSubArithmeticResidualAgainstRawImage) {
  AccelerationPolishProblem problem;
  problem.value(0) = -1e-15;
  problem.objectives = {Eigen::MatrixXd::Zero(1, 2)};
  problem.objectives[0](0, 0) = 1e8;
  const Eigen::VectorXd original = problem.value;
  // Arithmetic-only preprocessing would erase this otherwise visible change.
  EXPECT_DOUBLE_EQ(canonicalize_state_box_acceleration_roundoff(
                       original(0), problem.lower(0), problem.upper(0)),
                   0.0);
  EXPECT_FALSE(problem.apply());
  EXPECT_FALSE(problem.changed);
  EXPECT_TRUE(problem.value == original);
}

TEST(AccelerationStateBoxPolish, CertifiesComposedAxisCorrectionsTogether) {
  AccelerationPolishProblem problem;
  problem.value << -1e-15, -1e-13;
  problem.objectives = {Eigen::MatrixXd::Zero(1, 2)};
  problem.objectives[0] << 7e6, 7e4;
  const Eigen::VectorXd original = problem.value;

  // Each individual image change is below tolerance, but their combined
  // correction is 1.4e-8 and must be checked against the raw backend image.
  AccelerationPolishProblem preprocessed = problem;
  preprocessed.value(0) = canonicalize_state_box_acceleration_roundoff(
      original(0), problem.lower(0), problem.upper(0));
  ASSERT_TRUE(preprocessed.apply());
  EXPECT_TRUE(preprocessed.changed);
  EXPECT_GT((problem.objectives[0] * preprocessed.value -
             problem.objectives[0] * original).norm(),
            1e-8);

  EXPECT_FALSE(problem.apply());
  EXPECT_FALSE(problem.changed);
  EXPECT_TRUE(problem.value == original);
}

TEST(AccelerationStateBoxPolish, HandlesEmptyAndConstantHardRowSets) {
  AccelerationPolishProblem problem;
  problem.hard = Eigen::MatrixXd::Zero(0, 2);
  problem.hard_lower.resize(0);
  problem.hard_upper.resize(0);
  ASSERT_TRUE(problem.apply());
  EXPECT_TRUE(problem.changed);
  problem.value(0) = -1e-13;
  problem.hard = Eigen::MatrixXd::Zero(1, 2);
  problem.hard_lower = Eigen::VectorXd::Zero(1);
  problem.hard_upper = Eigen::VectorXd::Zero(1);
  ASSERT_TRUE(problem.apply());
  problem.value(0) = -1e-13;
  problem.hard_lower(0) = 1e-8;
  problem.hard_upper(0) = 2e-8;
  EXPECT_FALSE(problem.apply());
  EXPECT_DOUBLE_EQ(problem.value(0), -1e-13);
}

TEST(AccelerationStateBoxPolish, RejectsLargeResidualAndInvalidInputs) {
  for (int variant = 0; variant < 9; ++variant) {
    AccelerationPolishProblem problem;
    if (variant == 0) problem.value(0) = -1e-7;
    if (variant == 1) problem.upper.resize(1);
    if (variant == 2) problem.hard.resize(2, 3);
    if (variant == 3) problem.hard_lower(0) = std::numeric_limits<double>::infinity();
    if (variant == 4) problem.hard(0, 0) = std::numeric_limits<double>::quiet_NaN();
    if (variant == 5) problem.objectives = {Eigen::MatrixXd::Zero(1, 3)};
    if (variant == 6) problem.lower(0) = std::numeric_limits<double>::quiet_NaN();
    if (variant == 7) problem.hard_upper(0) = std::numeric_limits<double>::quiet_NaN();
    if (variant == 8) problem.value(0) = std::numeric_limits<double>::quiet_NaN();
    const Eigen::VectorXd original = problem.value;
    EXPECT_FALSE(problem.apply());
    EXPECT_FALSE(problem.changed);
    EXPECT_EQ(std::memcmp(problem.value.data(), original.data(),
                          sizeof(double) * original.size()), 0);
  }
}

TEST(AccelerationStateBoxPolish, DoesNotRelaxUncertifiedMeasuredState) {
  auto input = fixture();
  input.value = -1e-20;
  input.rate = -1e-16;
  input.state_bounds = active(0.0, 1.0);
  EXPECT_EQ(shape_state_box_row(input, 0.01).status, SolverStatus::kInfeasible);
  EXPECT_DOUBLE_EQ(input.value, -1e-20);
  EXPECT_DOUBLE_EQ(input.rate, -1e-16);
}

} // namespace
} // namespace embodik::detail
