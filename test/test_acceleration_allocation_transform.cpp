#include "acceleration_allocation_transform.hpp"

#include <gtest/gtest.h>

#include <limits>

namespace embodik::detail {
namespace {

constexpr double kTolerance = 1e-12;

Eigen::VectorXd vector(std::initializer_list<double> values) {
  Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
  Eigen::Index index = 0;
  for (double value : values) {
    result(index++) = value;
  }
  return result;
}

TEST(AccelerationAllocationTransformTest,
     IdentityAndUniformMetricsAreExactlyNeutral) {
  for (const double weight : {1.0, 0.25, 4.0, 1e100}) {
    const auto transform = prepare_acceleration_allocation_transform(
        Eigen::Vector3d::Constant(weight), Eigen::Vector3d::Zero(),
        /*variable_count=*/3);

    ASSERT_EQ(transform.status, SolverStatus::kSuccess) << transform.message;
    EXPECT_TRUE(transform.variable_scale.isOnes(0.0));
    EXPECT_TRUE(transform.inverse_variable_scale.isOnes(0.0));

    Eigen::Matrix<double, 2, 3> objective;
    objective << 1.0, 2.0, 3.0, -4.0, 5.0, -6.0;
    const auto transformed = transform.transform_objective_matrix(objective);
    ASSERT_TRUE(transformed.satisfied()) << transformed.message;
    EXPECT_TRUE(transformed.matrix.isApprox(objective, 0.0));
  }
}

TEST(AccelerationAllocationTransformTest,
     TransformsObjectiveAndBackendHardRowsWithReference) {
  const auto transform = prepare_acceleration_allocation_transform(
      vector({4.0, 1.0}), vector({0.5, -0.25}), /*variable_count=*/2);
  ASSERT_EQ(transform.status, SolverStatus::kSuccess) << transform.message;
  EXPECT_TRUE(transform.variable_scale.isApprox(
      vector({0.7071067811865476, 1.4142135623730951}), kTolerance));

  Eigen::Matrix2d objective;
  objective << 1.0, 2.0, -3.0, 4.0;
  const Eigen::Vector2d bias(0.1, -0.2);

  const auto transformed_objective =
      transform.transform_objective_matrix(objective);
  const auto transformed_bias = transform.transform_objective_bias(objective,
                                                                  bias);

  ASSERT_TRUE(transformed_objective.satisfied())
      << transformed_objective.message;
  ASSERT_TRUE(transformed_bias.satisfied()) << transformed_bias.message;
  EXPECT_TRUE(transformed_objective.matrix.isApprox(
      objective * transform.variable_scale.asDiagonal(), 0.0));
  EXPECT_TRUE(transformed_bias.vector.isApprox(
      bias + objective * transform.reference_acceleration, 0.0));

  Eigen::Matrix<double, 2, 2> hard_rows;
  hard_rows << 2.0, -1.0, -0.5, 3.0;
  const Eigen::Vector2d lower_shifted(-2.0, -3.0);
  const Eigen::Vector2d upper_shifted(2.0, 3.0);
  const auto transformed_hard =
      transform.transform_backend_hard_rows(hard_rows, lower_shifted,
                                            upper_shifted);

  ASSERT_TRUE(transformed_hard.satisfied()) << transformed_hard.message;
  const Eigen::Vector2d reference_shift =
      hard_rows * transform.reference_acceleration;
  EXPECT_TRUE(transformed_hard.coefficient_matrix.isApprox(
      hard_rows * transform.variable_scale.asDiagonal(), 0.0));
  EXPECT_TRUE(transformed_hard.lower_bounds.isApprox(
      lower_shifted - reference_shift, 0.0));
  EXPECT_TRUE(transformed_hard.upper_bounds.isApprox(
      upper_shifted - reference_shift, 0.0));

  const Eigen::Vector2d solver_variable(0.3, -0.6);
  const auto physical =
      transform.reconstruct_physical_acceleration(solver_variable);
  ASSERT_TRUE(physical.satisfied()) << physical.message;
  EXPECT_TRUE(physical.vector.isApprox(
      transform.reference_acceleration +
          transform.variable_scale.cwiseProduct(solver_variable),
      0.0));

  EXPECT_TRUE((objective * physical.vector + bias)
                  .isApprox(transformed_objective.matrix * solver_variable +
                                transformed_bias.vector,
                            kTolerance));
  EXPECT_TRUE((hard_rows * physical.vector)
                  .isApprox(transformed_hard.coefficient_matrix *
                                    solver_variable +
                                reference_shift,
                            kTolerance));
}

TEST(AccelerationAllocationTransformTest, ReconstructsAndInverseScalesResidual) {
  const auto transform = prepare_acceleration_allocation_transform(
      vector({9.0, 1.0, 4.0}), vector({0.2, -0.3, 0.4}),
      /*variable_count=*/3);
  ASSERT_TRUE(transform.satisfied()) << transform.message;

  const Eigen::Vector3d solver_variable(0.5, -0.25, 0.75);
  const auto physical =
      transform.reconstruct_physical_acceleration(solver_variable);
  ASSERT_TRUE(physical.satisfied()) << physical.message;
  const Eigen::Vector3d residual =
      physical.vector - transform.reference_acceleration;

  const auto recovered_solver_variable =
      transform.inverse_scale_physical_residual(residual);
  ASSERT_TRUE(recovered_solver_variable.satisfied())
      << recovered_solver_variable.message;
  EXPECT_TRUE(
      recovered_solver_variable.vector.isApprox(solver_variable, kTolerance));
}

TEST(AccelerationAllocationTransformTest,
     InvalidSizesNonFiniteAndNonPositiveMetricsFailClosed) {
  EXPECT_EQ(prepare_acceleration_allocation_transform(
                vector({1.0}), Eigen::Vector2d::Zero(), 2)
                .status,
            SolverStatus::kShapeMismatch);
  EXPECT_EQ(prepare_acceleration_allocation_transform(
                Eigen::Vector2d::Ones(), Eigen::Vector2d::Zero(), 0)
                .status,
            SolverStatus::kShapeMismatch);
  EXPECT_EQ(prepare_acceleration_allocation_transform(
                vector({1.0, std::numeric_limits<double>::quiet_NaN()}),
                Eigen::Vector2d::Zero(), 2)
                .status,
            SolverStatus::kNonFiniteInput);
  EXPECT_EQ(prepare_acceleration_allocation_transform(
                vector({1.0, 0.0}), Eigen::Vector2d::Zero(), 2)
                .status,
            SolverStatus::kInvalidInput);
  EXPECT_EQ(prepare_acceleration_allocation_transform(
                vector({1.0, -1.0}), Eigen::Vector2d::Zero(), 2)
                .status,
            SolverStatus::kInvalidInput);

  EXPECT_EQ(prepare_acceleration_allocation_transform(
                vector({std::numeric_limits<double>::min(),
                        std::numeric_limits<double>::max()}),
                Eigen::Vector2d::Zero(), 2)
                .status,
            SolverStatus::kInvalidInput);
}

TEST(AccelerationAllocationTransformTest,
     TransformMethodsRejectInvalidShapesAndNonfiniteInputs) {
  const auto transform = prepare_acceleration_allocation_transform(
      Eigen::Vector2d::Ones(), Eigen::Vector2d::Zero(),
      /*variable_count=*/2);
  ASSERT_TRUE(transform.satisfied()) << transform.message;

  EXPECT_EQ(transform.transform_objective_matrix(Eigen::MatrixXd::Ones(1, 3))
                .status,
            SolverStatus::kShapeMismatch);
  EXPECT_EQ(transform
                .transform_objective_bias(Eigen::MatrixXd::Ones(1, 2),
                                          Eigen::Vector2d::Ones())
                .status,
            SolverStatus::kShapeMismatch);
  EXPECT_EQ(transform
                .transform_backend_hard_rows(Eigen::MatrixXd::Ones(1, 2),
                                             Eigen::Vector2d::Zero(),
                                             Eigen::VectorXd::Zero(1))
                .status,
            SolverStatus::kShapeMismatch);
  EXPECT_EQ(transform.reconstruct_physical_acceleration(Eigen::Vector3d::Zero())
                .status,
            SolverStatus::kShapeMismatch);
  EXPECT_EQ(transform.inverse_scale_physical_residual(Eigen::Vector3d::Zero())
                .status,
            SolverStatus::kShapeMismatch);

  Eigen::MatrixXd nonfinite_matrix = Eigen::MatrixXd::Ones(1, 2);
  nonfinite_matrix(0, 1) = std::numeric_limits<double>::infinity();
  EXPECT_EQ(transform.transform_objective_matrix(nonfinite_matrix).status,
            SolverStatus::kNonFiniteInput);
  EXPECT_EQ(transform
                .transform_backend_hard_rows(nonfinite_matrix,
                                             Eigen::VectorXd::Zero(1),
                                             Eigen::VectorXd::Ones(1))
                .status,
            SolverStatus::kNonFiniteInput);
  EXPECT_EQ(transform
                .reconstruct_physical_acceleration(
                    vector({0.0, std::numeric_limits<double>::quiet_NaN()}))
                .status,
            SolverStatus::kNonFiniteInput);
}

TEST(AccelerationAllocationTransformTest,
     TransformedNonfiniteArithmeticFailsClosed) {
  const auto transform = prepare_acceleration_allocation_transform(
      Eigen::Vector2d::Ones(), Eigen::Vector2d::Constant(1e308),
      /*variable_count=*/2);
  ASSERT_TRUE(transform.satisfied()) << transform.message;

  Eigen::RowVector2d overflowing_row;
  overflowing_row << 1e308, 1e308;
  EXPECT_EQ(transform
                .transform_objective_bias(overflowing_row,
                                          Eigen::VectorXd::Zero(1))
                .status,
            SolverStatus::kNumericalError);
  EXPECT_EQ(transform
                .transform_backend_hard_rows(overflowing_row,
                                             Eigen::VectorXd::Zero(1),
                                             Eigen::VectorXd::Ones(1))
                .status,
            SolverStatus::kNumericalError);
}

} // namespace
} // namespace embodik::detail
