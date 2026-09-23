#include "generalized_constraint_set.hpp"

#include <gtest/gtest.h>

namespace embodik::detail {
namespace {

GeneralizedConstraintBlock make_block(
    const Eigen::MatrixXd &coefficients, const Eigen::VectorXd &lower,
    const Eigen::VectorXd &upper,
    const Eigen::VectorXd &affine_bias = Eigen::VectorXd{},
    const Eigen::VectorXd &softening = Eigen::VectorXd{}) {
  GeneralizedConstraintBlock block;
  block.coefficient_matrix = coefficients;
  block.physical_lower_bounds = lower;
  block.physical_upper_bounds = upper;
  block.affine_bias = affine_bias.size() == 0
                          ? Eigen::VectorXd::Zero(coefficients.rows())
                          : affine_bias;
  block.max_softening_factors = softening;
  return block;
}

TEST(GeneralizedConstraintSetTest,
     ConcatenatesRowsSofteningAndSourcesInInsertionOrder) {
  GeneralizedConstraintSet constraints(/*variable_count=*/2,
                                       /*row_capacity=*/3,
                                       /*track_sources=*/true);

  GeneralizedConstraintBlock identity = make_block(
      Eigen::Matrix2d::Identity(),
      (Eigen::Vector2d() << -0.2, -0.3).finished(),
      (Eigen::Vector2d() << 0.2, 0.1).finished());
  identity.row_sources = {
      {{"joint_velocity_model_limit", "nv", 0,
        GeneralizedConstraintBoundSide::kBoth}},
      {{"joint_velocity_model_limit", "nv", 1,
        GeneralizedConstraintBoundSide::kBoth}},
  };
  ASSERT_TRUE(constraints.append_block(identity));

  GeneralizedConstraintBlock coupled = make_block(
      (Eigen::Matrix<double, 1, 2>() << 1.0, 1.0).finished(),
      Eigen::VectorXd::Constant(1, -0.1),
      Eigen::VectorXd::Constant(1, 0.15), Eigen::VectorXd{},
      Eigen::VectorXd::Constant(1, 4.0));
  coupled.row_sources = {
      {{"user_linear_velocity", "configured", 0,
        GeneralizedConstraintBoundSide::kBoth}},
  };
  ASSERT_TRUE(constraints.append_block(coupled));
  ASSERT_TRUE(constraints.finalize());

  Eigen::Matrix<double, 3, 2> expected_coefficients;
  expected_coefficients << 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;
  EXPECT_TRUE(
      constraints.coefficient_matrix().isApprox(expected_coefficients, 0.0));
  EXPECT_TRUE(constraints.lower_bounds().isApprox(
      (Eigen::Vector3d() << -0.2, -0.3, -0.1).finished(), 0.0));
  EXPECT_TRUE(constraints.upper_bounds().isApprox(
      (Eigen::Vector3d() << 0.2, 0.1, 0.15).finished(), 0.0));
  EXPECT_TRUE(constraints.max_softening_factors().isApprox(
      (Eigen::Vector3d() << 1.0, 1.0, 4.0).finished(), 0.0));

  ASSERT_EQ(constraints.row_sources().size(), 3U);
  EXPECT_EQ(constraints.row_sources()[0][0].family,
            "joint_velocity_model_limit");
  EXPECT_EQ(constraints.row_sources()[2][0].family,
            "user_linear_velocity");
}

TEST(GeneralizedConstraintSetTest, ShiftsPhysicalBoundsByAffineBias) {
  GeneralizedConstraintSet constraints(/*variable_count=*/2,
                                       /*row_capacity=*/2);
  const Eigen::Vector2d bias(0.4, -0.2);
  ASSERT_TRUE(constraints.append_block(make_block(
      Eigen::Matrix2d::Identity(),
      (Eigen::Vector2d() << -1.0, -2.0).finished(),
      (Eigen::Vector2d() << 1.0, 2.0).finished(), bias)));
  ASSERT_TRUE(constraints.finalize());

  EXPECT_TRUE(constraints.lower_bounds().isApprox(
      (Eigen::Vector2d() << -1.4, -1.8).finished(), 0.0));
  EXPECT_TRUE(constraints.upper_bounds().isApprox(
      (Eigen::Vector2d() << 0.6, 2.2).finished(), 0.0));
}

TEST(GeneralizedConstraintSetTest, RejectsMalformedBlockAtomically) {
  GeneralizedConstraintSet constraints(/*variable_count=*/2,
                                       /*row_capacity=*/2,
                                       /*track_sources=*/true);
  ASSERT_TRUE(constraints.append_block(make_block(
      (Eigen::Matrix<double, 1, 2>() << 1.0, 0.0).finished(),
      Eigen::VectorXd::Constant(1, -1.0),
      Eigen::VectorXd::Constant(1, 1.0))));

  const Eigen::MatrixXd coefficients_before = constraints.coefficient_matrix();
  const Eigen::VectorXd lower_before = constraints.lower_bounds();
  const Eigen::VectorXd upper_before = constraints.upper_bounds();
  const Eigen::Index rows_before = constraints.row_count();

  GeneralizedConstraintBlock wrong_cols = make_block(
      Eigen::MatrixXd::Ones(1, 3), Eigen::VectorXd::Constant(1, -2.0),
      Eigen::VectorXd::Constant(1, 2.0));
  EXPECT_FALSE(constraints.append_block(wrong_cols));
  EXPECT_EQ(constraints.row_count(), rows_before);
  EXPECT_TRUE(
      constraints.coefficient_matrix().isApprox(coefficients_before, 0.0));
  EXPECT_TRUE(constraints.lower_bounds().isApprox(lower_before, 0.0));
  EXPECT_TRUE(constraints.upper_bounds().isApprox(upper_before, 0.0));

  GeneralizedConstraintBlock wrong_bias;
  wrong_bias.coefficient_matrix = Eigen::MatrixXd::Identity(1, 2);
  wrong_bias.physical_lower_bounds = Eigen::VectorXd::Constant(1, -1.0);
  wrong_bias.physical_upper_bounds = Eigen::VectorXd::Constant(1, 1.0);
  wrong_bias.affine_bias = Eigen::VectorXd::Zero(2);
  EXPECT_FALSE(constraints.append_block(wrong_bias));
  EXPECT_EQ(constraints.row_count(), rows_before);
}

TEST(GeneralizedConstraintSetTest,
     RejectsNegativeDimensionsWithoutAllocation) {
  GeneralizedConstraintSet negative_variables(/*variable_count=*/-1,
                                              /*row_capacity=*/1);
  EXPECT_FALSE(negative_variables.append_block(make_block(
      Eigen::MatrixXd::Identity(1, 1),
      Eigen::VectorXd::Constant(1, -1.0),
      Eigen::VectorXd::Constant(1, 1.0))));
  EXPECT_FALSE(negative_variables.finalize());
  EXPECT_EQ(negative_variables.coefficient_matrix().rows(), 0);
  EXPECT_EQ(negative_variables.coefficient_matrix().cols(), 0);
}

} // namespace
} // namespace embodik::detail
