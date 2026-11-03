#include <embodik/control_solver_kernel.hpp>
#include <gtest/gtest.h>

namespace {

TEST(LeastSquaresQuadratic, RecoversRedundantMinimumWithoutCheckpoint) {
  const Eigen::MatrixXd matrix = Eigen::MatrixXd::Ones(1, 2);
  const Eigen::VectorXd target = Eigen::VectorXd::Ones(1);
  const double ridge = 1e-12;
  Eigen::MatrixXd hessian = matrix.transpose() * matrix;
  hessian.diagonal().array() += ridge;
  const Eigen::VectorXd gradient_target = matrix.transpose() * target;
  const Eigen::MatrixXd C = Eigen::MatrixXd::Identity(2, 2);
  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(2, -3.0);
  const Eigen::VectorXd upper = -lower;
  const Eigen::VectorXd seed = Eigen::VectorXd::Zero(2);
  const embodik::detail::LeastSquaresQuadratic objective{matrix, target, ridge};
  embodik::VelocitySolverConfig config;
  config.epsilon = 1e-10;
  Eigen::VectorXd solution;
  unsigned int iterations;
  ASSERT_TRUE(embodik::detail::RecoverFeasibleWeightedQuadratic(
      hessian, gradient_target, C, lower, upper, config, &solution,
      &iterations, &seed, false, nullptr, &objective));
  EXPECT_LT(iterations, 10U);
  EXPECT_NEAR(solution(0), 0.5, 1e-8);
  EXPECT_NEAR(solution(1), 0.5, 1e-8);
  EXPECT_LE(embodik::detail::ComputeMaxLinearConstraintViolation(
      C, lower, upper, solution), config.epsilon);
}

TEST(LeastSquaresQuadratic, RejectsMalformedFactorsBeforeSearching) {
  const Eigen::MatrixXd H = Eigen::MatrixXd::Identity(2, 2);
  const Eigen::VectorXd g = Eigen::VectorXd::Zero(2);
  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(2, -3.0);
  const Eigen::VectorXd upper = -lower;
  const Eigen::VectorXd seed = Eigen::VectorXd::Zero(2);
  embodik::VelocitySolverConfig config;
  for (int invalid = 0; invalid < 7; ++invalid) {
    SCOPED_TRACE(invalid);
    Eigen::MatrixXd matrix = H;
    Eigen::VectorXd target = g;
    double ridge = 1e-12;
    switch (invalid) {
    case 0: matrix.resize(2, 3); matrix.setZero(); break;
    case 1: target.resize(3); target.setZero(); break;
    case 2: matrix(0, 0) = std::numeric_limits<double>::quiet_NaN(); break;
    case 3: target(0) = std::numeric_limits<double>::infinity(); break;
    case 4: ridge = std::numeric_limits<double>::quiet_NaN(); break;
    case 5: ridge = -1.0; break;
    case 6: ridge = std::numeric_limits<double>::infinity(); break;
    }
    const embodik::detail::LeastSquaresQuadratic objective{matrix, target, ridge};
    Eigen::VectorXd solution = Eigen::VectorXd::Constant(2, -7.0);
    unsigned int iterations = 99;
    std::string reason;
    EXPECT_FALSE(embodik::detail::RecoverFeasibleWeightedQuadratic(
        H, g, H, lower, upper, config, &solution, &iterations, &seed,
        false, &reason, &objective));
    EXPECT_EQ(iterations, 0U);
    EXPECT_EQ(reason, "invalid least-squares objective factors");
    EXPECT_EQ(solution(0), -7.0);
    EXPECT_EQ(solution(1), -7.0);
  }
}

} // namespace
