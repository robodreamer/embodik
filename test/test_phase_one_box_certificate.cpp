#include <embodik/control_solver_kernel.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>

namespace embodik::test {
namespace {

constexpr double kTolerance = 1e-8;

bool certificate(const Eigen::MatrixXd &matrix, const Eigen::VectorXd &lower,
                 const Eigen::VectorXd &upper) {
  return detail::UnitScalarBoxProvesLinearInfeasibility(matrix, lower, upper, kTolerance);
}

class PhaseOneBoxCertificateProof : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!std::numeric_limits<double>::is_iec559 ||
        std::numeric_limits<double>::radix != 2 ||
        std::numeric_limits<double>::digits != 53 ||
        sizeof(double) != sizeof(std::uint64_t) || FLT_EVAL_METHOD != 0 ||
        std::numeric_limits<long double>::digits < 64 ||
        std::numeric_limits<long double>::max_exponent <
            4 * std::numeric_limits<double>::max_exponent ||
        std::numeric_limits<long double>::min_exponent >
            4 * std::numeric_limits<double>::min_exponent) {
      GTEST_SKIP() << "The optional box certificate requires wide intermediates";
    }
  }
};

TEST_F(PhaseOneBoxCertificateProof, PositiveAndNegativeCoupledRowGaps) {
  Eigen::MatrixXd matrix(3, 2);
  matrix << 1, 0, 0, 1, 1, 1;
  Eigen::Vector3d lower(1, 2, 6), upper(2, 3, 7);
  EXPECT_TRUE(certificate(matrix, lower, upper));
  matrix.row(2) << -1, -2;
  lower(2) = -12;
  upper(2) = -9;
  EXPECT_TRUE(certificate(matrix, lower, upper));
  VelocitySolverConfig config;
  config.epsilon = kTolerance;
  Eigen::VectorXd witness;
  EXPECT_EQ(detail::FindHardConstraintFeasibleWitness(matrix, lower, upper, config, &witness),
            detail::HardConstraintFeasibilityStatus::kProvenInfeasible);
}

TEST_F(PhaseOneBoxCertificateProof, IntersectsSignedUnitRowsAndSkipsNonunitScalarRows) {
  Eigen::MatrixXd matrix(6, 2);
  matrix << 1, 0, -1, 0, 0, -1, 0, 1, 2, 0, 1, -1;
  Eigen::VectorXd lower(6), upper(6);
  lower << 1, -2, -3, 2, 2, 4;
  upper << 2, -1, -2, 3, 4, 5;
  EXPECT_TRUE(certificate(matrix, lower, upper));
  lower(5) = -2;
  upper(5) = 0;
  EXPECT_FALSE(certificate(matrix, lower, upper));
}

TEST_F(PhaseOneBoxCertificateProof, OneAndThreeDimensionalGaps) {
  Eigen::MatrixXd scalar(2, 1);
  scalar << 1, -1;
  EXPECT_TRUE(certificate(scalar, Eigen::Vector2d(1, -.5), Eigen::Vector2d(2, 0)));
  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(4, 3);
  matrix.topRows(3).setIdentity();
  matrix.row(3) << 2, -1, 3;
  Eigen::Vector4d lower(.5, .5, .5, -1), upper(1.5, 1.5, 1.5, .5);
  EXPECT_TRUE(certificate(matrix, lower, upper));
}

TEST(PhaseOneBoxCertificate, OriginalEpsilonAcceptedCornerIsNotRejected) {
  Eigen::MatrixXd matrix(3, 2);
  matrix << 1, 0, 0, 1, 1, 1;
  Eigen::Vector3d lower(0, 0, 2 + 1.25 * kTolerance), upper(1, 1, 3);
  const Eigen::Vector2d accepted = Eigen::Vector2d::Constant(1 + .25 * kTolerance);
  ASSERT_LE(detail::ComputeMaxLinearConstraintViolation(matrix, lower, upper, accepted), kTolerance);
  EXPECT_FALSE(certificate(matrix, lower, upper));
  // Here scalar-bound tolerance contributes negligibly to the coupled image:
  // acceptance depends on expanding the original coupled row by its epsilon.
  matrix.row(2).setConstant(1e-10);
  lower << 1, 1, 2e-10 + .75 * kTolerance;
  upper << 1, 1, 1;
  const Eigen::Vector2d corner = Eigen::Vector2d::Ones();
  ASSERT_LE(detail::ComputeMaxLinearConstraintViolation(matrix, lower, upper, corner), kTolerance);
  EXPECT_FALSE(certificate(matrix, lower, upper));
}

TEST(PhaseOneBoxCertificate, CoupledCancellationPreservesAcceptedNonzeroWitness) {
  Eigen::MatrixXd matrix(5, 3);
  matrix << 1, 0, 0, 0, -1, 0, 0, 0, 1, .1, 0, 0, 1, 1, -1;
  // Every coordinate has a signed-unit box. The exact final image is 1,
  // while a double reduction may lose the middle term to cancellation.
  const Eigen::Vector3d accepted(1e16, 1, 1e16);
  Eigen::VectorXd lower = matrix * accepted, upper = lower;
  for (Eigen::Index row = 0; row < matrix.rows(); ++row)
    lower(row) = upper(row) = matrix.row(row).dot(accepted);
  ASSERT_EQ(detail::ComputeMaxLinearConstraintViolation(matrix, lower, upper, accepted), 0);
  EXPECT_FALSE(certificate(matrix, lower, upper));
}

TEST_F(PhaseOneBoxCertificateProof, UnboundedImagesRemainUnprovedButOneSidedRowsCanProveGap) {
  const double infinity = std::numeric_limits<double>::infinity();
  Eigen::MatrixXd matrix(2, 2);
  matrix << 1, 0, 1, 1;
  EXPECT_FALSE(certificate(matrix, Eigen::Vector2d(1, -infinity), Eigen::Vector2d(2, 0)));
  Eigen::MatrixXd bounded(3, 2);
  bounded << 1, 0, 0, 1, 1, 1;
  EXPECT_TRUE(certificate(bounded, Eigen::Vector3d(1, 2, 6), Eigen::Vector3d(2, 3, infinity)));
  EXPECT_TRUE(certificate(bounded, Eigen::Vector3d(1, 2, -infinity), Eigen::Vector3d(2, 3, 0)));
  EXPECT_FALSE(certificate(bounded, Eigen::Vector3d(1, 2, -infinity), Eigen::Vector3d(infinity, 3, 0)));
}

TEST(PhaseOneBoxCertificate, OverflowAndSubnormalRowsGiveNoCertificate) {
  Eigen::MatrixXd matrix(3, 2);
  matrix << 1, 0, 0, 1, std::numeric_limits<double>::max(), std::numeric_limits<double>::max();
  EXPECT_FALSE(certificate(matrix, Eigen::Vector3d(.75, .75, -4), Eigen::Vector3d(1, 1, -1)));
  matrix.row(2).setConstant(std::numeric_limits<double>::denorm_min());
  EXPECT_FALSE(certificate(matrix, Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(2, 2, 2)));
  matrix.row(2) << std::numeric_limits<double>::quiet_NaN(), 1;
  EXPECT_FALSE(certificate(matrix, Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(2, 2, 2)));
}

TEST(PhaseOneBoxCertificate, UnderflowedProductsDoNotTurnAcceptedRowsIntoGaps) {
  Eigen::MatrixXd matrix(3, 2);
  matrix << 1, 0, 0, -1, 1e-200, 1e-200;
  const Eigen::Vector2d accepted(1e-200, 1e-200);
  const Eigen::Vector3d bounds = matrix * accepted;
  ASSERT_EQ(detail::ComputeMaxLinearConstraintViolation(matrix, bounds, bounds, accepted), 0);
  EXPECT_FALSE(certificate(matrix, bounds, bounds));
}

TEST(PhaseOneBoxCertificate, SubnormalColumnCannotMasqueradeAsSignedUnitScalarRow) {
  Eigen::MatrixXd matrix(3, 2);
  matrix << 1, std::numeric_limits<double>::min() / 2, 0, 1, 1, 0;
  const Eigen::Vector2d accepted(0, 1e308);
  const Eigen::Vector3d lower(1, 1e308, -.25), upper(2, 1e308, .25);
  ASSERT_EQ(detail::ComputeMaxLinearConstraintViolation(matrix, lower, upper, accepted), 0);
  EXPECT_FALSE(certificate(matrix, lower, upper));
}

TEST(PhaseOneBoxCertificate, GenericAcceptedWitnessesAcrossSignsAndScalesAreNeverRejected) {
  std::mt19937 random(1701);
  std::uniform_real_distribution<double> mantissa(-2., 2.);
  std::uniform_int_distribution<int> exponent(-450, 450);
  for (int trial = 0; trial < 1000; ++trial) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(7, 3);
    Eigen::Vector3d accepted;
    for (int col = 0; col < 3; ++col) {
      accepted(col) = std::ldexp(mantissa(random), exponent(random));
      matrix(col, col) = trial % 2 ? 1.0 : -1.0;
    }
    matrix.row(3) = -matrix.row(0);
    for (int row = 4; row < 7; ++row)
      for (int col = 0; col < 3; ++col)
        matrix(row, col) = std::ldexp(mantissa(random), exponent(random));
    Eigen::VectorXd bounds(7);
    for (int row = 0; row < 7; ++row) bounds(row) = matrix.row(row).dot(accepted);
    ASSERT_TRUE(bounds.allFinite());
    ASSERT_EQ(detail::ComputeMaxLinearConstraintViolation(matrix, bounds, bounds, accepted), 0);
    EXPECT_FALSE(certificate(matrix, bounds, bounds)) << "trial=" << trial;
  }
}

TEST(PhaseOneBoxCertificate, FeasibleZeroFastPathAndWitnessStayUnchanged) {
  Eigen::MatrixXd matrix(3, 2);
  matrix << 1, 0, 0, 1, 1, -1;
  Eigen::Vector3d lower = Eigen::Vector3d::Constant(-1), upper = -lower;
  VelocitySolverConfig config;
  Eigen::VectorXd witness = Eigen::VectorXd::Constant(2, 9);
  EXPECT_EQ(detail::FindHardConstraintFeasibleWitness(matrix, lower, upper, config, &witness),
            detail::HardConstraintFeasibilityStatus::kFeasible);
  EXPECT_TRUE(witness.isZero(0));
}

} // namespace
} // namespace embodik::test
