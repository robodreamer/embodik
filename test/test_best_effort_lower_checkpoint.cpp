#include <embodik/ik_baseline.hpp>
#include <gtest/gtest.h>
#include <limits>

namespace embodik::test {
namespace {
struct Checkpoint {
  Eigen::VectorXd velocity = Eigen::Vector2d(.25, .4);
  std::vector<Eigen::MatrixXd> tasks;
  std::vector<Eigen::VectorXd> targets, biases, images;
  std::vector<ObjectiveSolveConfig> policies;
  Eigen::MatrixXd C = Eigen::Matrix2d::Identity();
  Eigen::VectorXd lower = Eigen::Vector2d(-1, -1);
  Eigen::VectorXd upper = Eigen::Vector2d(1, 1);
  size_t first_unoptimized = 1;
  bool enabled = true;

  Checkpoint() {
    Eigen::MatrixXd primary(1,2), secondary(1,2);
    primary << 1,0; secondary << 0,1;
    tasks = {primary, secondary};
    targets = {Eigen::VectorXd::Constant(1,.5), Eigen::VectorXd::Constant(1,.8)};
    biases = {Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1)};
    images = {Eigen::VectorXd::Constant(1,.25)};
    ObjectiveSolveConfig primary_policy, secondary_policy;
    primary_policy.solve_mode = TaskSolveMode::kScale;
    primary_policy.allow_min_error_fallback = false;
    secondary_policy.solve_mode = TaskSolveMode::kMinError;
    secondary_policy.allow_min_error_fallback = false;
    policies = {primary_policy, secondary_policy};
  }
  bool valid() const {
    return detail::CanRetainBestEffortLowerTaskCheckpoint(
        enabled, velocity, images, first_unoptimized, tasks, targets, biases,
        policies, C, lower, upper, 1e-10, 1e-10);
  }
};

TEST(BestEffortLowerCheckpoint, RetainsNonzeroScaleImageWithoutOptimizingSuffix) {
  Checkpoint p;
  ASSERT_TRUE(p.valid());
  EXPECT_DOUBLE_EQ((p.tasks[0]*p.velocity)(0), .5*p.targets[0](0));
  EXPECT_GT((p.tasks[1]*p.velocity-p.targets[1]).norm(), .1);
}
TEST(BestEffortLowerCheckpoint, RetainsZeroSatisfiedScaleImage) {
  Checkpoint p; p.velocity(0)=0; p.images[0](0)=0;
  EXPECT_TRUE(p.valid());
}
TEST(BestEffortLowerCheckpoint, RetainsCompletedMinimumErrorImage) {
  Checkpoint p; p.policies[0].solve_mode=TaskSolveMode::kMinError;
  EXPECT_TRUE(p.valid());
}
TEST(BestEffortLowerCheckpoint, RankDeficientSuffixDoesNotInvalidateFiniteWitness) {
  Checkpoint p;
  p.tasks[1] = Eigen::Matrix2d::Zero();
  p.tasks[1](0, 1) = 1.0;
  p.targets[1] = Eigen::Vector2d(0.8, 0.0);
  p.biases[1] = Eigen::Vector2d::Zero();
  VelocitySolverConfig config;
  double condition = 1.0;
  const auto inverse = linalg::ComputeRegularizedInverse(
      config.regularization_config, p.tasks[1], &condition, true);
  EXPECT_TRUE(inverse.allFinite());
  EXPECT_TRUE(std::isinf(condition));
  EXPECT_TRUE(p.valid());
  p.enabled = false;
  EXPECT_FALSE(p.valid());
}
TEST(BestEffortLowerCheckpoint, RequiresExplicitOptIn) {
  Checkpoint p; p.enabled=false; EXPECT_FALSE(p.valid());
}
TEST(BestEffortLowerCheckpoint, DoesNotRetainFailedPrimaryOrUninitializedSeed) {
  Checkpoint p; p.images.clear(); p.first_unoptimized=0;
  EXPECT_FALSE(p.valid());
}
TEST(BestEffortLowerCheckpoint, RequiresEveryCompletedImage) {
  Checkpoint p; p.images.clear(); EXPECT_FALSE(p.valid());
}
TEST(BestEffortLowerCheckpoint, DoesNotOmitStrictLowerScaleTask) {
  Checkpoint p; p.policies.back().solve_mode=TaskSolveMode::kScale;
  EXPECT_FALSE(p.valid());
}
TEST(BestEffortLowerCheckpoint, DoesNotOmitStrictLaterScaleTask) {
  Checkpoint p; p.tasks.push_back(p.tasks.back()); p.targets.push_back(p.targets.back());
  p.biases.push_back(p.biases.back()); p.policies.push_back(p.policies.front());
  EXPECT_FALSE(p.valid());
}
TEST(BestEffortLowerCheckpoint, RejectsOriginalHardRowViolation) {
  Checkpoint p; p.upper(1)=.3; EXPECT_FALSE(p.valid());
}
TEST(BestEffortLowerCheckpoint, RejectsChangedHigherImage) {
  Checkpoint p; p.images[0](0)+=1e-5; EXPECT_FALSE(p.valid());
}
TEST(BestEffortLowerCheckpoint, RejectsNonfiniteCheckpoint) {
  Checkpoint p; p.velocity(1)=std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(p.valid());
}
TEST(BestEffortLowerCheckpoint, RejectsOverflowedHardRowProduct) {
  Checkpoint p; p.C(1,1)=1e308; p.velocity(1)=2; p.upper(1)=1e308;
  EXPECT_FALSE(p.valid());
}
TEST(BestEffortLowerCheckpoint, RejectsOverflowedUnoptimizedTaskError) {
  Checkpoint p; p.tasks.back()(0,1)=1e308; EXPECT_FALSE(p.valid());
}
TEST(BestEffortLowerCheckpoint, RejectsOverflowedCompletedImageNorm) {
  Checkpoint p; p.tasks[0](0,0)=1e308; p.images[0](0)=.25e308;
  p.targets[0](0)=.25e308; EXPECT_FALSE(p.valid());
}
TEST(BestEffortLowerCheckpoint, ProtectsAllCompletedImages) {
  Checkpoint p; p.tasks.push_back(p.tasks.back()); p.targets.push_back(p.targets.back());
  p.biases.push_back(p.biases.back());p.policies.push_back(p.policies.back());
  p.images.push_back(Eigen::VectorXd::Constant(1,.4));p.first_unoptimized=2;
  ASSERT_TRUE(p.valid());p.images[1](0)-=1e-5;EXPECT_FALSE(p.valid());
}
} // namespace
} // namespace embodik::test
