#include <embodik/ik_baseline.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace embodik::test {
namespace {

TEST(ProjectedInverseCacheTest, ExactHitRestoresInverseAndInfiniteCondition) {
  detail::ProjectedInverseCache cache;
  RegularizedInverseConfig config;
  const Eigen::MatrixXd input = Eigen::MatrixXd::Identity(2, 2);
  const Eigen::MatrixXd inverse = 3.0 * input;
  cache.Insert(config, input, true, inverse,
               std::numeric_limits<double>::infinity());
  double condition = 0;
  const auto *found = cache.Find(config, input, true, &condition);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ((*found - inverse).norm(), 0);
  EXPECT_TRUE(std::isinf(condition));
  detail::ProjectedInverseCache next_invocation;
  EXPECT_EQ(next_invocation.Find(config, input, true, nullptr), nullptr);
}

TEST(ProjectedInverseCacheTest, MatrixBitsShapeAndConfigurationArePartOfKey) {
  detail::ProjectedInverseCache cache;
  RegularizedInverseConfig config;
  const Eigen::MatrixXd input = Eigen::MatrixXd::Identity(2, 2);
  cache.Insert(config, input, true, input, 1);
  Eigen::MatrixXd changed = input;
  changed(0, 0) = std::nextafter(1.0, 2.0);
  EXPECT_EQ(cache.Find(config, changed, true, nullptr), nullptr);
  changed = input;
  changed(0, 1) = -0.0;
  EXPECT_EQ(cache.Find(config, changed, true, nullptr), nullptr);
  EXPECT_EQ(cache.Find(config, Eigen::MatrixXd::Ones(1, 4), true, nullptr),
            nullptr);
  auto changed_config = config;
  changed_config.epsilon *= 2;
  EXPECT_EQ(cache.Find(changed_config, input, true, nullptr), nullptr);
  changed_config = config;
  changed_config.regularization_factor += 1;
  EXPECT_EQ(cache.Find(changed_config, input, true, nullptr), nullptr);
  EXPECT_EQ(cache.Find(config, input, false, nullptr), nullptr);
}

TEST(ProjectedInverseCacheTest, ConfigurationSignedZeroRemainsDistinct) {
  detail::ProjectedInverseCache cache;
  RegularizedInverseConfig config;
  config.regularization_factor = 0.0;
  const Eigen::MatrixXd input = Eigen::MatrixXd::Identity(2, 2);
  cache.Insert(config, input, true, input, 1);
  config.regularization_factor = -0.0;
  EXPECT_EQ(cache.Find(config, input, true, nullptr), nullptr);
}

TEST(ProjectedInverseCacheTest, CopiesInputsAndEvictsOldestAtCapacity) {
  detail::ProjectedInverseCache cache;
  RegularizedInverseConfig config;
  Eigen::MatrixXd input = Eigen::MatrixXd::Ones(1, 1);
  cache.Insert(config, input, true, input, 1);
  input(0, 0) = 2;
  EXPECT_NE(cache.Find(config, Eigen::MatrixXd::Ones(1, 1), true, nullptr),
            nullptr);
  for (int value = 2; value <= 9; ++value) {
    input(0, 0) = value;
    cache.Insert(config, input, true, input, value);
  }
  EXPECT_EQ(cache.Find(config, Eigen::MatrixXd::Ones(1, 1), true, nullptr),
            nullptr);
  EXPECT_NE(cache.Find(config, input, true, nullptr), nullptr);
}

TEST(ProjectedInverseCacheTest, NonfiniteAndOversizedEntriesAreNotStored) {
  RegularizedInverseConfig config;
  const Eigen::MatrixXd input = Eigen::MatrixXd::Identity(2, 2);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  detail::ProjectedInverseCache cache;
  cache.Insert(config, input, true, Eigen::MatrixXd::Constant(2, 2, nan), 1);
  EXPECT_EQ(cache.Find(config, input, true, nullptr), nullptr);
  cache.Insert(config, input, true, input, nan);
  EXPECT_EQ(cache.Find(config, input, true, nullptr), nullptr);
  const Eigen::MatrixXd large = Eigen::MatrixXd::Identity(65, 65);
  cache.Insert(config, large, true, large, 1);
  EXPECT_EQ(cache.Find(config, large, true, nullptr), nullptr);
  auto invalid_config = config;
  invalid_config.epsilon = nan;
  cache.Insert(invalid_config, input, true, input, 1);
  EXPECT_EQ(cache.Find(invalid_config, input, true, nullptr), nullptr);
  const Eigen::MatrixXd nonfinite = Eigen::MatrixXd::Constant(2, 2, nan);
  cache.Insert(config, nonfinite, true, input, 1);
  EXPECT_EQ(cache.Find(config, nonfinite, true, nullptr), nullptr);
}

TEST(ProjectedInverseCacheTest, EmptyMatrixComparisonDoesNotReadNullStorage) {
  detail::ProjectedInverseCache cache;
  RegularizedInverseConfig config;
  const Eigen::MatrixXd empty(0, 2), inverse(2, 0);
  cache.Insert(config, empty, true, inverse, 1);
  EXPECT_NE(cache.Find(config, empty, true, nullptr), nullptr);
}

}  // namespace
}  // namespace embodik::test
