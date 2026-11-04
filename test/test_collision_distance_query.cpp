#include "collision_distance_query.hpp"
#include <coal/shape/geometric_shapes.h>
#include <gtest/gtest.h>
#include <limits>
#include <memory>

namespace embodik::test {
namespace {

pinocchio::GeometryObject object(
    const std::shared_ptr<coal::CollisionGeometry> &shape) {
  shape->computeLocalAABB();
  return pinocchio::GeometryObject("shape", 0, 0,
                                  pinocchio::SE3::Identity(), shape);
}

TEST(CollisionDistanceQueryTest, FiniteEvidenceMustRespectPositiveSeparation) {
  EXPECT_FALSE(detail::collision_distance_evidence_is_valid(0.005, 0.08));
  EXPECT_FALSE(detail::collision_distance_evidence_is_valid(-0.01, 0.08));
  EXPECT_TRUE(detail::collision_distance_evidence_is_valid(0.18, 0.08));
  EXPECT_TRUE(detail::collision_distance_evidence_is_valid(0.08, 0.08));
  EXPECT_FALSE(detail::collision_distance_evidence_is_valid(
      std::numeric_limits<double>::max(), 0.08));
}

TEST(CollisionDistanceQueryTest, OverlapOrUnavailableBoundsPreservePenetration) {
  EXPECT_TRUE(detail::collision_distance_evidence_is_valid(-0.02, 0.0));
  EXPECT_TRUE(detail::collision_distance_evidence_is_valid(
      -0.02, std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(detail::collision_distance_evidence_is_valid(
      -std::numeric_limits<double>::max(), 0.0));
}

TEST(CollisionDistanceQueryTest, EvidenceCheckRespectsQueryPrecisionOnly) {
  // A separated approximate query may lie slightly below an enclosing bound.
  // This allowance does not modify the reported distance or physical floors.
  EXPECT_TRUE(detail::collision_distance_evidence_is_valid(
      0.12 - 5e-12, 0.12, 1e-6));
  EXPECT_FALSE(detail::collision_distance_evidence_is_valid(0.005, 0.08, 1e-6));
  EXPECT_FALSE(detail::collision_distance_evidence_is_valid(0.12, 0.12, -1.0));
}

TEST(CollisionDistanceQueryTest, RotatedBoxesProvideConservativeWorldBound) {
  const auto box = object(std::make_shared<coal::Box>(2.0, 2.0, 2.0));
  const pinocchio::SE3 first(
      Eigen::AngleAxisd(std::acos(-1.0) / 4.0, Eigen::Vector3d::UnitZ())
          .toRotationMatrix(), Eigen::Vector3d::Zero());
  const pinocchio::SE3 second(Eigen::Matrix3d::Identity(),
                             Eigen::Vector3d(3.0, 0.0, 0.0));
  const auto bound = detail::collision_aabb_separation_lower_bound(
      box, first, box, second);
  ASSERT_TRUE(bound.has_value());
  const double exact_box_gap = 2.0 - std::sqrt(2.0);
  EXPECT_LE(*bound, exact_box_gap);
  EXPECT_NEAR(*bound, exact_box_gap, 1e-12);
  const auto overlap = detail::collision_aabb_separation_lower_bound(
      box, first, box, first);
  ASSERT_TRUE(overlap.has_value());
  EXPECT_DOUBLE_EQ(*overlap, 0.0);
}

TEST(CollisionDistanceQueryTest, SweptRadiusCannotOverstateSeparation) {
  auto shape = std::make_shared<coal::Box>(2.0, 2.0, 2.0);
  const auto box = object(shape);
  // Even a local box computed before a swept-radius edit must not provide an
  // overly large separation bound.
  shape->setSweptSphereRadius(0.25);
  const auto bound = detail::collision_aabb_separation_lower_bound(
      box, pinocchio::SE3::Identity(), box,
      pinocchio::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(3.0, 0.0, 0.0)));
  ASSERT_TRUE(bound.has_value());
  EXPECT_LE(*bound, 0.5);
  EXPECT_NEAR(*bound, 0.5, 1e-12);
  // Some Coal shapes already include the radius when refreshing their AABB.
  // Additional padding may weaken this auxiliary bound, never overstate it.
  shape->computeLocalAABB();
  const auto refreshed = detail::collision_aabb_separation_lower_bound(
      box, pinocchio::SE3::Identity(), box,
      pinocchio::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(3.0, 0.0, 0.0)));
  ASSERT_TRUE(refreshed.has_value());
  EXPECT_GE(*refreshed, 0.0);
  EXPECT_LE(*refreshed, 0.5);
}

TEST(CollisionDistanceQueryTest, InfiniteOrUninitializedBoundsAreUnavailable) {
  const auto plane = object(std::make_shared<coal::Plane>(
      Eigen::Vector3d::UnitZ(), 0.0));
  const auto sphere = object(std::make_shared<coal::Sphere>(1.0));
  EXPECT_FALSE(detail::collision_aabb_separation_lower_bound(
      plane, pinocchio::SE3::Identity(), sphere, pinocchio::SE3::Identity()));
  auto uninitialized = std::make_shared<coal::Sphere>(1.0);
  uninitialized->aabb_radius = -1.0;
  const pinocchio::GeometryObject missing("missing", 0, 0,
      pinocchio::SE3::Identity(), uninitialized);
  EXPECT_FALSE(detail::collision_aabb_separation_lower_bound(
      missing, pinocchio::SE3::Identity(), sphere, pinocchio::SE3::Identity()));
}

TEST(CollisionDistanceQueryTest, SharedQueryKeepsSignedContactAndCallerSettings) {
  pinocchio::GeometryModel model;
  model.addGeometryObject(object(std::make_shared<coal::Sphere>(1.0)));
  auto second = object(std::make_shared<coal::Sphere>(1.0));
  second.name = "second";
  model.addGeometryObject(second);
  model.addCollisionPair(pinocchio::CollisionPair(0, 1));
  pinocchio::GeometryData data(model);
  data.oMg[0] = pinocchio::SE3::Identity();
  data.oMg[1] = pinocchio::SE3(Eigen::Matrix3d::Identity(),
                              Eigen::Vector3d(1.5, 0.0, 0.0));
  const auto request_before = data.distanceRequests[0];
  detail::compute_policy_collision_distance(model, data, 0, true);
  EXPECT_NEAR(data.distanceResults[0].min_distance, -0.5, 1e-12);
  EXPECT_TRUE(data.distanceRequests[0] == request_before);
}

} // namespace
} // namespace embodik::test
