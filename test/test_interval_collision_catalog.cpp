#include "interval_collision_catalog.hpp"

#include <gtest/gtest.h>

namespace embodik::detail {
namespace {
#ifdef PINOCCHIO_WITH_HPP_FCL
using Collection = pinocchio::JointCollectionDefaultTpl<double, 0>;

struct Fixture {
  pinocchio::Model model;
  pinocchio::GeometryModel geometry;
  Fixture() {
    model.addJoint(0, Collection::JointModelRZ(), pinocchio::SE3::Identity(), "yaw");
    geometry.addGeometryObject(pinocchio::GeometryObject("box", 0, pinocchio::SE3::Identity(), std::make_shared<coal::Box>(2, 4, 6)));
    geometry.addGeometryObject(pinocchio::GeometryObject("sphere", 1, pinocchio::SE3::Identity(), std::make_shared<coal::Sphere>(.25)));
    geometry.addGeometryObject(pinocchio::GeometryObject("cylinder", 1, pinocchio::SE3::Identity(), std::make_shared<coal::Cylinder>(.5, 2)));
    geometry.addCollisionPair(pinocchio::CollisionPair(0, 1));
  }
};

TEST(IntervalCollisionCatalog, CopiesExactNativeDimensionsAndOrdering) {
  Fixture fixture;
  const auto catalog = make_interval_collision_catalog(fixture.model, fixture.geometry, {{"sphere", "box"}});
  ASSERT_EQ(catalog.joints.size(), 1);
  EXPECT_EQ(catalog.joint_names, std::vector<std::string>({"yaw"}));
  EXPECT_EQ(catalog.joints[0].kind, IntervalJointKind::kRevolute);
  EXPECT_EQ(catalog.joints[0].axis, Eigen::Vector3d::UnitZ());
  EXPECT_EQ(catalog.pairs, (std::vector<std::pair<int, int>>{{1, 0}}));
  EXPECT_EQ(catalog.geometries[0].shape.dimensions, std::vector<double>({1, 2, 3}));
  EXPECT_EQ(catalog.geometries[1].shape.dimensions, std::vector<double>({.25}));
  EXPECT_EQ(catalog.geometries[2].shape.dimensions, std::vector<double>({.5, 1}));
}

TEST(IntervalCollisionCatalog, OwnsStoredPlacementsAndSweptRadius) {
  Fixture fixture;
  fixture.model.jointPlacements[1].translation().x() = .125;
  fixture.geometry.geometryObjects[0].placement.rotation()(0, 0) = 2;
  auto *box = dynamic_cast<coal::Box *>(fixture.geometry.geometryObjects[0].geometry.get());
  box->setSweptSphereRadius(.01);
  const auto catalog = make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}});
  fixture.model.jointPlacements[1].translation().x() = 99;
  fixture.geometry.geometryObjects[0].placement.rotation()(0, 0) = 99;
  box->halfSide.x() = 99;
  EXPECT_TRUE(catalog.joints[0].placement.translation[0].point(.125));
  EXPECT_TRUE(catalog.geometries[0].placement.rotation[0].point(2));
  EXPECT_DOUBLE_EQ(catalog.geometries[0].shape.dimensions[0], 1);
  EXPECT_DOUBLE_EQ(catalog.geometries[0].shape.swept_radius, .01);
}

TEST(IntervalCollisionCatalog, RetainsExactNegativeCoordinateAxis) {
  Fixture fixture;
  fixture.model = pinocchio::Model();
  fixture.model.addJoint(0, Collection::JointModelRevoluteUnaligned(Eigen::Vector3d(0, 0, -1)), pinocchio::SE3::Identity(), "negative_yaw");
  const auto catalog = make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}});
  EXPECT_EQ(catalog.joints[0].axis, -Eigen::Vector3d::UnitZ());
}

TEST(IntervalCollisionCatalog, RejectsNonCoordinateRotationAndFloatingBase) {
  Fixture fixture;
  fixture.model = pinocchio::Model();
  fixture.model.addJoint(0, Collection::JointModelRevoluteUnaligned(Eigen::Vector3d(.6, .8, 0)), pinocchio::SE3::Identity(), "unaligned");
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}}), std::domain_error);
  fixture.model = pinocchio::Model();
  fixture.model.addJoint(0, Collection::JointModelFreeFlyer(), pinocchio::SE3::Identity(), "floating");
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}}), std::domain_error);
}

TEST(IntervalCollisionCatalog, RejectsAbsentDuplicateAndMalformedPairCatalogs) {
  Fixture fixture;
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {}), std::domain_error);
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "cylinder"}}), std::domain_error);
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "box"}}), std::domain_error);
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "missing"}}), std::domain_error);
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}, {"sphere", "box"}}), std::domain_error);
  fixture.geometry.geometryObjects[1].name = "box";
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}}), std::domain_error);
}

TEST(IntervalCollisionCatalog, RejectsUnsupportedGeometryAndBrokenCoordinateMap) {
  Fixture fixture;
  fixture.model.joints[1].setIndexes(1, 1, 0);
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}}), std::domain_error);
  fixture.model.joints[1].setIndexes(1, 0, 0);
  fixture.geometry.geometryObjects[0].geometry = std::make_shared<coal::Capsule>(.1, .2);
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}}), std::domain_error);
  fixture.geometry.geometryObjects[0].geometry.reset();
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}}), std::domain_error);
}

TEST(IntervalCollisionCatalog, RejectsOverflowingMalformedDimensionsWithoutArithmeticOverflow) {
  Fixture fixture;
  fixture.model.nq = fixture.model.nv = std::numeric_limits<int>::max();
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}}), std::domain_error);
}

TEST(IntervalCollisionCatalog, RejectsMalformedNativeTopologyAndDuplicateSourcePairs) {
  Fixture fixture;
  fixture.model.parents[1] = std::numeric_limits<pinocchio::JointIndex>::max();
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}}), std::domain_error);
  fixture.model.parents[1] = 0;
  fixture.geometry.collisionPairs.push_back(pinocchio::CollisionPair(0, 1));
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}}), std::domain_error);
  fixture.geometry.collisionPairs.pop_back();
  fixture.model.names.pop_back();
  EXPECT_THROW(make_interval_collision_catalog(fixture.model, fixture.geometry, {{"box", "sphere"}}), std::domain_error);
}

#else
TEST(IntervalCollisionCatalog, RejectsMissingCollisionSupport) {
  const pinocchio::Model model;
  const pinocchio::GeometryModel geometry;
  EXPECT_THROW(make_interval_collision_catalog(model, geometry, {}), std::domain_error);
}
#endif

} // namespace
} // namespace embodik::detail
