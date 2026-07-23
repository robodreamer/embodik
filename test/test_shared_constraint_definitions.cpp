#include <embodik/types.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace embodik {
namespace {

TEST(SharedConstraintDefinitionsTest, ContactTypeLivesInSharedTypes) {
  static_assert(std::is_enum_v<ContactType>);
  EXPECT_NE(ContactType::kPointContact, ContactType::kRigidContact);
}

TEST(SharedConstraintDefinitionsTest, GeometryDefinitionsHaveStableDefaults) {
  CollisionConstraintDefinition collision;
  EXPECT_DOUBLE_EQ(collision.min_distance, 0.05);
  EXPECT_TRUE(collision.include_pairs.empty());
  EXPECT_TRUE(collision.exclude_pairs.empty());
  EXPECT_TRUE(collision.pair_minimum_distances.empty());

  ComSupportPolygonConstraintDefinition com;
  EXPECT_EQ(com.frame_name, "world");
  EXPECT_DOUBLE_EQ(com.margin, 0.0);
  EXPECT_DOUBLE_EQ(com.proximity_fraction, 0.0);
  EXPECT_EQ(com.support_polygon.rows(), 0);
  EXPECT_EQ(com.support_polygon.cols(), 0);

  RelativePoseConstraintDefinition relative;
  EXPECT_TRUE(relative.lower_bounds.isZero());
  EXPECT_TRUE(relative.upper_bounds.isZero());
  EXPECT_TRUE(relative.axis_mask.isApprox(Eigen::Matrix<double, 6, 1>::Ones()));

  TightPointConstraintDefinition point;
  EXPECT_DOUBLE_EQ(point.position_epsilon, 1e-5);
  EXPECT_TRUE(point.target_point.isZero());
  EXPECT_TRUE(point.axis_mask.isApprox(Eigen::Vector3d::Ones()));

  TightFramePoseConstraintDefinition frame;
  EXPECT_DOUBLE_EQ(frame.position_epsilon, 1e-5);
  EXPECT_DOUBLE_EQ(frame.orientation_epsilon, 1e-4);
  EXPECT_TRUE(frame.target_pose.isApprox(Eigen::Matrix4d::Identity()));
  EXPECT_TRUE(frame.axis_mask.isApprox(Eigen::Matrix<double, 6, 1>::Ones()));
}

} // namespace
} // namespace embodik
