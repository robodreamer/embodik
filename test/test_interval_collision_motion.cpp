#include "interval_collision_motion.hpp"

#include <gtest/gtest.h>

namespace embodik::detail {
namespace {
#ifdef PINOCCHIO_WITH_HPP_FCL
using Collection = pinocchio::JointCollectionDefaultTpl<double, 0>;
struct Model {
  pinocchio::Model model;
  pinocchio::GeometryModel geometry;
  explicit Model(const Eigen::Vector3d &position) {
    model.addJoint(0, Collection::JointModelPX(), pinocchio::SE3::Identity(), "slide");
    auto placement = pinocchio::SE3::Identity();
    placement.translation() = position;
    geometry.addGeometryObject(pinocchio::GeometryObject("a", 1, placement, std::make_shared<coal::Sphere>(.25)));
    geometry.addGeometryObject(pinocchio::GeometryObject("b", 0, pinocchio::SE3::Identity(), std::make_shared<coal::Sphere>(.25)));
    geometry.addCollisionPair(pinocchio::CollisionPair(0, 1));
  }
};
Eigen::VectorXd scalar(double x) { return Eigen::VectorXd::Constant(1, x); }

TEST(IntervalCollisionMotion, ProvesEntireQuadraticWhenItsInteriorIsSafe) {
  const Model robot(Eigen::Vector3d(2, 0, 0));
  const auto result = prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(0), scalar(-1), scalar(2), 1, {1.2});
  EXPECT_TRUE(result.proved()) << result.message;
  EXPECT_EQ(result.visited_nodes, 1);
}
TEST(IntervalCollisionMotion, SafeEndpointsDoNotAdmitAnUnsafeInterior) {
  const Model robot(Eigen::Vector3d(2, 0, 0));
  const auto result = prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(0), scalar(-1), scalar(2), 1, {1.4});
  EXPECT_EQ(result.status, IntervalProofStatus::kUnproved);
  EXPECT_LE(result.visited_nodes, 32);
}
TEST(IntervalCollisionMotion, ReversedPairsUseCorrectHintOrientation) {
  const Model robot(Eigen::Vector3d(.5, .5, 0));
  for (const auto &pair : {std::make_pair(std::string("a"), std::string("b")),
                            std::make_pair(std::string("b"), std::string("a"))}) {
    const auto result = prove_collision_motion(robot.model, robot.geometry, {pair},
        scalar(0), scalar(0), scalar(0), 1, {.2});
    EXPECT_TRUE(result.proved()) << result.message;
  }
  // GJK distance/direction cannot admit a floor above actual clearance.
  const auto too_large = prove_collision_motion(robot.model, robot.geometry, {{"b", "a"}},
      scalar(0), scalar(0), scalar(0), 1, {.3});
  EXPECT_EQ(too_large.status, IntervalProofStatus::kUnproved);
}
TEST(IntervalCollisionMotion, RejectsMalformedFloorsAndState) {
  const Model robot(Eigen::Vector3d(2, 0, 0));
  for (double floor : {double(NAN), double(INFINITY), -1.0, 1e-10}) {
    const auto result = prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
        scalar(0), scalar(0), scalar(0), 1, {floor});
    EXPECT_EQ(result.status, IntervalProofStatus::kInvalidInput);
  }
  EXPECT_EQ(prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(0), Eigen::VectorXd::Zero(2), scalar(0), 1, {1.0}).status, IntervalProofStatus::kInvalidInput);
}

TEST(IntervalCollisionMotion, OptionalContainmentSharesTheCollisionTraversalAndBudget) {
  Model robot(Eigen::Vector3d(2, 0, 0));
  robot.model.inertias[1] = pinocchio::Inertia(1, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity());
  IntervalComContainmentPolicy com;
  com.halfspace_normals.resize(2, 2);
  com.halfspace_normals << 1, 0, -1, 0;
  com.halfspace_offsets = Eigen::Vector2d(.3, .3);
  const auto safe = prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(0), scalar(1), scalar(-2), 1, {1.2}, 1e-9, {}, &com);
  EXPECT_TRUE(safe.proved()) << safe.message;
  EXPECT_EQ(safe.visited_nodes, 1);
  EXPECT_EQ(safe.certified_intervals, 1);
  // Endpoint CoM is zero, but its quadratic interior reaches .25.
  com.halfspace_offsets(0) = .2;
  const auto unsafe = prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(0), scalar(1), scalar(-2), 1, {1.2}, 1e-9, {1, 24}, &com);
  EXPECT_EQ(unsafe.status, IntervalProofStatus::kUnproved);
  EXPECT_EQ(unsafe.visited_nodes, 1);
  const auto collision_only = prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(0), scalar(1), scalar(-2), 1, {1.2});
  EXPECT_TRUE(collision_only.proved());
}

TEST(IntervalCollisionMotion, ContainmentToleranceDoesNotAccumulateAcrossNodes) {
  Model robot(Eigen::Vector3d(2, 0, 0));
  robot.model.inertias[1] = pinocchio::Inertia(1, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity());
  IntervalComContainmentPolicy com;
  com.halfspace_normals = Eigen::Matrix<double, 1, 2>(1, 0);
  com.halfspace_offsets = scalar(.25);
  EXPECT_TRUE(prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(.25 + .5e-9), scalar(0), scalar(0), 1, {1.2}, 1e-9, {}, &com).proved());
  const auto rejected = prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(.25 + 2e-9), scalar(0), scalar(0), 1, {1.2}, 1e-9, {}, &com);
  EXPECT_EQ(rejected.status, IntervalProofStatus::kUnproved);
}

TEST(IntervalCollisionMotion, DefaultNullDoesNotRequireMassAndOptionalFailuresClose) {
  Model robot(Eigen::Vector3d(2, 0, 0));
  EXPECT_TRUE(prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(0), scalar(0), scalar(0), 1, {1.2}).proved());
  IntervalComContainmentPolicy com;
  com.halfspace_normals = Eigen::Matrix<double, 1, 2>(1, 0);
  com.halfspace_offsets = scalar(.25);
  EXPECT_EQ(prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(0), scalar(0), scalar(0), 1, {1.2}, 1e-9, {}, &com).status, IntervalProofStatus::kInvalidInput);
  robot.model.inertias[1] = pinocchio::Inertia(1e300, Eigen::Vector3d(1e300, 0, 0), Eigen::Matrix3d::Identity());
  const auto overflow = prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(0), scalar(0), scalar(0), 1, {1.2}, 1e-9, {}, &com);
  EXPECT_EQ(overflow.status, IntervalProofStatus::kUnproved);
  EXPECT_EQ(overflow.visited_nodes, 1);
}

TEST(IntervalCollisionMotion, ExplicitNullMatchesDefaultForSafeAndUnprovedPaths) {
  const Model robot(Eigen::Vector3d(2, 0, 0));
  for (double floor : {1.2, 1.4}) {
    const auto original = prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
        scalar(0), scalar(-1), scalar(2), 1, {floor});
    const auto explicit_null = prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
        scalar(0), scalar(-1), scalar(2), 1, {floor}, 1e-9, {}, nullptr);
    EXPECT_EQ(original.status, explicit_null.status);
    EXPECT_EQ(original.visited_nodes, explicit_null.visited_nodes);
    EXPECT_EQ(original.certified_intervals, explicit_null.certified_intervals);
    EXPECT_EQ(original.message, explicit_null.message);
  }
}

TEST(IntervalCollisionMotion, OptionalContainmentSnapshotsTheSameModelOnEveryInvocation) {
  Model robot(Eigen::Vector3d(2, 0, 0));
  robot.model.inertias[1] = pinocchio::Inertia(1, Eigen::Vector3d(.1, 0, 0), Eigen::Matrix3d::Identity());
  IntervalComContainmentPolicy com;
  com.halfspace_normals = Eigen::Matrix<double, 1, 2>(1, 0);
  com.halfspace_offsets = scalar(.25);
  EXPECT_TRUE(prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(0), scalar(0), scalar(0), 1, {1.2}, 1e-9, {}, &com).proved());
  robot.model.inertias[1].lever().x() = .4;
  EXPECT_EQ(prove_collision_motion(robot.model, robot.geometry, {{"a", "b"}},
      scalar(0), scalar(0), scalar(0), 1, {1.2}, 1e-9, {}, &com).status, IntervalProofStatus::kUnproved);
}
#else
TEST(IntervalCollisionMotion, RejectsMissingCollisionSupport) {
  const pinocchio::Model model;
  const pinocchio::GeometryModel geometry;
  EXPECT_FALSE(prove_collision_motion(model, geometry, {}, {}, {}, {}, 1, {}).proved());
}
#endif
} // namespace
} // namespace embodik::detail
