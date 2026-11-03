#include "interval_scalar_kinematics.hpp"

#include <gtest/gtest.h>

namespace embodik::detail {
namespace {

Eigen::VectorXd vector(double value) { return Eigen::VectorXd::Constant(1, value); }

TEST(IntervalScalarKinematics, QuadraticInteriorExtremumAndOwnedState) {
  auto q = vector(0.0), v = vector(-1.0), a = vector(2.0);
  const QuadraticIntervalMotion motion(q, v, a, 1.0);
  q(0) = 1000.0;
  const auto range = motion.position_range(0, 0.0, 1.0);
  EXPECT_LE(range.lower(), -0.25);
  EXPECT_GE(range.upper(), 0.0);
  EXPECT_GT(range.lower(), -0.25 - 1e-12);
  EXPECT_LT(range.upper(), 1e-12);
  EXPECT_THROW(motion.position_range(1, 0.0, 1.0), std::domain_error);
  EXPECT_THROW(motion.position_range(0, -0.1, 1.0), std::domain_error);
}

TEST(IntervalScalarKinematics, PrismaticTransformIncludesInteriorTurnaround) {
  IntervalScalarJoint joint;
  joint.kind = IntervalJointKind::kPrismatic;
  joint.axis = Eigen::Vector3d::UnitX();
  joint.placement = IntervalTransform::fixed(Eigen::Matrix3d::Identity(), Eigen::Vector3d(2, 0, 0));
  const IntervalScalarKinematics model({joint});
  const QuadraticIntervalMotion motion(vector(0), vector(-1), vector(2), 1);
  const auto snapshot = model.poses(motion, 0, 1, {1});
  EXPECT_LE(snapshot.at(1).translation[0].lower(), 1.75);
  EXPECT_GE(snapshot.at(1).translation[0].upper(), 2.0);
}

TEST(IntervalScalarKinematics, CoordinateYawRetainsFixedVerticalProjection) {
  IntervalScalarJoint joint;
  joint.placement = IntervalTransform::fixed(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0.5));
  const IntervalScalarKinematics model({joint});
  const QuadraticIntervalMotion motion(vector(-1), vector(2), vector(0), 1);
  const auto snapshot = model.poses(motion, 0, 1, {1});
  EXPECT_TRUE(snapshot.at(1).translation[2].point(0.5));
  EXPECT_TRUE(snapshot.at(1).rotation[8].point(1.0));
  EXPECT_TRUE(snapshot.at(1).rotation[6].point(0.0));
  EXPECT_TRUE(snapshot.at(1).rotation[7].point(0.0));
}

TEST(IntervalScalarKinematics, UnrequestedBranchesRemainUnavailable) {
  IntervalScalarJoint first, second;
  first.kind = second.kind = IntervalJointKind::kPrismatic;
  const IntervalScalarKinematics model({first, second});
  const auto zero = Eigen::VectorXd::Zero(2).eval();
  const QuadraticIntervalMotion motion(zero, zero, zero, 1);
  const auto snapshot = model.poses(motion, 0, 1, {1});
  EXPECT_NO_THROW(snapshot.at(0));
  EXPECT_NO_THROW(snapshot.at(1));
  EXPECT_THROW(snapshot.at(2), std::domain_error);
  EXPECT_THROW(model.poses(motion, 0, 1, {3}), std::domain_error);
}

TEST(IntervalScalarKinematics, RejectsInvalidTreesAxesAndMotionShapes) {
  IntervalScalarJoint joint;
  joint.parent = 1;
  EXPECT_THROW((void)IntervalScalarKinematics({joint}), std::domain_error);
  joint.parent = 0;
  joint.axis = Eigen::Vector3d(0.5, 0.5, 0.0);
  EXPECT_THROW((void)IntervalScalarKinematics({joint}), std::domain_error);
  EXPECT_THROW((void)QuadraticIntervalMotion(vector(0), Eigen::VectorXd::Zero(2), vector(0), 1), std::domain_error);
}

TEST(IntervalScalarKinematics, StoredLinearPlacementIsNotAssumedOrthogonal) {
  IntervalScalarJoint joint;
  joint.kind = IntervalJointKind::kPrismatic;
  joint.axis = Eigen::Vector3d::UnitX();
  joint.placement = IntervalTransform::fixed(2.0 * Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  const IntervalScalarKinematics model({joint});
  const QuadraticIntervalMotion motion(vector(1), vector(0), vector(0), 1);
  const auto snapshot = model.poses(motion, 0, 1, {1});
  EXPECT_LE(snapshot.at(1).translation[0].lower(), 2.0);
  EXPECT_GE(snapshot.at(1).translation[0].upper(), 2.0);
}

} // namespace
} // namespace embodik::detail
