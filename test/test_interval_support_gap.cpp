#include "interval_primitive_projection.hpp"

#include <gtest/gtest.h>

namespace embodik::detail {
namespace {
IntervalAttachedPrimitive ball(int parent, double x, double radius = .25) {
  return {parent, {IntervalPrimitiveKind::kSphere, {radius}, 0},
          IntervalTransform::fixed(Eigen::Matrix3d::Identity(), Eigen::Vector3d(x, 0, 0))};
}
QuadraticIntervalMotion segment(const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                                const Eigen::VectorXd &a, double dt = 1.) {
  return {q, v, a, dt};
}

TEST(IntervalSupportGap, SharedQuadraticPrismaticMotionCancelsBeforeTimeRanging) {
  IntervalScalarJoint first, second;
  first.kind = second.kind = IntervalJointKind::kPrismatic;
  first.axis = second.axis = Eigen::Vector3d::UnitX();
  second.parent = 1;
  const IntervalPrimitiveProjection projection({first, second}, {ball(2, 1.), ball(0, 0.)});
  Eigen::Vector2d acceleration(1., -1.);
  const auto motion = segment(Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), acceleration);
  const double direct = projection.lower_bounds(motion, 0, 1, {{0, 1}})[0];
  const double tightened = projection.lower_bounds(motion, 0, 1, {{0, 1}}, {}, true)[0];
  EXPECT_LT(direct, .01);
  EXPECT_GT(tightened, .5 - 1e-12);
  EXPECT_LE(tightened, .5);
  EXPECT_DOUBLE_EQ(direct, projection.lower_bounds(motion, 0, 1, {{0, 1}}, {}, false)[0]);
}

TEST(IntervalSupportGap, LowerQuadraticIncludesItsUnsafeInteriorMinimum) {
  IntervalScalarJoint joint;
  joint.kind = IntervalJointKind::kPrismatic;
  joint.axis = Eigen::Vector3d::UnitX();
  const IntervalPrimitiveProjection projection({joint}, {ball(1, .6), ball(0, 0.)});
  const auto motion = segment(Eigen::VectorXd::Zero(1), Eigen::VectorXd::Constant(1, -1.),
                              Eigen::VectorXd::Constant(1, 2.));
  const double value = projection.lower_bounds(motion, 0, 1, {{0, 1}}, {}, true)[0];
  EXPECT_LE(value, -.15);
  EXPECT_GT(value, -.15 - 1e-12);
  EXPECT_GT(projection.lower_bounds(motion, 0, 0, {{0, 1}}, {}, true)[0], .09);
  EXPECT_GT(projection.lower_bounds(motion, 1, 1, {{0, 1}}, {}, true)[0], .09);
}

TEST(IntervalSupportGap, LiteralNonrigidSphereAndSweptRadiusRemainInSupport) {
  IntervalScalarJoint joint;
  auto a = ball(0, 2.), b = ball(0, 0.);
  a.placement.rotation[0] = Interval(2.);
  a.shape.swept_radius = .1;
  const IntervalPrimitiveProjection projection({joint}, {a, b});
  const auto motion = segment(Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1));
  const double value = projection.lower_bounds(motion, 0, 1, {{0, 1}}, {}, true)[0];
  EXPECT_LE(value, 1.05);
  EXPECT_GT(value, 1.05 - 1e-12);
}

TEST(IntervalSupportGap, WholeIntervalAbsoluteCrossingRetainsDirectFallback) {
  using namespace interval_projection_internal;
  ScalarMotion whole{Interval(-1, 1), Interval(1), Interval(0)};
  ScalarMotion anchor{Interval(-1), Interval(1), Interval(0)};
  EXPECT_THROW(signed_absolute(anchor, whole), std::domain_error);
  whole[0] = Interval(0, 1);
  anchor[0] = Interval(0);
  EXPECT_DOUBLE_EQ(signed_absolute(anchor, whole)[0].lower(), 0.);
  IntervalScalarJoint joint;
  IntervalAttachedPrimitive box{1, {IntervalPrimitiveKind::kBox, {.5, .5, .1}, 0},
      IntervalTransform::fixed(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 1))};
  const IntervalPrimitiveProjection projection({joint}, {box, ball(0, 0.)});
  const auto motion = segment(Eigen::VectorXd::Constant(1, -.5), Eigen::VectorXd::Ones(1), Eigen::VectorXd::Zero(1));
  const double direct = projection.lower_bounds(motion, 0, 1, {{0, 1}})[0];
  const double value = projection.lower_bounds(motion, 0, 1, {{0, 1}}, {}, true)[0];
  EXPECT_GE(value, direct);
  EXPECT_LE(value, .65);
}

TEST(IntervalSupportGap, CylinderAxisAlignmentAndZeroDimensionsRemainSafe) {
  IntervalScalarJoint joint;
  IntervalAttachedPrimitive cylinder{1, {IntervalPrimitiveKind::kCylinder, {.1, .2}, 0},
      IntervalTransform::fixed(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 1))};
  const IntervalPrimitiveProjection projection({joint}, {cylinder, ball(0, 0.)});
  const auto motion = segment(Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1), Eigen::VectorXd::Zero(1));
  const double direct = projection.lower_bounds(motion, 0, 1, {{0, 1}})[0];
  const double value = projection.lower_bounds(motion, 0, 1, {{0, 1}}, {}, true)[0];
  EXPECT_GE(value, direct);
  EXPECT_LE(value, .55);
  auto zero = ball(0, 2., 0.);
  zero.placement.rotation = {};
  const IntervalPrimitiveProjection points({joint}, {zero, ball(0, 0., 0.)});
  EXPECT_GE(points.lower_bounds(motion, 0, 1, {{0, 1}}, {}, true)[0], 1.99);
}

TEST(IntervalSupportGap, ChangingPositiveCylinderRadialNormIncludesInteriorMinimum) {
  IntervalScalarJoint joint;
  joint.axis = Eigen::Vector3d::UnitY();
  auto fixed = ball(0, 0.);
  fixed.placement.translation[2] = Interval(1.);
  IntervalAttachedPrimitive cylinder{1, {IntervalPrimitiveKind::kCylinder, {.1, .2}, 0},
      IntervalTransform::identity()};
  const IntervalPrimitiveProjection projection({joint}, {fixed, cylinder});
  const auto motion = segment(Eigen::VectorXd::Constant(1, .2), Eigen::VectorXd::Constant(1, .5),
                              Eigen::VectorXd::Zero(1));
  const double direct = projection.lower_bounds(motion, 0, 1, {{0, 1}})[0];
  const double value = projection.lower_bounds(motion, 0, 1, {{0, 1}}, {}, true)[0];
  const double minimum = .75-std::sqrt(.05);
  EXPECT_LE(value, minimum);
  // Broad support-norm jets can be less useful than direct pose intervals.
  // Opt-in tightening promises a safe maximum, not universal improvement.
  EXPECT_GE(value, direct);
  EXPECT_GT(.75-.1*std::sin(.2)-.2*std::cos(.2), minimum);
  EXPECT_GT(.75-.1*std::sin(.7)-.2*std::cos(.7), minimum);
}

TEST(IntervalSupportGap, NarrowPositiveCylinderNormTightensWithoutMissingInteriorMinimum) {
  IntervalScalarJoint joint;
  joint.axis = Eigen::Vector3d::UnitY();
  auto fixed = ball(0, 0.);
  fixed.placement.translation[2] = Interval(1.);
  IntervalAttachedPrimitive cylinder{1, {IntervalPrimitiveKind::kCylinder, {.1, .2}, 0},
      IntervalTransform::identity()};
  const IntervalPrimitiveProjection projection({joint}, {fixed, cylinder});
  const auto motion = segment(Eigen::VectorXd::Constant(1, .4), Eigen::VectorXd::Constant(1, .1),
                              Eigen::VectorXd::Zero(1));
  const double direct = projection.lower_bounds(motion, 0, 1, {{0, 1}})[0];
  const double value = projection.lower_bounds(motion, 0, 1, {{0, 1}}, {}, true)[0];
  EXPECT_LE(value, .75-std::sqrt(.05));
  EXPECT_GT(value, .75-std::sqrt(.05)-.005);
  EXPECT_GT(value, direct);
}

TEST(IntervalSupportGap, PositiveNormRecurrenceEnclosesChangingFirstAndSecondDerivatives) {
  using namespace interval_projection_internal;
  DirectionMotion direction;
  direction[0] = {Interval(std::sin(.4)), Interval(.1*std::cos(.4)), Interval(-.01*std::sin(.4))};
  const auto value = norm_motion(direction, 2);
  for (int order = 0; order < 3; ++order) {
    EXPECT_LE(value[order].lower(), direction[0][order].lower());
    EXPECT_GE(value[order].upper(), direction[0][order].upper());
  }
}

TEST(IntervalSupportGap, NonunitHintsNegativeGapsAndExtremeScalesKeepDirectGuarantee) {
  IntervalScalarJoint joint;
  auto a = ball(0, .1), b = ball(0, 0.);
  a.placement.translation[1] = Interval(.1);
  const IntervalPrimitiveProjection projection({joint}, {a, b});
  const auto motion = segment(Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1));
  for (double scale : {1e-150, .1, 7., 1e150}) {
    const std::vector<Eigen::Vector3d> hints{scale * Eigen::Vector3d(1, 1, 0)};
    const double direct = projection.lower_bounds(motion, 0, 1, {{0, 1}}, hints)[0];
    const double value = projection.lower_bounds(motion, 0, 1, {{0, 1}}, hints, true)[0];
    EXPECT_GE(value, direct);
    EXPECT_LE(value, std::sqrt(.02) - .5);
  }
}

TEST(IntervalSupportGap, OptionalDerivativeOverflowDoesNotDiscardFiniteDirectBound) {
  IntervalScalarJoint joint;
  const IntervalPrimitiveProjection projection({joint}, {ball(1, 2.), ball(0, 0.)});
  const auto motion = segment(Eigen::VectorXd::Zero(1), Eigen::VectorXd::Constant(1, 1e100),
                              Eigen::VectorXd::Zero(1), 1e-100);
  const double direct = projection.lower_bounds(motion, 0, 1e-100, {{0, 1}})[0];
  EXPECT_DOUBLE_EQ(direct, projection.lower_bounds(motion, 0, 1e-100, {{0, 1}}, {}, true)[0]);
}

TEST(IntervalSupportGap, MalformedOriginalInputsAreNotHiddenByOptionalFallback) {
  IntervalScalarJoint joint;
  const IntervalPrimitiveProjection projection({joint}, {ball(0, 2.), ball(0, 0.)});
  const auto motion = segment(Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1));
  EXPECT_THROW(projection.lower_bounds(motion, 0, 1, {{0, 1}}, {Eigen::Vector3d::Zero()}, true), std::domain_error);
  EXPECT_THROW(projection.lower_bounds(motion, 0, 1, {{0, 3}}, {}, true), std::domain_error);
}
} // namespace
} // namespace embodik::detail
