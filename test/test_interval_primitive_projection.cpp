#include "interval_primitive_projection.hpp"

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace embodik::detail {
namespace {

IntervalAttachedPrimitive sphere(int parent, double x, double radius = 0.25) {
  return {parent, {IntervalPrimitiveKind::kSphere, {radius}, 0.0},
          IntervalTransform::fixed(Eigen::Matrix3d::Identity(), Eigen::Vector3d(x, 0, 0))};
}
QuadraticIntervalMotion motion(double q = 0, double v = 0, double a = 0) {
  return {Eigen::VectorXd::Constant(1, q), Eigen::VectorXd::Constant(1, v),
          Eigen::VectorXd::Constant(1, a), 1.0};
}

TEST(IntervalPrimitiveProjection, SupportIncludesRoundingAndSweptRadius) {
  const std::array<Interval, 3> direction{Interval(-3, 2), Interval(-4, 3), Interval(-2, 1)};
  const double norm = std::sqrt(29.0);
  const IntervalPrimitive box{IntervalPrimitiveKind::kBox, {1, 2, 3}, .5};
  EXPECT_GE(box.support_upper(direction), 17 + .5 * norm);
  const IntervalPrimitive ball{IntervalPrimitiveKind::kSphere, {2}, .5};
  EXPECT_GE(ball.support_upper(direction), 2.5 * norm);
  const IntervalPrimitive cylinder{IntervalPrimitiveKind::kCylinder, {2, 3}, .5};
  EXPECT_GE(cylinder.support_upper(direction), 16 + .5 * norm);
  const std::array<Interval, 3> zero{Interval(0), Interval(0), Interval(0)};
  EXPECT_DOUBLE_EQ(box.support_upper(zero), 0.0);
}

TEST(IntervalPrimitiveProjection, BoundsMotionInteriorNotOnlyEndpoints) {
  IntervalScalarJoint joint;
  joint.kind = IntervalJointKind::kPrismatic;
  joint.axis = Eigen::Vector3d::UnitX();
  const IntervalPrimitiveProjection model({joint}, {sphere(1, 2), sphere(0, 0)});
  const auto range = model.lower_bounds(motion(0, -1, 2), 0, 1, {{0, 1}, {1, 0}});
  ASSERT_EQ(range.size(), 2);
  EXPECT_LE(range[0], 1.25);
  EXPECT_GT(range[0], 1.25 - 1e-12);
  EXPECT_DOUBLE_EQ(range[0], range[1]);
  EXPECT_GT(model.lower_bounds(motion(0, -1, 2), 0, 0, {{0, 1}})[0], 1.49);
}

TEST(IntervalPrimitiveProjection, OwnsInputsAndUsesLiteralStoredMatrix) {
  IntervalScalarJoint joint;
  auto a = sphere(0, 2), b = sphere(0, 0);
  a.placement.rotation[0] = Interval(2.0);
  std::vector<IntervalAttachedPrimitive> geometries{a, b};
  const IntervalPrimitiveProjection model({joint}, geometries);
  geometries[0].placement.translation[0] = Interval(100);
  const double value = model.lower_bounds(motion(), 0, 1, {{0, 1}})[0];
  EXPECT_LE(value, 1.25);
  EXPECT_GT(value, 1.25 - 1e-12);
}

TEST(IntervalPrimitiveProjection, YawMotionRetainsVerticalBoxCylinderSeparation) {
  IntervalScalarJoint joint;
  IntervalAttachedPrimitive box{0, {IntervalPrimitiveKind::kBox, {.5, .5, .1}, 0},
      IntervalTransform::fixed(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 1))};
  IntervalAttachedPrimitive cylinder{1, {IntervalPrimitiveKind::kCylinder, {.1, .2}, 0},
      IntervalTransform::identity()};
  const IntervalPrimitiveProjection model({joint}, {box, cylinder});
  const double value = model.lower_bounds(motion(-1, 2), 0, 1, {{0, 1}})[0];
  EXPECT_LE(value, .7);
  EXPECT_GT(value, .7 - 1e-12);
}

#if defined(__GLIBC__) && defined(__USE_GNU)
TEST(IntervalPrimitiveProjection, InvalidMetadataDoesNotLeakCallerTraps) {
  EXPECT_EXIT(([] {
    const std::uint64_t bits = UINT64_C(0x7ff0000000000001);
    double signaling;
    std::memcpy(&signaling, &bits, sizeof(signaling));
    feenableexcept(FE_INVALID);
    bool shape_rejected = false, tree_rejected = false;
    try {
      IntervalPrimitive shape{IntervalPrimitiveKind::kSphere, {signaling}, 0};
      shape.validate();
    } catch (const std::domain_error &) { shape_rejected = true; }
    try {
      IntervalScalarJoint joint;
      joint.axis(0) = signaling;
      (void)IntervalScalarKinematics({joint});
    } catch (const std::domain_error &) { tree_rejected = true; }
    std::_Exit(shape_rejected && tree_rejected && (fegetexcept() & FE_INVALID) ? 0 : 1);
  }()), ::testing::ExitedWithCode(0), "");
}
#endif

TEST(IntervalPrimitiveProjection, ProposedDiagonalDirectionProvesMoreThanWorldAxes) {
  IntervalScalarJoint joint;
  auto a = sphere(0, 1), b = sphere(0, 0);
  a.placement.translation[1] = Interval(1);
  const IntervalPrimitiveProjection model({joint}, {a, b});
  const auto baseline = model.lower_bounds(motion(), 0, 1, {{0, 1}})[0];
  const auto diagonal = model.lower_bounds(motion(), 0, 1, {{0, 1}}, {Eigen::Vector3d(1, 1, 0)})[0];
  const auto scaled = model.lower_bounds(motion(), 0, 1, {{0, 1}}, {Eigen::Vector3d(7, 7, 0)})[0];
  EXPECT_LT(baseline, .51);
  EXPECT_LE(diagonal, std::sqrt(2.) - .5);
  EXPECT_GT(diagonal, .9142);
  EXPECT_NEAR(diagonal, scaled, 1e-12);
  // An adversarial hint cannot weaken the independently valid axis bounds.
  EXPECT_DOUBLE_EQ(model.lower_bounds(motion(), 0, 1, {{0, 1}}, {Eigen::Vector3d(-1, -1, 0)})[0], baseline);
  EXPECT_THROW(model.lower_bounds(motion(), 0, 1, {{0, 1}}, {Eigen::Vector3d::Zero()}), std::domain_error);
  EXPECT_THROW(model.lower_bounds(motion(), 0, 1, {{0, 1}}, {Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY()}), std::domain_error);
}

TEST(IntervalPrimitiveProjection, ProposedDirectionBoundsNegativeGapAndExtremeFiniteScales) {
  IntervalScalarJoint joint;
  auto a = sphere(0, .1), b = sphere(0, 0);
  a.placement.translation[1] = Interval(.1);
  const IntervalPrimitiveProjection overlap({joint}, {a, b});
  const double baseline = overlap.lower_bounds(motion(), 0, 1, {{0, 1}})[0];
  const double diagonal = overlap.lower_bounds(motion(), 0, 1, {{0, 1}}, {Eigen::Vector3d(1, 1, 0)})[0];
  EXPECT_GT(diagonal, baseline);
  EXPECT_GT(diagonal, -.3586);
  EXPECT_LE(diagonal, std::sqrt(.02) - .5);
  for (double scale : {1e-150, .1, 7., 1e150}) {
    const double value = overlap.lower_bounds(motion(), 0, 1, {{0, 1}}, {Eigen::Vector3d(scale, scale, 0)})[0];
    EXPECT_LE(value, std::sqrt(.02) - .5);
    EXPECT_NEAR(value, diagonal, 1e-12);
  }
  EXPECT_THROW(overlap.lower_bounds(motion(), 0, 1, {{0, 1}}, {Eigen::Vector3d(NAN, 0, 0)}), std::domain_error);
}

TEST(IntervalPrimitiveProjection, RejectsInvalidGeometryAndQueries) {
  IntervalScalarJoint joint;
  auto invalid = sphere(0, 0);
  invalid.shape.dimensions = {-1};
  EXPECT_THROW((void)IntervalPrimitiveProjection({joint}, {invalid}), std::domain_error);
  invalid = sphere(0, 0);
  invalid.shape.swept_radius = NAN;
  EXPECT_THROW((void)IntervalPrimitiveProjection({joint}, {invalid}), std::domain_error);
  invalid = sphere(0, 0);
  invalid.shape.kind = static_cast<IntervalPrimitiveKind>(-1);
  EXPECT_THROW((void)IntervalPrimitiveProjection({joint}, {invalid}), std::domain_error);
  invalid = sphere(2, 0);
  EXPECT_THROW((void)IntervalPrimitiveProjection({joint}, {invalid}), std::domain_error);
  const IntervalPrimitiveProjection model({joint}, {sphere(0, 0), sphere(1, 2)});
  EXPECT_THROW(model.lower_bounds(motion(), 0, 1, {}), std::domain_error);
  EXPECT_THROW(model.lower_bounds(motion(), 0, 1, {{0, 0}}), std::domain_error);
  EXPECT_THROW(model.lower_bounds(motion(), 0, 1, {{-1, 0}}), std::domain_error);
  EXPECT_THROW(model.lower_bounds(motion(), 0, 1, {{0, 2}}), std::domain_error);
  EXPECT_THROW(model.lower_bounds(motion(), 0, 2, {{0, 1}}), std::domain_error);
}

} // namespace
} // namespace embodik::detail
