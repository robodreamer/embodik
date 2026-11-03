#include "interval_com_containment.hpp"
#include "interval_proof.hpp"
#include "support_polygon_geometry.hpp"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <gtest/gtest.h>
#include <cstdlib>

namespace embodik::detail {
namespace {
using Collection = pinocchio::JointCollectionDefaultTpl<double, 0>;

Eigen::VectorXd scalar(double value) { return Eigen::VectorXd::Constant(1, value); }

IntervalComContainmentPolicy box(double bound = 2.0) {
  IntervalComContainmentPolicy policy;
  policy.halfspace_normals.resize(4, 2);
  policy.halfspace_normals << 1, 0, -1, 0, 0, 1, 0, -1;
  policy.halfspace_offsets = Eigen::VectorXd::Constant(4, bound);
  return policy;
}

pinocchio::Model slider(double mass = 1.0) {
  pinocchio::Model model;
  model.addJoint(0, Collection::JointModelPX(), pinocchio::SE3::Identity(), "slide");
  model.inertias[1] = pinocchio::Inertia(mass, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity());
  return model;
}

void expect_contains(const std::array<Interval, 3> &bounds, const Eigen::Vector3d &point,
                     double maximum_width = 1e-10) {
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_LE(bounds[axis].lower(), point(axis));
    EXPECT_GE(bounds[axis].upper(), point(axis));
    EXPECT_LT(bounds[axis].upper() - bounds[axis].lower(), maximum_width);
  }
}

TEST(IntervalComContainment, ModelSnapshotDoesNotRequireCollisionGeometry) {
  const auto model = slider();
  const auto catalog = make_interval_scalar_model_catalog(model);
  ASSERT_EQ(catalog.joints.size(), 1);
  EXPECT_EQ(catalog.joint_names, std::vector<std::string>({"slide"}));
  const IntervalComContainment com(model, box());
  const QuadraticIntervalMotion motion(scalar(.2), scalar(0), scalar(0), 1);
  expect_contains(com.position_bounds(motion, 0, 0), Eigen::Vector3d(.2, 0, 0));
}

TEST(IntervalComContainment, ZeroWidthMatchesPinocchioAcrossBranchedMassiveJoints) {
  pinocchio::Model model;
  auto placement = pinocchio::SE3::Identity();
  placement.translation() = Eigen::Vector3d(.1, -.2, .3);
  model.addJoint(0, Collection::JointModelRZ(), placement, "yaw");
  placement.translation() = Eigen::Vector3d(.3, .1, 0);
  model.addJoint(1, Collection::JointModelPX(), placement, "child");
  model.addJoint(0, Collection::JointModelPY(), pinocchio::SE3::Identity(), "branch");
  model.inertias[0] = pinocchio::Inertia(1000, Eigen::Vector3d(20, 30, 40), Eigen::Matrix3d::Identity());
  model.inertias[1] = pinocchio::Inertia(2, Eigen::Vector3d(.2, .3, -.1), Eigen::Matrix3d::Identity());
  model.inertias[2] = pinocchio::Inertia(3, Eigen::Vector3d(-.1, .2, .4), Eigen::Matrix3d::Identity());
  model.inertias[3] = pinocchio::Inertia(1, Eigen::Vector3d(.4, -.2, .1), Eigen::Matrix3d::Identity());
  const IntervalComContainment com(model, box());
  for (double yaw : {-1.0, 0.0, .7}) {
    Eigen::Vector3d q(yaw, .25, -.15);
    const QuadraticIntervalMotion motion(q, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 1);
    pinocchio::Data data(model);
    const Eigen::Vector3d native = pinocchio::centerOfMass(model, data, q, false);
    expect_contains(com.position_bounds(motion, 0, 0), native);
    EXPECT_DOUBLE_EQ(data.mass[0], 6);
  }
}

TEST(IntervalComContainment, OwnsMassLeverFrameAndHalfplanes) {
  auto model = slider(2);
  model.inertias[1].lever() = Eigen::Vector3d(.5, -.25, .1);
  auto frame = pinocchio::SE3::Identity();
  frame.translation().x() = .1;
  model.addFrame(pinocchio::Frame("support", 0, 0, frame, pinocchio::OP_FRAME), false);
  auto policy = box();
  policy.frame_name = "support";
  const IntervalComContainment com(model, policy);
  model.inertias[1].lever().setConstant(99);
  model.inertias[1].mass() = 99;
  model.frames.back().placement.translation().setConstant(99);
  model.jointPlacements[1].translation().setConstant(99);
  policy.halfspace_offsets.setConstant(-99);
  const QuadraticIntervalMotion motion(scalar(.2), scalar(0), scalar(0), 1);
  expect_contains(com.position_bounds(motion, 0, 0), Eigen::Vector3d(.6, -.25, .1));
  for (double residual : com.lower_bounds(motion, 0, 0)) EXPECT_GT(residual, 0);
}

TEST(IntervalComContainment, IgnoresZeroMassJointsAndUniverseInertia) {
  auto model = slider();
  model.addJoint(0, Collection::JointModelPY(), pinocchio::SE3::Identity(), "massless");
  model.inertias[0] = pinocchio::Inertia(1000, Eigen::Vector3d::Constant(99), Eigen::Matrix3d::Identity());
  model.inertias[2] = pinocchio::Inertia::Zero();
  model.inertias[2].lever().setConstant(10);
  const IntervalComContainment com(model, box());
  Eigen::Vector2d q(.3, 50);
  const QuadraticIntervalMotion motion(q, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), 1);
  expect_contains(com.position_bounds(motion, 0, 0), Eigen::Vector3d(.3, 0, 0));
}

TEST(IntervalComContainment, FixedFrameUsesLiteralTransposeRatherThanInverse) {
  auto model = slider();
  auto frame = pinocchio::SE3::Identity();
  frame.rotation() << 0, -2, 0, 1, 0, 0, 0, 0, 1;
  frame.translation() = Eigen::Vector3d(.1, .2, 0);
  model.addFrame(pinocchio::Frame("support", 0, 0, frame, pinocchio::OP_FRAME), false);
  auto policy = box();
  policy.frame_name = "support";
  const IntervalComContainment com(model, policy);
  const QuadraticIntervalMotion motion(scalar(.4), scalar(0), scalar(0), 1);
  expect_contains(com.position_bounds(motion, 0, 0), Eigen::Vector3d(-.2, -.6, 0));
}

TEST(IntervalComContainment, MovingSupportFrameIsEnclosedOverTheWholePath) {
  pinocchio::Model model;
  model.addJoint(0, Collection::JointModelRZ(), pinocchio::SE3::Identity(), "yaw");
  model.addJoint(0, Collection::JointModelPX(), pinocchio::SE3::Identity(), "slide");
  model.inertias[1] = pinocchio::Inertia(1, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity());
  model.inertias[2] = pinocchio::Inertia(1, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity());
  model.addFrame(pinocchio::Frame("support", 1, 0, pinocchio::SE3::Identity(), pinocchio::OP_FRAME), false);
  auto policy = box(.6);
  policy.frame_name = "support";
  const IntervalComContainment com(model, policy);
  const QuadraticIntervalMotion motion(Eigen::Vector2d(-1, 1), Eigen::Vector2d(2, 0), Eigen::Vector2d::Zero(), 1);
  const auto bounds = com.position_bounds(motion, 0, 1);
  for (int i = 0; i <= 20; ++i) {
    const double angle = -1.0 + .1 * i;
    EXPECT_LE(bounds[0].lower(), .5 * std::cos(angle));
    EXPECT_GE(bounds[0].upper(), .5 * std::cos(angle));
    EXPECT_LE(bounds[1].lower(), -.5 * std::sin(angle));
    EXPECT_GE(bounds[1].upper(), -.5 * std::sin(angle));
  }
  for (double t : {0.0, .25, 1.0}) {
    const Eigen::Vector2d q(-1 + 2 * t, 1);
    pinocchio::Data data(model);
    const Eigen::Vector3d native = pinocchio::centerOfMass(model, data, q, false);
    pinocchio::updateFramePlacements(model, data);
    const auto &frame = data.oMf[model.getFrameId("support")];
    expect_contains(com.position_bounds(motion, t, t),
        frame.rotation().transpose() * (native - frame.translation()));
  }
  const auto safe = prove_interval_inequalities(1, com.inequality_count(),
      [&](double lo, double hi) { return com.lower_bounds(motion, lo, hi); });
  EXPECT_TRUE(safe.proved()) << safe.message;
  policy.halfspace_offsets(0) = .4;
  const IntervalComContainment unsafe(model, policy);
  EXPECT_GT(unsafe.lower_bounds(motion, 0, 0)[0], 0);
  EXPECT_GT(unsafe.lower_bounds(motion, 1, 1)[0], 0);
  const auto failure = prove_interval_inequalities(1, unsafe.inequality_count(),
      [&](double lo, double hi) { return unsafe.lower_bounds(motion, lo, hi); });
  EXPECT_EQ(failure.status, IntervalProofStatus::kUnproved);
}

TEST(IntervalComContainment, NativePolygonPreparationRetainsMarginExactlyOnce) {
  ComSupportPolygonConstraintDefinition definition;
  definition.support_polygon.resize(4, 2);
  definition.support_polygon << -.234, -.238, .234, -.238, .234, .238, -.234, .238;
  definition.margin = .05;
  const auto prepared = prepare_support_polygon_geometry(definition, "test", "containment");
  ASSERT_TRUE(prepared.satisfied());
  auto policy = box();
  policy.halfspace_normals = prepared.geometry.halfspace_normals;
  policy.halfspace_offsets = prepared.geometry.halfspace_offsets;
  policy.tolerance = 0;
  const auto model = slider();
  const IntervalComContainment com(model, policy);
  const QuadraticIntervalMotion motion(scalar(.2), scalar(0), scalar(0), 1);
  const auto residuals = com.lower_bounds(motion, 0, 0);
  for (Eigen::Index row = 0; row < policy.halfspace_offsets.size(); ++row) {
    const double expected = policy.halfspace_offsets(row) - policy.halfspace_normals(row, 0) * .2;
    EXPECT_LE(residuals[row], expected);
    EXPECT_NEAR(residuals[row], expected, 1e-12);
  }
  EXPECT_NEAR(prepared.geometry.halfspace_offsets.minCoeff(), .2223, 1e-12);
}

TEST(IntervalComContainment, RejectsMalformedMassModelsPoliciesAndFrames) {
  auto model = slider();
  model.inertias.pop_back();
  EXPECT_THROW((void)IntervalComContainment(model, box()), std::domain_error);
  model = slider(-1);
  EXPECT_THROW((void)IntervalComContainment(model, box()), std::domain_error);
  model = slider(0);
  EXPECT_THROW((void)IntervalComContainment(model, box()), std::domain_error);
  model = slider();
  model.inertias[1].lever().x() = INFINITY;
  EXPECT_THROW((void)IntervalComContainment(model, box()), std::domain_error);
  model = slider();
  for (double value : {-1.0, double(NAN), double(INFINITY)}) {
    auto policy = box();
    policy.tolerance = value;
    EXPECT_THROW((void)IntervalComContainment(model, policy), std::domain_error);
  }
  auto policy = box();
  policy.frame_name = "absent";
  EXPECT_THROW((void)IntervalComContainment(model, policy), std::domain_error);
  policy.frame_name = "world";
  policy.halfspace_normals.row(0).setZero();
  EXPECT_THROW((void)IntervalComContainment(model, policy), std::domain_error);
  policy = box();
  policy.halfspace_offsets.resize(3);
  EXPECT_THROW((void)IntervalComContainment(model, policy), std::domain_error);
  policy = box();
  policy.halfspace_normals(0, 0) = NAN;
  EXPECT_THROW((void)IntervalComContainment(model, policy), std::domain_error);
  model.addFrame(pinocchio::Frame("broken", 1, 0, pinocchio::SE3::Identity(), pinocchio::OP_FRAME), false);
  model.frames.back().parentJoint = std::numeric_limits<pinocchio::JointIndex>::max();
  policy = box();
  policy.frame_name = "broken";
  EXPECT_THROW((void)IntervalComContainment(model, policy), std::domain_error);
}

TEST(IntervalComContainment, UnsupportedRoundingFailsWithoutChangingCallerMode) {
  const auto model = slider();
  const auto policy = box();
  const int saved = std::fegetround();
  struct Restore { int mode; ~Restore() { std::fesetround(mode); } } restore{saved};
  ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
  EXPECT_THROW((void)IntervalComContainment(model, policy), std::runtime_error);
  EXPECT_EQ(std::fegetround(), FE_UPWARD);
}

TEST(IntervalComContainment, ArithmeticFailureIsUnprovedAndPreservesCallerEnvironment) {
  auto model = slider(1e300);
  model.inertias[1].lever().x() = 1e300;
  const IntervalComContainment com(model, box());
  const QuadraticIntervalMotion motion(scalar(0), scalar(0), scalar(0), 1);
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  struct Restore { std::fenv_t environment; ~Restore() { std::fesetenv(&environment); } } restore{saved};
  std::feclearexcept(FE_ALL_EXCEPT);
  std::feraiseexcept(FE_INEXACT);
  const int flags = std::fetestexcept(FE_ALL_EXCEPT);
  const auto result = prove_interval_inequalities(1, com.inequality_count(),
      [&](double lo, double hi) { return com.lower_bounds(motion, lo, hi); });
  EXPECT_EQ(result.status, IntervalProofStatus::kUnproved);
  EXPECT_EQ(result.visited_nodes, 1);
  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
  EXPECT_EQ(std::fegetround(), FE_TONEAREST);
}

#if defined(__GLIBC__) && defined(__USE_GNU)
TEST(IntervalComContainment, InvalidMassMetadataMasksAndRestoresCallerTraps) {
  EXPECT_EXIT({
    auto model = slider();
    model.inertias[1].mass() = std::numeric_limits<double>::signaling_NaN();
    feenableexcept(FE_INVALID);
    bool rejected = false;
    try { (void)IntervalComContainment(model, box()); }
    catch (const std::domain_error &) { rejected = true; }
    std::_Exit(rejected && (fegetexcept() & FE_INVALID) ? 0 : 1);
  }, ::testing::ExitedWithCode(0), "");
}
#endif

} // namespace
} // namespace embodik::detail
