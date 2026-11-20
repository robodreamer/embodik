#include "interval_collision_motion.hpp"

#include <Eigen/Geometry>
#include <pinocchio/spatial/log.hpp>
#include <gtest/gtest.h>

namespace embodik::detail {
namespace {
using Collection = pinocchio::JointCollectionDefaultTpl<double, 0>;
Eigen::VectorXd scalar(double x) { return Eigen::VectorXd::Constant(1, x); }

pinocchio::Model model(bool prismatic = false) {
  pinocchio::Model result;
  if (prismatic)
    result.addJoint(0, Collection::JointModelPX(), pinocchio::SE3::Identity(), "slide");
  else
    result.addJoint(0, Collection::JointModelRZ(), pinocchio::SE3::Identity(), "yaw");
  result.inertias[1] = pinocchio::Inertia(1, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity());
  result.addFrame(pinocchio::Frame("tcp", 1, 0, pinocchio::SE3::Identity(), pinocchio::OP_FRAME), false);
  return result;
}

IntervalProtectedFramePosePolicy frame(bool position = true, bool orientation = true) {
  IntervalProtectedFramePosePolicy result;
  result.frame_name = "tcp";
  if (position) result.position_radius = .002;
  if (orientation) result.markley_trace_orientation_radius = .002;
  return result;
}

QuadraticIntervalMotion motion(double q, double v = 0, double a = 0) {
  return QuadraticIntervalMotion(scalar(q), scalar(v), scalar(a), 1);
}

TEST(IntervalProtectedPose, NearBoundaryZeroWidthRetainsTwoMillimeterTwoMilliradianBudgets) {
  auto p = frame();
  p.target_position.x() = .001987131203363346;
  const IntervalProtectedPoseContainment poses(model(), {{p}});
  const double angle = .0019998713977654597;
  const auto upper = poses.upper_bounds(motion(angle), 0, 0);
  ASSERT_EQ(upper.size(), 2);
  EXPECT_GE(upper[0], p.target_position.norm());
  EXPECT_LT(upper[0], .002);
  EXPECT_GE(upper[1], angle);
  EXPECT_NEAR(upper[1], 2 * std::tan(angle / 2), 1e-14);
  EXPECT_LT(upper[1], .002);
  for (double slack : poses.lower_bounds(motion(angle), 0, 0)) EXPECT_GT(slack, 0);
}

TEST(IntervalProtectedPose, LiteralNearlyRigidTargetUsesMarkleyRatherThanAssumedOrthogonality) {
  auto p = frame(false, true);
  p.target_rotation = Eigen::AngleAxisd(-.0012, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  p.target_rotation.col(0) *= 1 + 1e-15;
  const IntervalProtectedPoseContainment poses(model(), {{p}});
  const double q = .0003;
  const Eigen::Matrix3d actual = Eigen::AngleAxisd(q, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Matrix3d relative = p.target_rotation * actual.transpose();
  const Eigen::Vector3d s(relative(2,1)-relative(1,2), relative(0,2)-relative(2,0), relative(1,0)-relative(0,1));
  const double mathematical = 2 * std::atan2(s.norm(), 1 + relative.trace());
  const auto upper = poses.upper_bounds(motion(q), 0, 0);
  EXPECT_GE(upper[0], mathematical);
  EXPECT_LT(upper[0] - mathematical, 1e-9);
}

TEST(IntervalProtectedPose, SignedRotationAndFixedTcpAttachmentAreEnclosed) {
  for (double sign : {-1.0, 1.0}) {
    auto native = model();
    auto &fixed = native.frames[native.getFrameId("tcp")].placement;
    fixed.translation() = Eigen::Vector3d(.2, -.1, .3);
    fixed.rotation() = Eigen::AngleAxisd(.001, Eigen::Vector3d::UnitX()).toRotationMatrix();
    auto p = frame();
    p.target_position = fixed.translation();
    p.target_rotation = fixed.rotation();
    const IntervalProtectedPoseContainment poses(native, {{p}});
    const auto upper = poses.upper_bounds(motion(sign * .001), 0, 0);
    EXPECT_GE(upper[0], 2 * std::hypot(.2, .1) * std::sin(.0005));
    EXPECT_GE(upper[1], .001);
    EXPECT_LT(upper[1], .002);
  }
}

TEST(IntervalProtectedPose, SafePositionEndpointsDoNotCertifyInteriorExcursion) {
  auto p = frame(true, false); p.position_radius = .0019;
  const IntervalProtectedPoseContainment poses(model(true), {{p}});
  const auto segment = motion(0, .008, -.016);
  EXPECT_GT(poses.lower_bounds(segment, 0, 0)[0], 0);
  EXPECT_GT(poses.lower_bounds(segment, 1, 1)[0], 0);
  const auto proof = prove_interval_inequalities(1, 1,
      [&](double lo, double hi) { return poses.lower_bounds(segment, lo, hi); });
  EXPECT_EQ(proof.status, IntervalProofStatus::kUnproved);
  EXPECT_LE(proof.visited_nodes, 32);
}

TEST(IntervalProtectedPose, SafeOrientationEndpointsDoNotCertifyInteriorExcursion) {
  const IntervalProtectedPoseContainment poses(model(), {{frame(false, true)}});
  const auto segment = motion(0, .012, -.024);
  EXPECT_GT(poses.lower_bounds(segment, 0, 0)[0], 0);
  EXPECT_GT(poses.lower_bounds(segment, 1, 1)[0], 0);
  const auto proof = prove_interval_inequalities(1, 1,
      [&](double lo, double hi) { return poses.lower_bounds(segment, lo, hi); });
  EXPECT_EQ(proof.status, IntervalProofStatus::kUnproved);
}

TEST(IntervalProtectedPose, DisplacementBetweenTwoAndFiveMillimetersRemainsRejected) {
  const IntervalProtectedPoseContainment poses(model(true), {{frame(true, false)}});
  EXPECT_LT(poses.lower_bounds(motion(.003), 0, 0)[0], 0);
}

TEST(IntervalProtectedPose, PositionOnlyLiteralPlacementsDoNotAcquireRotationRestrictions) {
  auto native = model(true);
  native.jointPlacements[1].rotation() = 2 * Eigen::Matrix3d::Identity();
  const IntervalProtectedPoseContainment poses(native, {{frame(true, false)}});
  EXPECT_GE(poses.upper_bounds(motion(.0005), 0, 0)[0], .001);
}

TEST(IntervalProtectedPose, SvdPossibleOrNonTraceIntervalsAreUnproved) {
  auto p = frame(false, true); p.target_rotation(0,0) += 1e-7;
  const IntervalProtectedPoseContainment poses(model(), {{p}});
  const auto segment = motion(0);
  const auto unsupported = prove_interval_inequalities(1, 1,
      [&](double lo, double hi) { return poses.lower_bounds(segment, lo, hi); });
  EXPECT_EQ(unsupported.status, IntervalProofStatus::kUnproved);
  EXPECT_EQ(unsupported.visited_nodes, 1);
  const IntervalProtectedPoseContainment near(model(), {{frame(false, true)}});
  EXPECT_THROW(near.upper_bounds(motion(3), 0, 0), std::domain_error);
}

TEST(IntervalProtectedPose, MalformedPoliciesFailAndSourceConstantsAreOwned) {
  auto native = model(); auto p = frame();
  const IntervalProtectedPoseContainment poses(native, {{p}});
  const auto before = poses.upper_bounds(motion(.001), 0, 0);
  native.frames[native.getFrameId("tcp")].placement.translation().setConstant(99);
  p.target_position.setConstant(99); p.markley_trace_orientation_radius = 1;
  EXPECT_EQ(before, poses.upper_bounds(motion(.001), 0, 0));
  EXPECT_THROW((void)IntervalProtectedPoseContainment(model(), {}), std::domain_error);
  p = frame(); p.frame_name = "absent";
  EXPECT_THROW((void)IntervalProtectedPoseContainment(model(), {{p}}), std::domain_error);
  p = frame(false, true); p.target_rotation(0,0) = -1;
  EXPECT_THROW((void)IntervalProtectedPoseContainment(model(), {{p}}), std::domain_error);
  for (double invalid : {0.0, -1.0, double(INFINITY), double(NAN)}) {
    p = frame(); p.position_radius = invalid;
    EXPECT_THROW((void)IntervalProtectedPoseContainment(model(), {{p}}), std::domain_error);
    p = frame(); p.markley_trace_orientation_radius = invalid;
    EXPECT_THROW((void)IntervalProtectedPoseContainment(model(), {{p}}), std::domain_error);
  }
  p = frame(); p.orientation_comparison_tolerance = -1;
  EXPECT_THROW((void)IntervalProtectedPoseContainment(model(), {{p}}), std::domain_error);
}

TEST(IntervalProtectedPose, BlockOrderingAndComparisonTolerancesAreIndependent) {
  auto right = frame(false, true), left = frame();
  left.target_position.x() = .003;
  left.position_comparison_tolerance = .002;
  const IntervalProtectedPoseContainment poses(model(), {{right, left}});
  ASSERT_EQ(poses.inequality_count(), 3);
  const auto residual = poses.lower_bounds(motion(.0025), 0, 0);
  EXPECT_LT(residual[0], 0);
  EXPECT_GT(residual[1], 0);
  EXPECT_LT(residual[2], 0);
}

TEST(IntervalProtectedPose, RootFixedTcpIgnoresUnrelatedJointMotion) {
  auto native = model();
  native.addFrame(pinocchio::Frame("fixed", 0, 0, pinocchio::SE3::Identity(), pinocchio::OP_FRAME), false);
  auto p = frame(); p.frame_name = "fixed";
  const IntervalProtectedPoseContainment poses(native, {{p}});
  for (double residual : poses.lower_bounds(motion(0, .5, .1), 0, 1)) EXPECT_GT(residual, 0);
}

TEST(IntervalProtectedPose, UnsupportedEnvironmentFailsWithoutChangingCallerRounding) {
  const IntervalProtectedPoseContainment poses(model(), {{frame()}});
  const int previous = std::fegetround();
  struct Restore { int mode; ~Restore() { std::fesetround(mode); } } restore{previous};
  ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
  EXPECT_THROW(poses.lower_bounds(motion(0), 0, 0), std::runtime_error);
  EXPECT_EQ(std::fegetround(), FE_UPWARD);
}

IntervalProtectedFramePosePolicy pin_frame(bool position = false) {
  auto result = frame(position, false);
  result.pin_log_orientation_radius = .002;
  return result;
}

std::array<Interval, 9> literal_matrix(const Eigen::Matrix3d &matrix) {
  std::array<Interval, 9> result;
  for (int row=0; row<3; ++row)
    for (int col=0; col<3; ++col) result[3*row+col] = Interval(matrix(row,col));
  return result;
}

TEST(IntervalProtectedPose, PinLogNearBoundaryPositiveAndNegativeMatchInstalledFormula) {
  const IntervalProtectedPoseContainment poses(model(), {{pin_frame()}});
  for (double sign : {-1.0,1.0}) {
    const double angle = sign*.0019998713977654597;
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const double native = pinocchio::log3(rotation).norm();
    const auto upper = poses.upper_bounds(motion(angle),0,0);
    EXPECT_GE(upper[0], native);
    EXPECT_LT(upper[0]-native, 1e-9);
    EXPECT_LT(upper[0], .002);
    EXPECT_GT(poses.lower_bounds(motion(angle),0,0)[0],0);
  }
}

TEST(IntervalProtectedPose, PinLogRenormalizesLiteralNonorthogonalMatrices) {
  Eigen::Matrix3d relative;
  relative << 1.00001,.002,.0007, -.0011,.99999,.0013, .0002,-.0009,1.00002;
  const double upper = interval_protected_pose_internal::pin_log_norm_upper(literal_matrix(relative));
  const double native = pinocchio::log3(relative).norm();
  EXPECT_GE(upper,native);
  EXPECT_LT(upper-native,1e-7);
}

TEST(IntervalProtectedPose, PinLogUsesNativeReferenceTransposeMultiplicationOrder) {
  auto p = pin_frame(); p.pin_log_orientation_radius = .01;
  p.target_rotation(0,1) = .004;
  const IntervalProtectedPoseContainment poses(model(),{{p}});
  const Eigen::Matrix3d actual = Eigen::AngleAxisd(.001,Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Matrix3d correct = p.target_rotation.transpose()*actual;
  const Eigen::Matrix3d wrong = p.target_rotation*actual.transpose();
  const double native = pinocchio::log3(correct).norm();
  EXPECT_GT(std::abs(native-pinocchio::log3(wrong).norm()),.003);
  const double upper = poses.upper_bounds(motion(.001),0,0)[0];
  EXPECT_GE(upper,native);
  EXPECT_LT(upper-native,1e-7);
}

TEST(IntervalProtectedPose, PinLogUndefinedColumnsAndNegativeCosineRemainUnproved) {
  Eigen::Matrix3d relative = Eigen::Matrix3d::Identity(); relative.col(0).setZero();
  EXPECT_THROW(interval_protected_pose_internal::pin_log_norm_upper(literal_matrix(relative)),std::domain_error);
  relative = Eigen::AngleAxisd(2.4,Eigen::Vector3d::UnitZ()).toRotationMatrix();
  EXPECT_THROW(interval_protected_pose_internal::pin_log_norm_upper(literal_matrix(relative)),std::domain_error);
  // Parallel nonzero columns define a zero selected nominal Pin formula here.
  // Interval normalization cannot establish its boundary cosine>=0, so it is
  // conservatively unproved, rather than rejected by a new parallelism rule.
  relative = Eigen::Matrix3d::Zero(); relative.col(0)=relative.col(1)=Eigen::Vector3d::UnitY();
  EXPECT_DOUBLE_EQ(pinocchio::log3(relative).norm(),0);
  EXPECT_THROW(interval_protected_pose_internal::pin_log_norm_upper(literal_matrix(relative)),std::domain_error);
}

TEST(IntervalProtectedPose, PinLogSignedTcpAttachmentAndOwnedSourcesArePreserved) {
  for (double sign : {-1.0,1.0}) {
    auto native = model();
    auto &attachment = native.frames[native.getFrameId("tcp")].placement;
    attachment.translation() = Eigen::Vector3d(.2,-.1,.3);
    attachment.rotation() = Eigen::AngleAxisd(.001,Eigen::Vector3d::UnitX()).toRotationMatrix();
    auto p = pin_frame(true); p.target_position = attachment.translation(); p.target_rotation = attachment.rotation();
    const IntervalProtectedPoseContainment poses(native,{{p}});
    const auto before = poses.upper_bounds(motion(sign*.001),0,0);
    EXPECT_LT(before[0],.002); EXPECT_GE(before[1],.001); EXPECT_LT(before[1],.002);
    p.target_rotation.setZero(); attachment.rotation().setZero();
    EXPECT_EQ(before,poses.upper_bounds(motion(sign*.001),0,0));
  }
}

TEST(IntervalProtectedPose, PinLogPolicyIsExplicitExclusiveAndValidated) {
  auto p=pin_frame(); p.markley_trace_orientation_radius=.002;
  EXPECT_THROW((void)IntervalProtectedPoseContainment(model(),{{p}}),std::domain_error);
  for(double invalid : {0.0,-1.0,double(INFINITY),double(NAN)}) {
    p=pin_frame(); p.pin_log_orientation_radius=invalid;
    EXPECT_THROW((void)IntervalProtectedPoseContainment(model(),{{p}}),std::domain_error);
  }
}

#ifdef PINOCCHIO_WITH_HPP_FCL
TEST(IntervalProtectedPose, CombinedWalkerSharesBudgetAndPreservesAbsentPolicyResults) {
  const auto native = model(); pinocchio::GeometryModel geometry;
  auto fixed = pinocchio::SE3::Identity(); fixed.translation().x() = 2;
  geometry.addGeometryObject(pinocchio::GeometryObject("a", 0, fixed, std::make_shared<coal::Sphere>(.25)));
  geometry.addGeometryObject(pinocchio::GeometryObject("b", 0, pinocchio::SE3::Identity(), std::make_shared<coal::Sphere>(.25)));
  geometry.addCollisionPair(pinocchio::CollisionPair(0, 1));
  IntervalComContainmentPolicy com;
  com.halfspace_normals.resize(1,2); com.halfspace_normals << 1, 0;
  com.halfspace_offsets = scalar(1);
  const auto run = [&](const IntervalProtectedPosePolicy *poses) {
    return prove_collision_motion(native, geometry, {{"a", "b"}}, scalar(0), scalar(.012),
        scalar(-.024), 1, {1}, 1e-9, {1,24}, &com, poses);
  };
  const auto absent = prove_collision_motion(native, geometry, {{"a", "b"}}, scalar(0), scalar(.012),
      scalar(-.024), 1, {1}, 1e-9, {1,24}, &com);
  const auto explicit_null = run(nullptr);
  EXPECT_EQ(absent.status, explicit_null.status);
  EXPECT_EQ(absent.visited_nodes, explicit_null.visited_nodes);
  EXPECT_EQ(absent.certified_intervals, explicit_null.certified_intervals);
  EXPECT_EQ(absent.message, explicit_null.message);
  EXPECT_TRUE(absent.proved());
  const IntervalProtectedPosePolicy poses{{frame(false, true)}};
  const auto limited = run(&poses);
  EXPECT_EQ(limited.status, IntervalProofStatus::kUnproved);
  EXPECT_EQ(limited.visited_nodes, 1);
}
#endif
} // namespace
} // namespace embodik::detail
