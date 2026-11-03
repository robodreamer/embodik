#include "interval_collision_motion.hpp"

#include <gtest/gtest.h>

namespace embodik::detail {
namespace {
using Collection = pinocchio::JointCollectionDefaultTpl<double, 0>;
Eigen::VectorXd scalar(double x) { return Eigen::VectorXd::Constant(1, x); }
pinocchio::Model model(int size = 1) {
  pinocchio::Model result;
  for (int i = 0; i < size; ++i)
    result.addJoint(0, Collection::JointModelPX(), pinocchio::SE3::Identity(), "q"+std::to_string(i));
  return result;
}
IntervalScalarStatePolicy policy(int size = 1) {
  IntervalScalarStatePolicy result;
  for (int i = 0; i < size; ++i) result.coordinate_names.push_back("q"+std::to_string(i));
  result.position = IntervalScalarPositionBounds{Eigen::VectorXd::Constant(size,-1),
                                                Eigen::VectorXd::Constant(size,1),0};
  return result;
}
IntervalProofResult prove(const IntervalScalarStateContainment &bounds,
                           const QuadraticIntervalMotion &motion,
                           IntervalProofBudget budget = {}) {
  return prove_interval_inequalities(motion.dt(),bounds.inequality_count(),
      [&](double lo,double hi) { return bounds.lower_bounds(motion,lo,hi); },budget);
}

TEST(IntervalScalarState, PositionExtremumRejectsSafeEndpointsWithUnsafeInterior) {
  auto p=policy(); p.position->upper=scalar(.2);
  const QuadraticIntervalMotion motion(scalar(0),scalar(1),scalar(-2),1);
  const IntervalScalarStateContainment unsafe(model(),p);
  EXPECT_FALSE(prove(unsafe,motion).proved());
  EXPECT_GE(unsafe.lower_bounds(motion,0,0)[1],0);
  EXPECT_GE(unsafe.lower_bounds(motion,1,1)[1],0);
  p.position->upper=scalar(.251);
  EXPECT_TRUE(prove(IntervalScalarStateContainment(model(),p),motion).proved());
}

TEST(IntervalScalarState, VelocityAndDeclaredAccelerationAreSeparateCeilings) {
  auto p=policy(); p.velocity=IntervalScalarMagnitudeBounds{scalar(1.01),0};
  p.acceleration=IntervalScalarMagnitudeBounds{scalar(2),0};
  const QuadraticIntervalMotion motion(scalar(0),scalar(-1),scalar(2),1);
  EXPECT_EQ(IntervalScalarStateContainment(model(),p).inequality_count(),4);
  EXPECT_TRUE(prove(IntervalScalarStateContainment(model(),p),motion).proved());
  p.velocity->maximum=scalar(.99);
  EXPECT_FALSE(prove(IntervalScalarStateContainment(model(),p),motion,{1,24}).proved());
  p.velocity.reset(); p.acceleration->maximum=scalar(1.99);
  EXPECT_FALSE(prove(IntervalScalarStateContainment(model(),p),motion,{1,24}).proved());
  p.acceleration.reset();
  EXPECT_TRUE(prove(IntervalScalarStateContainment(model(),p),motion).proved());
}

TEST(IntervalScalarState, SignedAffinePositionHasItsOwnQuadraticExtremum) {
  auto p=policy(2); p.position.reset();
  IntervalAffinePositionBounds a;
  a.matrix.resize(1,2); a.matrix<<1,-1;
  a.bias=scalar(.1); a.lower=scalar(-.1); a.upper=scalar(.2);
  p.affine_position=a;
  const QuadraticIntervalMotion motion(Eigen::Vector2d(.1,.2),Eigen::Vector2d(2,1),
                                        Eigen::Vector2d(-4,-2),1);
  const auto bounds=motion.affine_position_range(Eigen::Vector2d(1,-1),.1,0,1);
  EXPECT_LE(bounds.lower(),0); EXPECT_GE(bounds.upper(),.25);
  const IntervalScalarStateContainment unsafe(model(2),p);
  EXPECT_EQ(unsafe.inequality_count(),2);
  EXPECT_FALSE(prove(unsafe,motion).proved());
  p.affine_position->upper=scalar(.251);
  EXPECT_TRUE(prove(IntervalScalarStateContainment(model(2),p),motion).proved());
}

TEST(IntervalScalarState, AffineCoefficientsEncloseLiteralSumWithoutRoundedDotReplacement) {
  const QuadraticIntervalMotion motion(Eigen::Vector3d(1e16,1,-1e16),
      Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),1);
  const auto value=motion.affine_position_range(Eigen::Vector3d::Ones(),0,0,1);
  EXPECT_LE(value.lower(),1); EXPECT_GE(value.upper(),1);
  // Exact acceleration cancels, but outward accumulation straddles zero.
  // This must take a safe range rather than divide by an interval containing 0.
  const QuadraticIntervalMotion cancellation(Eigen::Vector2d::Zero(),
      Eigen::Vector2d(1,-1),Eigen::Vector2d(1,-1),1);
  const auto zero=cancellation.affine_position_range(Eigen::Vector2d::Ones(),0,0,1);
  EXPECT_LE(zero.lower(),0); EXPECT_GE(zero.upper(),0);
  EXPECT_LT(zero.upper()-zero.lower(),1e-12);
}

TEST(IntervalScalarState, ComparisonAllowancesAreExplicitFixedAndFamilySpecific) {
  auto p=policy(); p.position->upper=scalar(0); p.position->comparison_tolerance=1e-9;
  const auto m=model();
  EXPECT_TRUE(prove(IntervalScalarStateContainment(m,p),
      QuadraticIntervalMotion(scalar(.5e-9),scalar(0),scalar(0),1)).proved());
  EXPECT_FALSE(prove(IntervalScalarStateContainment(m,p),
      QuadraticIntervalMotion(scalar(2e-9),scalar(0),scalar(0),1)).proved());
  p.velocity=IntervalScalarMagnitudeBounds{scalar(0),0};
  EXPECT_FALSE(prove(IntervalScalarStateContainment(m,p),
      QuadraticIntervalMotion(scalar(0),scalar(.5e-9),scalar(0),1),{1,24}).proved());
}

TEST(IntervalScalarState, PolicyAndMotionSourcesAreOwned) {
  auto m=model(); auto p=policy();
  const IntervalScalarStateContainment owned(m,p);
  auto q=scalar(0),v=scalar(0),a=scalar(0);
  const QuadraticIntervalMotion motion(q,v,a,1);
  p.position->upper=scalar(-1); p.coordinate_names[0]="changed"; m.names[1]="changed";
  q(0)=v(0)=a(0)=100;
  EXPECT_TRUE(prove(owned,motion).proved());
}

TEST(IntervalScalarState, MalformedUnsupportedAndNonfiniteInputsFailClosed) {
  const auto m=model();
  auto p=policy(); p.coordinate_names[0]="wrong";
  EXPECT_THROW(IntervalScalarStateContainment(m,p),std::domain_error);
  p=policy(); p.position->upper(0)=NAN;
  EXPECT_THROW(IntervalScalarStateContainment(m,p),std::domain_error);
  p=policy(); p.position->lower(0)=2;
  EXPECT_THROW(IntervalScalarStateContainment(m,p),std::domain_error);
  p=policy(); p.position->comparison_tolerance=-1;
  EXPECT_THROW(IntervalScalarStateContainment(m,p),std::domain_error);
  p=policy(); p.velocity=IntervalScalarMagnitudeBounds{scalar(-1),0};
  EXPECT_THROW(IntervalScalarStateContainment(m,p),std::domain_error);
  p=policy(); p.acceleration=IntervalScalarMagnitudeBounds{Eigen::Vector2d(1,1),0};
  EXPECT_THROW(IntervalScalarStateContainment(m,p),std::domain_error);
  p=policy(); p.affine_position=IntervalAffinePositionBounds{};
  EXPECT_THROW(IntervalScalarStateContainment(m,p),std::domain_error);
  p=policy(); p.position.reset();
  EXPECT_THROW(IntervalScalarStateContainment(m,p),std::domain_error);
  pinocchio::Model unsupported;
  unsupported.addJoint(0,Collection::JointModelFreeFlyer(),pinocchio::SE3::Identity(),"free");
  EXPECT_THROW(IntervalScalarStateContainment(unsupported,policy()),std::domain_error);
  const IntervalScalarStateContainment bounds(m,policy());
  EXPECT_THROW(bounds.lower_bounds(QuadraticIntervalMotion(Eigen::Vector2d::Zero(),
      Eigen::Vector2d::Zero(),Eigen::Vector2d::Zero(),1),0,1),std::domain_error);
  EXPECT_THROW(bounds.lower_bounds(QuadraticIntervalMotion(scalar(0),scalar(0),scalar(0),1),0,2),std::domain_error);
}

TEST(IntervalScalarState, AffinePositionDoesNotInventDerivativeOrBrakingPolicy) {
  auto p=policy(); p.position.reset();
  IntervalAffinePositionBounds a;
  a.matrix=Eigen::MatrixXd::Zero(1,1); a.bias=scalar(.5);
  a.lower=scalar(0); a.upper=scalar(1); p.affine_position=a;
  const QuadraticIntervalMotion fast(scalar(0),scalar(1e6),scalar(1e6),1);
  const IntervalScalarStateContainment bounds(model(),p);
  EXPECT_EQ(bounds.inequality_count(),2);
  EXPECT_TRUE(prove(bounds,fast).proved());
}

#ifdef PINOCCHIO_WITH_HPP_FCL
TEST(IntervalScalarState, SameCollisionComPoseTraversalIncludesEveryDeclaredFamily) {
  auto m=model();
  m.inertias[1]=pinocchio::Inertia(1,Eigen::Vector3d::Zero(),Eigen::Matrix3d::Identity());
  m.addFrame(pinocchio::Frame("tcp",1,0,pinocchio::SE3::Identity(),pinocchio::OP_FRAME),false);
  pinocchio::GeometryModel geometry;
  auto placement=pinocchio::SE3::Identity(); placement.translation().x()=2;
  geometry.addGeometryObject(pinocchio::GeometryObject("a",1,placement,std::make_shared<coal::Sphere>(.25)));
  geometry.addGeometryObject(pinocchio::GeometryObject("b",0,pinocchio::SE3::Identity(),std::make_shared<coal::Sphere>(.25)));
  geometry.addCollisionPair(pinocchio::CollisionPair(0,1));
  IntervalComContainmentPolicy com;
  com.halfspace_normals=Eigen::Matrix<double,1,2>(1,0); com.halfspace_offsets=scalar(.3);
  IntervalProtectedPosePolicy poses;
  IntervalProtectedFramePosePolicy frame; frame.frame_name="tcp";
  frame.position_radius=.3; frame.pin_log_orientation_radius=.002; poses.frames.push_back(frame);
  auto p=policy(); p.velocity=IntervalScalarMagnitudeBounds{scalar(1.01),0};
  p.acceleration=IntervalScalarMagnitudeBounds{scalar(2),0};
  p.affine_position=IntervalAffinePositionBounds{Eigen::MatrixXd::Identity(1,1),scalar(0),scalar(-1),scalar(.3),0};
  const auto run=[&](const IntervalScalarStatePolicy *s) {
    return prove_collision_motion(m,geometry,{{"a","b"}},scalar(0),scalar(1),scalar(-2),
        1,{1.2},1e-9,{1,24},&com,&poses,s);
  };
  const auto safe=run(&p);
  EXPECT_TRUE(safe.proved())<<safe.message; EXPECT_EQ(safe.visited_nodes,1);
  EXPECT_NE(safe.message.find("rounded native endpoints/connections not certified"),std::string::npos);
  p.affine_position->upper=scalar(.2);
  EXPECT_EQ(run(&p).status,IntervalProofStatus::kUnproved);
  p.affine_position->upper=scalar(.3); p.position->upper=scalar(.2);
  EXPECT_EQ(run(&p).status,IntervalProofStatus::kUnproved);
  p.position->upper=scalar(1); p.velocity->maximum=scalar(.9);
  EXPECT_EQ(run(&p).status,IntervalProofStatus::kUnproved);
  p.velocity->maximum=scalar(1.01); p.acceleration->maximum=scalar(1.9);
  EXPECT_EQ(run(&p).status,IntervalProofStatus::kUnproved);
  p.coordinate_names[0]="stale";
  EXPECT_EQ(run(&p).status,IntervalProofStatus::kInvalidInput);
  const auto absent=run(nullptr);
  const auto original=prove_collision_motion(m,geometry,{{"a","b"}},scalar(0),scalar(1),
      scalar(-2),1,{1.2},1e-9,{1,24},&com,&poses);
  EXPECT_EQ(absent.status,original.status); EXPECT_EQ(absent.visited_nodes,original.visited_nodes);
  EXPECT_EQ(absent.certified_intervals,original.certified_intervals); EXPECT_EQ(absent.message,original.message);
}
#endif
} // namespace
} // namespace embodik::detail
