#include "interval_collision_motion.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <gtest/gtest.h>
#include <cstdlib>

namespace embodik::detail {
namespace {
using Collection = pinocchio::JointCollectionDefaultTpl<double, 0>;
Eigen::VectorXd scalar(double x) { return Eigen::VectorXd::Constant(1,x); }
pinocchio::Model rotor() {
  pinocchio::Model model;
  model.addJoint(0, Collection::JointModelRZ(), pinocchio::SE3::Identity(), "yaw");
  model.inertias[1] = pinocchio::Inertia(1, Eigen::Vector3d::UnitX(), Eigen::Matrix3d::Identity());
  return model;
}
IntervalComContainmentPolicy policy(double rate = 10, double acceleration = 10) {
  IntervalComContainmentPolicy p;
  p.halfspace_normals.resize(1,2);
  p.halfspace_normals << 0,1;
  p.halfspace_offsets = scalar(2);
  p.derivatives = IntervalComDerivativePolicy{scalar(rate), scalar(acceleration), 1e-9};
  return p;
}
void contains(const Interval &x, double value, double tolerance = 1e-12) {
  EXPECT_LE(x.lower(), value+tolerance);
  EXPECT_GE(x.upper(), value-tolerance);
  EXPECT_LT(x.upper()-x.lower(), 1e-10);
}

TEST(IntervalComDerivatives, ZeroWidthBranchedMotionMatchesPinocchioAndJacobianBias) {
  auto model = rotor();
  auto placement = pinocchio::SE3::Identity();
  placement.translation() = Eigen::Vector3d(.2,-.3,.1);
  model.addJoint(1, Collection::JointModelPX(), placement, "slide");
  model.addJoint(0, Collection::JointModelRY(), placement, "branch");
  model.inertias[0] = pinocchio::Inertia(1000, Eigen::Vector3d::Constant(99), Eigen::Matrix3d::Identity());
  model.inertias[2] = pinocchio::Inertia(2, Eigen::Vector3d(.1,.2,.3), Eigen::Matrix3d::Identity());
  model.inertias[3] = pinocchio::Inertia(3, Eigen::Vector3d(.4,-.1,.2), Eigen::Matrix3d::Identity());
  auto frame = pinocchio::SE3::Identity();
  frame.rotation() << 0,-1,0, 1,0,0, 0,0,1;
  frame.translation() = Eigen::Vector3d(.2,.3,.4);
  model.addFrame(pinocchio::Frame("support",0,0,frame,pinocchio::OP_FRAME),false);
  const Eigen::Vector3d q(.3,-.2,.1), v(-.4,.2,.5), a(.2,-.3,.4);
  const QuadraticIntervalMotion motion(q,v,a,.2);
  for (const std::string name : {"world", "support"}) {
    auto p = policy(); p.frame_name = name;
    const IntervalComContainment com(model,p);
    for (double t : {0.0,.1,.2}) {
      const Eigen::Vector3d qt = q+t*v+.5*t*t*a, vt=v+t*a;
      pinocchio::Data data(model), zero(model);
      pinocchio::centerOfMass(model,data,qt,vt,a,false);
      pinocchio::centerOfMass(model,zero,qt,vt,Eigen::Vector3d::Zero(),false);
      const Eigen::Vector3d c=data.com[0], cd=data.vcom[0], cdd=data.acom[0], bias=zero.acom[0];
      pinocchio::jacobianCenterOfMass(model,zero,qt,false);
      EXPECT_NEAR((cdd-zero.Jcom*a-bias).norm(),0,1e-12);
      const auto result = com.motion_bounds(motion,t,t);
      Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
      if (name != "world") R = frame.rotation().transpose();
      const Eigen::Vector3d origin = name=="world" ? Eigen::Vector3d::Zero().eval() : frame.translation();
      for (int axis=0;axis<3;++axis) {
        contains(result[0][axis],(R*(c-origin))(axis));
        contains(result[1][axis],(R*cd)(axis));
        contains(result[2][axis],(R*cdd)(axis));
      }
    }
  }
}

TEST(IntervalComDerivatives, SingleRevoluteThirdAndFourthAreAnalyticTimeDerivatives) {
  const auto model=rotor();
  const IntervalComContainment com(model,policy());
  for (double v : {-.7,.7}) {
    const double q=.2,a=.3;
    const QuadraticIntervalMotion motion(scalar(q),scalar(v),scalar(a),1);
    const auto jets=com.motion_bounds(motion,0,0);
    contains(jets[3][0],-3*v*a*std::cos(q)+v*v*v*std::sin(q));
    contains(jets[3][1],-3*v*a*std::sin(q)-v*v*v*std::cos(q));
    contains(jets[4][0],-3*a*a*std::cos(q)+6*v*v*a*std::sin(q)+std::pow(v,4)*std::cos(q));
    contains(jets[4][1],-3*a*a*std::sin(q)-6*v*v*a*std::cos(q)+std::pow(v,4)*std::sin(q));
  }
}

TEST(IntervalComDerivatives, WholeIntervalHigherDerivativesEncloseVelocityReversal) {
  const auto model=rotor();
  const IntervalComContainment com(model,policy());
  const double initial_angle=.2,initial_rate=.7,a=-1.6;
  const QuadraticIntervalMotion motion(scalar(initial_angle),scalar(initial_rate),scalar(a),1);
  const auto whole=com.motion_bounds(motion,0,1);
  // Check both sides of the actual zero-rate point, where odd powers change
  // sign; all expected derivatives are closed-form, not finite differences.
  for(double t : {0.0,.2,-initial_rate/a,.7,1.0}) {
    const double q=initial_angle+t*initial_rate+.5*t*t*a,v=initial_rate+t*a;
    const std::array<double,2> third={-3*v*a*std::cos(q)+std::pow(v,3)*std::sin(q),
                                    -3*v*a*std::sin(q)-std::pow(v,3)*std::cos(q)};
    const std::array<double,2> fourth={-3*a*a*std::cos(q)+6*v*v*a*std::sin(q)+std::pow(v,4)*std::cos(q),
                                     -3*a*a*std::sin(q)-6*v*v*a*std::cos(q)+std::pow(v,4)*std::sin(q)};
    for(int axis=0;axis<2;++axis) {
      EXPECT_LE(whole[3][axis].lower(),third[axis]+1e-12);
      EXPECT_GE(whole[3][axis].upper(),third[axis]-1e-12);
      EXPECT_LE(whole[4][axis].lower(),fourth[axis]+1e-12);
      EXPECT_GE(whole[4][axis].upper(),fourth[axis]-1e-12);
    }
  }
}

TEST(IntervalComDerivatives, SignedAxisAndBinomialCompositionRetainAllCrossTerms) {
  for (double sign : {-1.0,1.0}) {
    IntervalScalarJoint first, second;
    first.axis=second.axis=sign*Eigen::Vector3d::UnitZ(); second.parent=1;
    const IntervalScalarKinematics tree({first,second});
    const Eigen::Vector2d q(.1,.2),v(.2,.5),a(.1,.3);
    const QuadraticIntervalMotion motion(q,v,a,1);
    const auto jets=tree.motion_jets(motion,0,0,{2});
    const double angle=sign*q.sum(), rate=sign*v.sum(), acc=sign*a.sum();
    contains(jets.at(2)[3].rotation[0],-3*rate*acc*std::cos(angle)+std::pow(rate,3)*std::sin(angle));
    contains(jets.at(2)[4].rotation[0],-3*acc*acc*std::cos(angle)+6*rate*rate*acc*std::sin(angle)+std::pow(rate,4)*std::cos(angle));
    EXPECT_THROW(jets.at(3),std::domain_error);
  }
}

TEST(IntervalComDerivatives, PrismaticMotionHasExactZeroHigherDerivatives) {
  IntervalScalarJoint joint; joint.kind=IntervalJointKind::kPrismatic;
  joint.axis=Eigen::Vector3d(2,-1,.5);
  joint.placement=IntervalTransform::fixed(2*Eigen::Matrix3d::Identity(),Eigen::Vector3d::Ones());
  const IntervalScalarKinematics tree({joint});
  const QuadraticIntervalMotion motion(scalar(.2),scalar(-1),scalar(2),1);
  const auto jets=tree.motion_jets(motion,0,1,{1});
  for (int axis=0;axis<3;++axis) {
    for(int order : {3,4}) EXPECT_TRUE(jets.at(1)[order].translation[axis].point(0));
    contains(jets.at(1)[2].translation[axis],4*joint.axis(axis));
  }
  EXPECT_LE(jets.at(1)[0].translation[0].lower(),.8);
  EXPECT_THROW(tree.motion_jets(motion,-1,1,{1}),std::domain_error);
  EXPECT_THROW(tree.motion_jets(motion,0,1,{2}),std::domain_error);
}

TEST(IntervalComDerivatives, RotatingSlidingAndStaticBranchesHaveAnalyticHigherCrossTerms) {
  auto model=rotor();model.inertias[1]=pinocchio::Inertia::Zero();
  model.addJoint(1,Collection::JointModelPX(),pinocchio::SE3::Identity(),"slide");
  model.inertias[2]=pinocchio::Inertia(1,Eigen::Vector3d::UnitX(),Eigen::Matrix3d::Identity());
  model.addJoint(0,Collection::JointModelPY(),pinocchio::SE3::Identity(),"branch");
  model.inertias[3]=pinocchio::Inertia(1,Eigen::Vector3d::UnitY(),Eigen::Matrix3d::Identity());
  const IntervalComContainment com(model,policy());
  const double angle=.2,v=.7,a=.3,r=1.3,rv=.4,ra=.2;
  const QuadraticIntervalMotion motion(Eigen::Vector3d(angle,r-1,0),Eigen::Vector3d(v,rv,0),Eigen::Vector3d(a,ra,0),1);
  const double f1=-v*std::sin(angle), f2=-a*std::sin(angle)-v*v*std::cos(angle);
  const double f3=-3*v*a*std::cos(angle)+v*v*v*std::sin(angle);
  const double f4=-3*a*a*std::cos(angle)+6*v*v*a*std::sin(angle)+std::pow(v,4)*std::cos(angle);
  const auto jets=com.motion_bounds(motion,0,0);
  contains(jets[3][0],.5*(r*f3+3*rv*f2+3*ra*f1));
  contains(jets[4][0],.5*(r*f4+4*rv*f3+6*ra*f2));
}

TEST(IntervalComDerivatives, CenteredCoordinatesEncloseCancellationWithoutRoundingCenterState) {
  IntervalScalarJoint joint; joint.kind=IntervalJointKind::kPrismatic; joint.axis=Eigen::Vector3d::UnitX();
  const IntervalScalarKinematics tree({joint});
  const QuadraticIntervalMotion motion(scalar(1e16),scalar(-1e16),scalar(2),1);
  const auto jets=tree.motion_jets(motion,1,1,{1});
  EXPECT_LE(jets.at(1)[0].translation[0].lower(),1);
  EXPECT_GE(jets.at(1)[0].translation[0].upper(),1);
}

TEST(IntervalComDerivatives, InteriorRateViolationCannotBeAdmittedFromSafeEndpoints) {
  const auto model=rotor();
  const IntervalComContainment com(model,policy(1.5,10));
  const QuadraticIntervalMotion motion(scalar(-1),scalar(2),scalar(0),1);
  for(double t : {0.0,1.0}) for(double residual : com.lower_bounds(motion,t,t)) EXPECT_GT(residual,0);
  const auto proof=prove_interval_inequalities(1,com.inequality_count(),
      [&](double lo,double hi){return com.lower_bounds(motion,lo,hi);});
  EXPECT_EQ(proof.status,IntervalProofStatus::kUnproved);
  EXPECT_LE(proof.visited_nodes,32);
}

TEST(IntervalComDerivatives, InteriorAccelerationViolationCannotBeAdmittedFromSafeEndpoints) {
  const auto model=rotor();
  const double pi=std::acos(-1.0);
  const IntervalComContainment com(model,policy(4,.5*pi*pi));
  const QuadraticIntervalMotion motion(scalar(0),scalar(pi),scalar(0),1);
  // c_y=sin(pi*t): rate is within 4 everywhere; acceleration is zero at
  // both endpoints but reaches -pi^2 at t=.5, exceeding its declared ceiling.
  for(double t : {0.0,1.0})
    for(double residual : com.lower_bounds(motion,t,t)) EXPECT_GT(residual,0);
  const auto interior=com.lower_bounds(motion,.5,.5);
  ASSERT_EQ(interior.size(),5);
  EXPECT_LT(interior[4],0);
  const auto proof=prove_interval_inequalities(1,com.inequality_count(),
      [&](double lo,double hi){return com.lower_bounds(motion,lo,hi);});
  EXPECT_EQ(proof.status,IntervalProofStatus::kUnproved);
  EXPECT_LE(proof.visited_nodes,32);
}

TEST(IntervalComDerivatives, TaylorUpperIncludesInteriorMaximumAndIntersectionFailsClosed) {
  const auto upper=interval_com_internal::taylor_quadratic_upper(Interval(0),Interval(1),Interval(-2),Interval(1));
  EXPECT_GE(upper,.25); EXPECT_LT(upper,.25+1e-12);
  EXPECT_THROW(interval_com_internal::intersect(Interval(0,1),Interval(2,3)),std::domain_error);
  EXPECT_THROW(interval_com_internal::taylor_quadratic_upper(Interval(0),Interval(1),Interval(-2),Interval(-1)),std::domain_error);
}

TEST(IntervalComDerivatives, MovingContainmentStaysSupportedButDerivativeRequestsAreRejected) {
  auto model=rotor();
  model.addFrame(pinocchio::Frame("moving",1,0,pinocchio::SE3::Identity(),pinocchio::OP_FRAME),false);
  auto p=policy();p.frame_name="moving";
  EXPECT_THROW((void)IntervalComContainment(model,p),std::domain_error);
  p.derivatives.reset();
  const IntervalComContainment com(model,p);
  const QuadraticIntervalMotion motion(scalar(0),scalar(.1),scalar(.1),1);
  EXPECT_NO_THROW(com.position_bounds(motion,0,1));
  EXPECT_THROW(com.motion_bounds(motion,0,1),std::domain_error);
}

TEST(IntervalComDerivatives, DerivativePolicyIsOwnedAndToleranceIsSeparateFromContainment) {
  auto model=rotor();auto p=policy(2,.5);
  const IntervalComContainment com(model,p);
  p.derivatives->rate_limits(0)=.01; model.inertias[1].lever().setConstant(99);
  const QuadraticIntervalMotion motion(scalar(0),scalar(1),scalar(0),.1);
  const auto bounds=com.motion_bounds(motion,0,0); contains(bounds[1][1],1);
  auto baseline=policy(); baseline.derivatives.reset(); baseline.halfspace_offsets(0)=-1e-6;baseline.tolerance=0;
  auto extended=baseline;extended.derivatives=IntervalComDerivativePolicy{scalar(10),scalar(10),100};
  model=rotor();
  const IntervalComContainment original(model,baseline), modified(model,extended);
  EXPECT_DOUBLE_EQ(original.lower_bounds(motion,0,0)[0],modified.lower_bounds(motion,0,0)[0]);
  EXPECT_LT(modified.lower_bounds(motion,0,0)[0],0);
}

TEST(IntervalComDerivatives, RejectsMalformedCeilingsAndTolerance) {
  const auto model=rotor();
  for(double invalid : {0.0,-1.0,double(INFINITY),double(NAN)}) {
    auto p=policy(); p.derivatives->rate_limits(0)=invalid;
    EXPECT_THROW((void)IntervalComContainment(model,p),std::domain_error);
    p=policy();p.derivatives->acceleration_limits(0)=invalid;
    EXPECT_THROW((void)IntervalComContainment(model,p),std::domain_error);
  }
  auto p=policy();p.derivatives->rate_limits.resize(0);
  EXPECT_THROW((void)IntervalComContainment(model,p),std::domain_error);
  for(double invalid : {-1.0,double(INFINITY),double(NAN)}) {
    p=policy();p.derivatives->comparison_tolerance=invalid;
    EXPECT_THROW((void)IntervalComContainment(model,p),std::domain_error);
  }
}

TEST(IntervalComDerivatives, UnsupportedRoundingDoesNotChangeTheCallerEnvironment) {
  const IntervalComContainment com(rotor(),policy());
  const QuadraticIntervalMotion motion(scalar(0),scalar(1),scalar(0),1);
  const int previous=std::fegetround();
  struct Restore{int mode;~Restore(){std::fesetround(mode);}}restore{previous};
  ASSERT_EQ(std::fesetround(FE_UPWARD),0);
  EXPECT_THROW(com.motion_bounds(motion,0,0),std::runtime_error);
  EXPECT_EQ(std::fegetround(),FE_UPWARD);
}

TEST(IntervalComDerivatives, ArithmeticOverflowIsUnprovedAndPreservesFloatingEnvironment) {
  const auto model=rotor();const IntervalComContainment com(model,policy());
  const QuadraticIntervalMotion motion(scalar(0),scalar(1e100),scalar(0),1);
  std::fenv_t saved; ASSERT_EQ(std::fegetenv(&saved),0);
  struct Restore{std::fenv_t previous;~Restore(){std::fesetenv(&previous);}}restore{saved};
  std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INEXACT);
  const int flags=std::fetestexcept(FE_ALL_EXCEPT);
  const auto proof=prove_interval_inequalities(1,com.inequality_count(),
      [&](double,double){return com.lower_bounds(motion,0,0);});
  EXPECT_EQ(proof.status,IntervalProofStatus::kUnproved);
  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT),flags);
}

#if defined(__GLIBC__) && defined(__USE_GNU)
TEST(IntervalComDerivatives, InvalidDerivativeMetadataMasksAndRestoresCallerTraps) {
  EXPECT_EXIT({
    const auto model=rotor();auto p=policy();
    p.derivatives->rate_limits(0)=std::numeric_limits<double>::signaling_NaN();
    feenableexcept(FE_INVALID);bool rejected=false;
    try{(void)IntervalComContainment(model,p);}catch(const std::domain_error&){rejected=true;}
    std::_Exit(rejected&&(fegetexcept()&FE_INVALID)?0:1);
  },::testing::ExitedWithCode(0),"");
}
#endif

#ifdef PINOCCHIO_WITH_HPP_FCL
TEST(IntervalComDerivatives, CombinedCollisionConsumerUsesOneBudgetAndDeclaredCeilings) {
  const auto model=rotor();pinocchio::GeometryModel geometry;
  auto placement=pinocchio::SE3::Identity();placement.translation().x()=2;
  geometry.addGeometryObject(pinocchio::GeometryObject("a",0,placement,std::make_shared<coal::Sphere>(.25)));
  geometry.addGeometryObject(pinocchio::GeometryObject("b",0,pinocchio::SE3::Identity(),std::make_shared<coal::Sphere>(.25)));
  geometry.addCollisionPair(pinocchio::CollisionPair(0,1));
  auto p=policy(1.5,10);
  const auto collision=prove_collision_motion(model,geometry,{{"a","b"}},scalar(-1),scalar(2),scalar(0),1,{1.0});
  EXPECT_TRUE(collision.proved());
  const auto limited=prove_collision_motion(model,geometry,{{"a","b"}},scalar(-1),scalar(2),scalar(0),1,{1.0},1e-9,{1,24},&p);
  EXPECT_EQ(limited.status,IntervalProofStatus::kUnproved);EXPECT_EQ(limited.visited_nodes,1);
  p.derivatives->rate_limits(0)=3;
  const auto allowed=prove_collision_motion(model,geometry,{{"a","b"}},scalar(-1),scalar(2),scalar(0),1,{1.0},1e-9,{32,24},&p);
  EXPECT_TRUE(allowed.proved())<<allowed.message;
}
#endif
} // namespace
} // namespace embodik::detail
