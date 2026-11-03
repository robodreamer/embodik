#pragma once

#include "interval_scalar_kinematics.hpp"

#include <pinocchio/multibody/model.hpp>
#include <boost/variant/get.hpp>
#include <set>
#include <string>

namespace embodik::detail {

// Model-only owned constants, shared by interval collision and CoM consumers.
struct IntervalScalarModelCatalog {
  std::vector<IntervalScalarJoint> joints;
  std::vector<std::string> joint_names;
};

inline IntervalScalarModelCatalog make_interval_scalar_model_catalog(
    const pinocchio::Model &model) {
  const ScopedIntervalEnvironment environment;
  if (model.nq <= 0 || model.nq != model.nv ||
      static_cast<long long>(model.njoints) != static_cast<long long>(model.nq) + 1 ||
      model.joints.size() != static_cast<std::size_t>(model.njoints) ||
      model.parents.size() != static_cast<std::size_t>(model.njoints) ||
      model.names.size() != static_cast<std::size_t>(model.njoints) ||
      model.jointPlacements.size() != static_cast<std::size_t>(model.njoints))
    throw std::domain_error("additive ordered scalar model required");
  IntervalScalarModelCatalog result;
  std::set<std::string> joint_names;
  for (int i = 1; i < model.njoints; ++i) {
    const auto &source = model.joints[i];
    if (source.nq() != 1 || source.nv() != 1 || source.idx_q() != i - 1 || source.idx_v() != i - 1 ||
        model.names[i].empty() || !joint_names.insert(model.names[i]).second)
      throw std::domain_error("native scalar coordinate ordering or names are invalid");
    if (model.parents[i] >= static_cast<pinocchio::JointIndex>(i))
      throw std::domain_error("native joint parent must precede its child");
    IntervalScalarJoint joint;
    joint.parent = static_cast<int>(model.parents[i]);
    const std::string name = source.shortname();
    joint.axis.setZero();
    if (name == "JointModelRX" || name == "JointModelRY" || name == "JointModelRZ" ||
        name == "JointModelPX" || name == "JointModelPY" || name == "JointModelPZ") {
      const auto axis = std::string("XYZ").find(name.back());
      joint.axis(static_cast<Eigen::Index>(axis)) = 1.0;
      joint.kind = name[10] == 'R' ? IntervalJointKind::kRevolute : IntervalJointKind::kPrismatic;
    } else {
      using Collection = pinocchio::JointCollectionDefaultTpl<double, 0>;
      if (const auto *revolute = boost::get<Collection::JointModelRevoluteUnaligned>(&source.toVariant())) {
        joint.axis = revolute->axis;
        joint.kind = IntervalJointKind::kRevolute;
      } else if (const auto *prismatic = boost::get<Collection::JointModelPrismaticUnaligned>(&source.toVariant())) {
        joint.axis = prismatic->axis;
        joint.kind = IntervalJointKind::kPrismatic;
      } else throw std::domain_error("unsupported native scalar joint family");
    }
    joint.placement = IntervalTransform::fixed(model.jointPlacements[i].rotation(), model.jointPlacements[i].translation());
    result.joints.push_back(joint);
    result.joint_names.push_back(model.names[i]);
  }
  (void)IntervalScalarKinematics(result.joints);
  return result;
}

} // namespace embodik::detail
