#include "collision_constraint_differential.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#ifdef PINOCCHIO_WITH_HPP_FCL
#include <coal/shape/geometric_shapes.h>
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace embodik::detail {

struct CollisionDifferentialScratch::Impl {
  explicit Impl(const RobotModel &robot) : data(robot.model()) {
    if (robot.collision_model() != nullptr) {
      geometry_data =
          std::make_unique<pinocchio::GeometryData>(*robot.collision_model());
    }
    model_identity = &robot.model();
    geometry_model_identity = robot.collision_model();
    nq = robot.nq();
    nv = robot.nv();
    geometry_count = robot.collision_model() == nullptr
                         ? 0
                         : robot.collision_model()->geometryObjects.size();
    pair_count = robot.collision_model() == nullptr
                     ? 0
                     : robot.collision_model()->collisionPairs.size();
  }

  bool matches(const RobotModel &robot, std::string *message) const {
    if (&robot.model() != model_identity ||
        robot.collision_model() != geometry_model_identity ||
        robot.nq() != nq || robot.nv() != nv) {
      *message = "collision differential scratch belongs to another robot";
      return false;
    }
    const auto *geometry_model = robot.collision_model();
    if (geometry_model == nullptr) {
      if (geometry_model_identity != nullptr) {
        *message = "collision differential geometry topology changed";
        return false;
      }
      return true;
    }
    if (geometry_model->geometryObjects.size() != geometry_count ||
        geometry_model->collisionPairs.size() != pair_count) {
      *message = "collision differential geometry topology changed";
      return false;
    }
    return true;
  }

  pinocchio::Data data;
  std::unique_ptr<pinocchio::GeometryData> geometry_data;
  const pinocchio::Model *model_identity = nullptr;
  const pinocchio::GeometryModel *geometry_model_identity = nullptr;
  int nq = 0;
  int nv = 0;
  std::size_t geometry_count = 0;
  std::size_t pair_count = 0;
};

struct CollisionDifferentialAccess {
  static CollisionDifferentialScratch::Impl *
  get(CollisionDifferentialScratch &scratch) {
    return scratch.impl_.get();
  }

  static const CollisionDifferentialScratch::Impl *
  get(const CollisionDifferentialScratch &scratch) {
    return scratch.impl_.get();
  }
};

CollisionDifferentialScratch::CollisionDifferentialScratch(
    const RobotModel &robot)
    : impl_(std::make_unique<Impl>(robot)) {}

CollisionDifferentialScratch::~CollisionDifferentialScratch() = default;
CollisionDifferentialScratch::CollisionDifferentialScratch(
    CollisionDifferentialScratch &&) noexcept = default;
CollisionDifferentialScratch &CollisionDifferentialScratch::operator=(
    CollisionDifferentialScratch &&) noexcept = default;

SolverStatus validate_collision_differential_model(const RobotModel &robot,
                                                   std::string *message) {
  if (message != nullptr) {
    message->clear();
  }
  if (robot.is_floating_base()) {
    if (message != nullptr) {
      *message =
          "collision differential currently requires a fixed-base model";
    }
    return SolverStatus::kInvalidInput;
  }
  const auto &model = robot.model();
  const auto joint_count =
      static_cast<pinocchio::JointIndex>(model.njoints);
  for (pinocchio::JointIndex joint = 1; joint < joint_count; ++joint) {
    if (model.nvs[joint] > 0 &&
        (model.nqs[joint] != 1 || model.nvs[joint] != 1)) {
      if (message != nullptr) {
        *message =
            "collision differential currently requires scalar moving joints";
      }
      return SolverStatus::kInvalidInput;
    }
  }
  return SolverStatus::kSuccess;
}

namespace {

constexpr double kCenterSeparationTolerance = 1e-12;

std::string canonical_pair_key(const std::string &object_a,
                               const std::string &object_b) {
  return object_a <= object_b ? object_a + "|" + object_b
                              : object_b + "|" + object_a;
}

BoundedCollisionPairDifferentialResult pair_failure(SolverStatus status,
                                                    std::string message) {
  BoundedCollisionPairDifferentialResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

BoundedCollisionPairDifferentialsResult batch_failure(
    SolverStatus status, std::string message,
    const BoundedCollisionPairDifferentialsResult &partial) {
  BoundedCollisionPairDifferentialsResult result;
  result.status = status;
  result.message = std::move(message);
  result.pair_evaluations = partial.pair_evaluations;
  result.active_pairs_considered = partial.active_pairs_considered;
  return result;
}

SolverStatus validate_inputs(const RobotModel &robot,
                             const CollisionDifferentialScratch &scratch,
                             const Eigen::VectorXd &q,
                             const Eigen::VectorXd &dq,
                             std::string *message) {
  const auto *impl = CollisionDifferentialAccess::get(scratch);
  if (impl == nullptr || q.size() != robot.nq() ||
      dq.size() != robot.nv()) {
    *message = "collision differential state dimensions are inconsistent";
    return SolverStatus::kInvalidInput;
  }
  if (!q.allFinite() || !dq.allFinite()) {
    *message = "collision differential state must be finite";
    return SolverStatus::kNonFiniteInput;
  }
  const auto model_status =
      validate_collision_differential_model(robot, message);
  if (model_status != SolverStatus::kSuccess) {
    return model_status;
  }
  if (!impl->matches(robot, message)) {
    return SolverStatus::kInvalidInput;
  }
  if (robot.collision_model() == nullptr ||
      impl->geometry_data == nullptr) {
    *message = "collision differential requires collision geometry";
    return SolverStatus::kInvalidInput;
  }
  return SolverStatus::kSuccess;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> point_jacobian(
    const pinocchio::Model &model, pinocchio::Data &data,
    pinocchio::JointIndex parent_joint,
    const Eigen::Vector3d &point_world) {
  Eigen::Matrix<double, 3, Eigen::Dynamic> linear(3, model.nv);
  linear.setZero();
  if (parent_joint == 0) {
    return linear;
  }

  Eigen::Matrix<double, 6, Eigen::Dynamic> joint_jacobian(6, model.nv);
  joint_jacobian.setZero();
  pinocchio::getJointJacobian(model, data, parent_joint,
                              pinocchio::LOCAL_WORLD_ALIGNED,
                              joint_jacobian);
  linear = joint_jacobian.topRows<3>();
  const auto angular = joint_jacobian.bottomRows<3>();
  const Eigen::Vector3d offset =
      point_world - data.oMi[parent_joint].translation();
  for (Eigen::Index column = 0; column < linear.cols(); ++column) {
    linear.col(column) += angular.col(column).cross(offset);
  }
  return linear;
}

bool pair_is_active(const pinocchio::GeometryData &geometry_data,
                    std::size_t pair_index) {
  return geometry_data.activeCollisionPairs.empty() ||
         geometry_data.activeCollisionPairs[pair_index];
}

std::vector<std::size_t> active_pair_indices(
    const pinocchio::GeometryModel &geometry_model,
    const pinocchio::GeometryData &geometry_data) {
  std::vector<std::size_t> indices;
  indices.reserve(geometry_model.collisionPairs.size());
  for (std::size_t pair_index = 0;
       pair_index < geometry_model.collisionPairs.size(); ++pair_index) {
    if (pair_is_active(geometry_data, pair_index)) {
      indices.push_back(pair_index);
    }
  }
  return indices;
}

#ifdef PINOCCHIO_WITH_HPP_FCL
bool valid_sphere(const coal::Sphere &sphere) {
  return std::isfinite(sphere.radius) && sphere.radius >= 0.0 &&
         std::isfinite(sphere.getSweptSphereRadius()) &&
         sphere.getSweptSphereRadius() >= 0.0;
}

double effective_radius(const coal::Sphere &sphere) {
  return sphere.radius + sphere.getSweptSphereRadius();
}
#endif

BoundedCollisionPairDifferentialResult evaluate_prepared_sphere_pair(
    const RobotModel &robot, CollisionDifferentialScratch::Impl &scratch,
    std::size_t pair_index, const Eigen::VectorXd &dq,
    bool require_active_pair) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  const auto *geometry_model = robot.collision_model();
  auto *geometry_data = scratch.geometry_data.get();
  if (geometry_model == nullptr || geometry_data == nullptr ||
      pair_index >= geometry_model->collisionPairs.size()) {
    return pair_failure(SolverStatus::kInvalidInput,
                        "collision pair index is out of range");
  }
  if (require_active_pair && !pair_is_active(*geometry_data, pair_index)) {
    return pair_failure(SolverStatus::kInvalidInput,
                        "collision pair is inactive");
  }
  const auto &pair = geometry_model->collisionPairs[pair_index];
  if (pair.first >= geometry_model->geometryObjects.size() ||
      pair.second >= geometry_model->geometryObjects.size()) {
    return pair_failure(SolverStatus::kInvalidInput,
                        "collision pair geometry index was invalid");
  }
  const auto &object_a = geometry_model->geometryObjects[pair.first];
  const auto &object_b = geometry_model->geometryObjects[pair.second];
  const auto *sphere_a =
      dynamic_cast<const coal::Sphere *>(object_a.geometry.get());
  const auto *sphere_b =
      dynamic_cast<const coal::Sphere *>(object_b.geometry.get());
  if (sphere_a == nullptr || sphere_b == nullptr) {
    return pair_failure(
        SolverStatus::kInvalidInput,
        "bounded collision differential currently supports only "
        "sphere-sphere pairs");
  }
  if (!valid_sphere(*sphere_a) || !valid_sphere(*sphere_b)) {
    return pair_failure(SolverStatus::kInvalidInput,
                        "sphere collision geometry was invalid");
  }

  const Eigen::Vector3d center_a = geometry_data->oMg[pair.first].translation();
  const Eigen::Vector3d center_b =
      geometry_data->oMg[pair.second].translation();
  const Eigen::Vector3d center_delta = center_b - center_a;
  const double center_distance = center_delta.norm();
  const double center_scale =
      std::max({1.0, center_a.norm(), center_b.norm()});
  if (!center_a.allFinite() || !center_b.allFinite() ||
      !center_delta.allFinite() || !std::isfinite(center_distance) ||
      center_distance <= kCenterSeparationTolerance * center_scale) {
    return pair_failure(
        SolverStatus::kNumericalError,
        "sphere-sphere distance has coincident or numerically "
        "indistinguishable centers");
  }

  const auto &model = robot.model();
  const Eigen::Vector3d normal = center_delta / center_distance;
  const double radius_a = effective_radius(*sphere_a);
  const double radius_b = effective_radius(*sphere_b);
  const double signed_distance = center_distance - radius_a - radius_b;
  const auto jacobian_a =
      point_jacobian(model, scratch.data, object_a.parentJoint, center_a);
  const auto jacobian_b =
      point_jacobian(model, scratch.data, object_b.parentJoint, center_b);
  const Eigen::Matrix<double, 3, Eigen::Dynamic> relative_jacobian =
      jacobian_b - jacobian_a;
  const Eigen::Vector3d relative_velocity = relative_jacobian * dq;
  const double rate = normal.dot(relative_velocity);
  const double tangent_speed_squared =
      std::max(0.0, relative_velocity.squaredNorm() - rate * rate);
  const Eigen::Vector3d acceleration_a =
      pinocchio::getFrameClassicalAcceleration(
          model, scratch.data, object_a.parentJoint, object_a.placement,
          pinocchio::LOCAL_WORLD_ALIGNED)
          .linear();
  const Eigen::Vector3d acceleration_b =
      pinocchio::getFrameClassicalAcceleration(
          model, scratch.data, object_b.parentJoint, object_b.placement,
          pinocchio::LOCAL_WORLD_ALIGNED)
          .linear();
  const Eigen::Vector3d relative_acceleration =
      acceleration_b - acceleration_a;
  const double affine_bias =
      normal.dot(relative_acceleration) +
      tangent_speed_squared / center_distance;
  const Eigen::RowVectorXd jacobian =
      normal.transpose() * relative_jacobian;

  if (!normal.allFinite() || !relative_jacobian.allFinite() ||
      !relative_velocity.allFinite() || !relative_acceleration.allFinite() ||
      !std::isfinite(radius_a) || !std::isfinite(radius_b) ||
      !std::isfinite(signed_distance) || !std::isfinite(rate) ||
      !std::isfinite(affine_bias) || !jacobian.allFinite()) {
    return pair_failure(SolverStatus::kNumericalError,
                        "sphere-sphere differential was non-finite");
  }

  CollisionPairDifferential nominal;
  nominal.pair_index = pair_index;
  nominal.object_a = object_a.name;
  nominal.object_b = object_b.name;
  nominal.pair_key = canonical_pair_key(object_a.name, object_b.name);
  nominal.center_a_world = center_a;
  nominal.center_b_world = center_b;
  nominal.point_a_world = center_a + radius_a * normal;
  nominal.point_b_world = center_b - radius_b * normal;
  nominal.normal_a_to_b_world = normal;
  nominal.relative_center_jacobian = relative_jacobian;
  nominal.relative_center_velocity_world = relative_velocity;
  nominal.relative_center_affine_acceleration_world = relative_acceleration;
  nominal.signed_distance = signed_distance;
  nominal.jacobian = jacobian;
  nominal.rate = rate;
  nominal.affine_bias = affine_bias;

  BoundedCollisionPairDifferential bounded;
  bounded.nominal = std::move(nominal);
  bounded.affine_bias_error_bound = 0.0;
  bounded.signed_distance_lower_bound = signed_distance;
  bounded.signed_distance_upper_bound = signed_distance;
  bounded.rate_lower_bound = rate;
  bounded.rate_upper_bound = rate;
  bounded.proof_kind =
      CollisionDifferentialProofKind::kAnalyticSphereSphere;

  BoundedCollisionPairDifferentialResult result;
  result.differential = std::move(bounded);
  return result;
#else
  (void)robot;
  (void)scratch;
  (void)pair_index;
  (void)dq;
  return pair_failure(SolverStatus::kInvalidInput,
                      "collision differential requires hpp-fcl support");
#endif
}

SolverStatus update_collision_kinematics(const RobotModel &robot,
                                         CollisionDifferentialScratch &scratch,
                                         const Eigen::VectorXd &q,
                                         const Eigen::VectorXd &dq,
                                         std::string *message) {
  auto *impl = CollisionDifferentialAccess::get(scratch);
  const auto *geometry_model = robot.collision_model();
  if (impl == nullptr || geometry_model == nullptr ||
      impl->geometry_data == nullptr) {
    *message = "collision differential requires collision geometry";
    return SolverStatus::kInvalidInput;
  }
  const auto &model = robot.model();
  if (impl->geometry_data->activeCollisionPairs.size() !=
      geometry_model->collisionPairs.size()) {
    *message = "collision differential active-pair topology is inconsistent";
    return SolverStatus::kInvalidInput;
  }
  const auto *robot_collision_data = robot.collision_data();
  if (robot_collision_data == nullptr ||
      robot_collision_data->activeCollisionPairs.size() !=
          geometry_model->collisionPairs.size()) {
    *message =
        "collision differential robot active-pair topology is inconsistent";
    return SolverStatus::kInvalidInput;
  }
  impl->geometry_data->activeCollisionPairs =
      robot_collision_data->activeCollisionPairs;
  try {
    pinocchio::forwardKinematics(
        model, impl->data, q, dq, Eigen::VectorXd::Zero(model.nv));
    pinocchio::computeJointJacobians(model, impl->data, q);
    pinocchio::updateFramePlacements(model, impl->data);
    pinocchio::updateGeometryPlacements(model, impl->data, *geometry_model,
                                        *impl->geometry_data);
  } catch (const std::exception &error) {
    *message =
        std::string("collision differential kinematics update failed: ") +
        error.what();
    return SolverStatus::kNumericalError;
  }
  return SolverStatus::kSuccess;
}

} // namespace

BoundedCollisionPairDifferentialResult
evaluate_bounded_collision_pair_differential_at_state(
    const RobotModel &robot, CollisionDifferentialScratch &scratch,
    std::size_t pair_index, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq) {
  std::string message;
  auto status = validate_inputs(robot, scratch, q, dq, &message);
  if (status != SolverStatus::kSuccess) {
    return pair_failure(status, message);
  }
  status = update_collision_kinematics(robot, scratch, q, dq, &message);
  if (status != SolverStatus::kSuccess) {
    return pair_failure(status, message);
  }
  return evaluate_prepared_sphere_pair(
      robot, *CollisionDifferentialAccess::get(scratch), pair_index, dq, true);
}

BoundedCollisionPairDifferentialsResult
evaluate_bounded_collision_pair_differentials_at_state(
    const RobotModel &robot, CollisionDifferentialScratch &scratch,
    const std::vector<std::size_t> &ordered_pair_indices,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq) {
  BoundedCollisionPairDifferentialsResult result;
  std::string message;
  auto status = validate_inputs(robot, scratch, q, dq, &message);
  if (status != SolverStatus::kSuccess) {
    return batch_failure(status, message, result);
  }
  status = update_collision_kinematics(robot, scratch, q, dq, &message);
  if (status != SolverStatus::kSuccess) {
    return batch_failure(status, message, result);
  }
  result.active_pairs_considered = ordered_pair_indices.size();
  result.differentials.reserve(ordered_pair_indices.size());
  for (std::size_t pair_index : ordered_pair_indices) {
    ++result.pair_evaluations;
    auto pair_result =
        evaluate_prepared_sphere_pair(
            robot, *CollisionDifferentialAccess::get(scratch), pair_index, dq,
            true);
    if (!pair_result.satisfied()) {
      return batch_failure(pair_result.status, pair_result.message, result);
    }
    result.differentials.push_back(std::move(*pair_result.differential));
  }
  if (result.differentials.empty()) {
    return batch_failure(SolverStatus::kInvalidInput,
                         "collision differential selected no active pairs",
                         result);
  }
  return result;
}

BoundedCollisionPairDifferentialsResult
evaluate_selected_bounded_collision_pair_differentials_at_state(
    const RobotModel &robot, CollisionDifferentialScratch &scratch,
    const std::vector<std::size_t> &ordered_pair_indices,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq) {
  BoundedCollisionPairDifferentialsResult result;
  std::string message;
  auto status = validate_inputs(robot, scratch, q, dq, &message);
  if (status != SolverStatus::kSuccess) {
    return batch_failure(status, message, result);
  }
  status = update_collision_kinematics(robot, scratch, q, dq, &message);
  if (status != SolverStatus::kSuccess) {
    return batch_failure(status, message, result);
  }
  result.active_pairs_considered = ordered_pair_indices.size();
  result.differentials.reserve(ordered_pair_indices.size());
  for (std::size_t pair_index : ordered_pair_indices) {
    ++result.pair_evaluations;
    auto pair_result = evaluate_prepared_sphere_pair(
        robot, *CollisionDifferentialAccess::get(scratch), pair_index, dq,
        false);
    if (!pair_result.satisfied()) {
      return batch_failure(pair_result.status, pair_result.message, result);
    }
    result.differentials.push_back(std::move(*pair_result.differential));
  }
  if (result.differentials.empty()) {
    return batch_failure(SolverStatus::kInvalidInput,
                         "collision differential selected no catalog pairs",
                         result);
  }
  return result;
}

BoundedCollisionPairDifferentialsResult
evaluate_all_active_bounded_collision_pair_differentials_at_state(
    const RobotModel &robot, CollisionDifferentialScratch &scratch,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq) {
  std::string message;
  BoundedCollisionPairDifferentialsResult result;
  auto status = validate_inputs(robot, scratch, q, dq, &message);
  if (status != SolverStatus::kSuccess) {
    return batch_failure(status, message, result);
  }
  status = update_collision_kinematics(robot, scratch, q, dq, &message);
  if (status != SolverStatus::kSuccess) {
    return batch_failure(status, message, result);
  }
  const auto *geometry_model = robot.collision_model();
  const auto *impl = CollisionDifferentialAccess::get(scratch);
  if (geometry_model == nullptr || impl == nullptr ||
      impl->geometry_data == nullptr) {
    return batch_failure(SolverStatus::kInvalidInput,
                         "collision differential requires collision geometry",
                         result);
  }
  const auto active_pairs =
      active_pair_indices(*geometry_model, *impl->geometry_data);
  result.active_pairs_considered = active_pairs.size();
  result.differentials.reserve(active_pairs.size());
  for (std::size_t pair_index : active_pairs) {
    ++result.pair_evaluations;
    auto pair_result =
        evaluate_prepared_sphere_pair(
            robot, *CollisionDifferentialAccess::get(scratch), pair_index, dq,
            true);
    if (!pair_result.satisfied()) {
      return batch_failure(pair_result.status, pair_result.message, result);
    }
    result.differentials.push_back(std::move(*pair_result.differential));
  }
  if (result.differentials.empty()) {
    return batch_failure(SolverStatus::kInvalidInput,
                         "collision differential found no active pairs",
                         result);
  }
  return result;
}

} // namespace embodik::detail
