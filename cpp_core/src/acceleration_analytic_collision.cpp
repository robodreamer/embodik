#include "acceleration_analytic_collision.hpp"

#ifdef PINOCCHIO_WITH_HPP_FCL
#include <coal/shape/geometric_shapes.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace embodik::detail {
namespace {

constexpr double kConstraintTolerance = 1e-8;
constexpr double kBoundaryTolerance = 1e-10;
constexpr int kMaximumPathDepth = 24;

template <typename Result>
Result failure(SolverStatus status, std::string message) {
  Result result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

std::string canonical_pair_key(const std::string &object_a,
                               const std::string &object_b) {
  return object_a <= object_b ? object_a + "|" + object_b
                              : object_b + "|" + object_a;
}

bool finite_nonnegative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

bool finite_positive(double value) {
  return std::isfinite(value) && value > 0.0;
}

bool all_moving_joints_are_prismatic(const RobotModel &robot) {
  const auto &model = robot.model();
  for (pinocchio::JointIndex joint = 1;
       joint < static_cast<pinocchio::JointIndex>(model.njoints); ++joint) {
    if (model.nvs[joint] == 0) {
      continue;
    }
    if (model.nqs[joint] != 1 || model.nvs[joint] != 1) {
      return false;
    }
    const std::string name = model.joints[joint].shortname();
    if (name != "JointModelPX" && name != "JointModelPY" &&
        name != "JointModelPZ" &&
        name != "JointModelPrismaticUnaligned") {
      return false;
    }
  }
  return true;
}

bool validate_policy(const CollisionConstraintAccelerationPolicy &policy,
                     std::string *message) {
  if (!finite_nonnegative(policy.activation_margin) ||
      !finite_positive(policy.maximum_approach_rate) ||
      !finite_positive(policy.maximum_inward_acceleration) ||
      !finite_positive(policy.minimum_braking_acceleration) ||
      !finite_positive(policy.recovery_scale) ||
      !finite_positive(policy.minimum_recovery_rate) ||
      !finite_positive(policy.maximum_recovery_rate) ||
      policy.maximum_recovery_rate < policy.minimum_recovery_rate) {
    *message =
        "native collision acceleration policy contains invalid magnitudes";
    return false;
  }
  switch (policy.outside_policy) {
  case CollisionConstraintOutsidePolicy::kReject:
  case CollisionConstraintOutsidePolicy::kRecoverNonWorsening:
    return true;
  }
  *message = "native collision outside policy is invalid";
  return false;
}

bool collect_exact_pair_keys(const std::vector<CollisionGeometryPair> &pairs,
                             const std::string &label,
                             std::unordered_set<std::string> *keys,
                             std::string *message) {
  for (const auto &pair : pairs) {
    if (pair.geometry_a.empty() || pair.geometry_b.empty() ||
        pair.geometry_a == pair.geometry_b) {
      *message = label + " contains an invalid exact geometry pair";
      return false;
    }
    const auto key = canonical_pair_key(pair.geometry_a, pair.geometry_b);
    if (!keys->insert(key).second) {
      *message = label + " contains a duplicate exact geometry pair";
      return false;
    }
  }
  return true;
}

bool approximately_equal(double lhs, double rhs) {
  const double tolerance =
      kConstraintTolerance * (1.0 + std::max(std::abs(lhs), std::abs(rhs)));
  return std::abs(lhs - rhs) <= tolerance;
}

bool approximately_equal(const Eigen::Vector3d &lhs,
                         const Eigen::Vector3d &rhs) {
  const double tolerance =
      kConstraintTolerance * (1.0 + std::max(lhs.norm(), rhs.norm()));
  return (lhs - rhs).norm() <= tolerance;
}

std::pair<double, double> affine_range(
    const Eigen::Ref<const Eigen::RowVectorXd> &row, double bias,
    const Eigen::VectorXd &lower, const Eigen::VectorXd &upper) {
  double minimum = bias;
  double maximum = bias;
  for (Eigen::Index column = 0; column < row.cols(); ++column) {
    if (row(column) >= 0.0) {
      minimum += row(column) * lower(column);
      maximum += row(column) * upper(column);
    } else {
      minimum += row(column) * upper(column);
      maximum += row(column) * lower(column);
    }
  }
  return {minimum, maximum};
}

SolverStatus collision_status(SolverStatus status) {
  return status == SolverStatus::kInfeasible
             ? SolverStatus::kCollisionViolated
             : status;
}

bool affine_constraint_accepts(
    const AffineAccelerationConstraint &constraint,
    const Eigen::VectorXd &candidate) {
  if (constraint.coefficient_matrix.rows() == 0) {
    return true;
  }
  const Eigen::VectorXd physical =
      constraint.coefficient_matrix * candidate + constraint.affine_bias;
  if (!physical.allFinite()) {
    return false;
  }
  for (Eigen::Index row = 0; row < physical.size(); ++row) {
    const bool lower_active =
        constraint.lower_bound_active.empty() ||
        constraint.lower_bound_active[static_cast<std::size_t>(row)];
    const bool upper_active =
        constraint.upper_bound_active.empty() ||
        constraint.upper_bound_active[static_cast<std::size_t>(row)];
    if ((lower_active &&
         physical(row) < constraint.lower_bounds(row) -
                             kConstraintTolerance) ||
        (upper_active &&
         physical(row) > constraint.upper_bounds(row) +
                             kConstraintTolerance)) {
      return false;
    }
  }
  return true;
}

struct PathProof {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  double lowest_certified_lower_bound =
      std::numeric_limits<double>::infinity();
  std::uint64_t visited_nodes = 0;
  std::uint64_t certified_intervals = 0;
};

PathProof certify_quadratic_sphere_path(
    const Eigen::Vector3d &relative_position,
    const Eigen::Vector3d &relative_velocity,
    const Eigen::Vector3d &relative_acceleration, double combined_radius,
    double required_floor, double dt, const std::string &pair_key) {
  struct Interval {
    double lower = 0.0;
    double upper = 0.0;
    int depth = 0;
  };

  PathProof proof;
  std::vector<Interval> pending{{0.0, dt, 0}};
  const auto position_at = [&](double time) {
    return relative_position + time * relative_velocity +
           0.5 * time * time * relative_acceleration;
  };
  const auto velocity_at = [&](double time) {
    return relative_velocity + time * relative_acceleration;
  };
  const double floor_tolerance =
      kBoundaryTolerance *
      (1.0 + std::max(std::abs(required_floor), combined_radius));

  while (!pending.empty()) {
    const Interval interval = pending.back();
    pending.pop_back();
    ++proof.visited_nodes;
    const double midpoint = 0.5 * (interval.lower + interval.upper);
    const double half_width = 0.5 * (interval.upper - interval.lower);
    const Eigen::Vector3d midpoint_position = position_at(midpoint);
    const Eigen::Vector3d midpoint_velocity = velocity_at(midpoint);
    const double midpoint_distance =
        midpoint_position.norm() - combined_radius;
    const double endpoint_lower =
        position_at(interval.lower).norm() - combined_radius;
    const double endpoint_upper =
        position_at(interval.upper).norm() - combined_radius;
    if (!midpoint_position.allFinite() || !midpoint_velocity.allFinite() ||
        !std::isfinite(midpoint_distance) ||
        !std::isfinite(endpoint_lower) || !std::isfinite(endpoint_upper)) {
      return failure<PathProof>(
          SolverStatus::kNumericalError,
          "native collision path proof for '" + pair_key +
              "' produced non-finite evidence");
    }
    if (std::min({midpoint_distance, endpoint_lower, endpoint_upper}) <
        required_floor - floor_tolerance) {
      return failure<PathProof>(
          SolverStatus::kCollisionViolated,
          "native collision path for '" + pair_key +
              "' crosses its certified distance floor");
    }

    const double motion_radius =
        midpoint_velocity.norm() * half_width +
        0.5 * relative_acceleration.norm() * half_width * half_width;
    const double signed_distance_lower_bound =
        midpoint_position.norm() - motion_radius - combined_radius;
    if (!std::isfinite(motion_radius) ||
        !std::isfinite(signed_distance_lower_bound)) {
      return failure<PathProof>(
          SolverStatus::kNumericalError,
          "native collision path bound for '" + pair_key +
              "' was non-finite");
    }
    if (signed_distance_lower_bound >= required_floor - floor_tolerance) {
      const double canonical_lower_bound =
          signed_distance_lower_bound < required_floor
              ? required_floor
              : signed_distance_lower_bound;
      proof.lowest_certified_lower_bound =
          std::min(proof.lowest_certified_lower_bound,
                   canonical_lower_bound);
      ++proof.certified_intervals;
      continue;
    }
    if (interval.depth >= kMaximumPathDepth) {
      return failure<PathProof>(
          SolverStatus::kNoProgress,
          "native collision path for '" + pair_key +
              "' could not be continuously certified");
    }
    pending.push_back(
        {midpoint, interval.upper, interval.depth + 1});
    pending.push_back(
        {interval.lower, midpoint, interval.depth + 1});
  }
  return proof;
}

} // namespace

CompiledAnalyticCollisionConstraint compile_analytic_collision_constraint(
    const RobotModel &robot, const CollisionConstraintDefinition &definition,
    const CollisionConstraintAccelerationPolicy &policy) {
#ifndef PINOCCHIO_WITH_HPP_FCL
  (void)robot;
  (void)definition;
  (void)policy;
  return failure<CompiledAnalyticCollisionConstraint>(
      SolverStatus::kInvalidInput,
      "native collision constraints require hpp-fcl support");
#else
  std::string message;
  if (!finite_nonnegative(definition.min_distance)) {
    return failure<CompiledAnalyticCollisionConstraint>(
        SolverStatus::kInvalidInput,
        "native collision minimum distance must be finite and nonnegative");
  }
  if (!validate_policy(policy, &message)) {
    return failure<CompiledAnalyticCollisionConstraint>(
        SolverStatus::kInvalidInput, message);
  }
  if (robot.is_floating_base() || robot.nq() != robot.nv() ||
      !all_moving_joints_are_prismatic(robot)) {
    return failure<CompiledAnalyticCollisionConstraint>(
        SolverStatus::kInvalidInput,
        "native collision certification requires fixed-base all-prismatic "
        "scalar joints");
  }
  const auto *geometry_model = robot.collision_model();
  if (!robot.has_collision_geometry() || geometry_model == nullptr ||
      geometry_model->collisionPairs.empty()) {
    return failure<CompiledAnalyticCollisionConstraint>(
        SolverStatus::kInvalidInput,
        "native collision certification requires collision pairs");
  }

  std::unordered_set<std::string> include_keys;
  std::unordered_set<std::string> exclude_keys;
  if (!collect_exact_pair_keys(definition.include_pairs, "include_pairs",
                               &include_keys, &message) ||
      !collect_exact_pair_keys(definition.exclude_pairs, "exclude_pairs",
                               &exclude_keys, &message)) {
    return failure<CompiledAnalyticCollisionConstraint>(
        SolverStatus::kInvalidInput, message);
  }
  for (const auto &key : include_keys) {
    if (exclude_keys.find(key) != exclude_keys.end()) {
      return failure<CompiledAnalyticCollisionConstraint>(
          SolverStatus::kInvalidInput,
          "native collision pair cannot be both included and excluded");
    }
  }

  std::unordered_map<std::string, double> pair_minimum_distances;
  for (const auto &override : definition.pair_minimum_distances) {
    if (!finite_nonnegative(override.min_distance) ||
        override.pair.geometry_a.empty() ||
        override.pair.geometry_b.empty() ||
        override.pair.geometry_a == override.pair.geometry_b) {
      return failure<CompiledAnalyticCollisionConstraint>(
          SolverStatus::kInvalidInput,
          "native collision pair minimum distance is invalid");
    }
    const auto key = canonical_pair_key(override.pair.geometry_a,
                                        override.pair.geometry_b);
    if (!pair_minimum_distances.emplace(key, override.min_distance).second) {
      return failure<CompiledAnalyticCollisionConstraint>(
          SolverStatus::kInvalidInput,
          "native collision pair minimum distance is duplicated");
    }
  }

  std::unordered_set<std::string> catalog_keys;
  std::unordered_set<std::string> selected_keys;
  CompiledAnalyticCollisionConstraint compiled;
  for (std::size_t pair_index = 0;
       pair_index < geometry_model->collisionPairs.size(); ++pair_index) {
    const auto &pair = geometry_model->collisionPairs[pair_index];
    if (pair.first >= geometry_model->geometryObjects.size() ||
        pair.second >= geometry_model->geometryObjects.size()) {
      return failure<CompiledAnalyticCollisionConstraint>(
          SolverStatus::kInvalidInput,
          "native collision catalog contains an invalid geometry index");
    }
    const auto &object_a = geometry_model->geometryObjects[pair.first];
    const auto &object_b = geometry_model->geometryObjects[pair.second];
    const auto key = canonical_pair_key(object_a.name, object_b.name);
    if (!catalog_keys.insert(key).second) {
      return failure<CompiledAnalyticCollisionConstraint>(
          SolverStatus::kInvalidInput,
          "native collision catalog contains a duplicate unordered pair");
    }
    const bool selected =
        (include_keys.empty() || include_keys.find(key) != include_keys.end()) &&
        exclude_keys.find(key) == exclude_keys.end();
    if (!selected) {
      continue;
    }
    selected_keys.insert(key);
    const auto *sphere_a =
        dynamic_cast<const coal::Sphere *>(object_a.geometry.get());
    const auto *sphere_b =
        dynamic_cast<const coal::Sphere *>(object_b.geometry.get());
    if (sphere_a == nullptr || sphere_b == nullptr) {
      return failure<CompiledAnalyticCollisionConstraint>(
          SolverStatus::kInvalidInput,
          "native collision certification supports only sphere-sphere pairs");
    }
    const double radius_a =
        sphere_a->radius + sphere_a->getSweptSphereRadius();
    const double radius_b =
        sphere_b->radius + sphere_b->getSweptSphereRadius();
    if (!finite_nonnegative(radius_a) || !finite_nonnegative(radius_b)) {
      return failure<CompiledAnalyticCollisionConstraint>(
          SolverStatus::kInvalidInput,
          "native collision catalog contains an invalid sphere radius");
    }
    compiled.pair_indices.push_back(pair_index);
    const auto override = pair_minimum_distances.find(key);
    compiled.minimum_distances.push_back(
        override == pair_minimum_distances.end() ? definition.min_distance
                                                 : override->second);
    compiled.active_pairs.push_back({object_a.name, object_b.name});
  }

  for (const auto &key : include_keys) {
    if (catalog_keys.find(key) == catalog_keys.end()) {
      return failure<CompiledAnalyticCollisionConstraint>(
          SolverStatus::kInvalidInput,
          "native collision include pair is not in the collision catalog");
    }
  }
  for (const auto &key : exclude_keys) {
    if (catalog_keys.find(key) == catalog_keys.end()) {
      return failure<CompiledAnalyticCollisionConstraint>(
          SolverStatus::kInvalidInput,
          "native collision exclude pair is not in the collision catalog");
    }
  }
  for (const auto &entry : pair_minimum_distances) {
    if (catalog_keys.find(entry.first) == catalog_keys.end()) {
      return failure<CompiledAnalyticCollisionConstraint>(
          SolverStatus::kInvalidInput,
          "native collision distance override pair is not in the catalog");
    }
    if (selected_keys.find(entry.first) == selected_keys.end()) {
      return failure<CompiledAnalyticCollisionConstraint>(
          SolverStatus::kInvalidInput,
          "native collision distance override pair is not selected");
    }
  }
  if (!compiled.satisfied()) {
    return failure<CompiledAnalyticCollisionConstraint>(
        SolverStatus::kInvalidInput,
        "native collision filtering selected no sphere-sphere pairs");
  }
  return compiled;
#endif
}

PreparedAnalyticCollisionConstraint prepare_analytic_collision_constraint(
    const RobotModel &robot, CollisionDifferentialScratch &scratch,
    const CompiledAnalyticCollisionConstraint &compiled,
    const CollisionConstraintAccelerationPolicy &policy,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq, double dt,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    const std::string &state_label) {
  if (!compiled.satisfied() || dt <= 0.0 || !std::isfinite(dt) ||
      joint_acceleration_lower.size() != robot.nv() ||
      joint_acceleration_upper.size() != robot.nv() ||
      !joint_acceleration_lower.allFinite() ||
      !joint_acceleration_upper.allFinite() ||
      (joint_acceleration_lower.array() > joint_acceleration_upper.array())
          .any() ||
      state_label.empty()) {
    return failure<PreparedAnalyticCollisionConstraint>(
        SolverStatus::kInvalidInput,
        "native collision preparation inputs are invalid");
  }

  const auto evaluated =
      evaluate_selected_bounded_collision_pair_differentials_at_state(
          robot, scratch, compiled.pair_indices, q, dq);
  if (!evaluated.satisfied()) {
    return failure<PreparedAnalyticCollisionConstraint>(
        evaluated.status, state_label + " native collision differential: " +
                              evaluated.message);
  }

  PreparedAnalyticCollisionConstraint prepared;
  prepared.pair_evaluations = evaluated.pair_evaluations;
  prepared.diagnostics.reserve(evaluated.differentials.size());
  prepared.pair_differentials.reserve(evaluated.differentials.size());
  std::vector<std::size_t> active_indices;
  std::vector<StateBoxRowInput> active_rows;

  for (std::size_t pair = 0; pair < evaluated.differentials.size(); ++pair) {
    const auto &bounded = evaluated.differentials[pair];
    const auto &differential = bounded.nominal;
    const auto expected_key = canonical_pair_key(
        compiled.active_pairs[pair].geometry_a,
        compiled.active_pairs[pair].geometry_b);
    if (differential.pair_index != compiled.pair_indices[pair] ||
        differential.pair_key != expected_key) {
      return failure<PreparedAnalyticCollisionConstraint>(
          SolverStatus::kInvalidInput,
          state_label +
              " native collision catalog changed after configuration");
    }
    const double minimum_distance = compiled.minimum_distances[pair];
    const double boundary_tolerance =
        kBoundaryTolerance *
        (1.0 + std::max(std::abs(differential.signed_distance),
                        minimum_distance));

    AccelerationAnalyticCollisionPairDiagnostics diagnostics;
    diagnostics.pair_index = differential.pair_index;
    diagnostics.pair_key = differential.pair_key;
    diagnostics.object_a = differential.object_a;
    diagnostics.object_b = differential.object_b;
    diagnostics.minimum_distance = minimum_distance;
    diagnostics.current_signed_distance = differential.signed_distance;
    diagnostics.current_signed_distance_lower_bound =
        bounded.signed_distance_lower_bound;
    diagnostics.current_signed_distance_upper_bound =
        bounded.signed_distance_upper_bound;
    diagnostics.current_rate_lower_bound = bounded.rate_lower_bound;
    diagnostics.current_rate_upper_bound = bounded.rate_upper_bound;

    double state_floor = minimum_distance;
    double minimum_rate = -policy.maximum_approach_rate;
    if (differential.signed_distance <
        minimum_distance - boundary_tolerance) {
      diagnostics.regime =
          AccelerationCollisionRegime::kBelowLimitRecovery;
      if (policy.outside_policy == CollisionConstraintOutsidePolicy::kReject) {
        prepared.diagnostics.push_back(std::move(diagnostics));
        prepared.status = SolverStatus::kCollisionViolated;
        prepared.message = state_label + " native collision pair '" +
                           differential.pair_key +
                           "' is below its configured minimum distance";
        return prepared;
      }
      state_floor = differential.signed_distance;
    } else if (differential.signed_distance <=
               minimum_distance + boundary_tolerance) {
      diagnostics.regime = AccelerationCollisionRegime::kExactFloor;
      state_floor = differential.signed_distance;
    } else {
      diagnostics.regime = AccelerationCollisionRegime::kStrictInterior;
    }

    const auto range =
        affine_range(differential.jacobian.row(0), differential.affine_bias,
                     joint_acceleration_lower, joint_acceleration_upper);
    if (!std::isfinite(range.first) || !std::isfinite(range.second) ||
        range.second <
            policy.minimum_braking_acceleration - kConstraintTolerance) {
      prepared.diagnostics.push_back(std::move(diagnostics));
      prepared.status = SolverStatus::kCollisionViolated;
      prepared.message = state_label + " native collision pair '" +
                         differential.pair_key +
                         "' lacks outward braking authority";
      return prepared;
    }

    if (diagnostics.regime ==
        AccelerationCollisionRegime::kBelowLimitRecovery) {
      const auto recovery = compute_reachable_recovery_rate(
          minimum_distance - differential.signed_distance, differential.rate,
          dt, policy.recovery_scale, policy.minimum_recovery_rate,
          policy.maximum_recovery_rate, range.second, boundary_tolerance);
      if (!recovery.satisfied()) {
        prepared.diagnostics.push_back(std::move(diagnostics));
        prepared.status = recovery.status;
        prepared.message = state_label + " native collision pair '" +
                           differential.pair_key +
                           "' recovery rate: " + recovery.message;
        return prepared;
      }
      minimum_rate = recovery.rate;
    }

    bool active = !policy.proximity_activation_enabled ||
                  diagnostics.regime !=
                      AccelerationCollisionRegime::kStrictInterior;
    if (!active) {
      const double maximum_reachable_inward_acceleration =
          std::max(0.0, -range.first);
      const double conservative_inward_displacement =
          std::max(0.0, -differential.rate) * dt +
          0.5 * maximum_reachable_inward_acceleration * dt * dt;
      active =
          differential.signed_distance - conservative_inward_displacement <=
          minimum_distance + policy.activation_margin;
    }
    diagnostics.state_rate_shaping_active = active;
    diagnostics.braking_witness_required = active;
    prepared.diagnostics.push_back(std::move(diagnostics));
    prepared.pair_differentials.push_back(differential);
    if (!active) {
      continue;
    }

    StateBoxRowInput row;
    row.value = differential.signed_distance;
    row.rate = differential.rate;
    row.state_bounds = {state_floor, 0.0, true, false};
    row.rate_bounds = {minimum_rate, 0.0, true, false};
    row.acceleration_bounds = {
        -policy.maximum_inward_acceleration, range.second, true, true};
    row.lower_braking_acceleration =
        policy.minimum_braking_acceleration;
    row.upper_braking_acceleration =
        policy.minimum_braking_acceleration;
    active_indices.push_back(pair);
    active_rows.push_back(row);
  }

  if (active_indices.empty()) {
    return prepared;
  }

  LinearizedStateBoxInput input;
  input.source_id = "native_collision:" + state_label;
  input.coefficient_matrix.resize(
      static_cast<Eigen::Index>(active_indices.size()), robot.nv());
  input.affine_bias.resize(static_cast<Eigen::Index>(active_indices.size()));
  input.rows = std::move(active_rows);
  for (Eigen::Index row = 0;
       row < static_cast<Eigen::Index>(active_indices.size()); ++row) {
    const auto &differential =
        prepared.pair_differentials[active_indices[static_cast<std::size_t>(
            row)]];
    input.coefficient_matrix.row(row) = differential.jacobian;
    input.affine_bias(row) = differential.affine_bias;
  }

  auto shaped =
      prepare_linearized_state_box(input, dt, robot.nv());
  if (shaped.status != SolverStatus::kSuccess) {
    prepared.status = collision_status(shaped.status);
    prepared.message = state_label + " native collision state box: " +
                       shaped.message;
    return prepared;
  }
  prepared.physical_constraint = std::move(shaped.physical_constraint);
  prepared.braking_rows.reserve(active_indices.size());
  for (Eigen::Index row = 0;
       row < static_cast<Eigen::Index>(active_indices.size()); ++row) {
    prepared.braking_rows.push_back(
        {static_cast<int>(row), policy.minimum_braking_acceleration,
         AccelerationBrakingDirection::kIncreaseCoordinate});
  }
  const auto witness = validate_common_acceleration_braking_witness(
      prepared.physical_constraint, joint_acceleration_lower,
      joint_acceleration_upper, prepared.braking_rows,
      state_label + " native collision");
  if (!witness.satisfied()) {
    prepared.status = collision_status(witness.status);
    prepared.message = witness.message;
  }
  return prepared;
}

AnalyticCollisionStepCertificate certify_analytic_collision_step(
    const RobotModel &robot, CollisionDifferentialScratch &scratch,
    const CompiledAnalyticCollisionConstraint &compiled,
    const CollisionConstraintAccelerationPolicy &policy,
    const PreparedAnalyticCollisionConstraint &current,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &accepted_acceleration, double dt,
    const Eigen::VectorXd &q_next, const Eigen::VectorXd &dq_next,
    const Eigen::VectorXd &predicted_joint_acceleration_lower,
    const Eigen::VectorXd &predicted_joint_acceleration_upper) {
  if (!current.satisfied() ||
      current.pair_differentials.size() != compiled.pair_indices.size() ||
      accepted_acceleration.size() != robot.nv() ||
      !accepted_acceleration.allFinite()) {
    return failure<AnalyticCollisionStepCertificate>(
        SolverStatus::kInvalidInput,
        "native collision step certificate inputs are invalid");
  }

  auto predicted = prepare_analytic_collision_constraint(
      robot, scratch, compiled, policy, q_next, dq_next, dt,
      predicted_joint_acceleration_lower,
      predicted_joint_acceleration_upper, "predicted-state");
  if (!predicted.satisfied()) {
    return failure<AnalyticCollisionStepCertificate>(
        predicted.status, predicted.message);
  }
  if ((accepted_acceleration.array() <
       predicted_joint_acceleration_lower.array() - kConstraintTolerance)
          .any() ||
      (accepted_acceleration.array() >
       predicted_joint_acceleration_upper.array() + kConstraintTolerance)
          .any() ||
      !affine_constraint_accepts(predicted.physical_constraint,
                                 accepted_acceleration)) {
    return failure<AnalyticCollisionStepCertificate>(
        SolverStatus::kCollisionViolated,
        "accepted acceleration is not executable in the predicted native "
        "collision hard set");
  }

  const double half_dt = 0.5 * dt;
  Eigen::VectorXd q_mid;
  try {
    q_mid = robot.integrate(
        q, half_dt * dq +
               0.5 * half_dt * half_dt * accepted_acceleration);
  } catch (const std::exception &error) {
    return failure<AnalyticCollisionStepCertificate>(
        SolverStatus::kNumericalError,
        std::string("native collision midpoint integration failed: ") +
            error.what());
  }
  const Eigen::VectorXd dq_mid =
      dq + half_dt * accepted_acceleration;
  const auto midpoint =
      evaluate_selected_bounded_collision_pair_differentials_at_state(
          robot, scratch, compiled.pair_indices, q_mid, dq_mid);
  if (!midpoint.satisfied()) {
    return failure<AnalyticCollisionStepCertificate>(
        midpoint.status,
        "native collision midpoint differential: " + midpoint.message);
  }
  if (midpoint.differentials.size() != compiled.pair_indices.size() ||
      predicted.pair_differentials.size() != compiled.pair_indices.size()) {
    return failure<AnalyticCollisionStepCertificate>(
        SolverStatus::kShapeMismatch,
        "native collision path evidence has inconsistent pair counts");
  }

  AnalyticCollisionStepCertificate certificate;
  certificate.diagnostics = current.diagnostics;
  certificate.pair_evaluations =
      predicted.pair_evaluations + midpoint.pair_evaluations;
  for (std::size_t pair = 0; pair < compiled.pair_indices.size(); ++pair) {
    const auto &initial = current.pair_differentials[pair];
    const auto &middle = midpoint.differentials[pair].nominal;
    const auto &endpoint = predicted.pair_differentials[pair];
    if (initial.pair_key != middle.pair_key ||
        initial.pair_key != endpoint.pair_key) {
      return failure<AnalyticCollisionStepCertificate>(
          SolverStatus::kShapeMismatch,
          "native collision path pair ordering changed during certification");
    }
    const Eigen::Vector3d relative_position =
        initial.center_b_world - initial.center_a_world;
    const Eigen::Vector3d relative_velocity =
        initial.relative_center_velocity_world;
    const Eigen::Vector3d relative_acceleration =
        initial.relative_center_jacobian * accepted_acceleration +
        initial.relative_center_affine_acceleration_world;
    const double combined_radius =
        relative_position.norm() - initial.signed_distance;
    const auto expected_position = [&](double time) {
      return relative_position + time * relative_velocity +
             0.5 * time * time * relative_acceleration;
    };
    const auto expected_velocity = [&](double time) {
      return relative_velocity + time * relative_acceleration;
    };
    const Eigen::Vector3d middle_position =
        middle.center_b_world - middle.center_a_world;
    const Eigen::Vector3d endpoint_position =
        endpoint.center_b_world - endpoint.center_a_world;
    if (!finite_nonnegative(combined_radius) ||
        !approximately_equal(middle_position, expected_position(half_dt)) ||
        !approximately_equal(endpoint_position, expected_position(dt)) ||
        !approximately_equal(middle.relative_center_velocity_world,
                             expected_velocity(half_dt)) ||
        !approximately_equal(endpoint.relative_center_velocity_world,
                             expected_velocity(dt)) ||
        !approximately_equal(
            middle.signed_distance,
            expected_position(half_dt).norm() - combined_radius) ||
        !approximately_equal(
            endpoint.signed_distance,
            expected_position(dt).norm() - combined_radius)) {
      return failure<AnalyticCollisionStepCertificate>(
          SolverStatus::kNoProgress,
          "native collision pair '" + initial.pair_key +
              "' did not satisfy the constant-acceleration sphere-center "
              "model");
    }

    const auto regime = certificate.diagnostics[pair].regime;
    const double required_floor =
        regime == AccelerationCollisionRegime::kStrictInterior
            ? compiled.minimum_distances[pair]
            : initial.signed_distance;
    auto path = certify_quadratic_sphere_path(
        relative_position, relative_velocity, relative_acceleration,
        combined_radius, required_floor, dt, initial.pair_key);
    certificate.path_visited_nodes += path.visited_nodes;
    certificate.certified_intervals += path.certified_intervals;
    if (path.status != SolverStatus::kSuccess) {
      certificate.status = path.status;
      certificate.message = path.message;
      return certificate;
    }
    auto &diagnostics = certificate.diagnostics[pair];
    diagnostics.endpoint_signed_distance_lower_bound =
        endpoint.signed_distance;
    diagnostics.lowest_path_signed_distance_lower_bound =
        path.lowest_certified_lower_bound;
    diagnostics.step_certified = true;
  }
  certificate.endpoint_validated = true;
  certificate.step_certified = true;
  return certificate;
}

} // namespace embodik::detail
