#include "acceleration_relative_pose_constraint.hpp"

#include "acceleration_geometric_constraint_policy.hpp"
#include "geometric_constraint_differential.hpp"
#include "so3_log_differential.hpp"

#include <pinocchio/multibody/data.hpp>

#include <exception>
#include <string>
#include <unordered_set>
#include <utility>

namespace embodik::detail {
namespace {

RelativePoseConstraintResult failure(SolverStatus status, std::string message) {
  RelativePoseConstraintResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

RelativePoseConstraintResult as_relative_result(
    const ScalarGeometricConstraintResult &result) {
  return failure(result.status, result.message);
}

std::string identity(const RelativePoseAccelerationConstraint &constraint) {
  return "relative pose constraint '" + constraint.source_id + "'";
}

bool has_rotation_axis(const std::vector<int> &active_axes) {
  for (int axis : active_axes) {
    if (axis >= 3) {
      return true;
    }
  }
  return false;
}

ScalarGeometricConstraintSpecification make_scalar_specification(
    const RelativePoseAccelerationConstraint &constraint,
    std::vector<int> active_axes) {
  ScalarGeometricConstraintSpecification specification;
  specification.family_label = "relative pose constraint";
  specification.source_id = constraint.source_id;
  specification.dimension = 6;
  specification.active_axes = std::move(active_axes);
  specification.state_lower_bounds = constraint.definition.lower_bounds;
  specification.state_upper_bounds = constraint.definition.upper_bounds;
  specification.policy = constraint.policy;
  return specification;
}

ScalarGeometricCoordinateSample
pose_coordinate_sample(const GeometricCoordinateDifferential &differential,
                       const Eigen::VectorXd &ddq) {
  ScalarGeometricCoordinateSample sample;
  sample.value = differential.value;
  sample.rate = differential.rate;
  sample.acceleration = differential.jacobian * ddq + differential.affine_bias;
  sample.so3_rotation = differential.so3_rotation;
  return sample;
}

GeometricCoordinateDifferential promote_relative_translation(
    const GeometricCoordinateDifferential &translation,
    Eigen::Index variable_count) {
  return promote_translation_to_pose_differential(translation, variable_count);
}

GeometricCoordinateDifferential evaluate_pose_differential(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b, bool rotation_active) {
  if (rotation_active) {
    return evaluate_relative_pose_differential(robot, frame_a, frame_b);
  }
  return promote_relative_translation(
      evaluate_relative_pose_translation_differential(robot, frame_a, frame_b),
      robot.nv());
}

GeometricCoordinateDifferential evaluate_pose_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_a, const std::string &frame_b,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    bool rotation_active) {
  if (rotation_active) {
    return evaluate_relative_pose_differential_at_state(
        robot, scratch, frame_a, frame_b, q, dq);
  }
  return promote_relative_translation(
      evaluate_relative_pose_translation_differential_at_state(
          robot, scratch, frame_a, frame_b, q, dq),
      dq.size());
}

ScalarGeometricDifferential make_scalar_differential(
    const GeometricCoordinateDifferential &differential,
    const Eigen::VectorXd &ddq) {
  ScalarGeometricDifferential scalar;
  scalar.coefficient_matrix = differential.jacobian;
  scalar.affine_bias = differential.affine_bias;
  scalar.coordinates = pose_coordinate_sample(differential, ddq);
  return scalar;
}

void collect_ancestor_joints(const pinocchio::Model &model,
                             pinocchio::JointIndex joint,
                             std::unordered_set<pinocchio::JointIndex> *out) {
  while (joint > 0) {
    out->insert(joint);
    joint = model.parents[joint];
  }
}

RelativePoseConstraintResult validate_relative_pose_constraint(
    const RelativePoseAccelerationConstraint &constraint,
    const RobotModel &robot, std::vector<int> *active_axes,
    bool *rotation_active) {
  if (constraint.source_id.empty()) {
    return failure(SolverStatus::kInvalidInput,
                   "relative pose constraint source_id must not be empty");
  }
  const auto &definition = constraint.definition;
  if (definition.frame_a.empty() || definition.frame_b.empty()) {
    return failure(SolverStatus::kInvalidInput,
                   identity(constraint) +
                       " frame_a and frame_b must not be empty");
  }
  if (definition.frame_a == definition.frame_b) {
    return failure(SolverStatus::kInvalidInput,
                   identity(constraint) +
                       " frame_a and frame_b must be distinct");
  }
  if (!robot.has_frame(definition.frame_a) ||
      !robot.has_frame(definition.frame_b)) {
    return failure(SolverStatus::kInvalidInput,
                   identity(constraint) +
                       " frame_a and frame_b must identify robot frames");
  }
  if (!definition.lower_bounds.allFinite() ||
      !definition.upper_bounds.allFinite()) {
    return failure(SolverStatus::kNonFiniteInput,
                   identity(constraint) +
                       " bounds must contain only finite values");
  }
  if ((definition.lower_bounds.array() > definition.upper_bounds.array())
          .any()) {
    return failure(SolverStatus::kInvalidInput,
                   identity(constraint) +
                       " lower bounds exceed upper bounds");
  }
  const Eigen::VectorXd axis_mask = definition.axis_mask;
  auto axis_validation = validate_geometric_constraint_axis_mask(
      "relative pose constraint", constraint.source_id, axis_mask, 6);
  if (!axis_validation.satisfied()) {
    return failure(axis_validation.status, axis_validation.message);
  }
  auto policy_validation = validate_geometric_constraint_policy(
      "relative pose constraint", constraint.source_id, constraint.policy, 6);
  if (!policy_validation.satisfied()) {
    return failure(policy_validation.status, policy_validation.message);
  }
  *active_axes = std::move(axis_validation.active_axes);
  *rotation_active = has_rotation_axis(*active_axes);
  return {};
}

RelativePoseConstraintResult validate_nonlinear_path(
    const PreparedRelativePoseConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &ddq, double dt) {
  pinocchio::Data scratch(robot.model());
  const ScalarGeometricPathValidationOptions options;
  const auto &definition = prepared.specification.definition;
  const auto path = validate_scalar_geometric_path(
      prepared.scalar.specification, dq, ddq, dt, options,
      [&](const ScalarGeometricPathPoint &path_point,
          ScalarGeometricCoordinateSample *coordinates) {
        Eigen::VectorXd sample_q;
        try {
          sample_q = robot.integrate(q, path_point.tangent);
        } catch (const std::exception &error) {
          return ScalarGeometricConstraintResult{
              SolverStatus::kNumericalError,
              "accepted " + identity(prepared.specification) +
                  " nonlinear path integration failed: " + error.what()};
        }
        try {
          const auto differential = evaluate_pose_differential_at_state(
              robot, scratch, definition.frame_a, definition.frame_b, sample_q,
              path_point.rate, prepared.rotation_active);
          *coordinates = pose_coordinate_sample(differential, ddq);
        } catch (const std::domain_error &error) {
          return ScalarGeometricConstraintResult{
              SolverStatus::kNumericalError,
              "accepted " + identity(prepared.specification) +
                  " nonlinear path entered the SO(3) near-pi chart guard: " +
                  error.what()};
        } catch (const std::exception &error) {
          return ScalarGeometricConstraintResult{
              SolverStatus::kNumericalError,
              "accepted " + identity(prepared.specification) +
                  " nonlinear path evaluation failed: " + error.what()};
        }
        return ScalarGeometricConstraintResult{};
      },
      [&](const ScalarGeometricPathSegment &segment) {
        if (!prepared.rotation_active) {
          return ScalarGeometricConstraintResult{};
        }
        const auto guard = validate_so3_log_rotation_segment(
            segment, prepared.angular_path_support_mask);
        if (!guard.satisfied()) {
          return ScalarGeometricConstraintResult{
              guard.status, "accepted " + identity(prepared.specification) +
                                " nonlinear path segment failed SO(3) chart "
                                "guard: " +
                                guard.message};
        }
        return ScalarGeometricConstraintResult{};
      });
  return as_relative_result(path);
}

} // namespace

Eigen::VectorXd relative_pose_angular_path_support_mask(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b) {
  Eigen::VectorXd mask = Eigen::VectorXd::Zero(robot.nv());
  if (!robot.has_frame(frame_a) || !robot.has_frame(frame_b)) {
    return mask;
  }

  const auto &model = robot.model();
  const pinocchio::FrameIndex frame_a_id = model.getFrameId(frame_a);
  const pinocchio::FrameIndex frame_b_id = model.getFrameId(frame_b);
  std::unordered_set<pinocchio::JointIndex> ancestors_a;
  std::unordered_set<pinocchio::JointIndex> ancestors_b;
  collect_ancestor_joints(model, model.frames[frame_a_id].parentJoint,
                          &ancestors_a);
  collect_ancestor_joints(model, model.frames[frame_b_id].parentJoint,
                          &ancestors_b);

  for (pinocchio::JointIndex joint = 1; joint < model.joints.size(); ++joint) {
    const bool in_a = ancestors_a.count(joint) != 0U;
    const bool in_b = ancestors_b.count(joint) != 0U;
    if (in_a == in_b) {
      continue;
    }
    const auto &joint_model = model.joints[joint];
    const int start = joint_model.idx_v();
    const int count = joint_model.nv();
    for (int offset = 0; offset < count; ++offset) {
      const int velocity_index = start + offset;
      if (velocity_index >= 0 && velocity_index < mask.size()) {
        mask(velocity_index) = 1.0;
      }
    }
  }
  return mask;
}

RelativePosePreparationResult prepare_relative_pose_constraint(
    const RelativePoseAccelerationConstraint &constraint,
    const RobotModel &robot, double dt,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper) {
  RelativePosePreparationResult result;
  std::vector<int> active_axes;
  bool rotation_active = false;
  const auto validation = validate_relative_pose_constraint(
      constraint, robot, &active_axes, &rotation_active);
  if (!validation.satisfied()) {
    result.status = validation.status;
    result.message = validation.message;
    return result;
  }

  GeometricCoordinateDifferential differential;
  try {
    differential = evaluate_pose_differential(
        robot, constraint.definition.frame_a, constraint.definition.frame_b,
        rotation_active);
  } catch (const std::domain_error &error) {
    result.status = SolverStatus::kInvalidInput;
    result.message = identity(constraint) +
                     " is inside the SO(3) near-pi chart guard: " +
                     error.what();
    return result;
  } catch (const std::exception &error) {
    result.status = SolverStatus::kNumericalError;
    result.message = identity(constraint) +
                     " evaluation failed: " + error.what();
    return result;
  }

  if (differential.jacobian.rows() != 6 ||
      differential.jacobian.cols() != robot.nv() ||
      differential.affine_bias.size() != 6 ||
      differential.value.size() != 6 || differential.rate.size() != 6 ||
      !differential.value.allFinite() || !differential.rate.allFinite() ||
      !differential.jacobian.allFinite() ||
      !differential.affine_bias.allFinite()) {
    result.status = SolverStatus::kNumericalError;
    result.message = identity(constraint) +
                     " differential was non-finite or inconsistent";
    return result;
  }

  const Eigen::VectorXd zero_acceleration =
      Eigen::VectorXd::Zero(robot.nv());
  const auto scalar_specification =
      make_scalar_specification(constraint, active_axes);
  auto scalar_preparation = prepare_scalar_geometric_constraint(
      scalar_specification, make_scalar_differential(differential,
                                                     zero_acceleration),
      dt, robot.nv(), joint_acceleration_lower, joint_acceleration_upper);
  if (!scalar_preparation.satisfied()) {
    result.status = scalar_preparation.status;
    result.message = scalar_preparation.message;
    return result;
  }

  PreparedRelativePoseConstraint prepared;
  prepared.specification = constraint;
  prepared.scalar = std::move(scalar_preparation.prepared.value());
  prepared.active_axes = std::move(active_axes);
  prepared.rotation_active = rotation_active;
  prepared.angular_path_support_mask =
      rotation_active ? relative_pose_angular_path_support_mask(
                            robot, constraint.definition.frame_a,
                            constraint.definition.frame_b)
                      : Eigen::VectorXd::Zero(robot.nv());
  result.prepared = std::move(prepared);
  return result;
}

RelativePoseConstraintResult validate_relative_pose_constraint_acceptance(
    const PreparedRelativePoseConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &accepted_acceleration, double dt,
    const Eigen::VectorXd &predicted_joint_acceleration_lower,
    const Eigen::VectorXd &predicted_joint_acceleration_upper) {
  const auto linearized = validate_scalar_geometric_linearized_acceptance(
      prepared.scalar, accepted_acceleration, dt);
  if (!linearized.satisfied()) {
    return as_relative_result(linearized);
  }

  const auto nonlinear =
      validate_nonlinear_path(prepared, robot, q, dq, accepted_acceleration,
                              dt);
  if (!nonlinear.satisfied()) {
    return nonlinear;
  }

  Eigen::VectorXd q_next;
  try {
    q_next =
        robot.integrate(q, dt * dq + 0.5 * dt * dt * accepted_acceleration);
  } catch (const std::exception &error) {
    return failure(SolverStatus::kNumericalError,
                   "accepted " + identity(prepared.specification) +
                       " predicted support integration failed: " +
                       error.what());
  }
  const Eigen::VectorXd dq_next = dq + dt * accepted_acceleration;
  const auto &definition = prepared.specification.definition;
  try {
    pinocchio::Data scratch(robot.model());
    const auto predicted = evaluate_pose_differential_at_state(
        robot, scratch, definition.frame_a, definition.frame_b, q_next,
        dq_next, prepared.rotation_active);
    const auto support = validate_scalar_geometric_joint_box_support(
        prepared.scalar.specification,
        make_scalar_differential(predicted, accepted_acceleration),
        predicted_joint_acceleration_lower, predicted_joint_acceleration_upper,
        "predicted next-state");
    if (!support.satisfied()) {
      return failure(SolverStatus::kNumericalError, support.message);
    }
  } catch (const std::domain_error &error) {
    return failure(SolverStatus::kNumericalError,
                   "accepted " + identity(prepared.specification) +
                       " predicted next-state entered the SO(3) near-pi chart "
                       "guard: " +
                       error.what());
  } catch (const std::exception &error) {
    return failure(SolverStatus::kNumericalError,
                   "accepted " + identity(prepared.specification) +
                       " predicted support evaluation failed: " +
                       error.what());
  }
  return {};
}

} // namespace embodik::detail
