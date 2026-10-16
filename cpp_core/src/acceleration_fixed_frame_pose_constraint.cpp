#include "acceleration_fixed_frame_pose_constraint.hpp"

#include "acceleration_geometric_constraint_policy.hpp"
#include "frame_kinematic_differential.hpp"
#include "geometric_constraint_differential.hpp"
#include "so3_log_differential.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <string>
#include <utility>

namespace embodik::detail {
namespace {

FixedFramePoseConstraintResult failure(SolverStatus status,
                                       std::string message) {
  FixedFramePoseConstraintResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

FixedFramePoseConstraintResult as_pose_result(
    const ScalarGeometricConstraintResult &result) {
  auto converted = failure(result.status, result.message);
  converted.violation = result.violation;
  return converted;
}

std::string identity(const FixedFramePoseConstraintRecord &constraint) {
  return constraint.family_label + " '" + constraint.source_id + "'";
}

bool has_rotation_axis(const std::vector<int> &active_axes) {
  for (int axis : active_axes) {
    if (axis >= 3) {
      return true;
    }
  }
  return false;
}

FixedFramePoseConstraintResult pose_from_matrix(
    const std::string &family_label, const std::string &source_id,
    const Eigen::Matrix4d &matrix, pinocchio::SE3 *pose) {
  if (!matrix.allFinite()) {
    return failure(SolverStatus::kNonFiniteInput,
                   family_label + " '" + source_id +
                       "' reference pose must be finite");
  }
  const Eigen::RowVector4d expected_bottom(0.0, 0.0, 0.0, 1.0);
  if (!matrix.row(3).isApprox(expected_bottom, 1e-10)) {
    return failure(SolverStatus::kInvalidInput,
                   family_label + " '" + source_id +
                       "' reference pose must be homogeneous");
  }
  const Eigen::Matrix3d rotation = matrix.topLeftCorner<3, 3>();
  if (!is_valid_so3_rotation(rotation)) {
    return failure(SolverStatus::kInvalidInput,
                   family_label + " '" + source_id +
                       "' reference pose rotation is invalid");
  }
  *pose = pinocchio::SE3(rotation, matrix.topRightCorner<3, 1>());
  return {};
}

ScalarGeometricConstraintSpecification make_scalar_specification(
    const FixedFramePoseConstraintRecord &constraint,
    std::vector<int> active_axes) {
  ScalarGeometricConstraintSpecification specification;
  specification.family_label = constraint.family_label;
  specification.source_id = constraint.source_id;
  specification.dimension = 6;
  specification.active_axes = std::move(active_axes);
  specification.state_lower_bounds = constraint.lower_bounds;
  specification.state_upper_bounds = constraint.upper_bounds;
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

GeometricCoordinateDifferential translation_only_differential(
    const FrameKinematicDifferential &frame,
    const pinocchio::SE3 &reference_pose, const Eigen::VectorXd &velocity) {
  GeometricCoordinateDifferential translation;
  translation.value =
      frame.pose.translation() - reference_pose.translation();
  translation.jacobian = frame.jacobian.topRows<3>();
  translation.rate = translation.jacobian * velocity;
  translation.affine_bias = frame.affine_bias.head<3>();
  return promote_translation_to_pose_differential(translation, velocity.size());
}

GeometricCoordinateDifferential evaluate_pose_differential(
    const RobotModel &robot, const std::string &frame_name,
    const pinocchio::SE3 &reference_pose, bool rotation_active) {
  if (rotation_active) {
    return evaluate_fixed_frame_pose_differential(robot, frame_name,
                                                  reference_pose);
  }
  return translation_only_differential(
      evaluate_frame_kinematic_differential(robot, frame_name), reference_pose,
      robot.get_current_velocity());
}

GeometricCoordinateDifferential evaluate_pose_differential_at_state(
    const RobotModel &robot, pinocchio::Data &scratch,
    const std::string &frame_name, const pinocchio::SE3 &reference_pose,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    bool rotation_active) {
  if (rotation_active) {
    return evaluate_fixed_frame_pose_differential_at_state(
        robot, scratch, frame_name, reference_pose, q, dq);
  }
  return translation_only_differential(
      evaluate_frame_kinematic_differential_at_state(robot, scratch, frame_name,
                                                     q, dq),
      reference_pose, dq);
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

FixedFramePoseConstraintResult validate_nonlinear_path(
    const PreparedFixedFramePoseConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &ddq, double dt,
    const pinocchio::SE3 &reference_pose) {
  pinocchio::Data scratch(robot.model());
  const ScalarGeometricPathValidationOptions options;
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
              robot, scratch, prepared.specification.frame_name, reference_pose,
              sample_q, path_point.rate, prepared.rotation_active);
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
        const auto guard = validate_so3_log_rotation_segment(segment);
        if (!guard.satisfied()) {
          return ScalarGeometricConstraintResult{
              guard.status, "accepted " + identity(prepared.specification) +
                                " nonlinear path segment failed SO(3) chart "
                                "guard: " +
                                guard.message};
        }
        return ScalarGeometricConstraintResult{};
      });
  return as_pose_result(path);
}

FixedFramePoseConstraintResult validate_fixed_frame_pose_constraint(
    const FixedFramePoseConstraintRecord &constraint, const RobotModel &robot,
    std::vector<int> *active_axes, bool *rotation_active,
    pinocchio::SE3 *reference_pose) {
  if (constraint.family_label.empty() || constraint.source_id.empty()) {
    return failure(SolverStatus::kInvalidInput,
                   "fixed frame pose constraint identity is invalid");
  }
  if (constraint.frame_name.empty() || !robot.has_frame(constraint.frame_name)) {
    return failure(SolverStatus::kInvalidInput,
                   identity(constraint) +
                       " frame_name must identify a robot frame");
  }
  auto pose_result = pose_from_matrix(constraint.family_label,
                                      constraint.source_id,
                                      constraint.reference_pose,
                                      reference_pose);
  if (!pose_result.satisfied()) {
    return pose_result;
  }
  if (!constraint.lower_bounds.allFinite() ||
      !constraint.upper_bounds.allFinite()) {
    return failure(SolverStatus::kNonFiniteInput,
                   identity(constraint) +
                       " bounds must contain only finite values");
  }
  if (constraint.family_label == "tight frame pose constraint") {
    const bool positive_epsilons =
        constraint.lower_bounds.head<3>().isConstant(constraint.lower_bounds(0),
                                                     0.0) &&
        constraint.upper_bounds.head<3>().isConstant(constraint.upper_bounds(0),
                                                     0.0) &&
        constraint.lower_bounds.tail<3>().isConstant(constraint.lower_bounds(3),
                                                     0.0) &&
        constraint.upper_bounds.tail<3>().isConstant(constraint.upper_bounds(3),
                                                     0.0) &&
        constraint.lower_bounds(0) < 0.0 && constraint.upper_bounds(0) > 0.0 &&
        constraint.lower_bounds(3) < 0.0 && constraint.upper_bounds(3) > 0.0;
    if (!positive_epsilons) {
      return failure(SolverStatus::kInvalidInput,
                     identity(constraint) +
                         " epsilons must be finite and greater than zero");
    }
  }
  if ((constraint.lower_bounds.array() > constraint.upper_bounds.array())
          .any()) {
    return failure(SolverStatus::kInvalidInput,
                   identity(constraint) +
                       " lower bounds exceed upper bounds");
  }
  const Eigen::VectorXd axis_mask = constraint.axis_mask;
  auto axis_validation = validate_geometric_constraint_axis_mask(
      constraint.family_label, constraint.source_id, axis_mask, 6);
  if (!axis_validation.satisfied()) {
    return failure(axis_validation.status, axis_validation.message);
  }
  auto policy_validation = validate_geometric_constraint_policy(
      constraint.family_label, constraint.source_id, constraint.policy, 6);
  if (!policy_validation.satisfied()) {
    return failure(policy_validation.status, policy_validation.message);
  }
  *active_axes = std::move(axis_validation.active_axes);
  *rotation_active = has_rotation_axis(*active_axes);
  return {};
}

} // namespace

FixedFramePoseConstraintRecord make_tight_frame_pose_record(
    const TightFramePoseAccelerationConstraint &constraint) {
  FixedFramePoseConstraintRecord record;
  record.family_label = "tight frame pose constraint";
  record.source_id = constraint.source_id;
  record.frame_name = constraint.definition.frame_name;
  record.reference_pose = constraint.definition.target_pose;
  record.lower_bounds.head<3>().setConstant(
      -constraint.definition.position_epsilon);
  record.upper_bounds.head<3>().setConstant(
      constraint.definition.position_epsilon);
  record.lower_bounds.tail<3>().setConstant(
      -constraint.definition.orientation_epsilon);
  record.upper_bounds.tail<3>().setConstant(
      constraint.definition.orientation_epsilon);
  record.axis_mask = constraint.definition.axis_mask;
  record.policy = constraint.policy;
  return record;
}

FixedFramePoseConstraintResult make_torso_pose_bound_record(
    const TorsoPoseBoundAccelerationConstraint &constraint,
    FixedFramePoseConstraintRecord *record) {
  if (!record) {
    return failure(SolverStatus::kInvalidInput,
                   "torso pose bound constraint output record is missing");
  }
  if (!constraint.definition.reference_pose.has_value()) {
    return failure(SolverStatus::kInvalidInput,
                   "torso pose bound constraint '" + constraint.source_id +
                       "' reference_pose is required");
  }
  record->family_label = "torso pose bound constraint";
  record->source_id = constraint.source_id;
  record->frame_name = constraint.definition.frame_name;
  record->reference_pose = constraint.definition.reference_pose.value();
  record->lower_bounds = constraint.definition.lower_bounds;
  record->upper_bounds = constraint.definition.upper_bounds;
  record->axis_mask = constraint.definition.axis_mask;
  record->policy = constraint.policy;
  return {};
}

FixedFramePosePreparationResult prepare_fixed_frame_pose_constraint(
    const FixedFramePoseConstraintRecord &constraint, const RobotModel &robot,
    double dt, const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper) {
  FixedFramePosePreparationResult result;
  std::vector<int> active_axes;
  bool rotation_active = false;
  pinocchio::SE3 reference_pose;
  const auto validation = validate_fixed_frame_pose_constraint(
      constraint, robot, &active_axes, &rotation_active, &reference_pose);
  if (!validation.satisfied()) {
    result.status = validation.status;
    result.message = validation.message;
    return result;
  }

  GeometricCoordinateDifferential differential;
  try {
    differential = evaluate_pose_differential(
        robot, constraint.frame_name, reference_pose, rotation_active);
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

  PreparedFixedFramePoseConstraint prepared;
  prepared.specification = constraint;
  prepared.scalar = std::move(scalar_preparation.prepared.value());
  prepared.active_axes = std::move(active_axes);
  prepared.rotation_active = rotation_active;
  result.prepared = std::move(prepared);
  return result;
}

FixedFramePoseConstraintResult validate_fixed_frame_pose_constraint_acceptance(
    const PreparedFixedFramePoseConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &accepted_acceleration, double dt,
    const Eigen::VectorXd &predicted_joint_acceleration_lower,
    const Eigen::VectorXd &predicted_joint_acceleration_upper) {
  const auto linearized = validate_scalar_geometric_linearized_acceptance(
      prepared.scalar, accepted_acceleration, dt);
  if (!linearized.satisfied()) {
    return as_pose_result(linearized);
  }

  pinocchio::SE3 reference_pose;
  auto pose_result = pose_from_matrix(prepared.specification.family_label,
                                      prepared.specification.source_id,
                                      prepared.specification.reference_pose,
                                      &reference_pose);
  if (!pose_result.satisfied()) {
    return pose_result;
  }

  const auto nonlinear =
      validate_nonlinear_path(prepared, robot, q, dq, accepted_acceleration,
                              dt, reference_pose);
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
  try {
    pinocchio::Data scratch(robot.model());
    const auto predicted = evaluate_pose_differential_at_state(
        robot, scratch, prepared.specification.frame_name, reference_pose,
        q_next, dq_next, prepared.rotation_active);
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

FixedFramePoseConstraintResult linearize_fixed_frame_pose_violation(
    const PreparedFixedFramePoseConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &acceleration, const ScalarGeometricViolation &violation,
    AffineAccelerationConstraint *refinement) {
  const int axis = violation.axis;
  const double time = violation.time;
  if (!refinement || axis < 0 || axis >= 6 || !(time > 0.0) ||
      !std::isfinite(time) || q.size() != robot.nq() ||
      dq.size() != robot.nv() || acceleration.size() != robot.nv() ||
      !q.allFinite() || !dq.allFinite() || !acceleration.allFinite()) {
    return failure(SolverStatus::kInvalidInput,
                   "frame pose refinement inputs are invalid");
  }
  try {
    const auto &model = robot.model();
    const Eigen::VectorXd tangent = time * dq + 0.5 * time * time * acceleration;
    const Eigen::VectorXd sample_q = robot.integrate(q, tangent);
    const Eigen::VectorXd sample_dq = dq + time * acceleration;
    pinocchio::Data scratch(model);
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(robot.nv());
    pinocchio::computeForwardKinematicsDerivatives(model, scratch, sample_q,
                                                 sample_dq, zero);
    Eigen::MatrixXd rate_q = Eigen::MatrixXd::Zero(6, robot.nv());
    Eigen::MatrixXd rate_v = rate_q;
    const auto frame_id = model.getFrameId(prepared.specification.frame_name);
    pinocchio::getFrameVelocityDerivatives(
        model, scratch, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, rate_q, rate_v);
    // Pinocchio's spatial derivative holds the expression origin fixed.
    // The point's classical linear velocity also changes as that origin moves.
    const Eigen::Vector3d angular_rate = rate_v.bottomRows<3>() * sample_dq;
    for (Eigen::Index joint = 0; joint < robot.nv(); ++joint) {
      rate_q.col(joint).head<3>() +=
          angular_rate.cross(rate_v.col(joint).head<3>());
    }
    Eigen::MatrixXd integration = Eigen::MatrixXd::Zero(robot.nv(), robot.nv());
    pinocchio::dIntegrate(model, q, tangent, integration, pinocchio::ARG1);
    const Eigen::MatrixXd position_a = 0.5 * time * time * integration;
    Eigen::RowVectorXd coordinate_gradient;
    Eigen::RowVectorXd rate_gradient;
    double coordinate;
    double rate;
    if (axis < 3) {
      coordinate_gradient = rate_v.row(axis) * position_a;
      rate_gradient = rate_q.row(axis) * position_a + time * rate_v.row(axis);
      coordinate = scratch.oMf[frame_id].translation()(axis) -
                   prepared.specification.reference_pose(axis, 3);
      rate = rate_v.row(axis).dot(sample_dq);
    } else {
      const Eigen::Matrix3d body_rotation =
          scratch.oMf[frame_id].rotation().transpose();
      const Eigen::MatrixXd body_jacobian =
          body_rotation * rate_v.bottomRows<3>();
      const Eigen::Vector3d body_rate = body_jacobian * sample_dq;
      Eigen::MatrixXd body_rate_q = body_rotation * rate_q.bottomRows<3>();
      for (Eigen::Index joint = 0; joint < robot.nv(); ++joint) {
        body_rate_q.col(joint) +=
            body_rate.cross(body_jacobian.col(joint).head<3>());
      }
      const Eigen::Matrix3d error_rotation =
          prepared.specification.reference_pose.topLeftCorner<3, 3>().transpose() *
          scratch.oMf[frame_id].rotation();
      const auto log = evaluate_so3_log_differential(
          error_rotation, body_jacobian, Eigen::Vector3d::Zero(), sample_dq);
      const Eigen::MatrixXd log_rate_q =
          evaluate_so3_log_rate_configuration_gradient(
              error_rotation, body_jacobian, body_rate, body_rate_q);
      coordinate_gradient = log.jacobian.row(axis - 3) * position_a;
      rate_gradient = log_rate_q.row(axis - 3) * position_a +
                      time * log.jacobian.row(axis - 3);
      coordinate = log.value(axis - 3);
      rate = log.rate(axis - 3);
    }
    const double direction = violation.upper ? 1.0 : -1.0;
    const double bound = violation.upper
        ? prepared.specification.upper_bounds(axis)
        : prepared.specification.lower_bounds(axis);
    double residual = direction * (coordinate - bound);
    Eigen::RowVectorXd gradient = direction * coordinate_gradient;
    if (violation.kind == ScalarGeometricViolationKind::kBrakingViability) {
      const auto &policy = prepared.specification.policy;
      const double braking = violation.upper
          ? policy.upper_braking_accelerations(axis)
          : policy.lower_braking_accelerations(axis);
      if (!(direction * rate > 0.0)) {
        return failure(SolverStatus::kNumericalError,
                       "frame pose refinement braking side is inconsistent");
      }
      residual = rate * rate + 2.0 * braking * residual;
      gradient = 2.0 * rate * rate_gradient +
                 2.0 * braking * direction * coordinate_gradient;
    }
    if (!std::isfinite(residual) || residual <= 0.0 || !gradient.allFinite() ||
        !(gradient.norm() > 1e-14)) {
      return failure(SolverStatus::kNumericalError,
                     "frame pose refinement has no finite violated local model");
    }
    // A local model is not a nonlinear certificate. Retain every original row,
    // bound the per-call correction, and require the complete acceptance pass.
    const Eigen::Index rows = robot.nv() + 1;
    refinement->coefficient_matrix = Eigen::MatrixXd::Zero(rows, robot.nv());
    refinement->coefficient_matrix.row(0) = gradient;
    refinement->coefficient_matrix.bottomRows(robot.nv()).setIdentity();
    refinement->affine_bias = Eigen::VectorXd::Zero(rows);
    refinement->affine_bias(0) = residual - gradient.dot(acceleration);
    refinement->lower_bounds = Eigen::VectorXd::Zero(rows);
    refinement->upper_bounds = Eigen::VectorXd::Zero(rows);
    const double radius = std::max(0.01, 0.1 * std::max(1.0, acceleration.cwiseAbs().maxCoeff()));
    refinement->lower_bounds.tail(robot.nv()) = acceleration.array() - radius;
    refinement->upper_bounds.tail(robot.nv()) = acceleration.array() + radius;
    refinement->lower_bound_active.assign(rows, true);
    refinement->lower_bound_active[0] = false;
    refinement->upper_bound_active.assign(rows, true);
  } catch (const std::exception &error) {
    return failure(SolverStatus::kNumericalError,
                   std::string("frame pose refinement failed: ") + error.what());
  }
  return {};
}

FixedFramePoseConstraintResult linearize_fixed_frame_pose_violations(
    const PreparedFixedFramePoseConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &acceleration, double dt,
    const ScalarGeometricViolation &trigger,
    AffineAccelerationConstraint *refinement) {
  if (!(dt > 0.0) || !std::isfinite(dt)) {
    return failure(SolverStatus::kInvalidInput,
                   "frame pose batch refinement time is invalid");
  }
  auto original = linearize_fixed_frame_pose_violation(
      prepared, robot, q, dq, acceleration, trigger, refinement);
  if (!original.satisfied()) return original;
  std::vector<AffineAccelerationConstraint> extra_rows;
  try {
    pinocchio::SE3 reference;
    const auto valid_reference = pose_from_matrix(
        prepared.specification.family_label, prepared.specification.source_id,
        prepared.specification.reference_pose, &reference);
    if (!valid_reference.satisfied()) return valid_reference;
    // Path sampling can round an endpoint time one ulp above dt. Preserve its
    // mandatory triggering cut and evaluate optional endpoint faces at dt.
    std::vector<double> times{dt};
    if (trigger.time < dt) times.insert(times.begin(), trigger.time);
    pinocchio::Data scratch(robot.model());
    for (double time : times) {
      const auto sample = evaluate_pose_differential_at_state(
          robot, scratch, prepared.specification.frame_name, reference,
          robot.integrate(q, time * dq + 0.5 * time * time * acceleration),
          dq + time * acceleration, prepared.rotation_active);
      const auto coordinates = pose_coordinate_sample(sample, acceleration);
      for (int axis : prepared.active_axes) {
        auto specification = prepared.scalar.specification;
        specification.active_axes = {axis};
        const auto acceptance = validate_scalar_geometric_sample_acceptance(
            specification, coordinates, time == dt);
        if (!acceptance.violation) continue;
        auto violation = *acceptance.violation;
        violation.time = time;
        if (violation.axis == trigger.axis && violation.kind == trigger.kind &&
            violation.upper == trigger.upper &&
            (time == trigger.time || (time == dt && trigger.time >= dt))) continue;
        AffineAccelerationConstraint row;
        const auto local = linearize_fixed_frame_pose_violation(
            prepared, robot, q, dq, acceleration, violation, &row);
        if (local.satisfied()) extra_rows.push_back(std::move(row));
      }
    }
  } catch (const std::exception &) {
    // Optional extra cuts do not replace the valid triggering model. The
    // complete original acceptance pass still decides whether to emit motion.
    return original;
  }
  if (extra_rows.empty()) return original;
  const auto original_row = *refinement;
  const Eigen::Index count = 1 + extra_rows.size();
  const Eigen::Index rows = count + robot.nv();
  refinement->coefficient_matrix = Eigen::MatrixXd::Zero(rows, robot.nv());
  refinement->affine_bias = Eigen::VectorXd::Zero(rows);
  refinement->lower_bounds = Eigen::VectorXd::Zero(rows);
  refinement->upper_bounds = Eigen::VectorXd::Zero(rows);
  refinement->lower_bound_active.assign(rows, true);
  refinement->upper_bound_active.assign(rows, true);
  for (Eigen::Index row = 0; row < count; ++row) {
    const auto &model = row == 0 ? original_row : extra_rows[row - 1];
    refinement->coefficient_matrix.row(row) = model.coefficient_matrix.row(0);
    refinement->affine_bias(row) = model.affine_bias(0);
    refinement->lower_bound_active[row] = false;
  }
  refinement->coefficient_matrix.bottomRows(robot.nv()).setIdentity();
  refinement->lower_bounds.tail(robot.nv()) = original_row.lower_bounds.tail(robot.nv());
  refinement->upper_bounds.tail(robot.nv()) = original_row.upper_bounds.tail(robot.nv());
  return original;
}

} // namespace embodik::detail
