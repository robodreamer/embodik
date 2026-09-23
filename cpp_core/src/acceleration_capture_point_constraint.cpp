#include "acceleration_capture_point_constraint.hpp"

#include "acceleration_com_support_polygon_constraint.hpp"
#include "geometric_constraint_differential.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>

namespace embodik::detail {
namespace {

constexpr char kFamilyLabel[] = "capture-point acceleration constraint";
constexpr double kConstraintTolerance = 1e-8;

CapturePointConstraintAssembly failure(SolverStatus status,
                                       std::string message) {
  CapturePointConstraintAssembly result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

CapturePointConstraintResult validation_failure(SolverStatus status,
                                                std::string message) {
  return {status, std::move(message)};
}

std::string identity(const CapturePointAccelerationConstraint &constraint) {
  return std::string(kFamilyLabel) + " '" + constraint.source_id + "'";
}

double tolerance(double lhs, double rhs) {
  return std::max(kConstraintTolerance,
                  1e-12 * (1.0 + std::max(std::abs(lhs), std::abs(rhs))));
}

CapturePointConstraintResult validate_basic(
    const CapturePointAccelerationConstraint &constraint,
    const RobotModel &robot) {
  if (constraint.source_id.empty()) {
    return validation_failure(SolverStatus::kInvalidInput,
                              std::string(kFamilyLabel) +
                                  " source_id must not be empty");
  }
  if (constraint.definition.frame_name.empty() ||
      (constraint.definition.frame_name != "world" &&
       !robot.has_frame(constraint.definition.frame_name))) {
    return validation_failure(
        SolverStatus::kInvalidInput,
        identity(constraint) +
            " frame_name must identify world or a robot frame");
  }
  if (!is_structurally_fixed_support_frame(
          robot, constraint.definition.frame_name)) {
    return validation_failure(
        SolverStatus::kInvalidInput,
        identity(constraint) +
            " support frame must be world or structurally root-fixed; moving "
            "frames are unsupported");
  }
  if (constraint.omega.has_value() &&
      (!std::isfinite(*constraint.omega) || *constraint.omega <= 0.0)) {
    return validation_failure(SolverStatus::kInvalidInput,
                              identity(constraint) +
                                  " omega must be positive when provided");
  }
  return {};
}

double resolve_omega(const CapturePointAccelerationConstraint &constraint,
                     const RobotModel &robot,
                     const GeometricCoordinateDifferential &com,
                     SolverStatus *status, std::string *message) {
  if (constraint.omega.has_value()) {
    return *constraint.omega;
  }
  const double height = com.value.z();
  const Eigen::Vector3d gravity_world = robot.get_gravity();
  Eigen::Vector3d gravity = gravity_world;
  if (constraint.definition.frame_name != "world") {
    const auto frame_pose = robot.get_frame_pose(constraint.definition.frame_name);
    gravity = frame_pose.rotation().transpose() * gravity_world;
  }
  const double vertical_gravity = gravity.z();
  if (!std::isfinite(height) || !std::isfinite(vertical_gravity) ||
      height <= 0.0 || vertical_gravity >= 0.0) {
    *status = SolverStatus::kInvalidInput;
    *message = identity(constraint) +
               " cannot derive omega from nonpositive support-frame CoM "
               "height or non-downward gravity";
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double omega_squared = -vertical_gravity / height;
  if (!std::isfinite(omega_squared) || omega_squared <= 0.0) {
    *status = SolverStatus::kInvalidInput;
    *message = identity(constraint) + " derived omega is not positive";
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::sqrt(omega_squared);
}

bool point_inside(const SupportPolygonGeometry &geometry,
                  const Eigen::Vector2d &point) {
  const Eigen::VectorXd signed_distance =
      geometry.halfspace_normals * point - geometry.halfspace_offsets;
  if (!signed_distance.allFinite()) {
    return false;
  }
  for (Eigen::Index row = 0; row < signed_distance.size(); ++row) {
    if (signed_distance(row) > tolerance(signed_distance(row), 0.0)) {
      return false;
    }
  }
  return true;
}

void fill_diagnostic(const PreparedCapturePointConstraint &prepared,
                     const Eigen::Vector2d &capture_point,
                     bool postvalidated,
                     CapturePointAccelerationDiagnostics *diagnostic) {
  if (diagnostic == nullptr) {
    return;
  }
  diagnostic->source_id = prepared.specification.source_id;
  diagnostic->predicted_point_xy = capture_point;
  diagnostic->half_plane_slacks =
      prepared.geometry.halfspace_offsets -
      prepared.geometry.halfspace_normals * capture_point;
  diagnostic->min_slack = diagnostic->half_plane_slacks.size() > 0 &&
                                  diagnostic->half_plane_slacks.allFinite()
                              ? diagnostic->half_plane_slacks.minCoeff()
                              : std::numeric_limits<double>::quiet_NaN();
  diagnostic->frozen_omega = prepared.frozen_omega;
  diagnostic->postvalidated = postvalidated;
}

} // namespace

CapturePointConstraintAssembly make_capture_point_constraints(
    const RobotModel &robot,
    const std::vector<CapturePointAccelerationConstraint> &constraints,
    double dt, std::unordered_set<std::string> *source_ids) {
  CapturePointConstraintAssembly assembly;
  assembly.constraints.reserve(constraints.size());
  assembly.prepared_constraints.reserve(constraints.size());

  for (const auto &constraint : constraints) {
    const auto basic = validate_basic(constraint, robot);
    if (!basic.satisfied()) {
      return failure(basic.status, basic.message);
    }
    if (!source_ids->insert(constraint.source_id).second) {
      return failure(SolverStatus::kInvalidInput,
                     "duplicate acceleration constraint source_id '" +
                         constraint.source_id + "'");
    }
    const auto geometry = prepare_support_polygon_geometry(
        constraint.definition, kFamilyLabel, constraint.source_id);
    if (!geometry.satisfied()) {
      return failure(geometry.status, geometry.message);
    }

    GeometricCoordinateDifferential com;
    try {
      com = evaluate_com_in_frame_differential(
          robot, constraint.definition.frame_name);
    } catch (const std::exception &error) {
      return failure(SolverStatus::kNumericalError,
                     identity(constraint) +
                         " current CoM evaluation failed: " + error.what());
    }
    if (com.value.size() != 3 || com.rate.size() != 3 ||
        com.jacobian.rows() != 3 || com.jacobian.cols() != robot.nv() ||
        com.affine_bias.size() != 3 || !com.value.allFinite() ||
        !com.rate.allFinite() || !com.jacobian.allFinite() ||
        !com.affine_bias.allFinite()) {
      return failure(SolverStatus::kNumericalError,
                     identity(constraint) +
                         " produced non-finite CoM dynamics");
    }

    SolverStatus omega_status = SolverStatus::kSuccess;
    std::string omega_message;
    const double omega =
        resolve_omega(constraint, robot, com, &omega_status, &omega_message);
    if (omega_status != SolverStatus::kSuccess) {
      return failure(omega_status, omega_message);
    }

    const double acceleration_scale = 0.5 * dt * dt + dt / omega;
    AffineAccelerationConstraint affine;
    affine.source_id = constraint.source_id;
    affine.coefficient_matrix =
        acceleration_scale * geometry.geometry.halfspace_normals *
        com.jacobian.topRows<2>();
    affine.affine_bias =
        geometry.geometry.halfspace_normals *
            (com.value.head<2>() + dt * com.rate.head<2>() +
             com.rate.head<2>() / omega +
             acceleration_scale * com.affine_bias.head<2>());
    affine.lower_bounds =
        Eigen::VectorXd::Constant(geometry.geometry.halfspace_offsets.size(),
                                  -1.0);
    affine.upper_bounds = geometry.geometry.halfspace_offsets;
    affine.lower_bound_active =
        std::vector<bool>(static_cast<std::size_t>(affine.upper_bounds.size()),
                          false);
    if (!affine.coefficient_matrix.allFinite() ||
        !affine.affine_bias.allFinite()) {
      return failure(SolverStatus::kNumericalError,
                     identity(constraint) +
                         " produced non-finite affine rows");
    }

    PreparedCapturePointConstraint prepared;
    prepared.specification = constraint;
    prepared.geometry = geometry.geometry;
    prepared.physical_constraint = affine;
    prepared.frozen_omega = omega;
    assembly.row_count += affine.coefficient_matrix.rows();
    assembly.constraints.push_back(std::move(affine));
    assembly.prepared_constraints.push_back(std::move(prepared));
  }
  return assembly;
}

CapturePointConstraintResult validate_capture_point_constraint_acceptance(
    const PreparedCapturePointConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q_solution, const Eigen::VectorXd &dq_next,
    CapturePointAccelerationDiagnostics *diagnostic) {
  GeometricCoordinateDifferential com;
  try {
    com = evaluate_com_in_frame_differential_at_state(
        robot, prepared.specification.definition.frame_name, q_solution,
        dq_next);
  } catch (const std::exception &error) {
    return validation_failure(
        SolverStatus::kNumericalError,
        identity(prepared.specification) +
            " accepted capture-point evaluation failed: " + error.what());
  }
  if (com.value.size() != 3 || com.rate.size() != 3 ||
      !com.value.allFinite() || !com.rate.allFinite() ||
      !std::isfinite(prepared.frozen_omega) || prepared.frozen_omega <= 0.0) {
    return validation_failure(
        SolverStatus::kNumericalError,
        identity(prepared.specification) +
            " accepted capture-point state is non-finite");
  }
  const Eigen::Vector2d capture_point =
      com.value.head<2>() + com.rate.head<2>() / prepared.frozen_omega;
  const bool accepted =
      capture_point.allFinite() && point_inside(prepared.geometry, capture_point);
  fill_diagnostic(prepared, capture_point, accepted, diagnostic);
  if (!accepted) {
    return validation_failure(
        SolverStatus::kNumericalError,
        identity(prepared.specification) +
            " accepted capture point violates support polygon");
  }
  return {};
}

} // namespace embodik::detail
