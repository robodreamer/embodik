#include "acceleration_zmp_constraint.hpp"

#include "acceleration_com_support_polygon_constraint.hpp"
#include "geometric_constraint_differential.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>

namespace embodik::detail {
namespace {

constexpr char kFamilyLabel[] = "ZMP acceleration constraint";
constexpr double kConstraintTolerance = 1e-8;

ZmpConstraintAssembly failure(SolverStatus status, std::string message) {
  ZmpConstraintAssembly result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

ZmpConstraintResult validation_failure(SolverStatus status,
                                       std::string message) {
  return {status, std::move(message)};
}

std::string identity(const ZmpAccelerationConstraint &constraint) {
  return std::string(kFamilyLabel) + " '" + constraint.source_id + "'";
}

double tolerance(double lhs, double rhs) {
  return std::max(kConstraintTolerance,
                  1e-12 * (1.0 + std::max(std::abs(lhs), std::abs(rhs))));
}

struct SupportFrameCentroidalRateRows {
  Eigen::Matrix<double, 6, Eigen::Dynamic> matrix;
  Eigen::Matrix<double, 6, 1> bias;
  Eigen::Vector3d com;
  Eigen::Vector3d gravity;
};

ZmpConstraintResult validate_basic(const ZmpAccelerationConstraint &constraint,
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
  if (!std::isfinite(constraint.fz_min) || constraint.fz_min <= 0.0) {
    return validation_failure(SolverStatus::kInvalidInput,
                              identity(constraint) +
                                  " fz_min must be positive");
  }
  return {};
}

SupportFrameCentroidalRateRows evaluate_rows(
    const RobotModel &robot, const ZmpAccelerationConstraint &constraint,
    const Eigen::VectorXd *q = nullptr, const Eigen::VectorXd *dq = nullptr) {
  SupportFrameCentroidalRateRows rows;
  const Eigen::Matrix<double, 6, Eigen::Dynamic> ag =
      q == nullptr ? robot.get_centroidal_momentum_matrix()
                   : robot.compute_centroidal_momentum_matrix(*q, *dq);
  const Eigen::Matrix<double, 6, 1> bias =
      q == nullptr ? robot.get_centroidal_momentum_matrix_bias()
                   : robot.compute_centroidal_momentum_matrix_bias(*q, *dq);
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  if (constraint.definition.frame_name != "world") {
    const auto pose = robot.get_frame_pose(constraint.definition.frame_name);
    rotation = pose.rotation().transpose();
  }
  rows.matrix.resize(6, robot.nv());
  rows.matrix.topRows<3>() = rotation * ag.topRows<3>();
  rows.matrix.bottomRows<3>() = rotation * ag.bottomRows<3>();
  rows.bias.head<3>() = rotation * bias.head<3>();
  rows.bias.tail<3>() = rotation * bias.tail<3>();
  rows.com = q == nullptr
                 ? evaluate_com_in_frame_differential(
                       robot, constraint.definition.frame_name)
                       .value
                 : evaluate_com_in_frame_differential_at_state(
                       robot, constraint.definition.frame_name, *q, *dq)
                       .value;
  rows.gravity = rotation * robot.get_gravity();
  return rows;
}

Eigen::Vector2d zmp_from_rows(const SupportFrameCentroidalRateRows &rows,
                              const Eigen::VectorXd &ddq, double total_mass,
                              double *fz) {
  const Eigen::Matrix<double, 6, 1> hdot = rows.matrix * ddq + rows.bias;
  const Eigen::Vector3d force = hdot.head<3>() - total_mass * rows.gravity;
  *fz = force.z();
  const double mx = hdot(3);
  const double my = hdot(4);
  return {rows.com.x() - (my + rows.com.z() * force.x()) / *fz,
          rows.com.y() + (mx - rows.com.z() * force.y()) / *fz};
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

void fill_diagnostic(const PreparedZmpConstraint &prepared,
                     const Eigen::Vector2d &zmp, double force_z,
                     bool postvalidated,
                     ZmpAccelerationDiagnostics *diagnostic) {
  if (diagnostic == nullptr) {
    return;
  }
  diagnostic->source_id = prepared.specification.source_id;
  diagnostic->predicted_point_xy = zmp;
  diagnostic->half_plane_slacks =
      prepared.geometry.halfspace_offsets -
      prepared.geometry.halfspace_normals * zmp;
  diagnostic->min_slack = diagnostic->half_plane_slacks.size() > 0 &&
                                  diagnostic->half_plane_slacks.allFinite()
                              ? diagnostic->half_plane_slacks.minCoeff()
                              : std::numeric_limits<double>::quiet_NaN();
  diagnostic->force_z = force_z;
  diagnostic->postvalidated = postvalidated;
}

} // namespace

ZmpConstraintAssembly make_zmp_constraints(
    const RobotModel &robot,
    const std::vector<ZmpAccelerationConstraint> &constraints,
    std::unordered_set<std::string> *source_ids) {
  ZmpConstraintAssembly assembly;
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

    SupportFrameCentroidalRateRows rows;
    try {
      rows = evaluate_rows(robot, constraint);
    } catch (const std::exception &error) {
      return failure(SolverStatus::kNumericalError,
                     identity(constraint) +
                         " centroidal-rate evaluation failed: " +
                         error.what());
    }
    if (rows.matrix.rows() != 6 || rows.matrix.cols() != robot.nv() ||
        !rows.matrix.allFinite() || !rows.bias.allFinite() ||
        !rows.com.allFinite() || !rows.gravity.allFinite()) {
      return failure(SolverStatus::kNumericalError,
                     identity(constraint) +
                         " produced non-finite centroidal-rate rows");
    }

    const Eigen::Index edge_rows = geometry.geometry.halfspace_offsets.size();
    AffineAccelerationConstraint affine;
    affine.source_id = constraint.source_id;
    affine.coefficient_matrix.resize(edge_rows + 1, robot.nv());
    affine.affine_bias.resize(edge_rows + 1);
    affine.lower_bounds.resize(edge_rows + 1);
    affine.upper_bounds.resize(edge_rows + 1);
    affine.lower_bound_active =
        std::vector<bool>(static_cast<std::size_t>(edge_rows + 1), false);
    affine.upper_bound_active =
        std::vector<bool>(static_cast<std::size_t>(edge_rows + 1), true);

    const Eigen::RowVectorXd fx_matrix = rows.matrix.row(0);
    const Eigen::RowVectorXd fy_matrix = rows.matrix.row(1);
    const Eigen::RowVectorXd fz_matrix = rows.matrix.row(2);
    const double total_mass = robot.get_total_mass();
    const double fx_bias = rows.bias(0) - total_mass * rows.gravity.x();
    const double fy_bias = rows.bias(1) - total_mass * rows.gravity.y();
    const double fz_bias = rows.bias(2) - total_mass * rows.gravity.z();
    affine.coefficient_matrix.row(0) = fz_matrix;
    affine.affine_bias(0) = fz_bias;
    affine.lower_bounds(0) = constraint.fz_min;
    affine.upper_bounds(0) = 0.0;
    affine.lower_bound_active[0] = true;
    affine.upper_bound_active[0] = false;

    for (Eigen::Index edge = 0; edge < edge_rows; ++edge) {
      const double nx = geometry.geometry.halfspace_normals(edge, 0);
      const double ny = geometry.geometry.halfspace_normals(edge, 1);
      const double c_offset =
          nx * rows.com.x() + ny * rows.com.y() -
          geometry.geometry.halfspace_offsets(edge);
      const Eigen::Index row = edge + 1;
      affine.coefficient_matrix.row(row) =
          c_offset * fz_matrix - nx * rows.matrix.row(4) -
          nx * rows.com.z() * fx_matrix + ny * rows.matrix.row(3) -
          ny * rows.com.z() * fy_matrix;
      affine.affine_bias(row) =
          c_offset * fz_bias - nx * rows.bias(4) -
          nx * rows.com.z() * fx_bias + ny * rows.bias(3) -
          ny * rows.com.z() * fy_bias;
      affine.lower_bounds(row) = -1.0;
      affine.upper_bounds(row) = 0.0;
    }
    if (!affine.coefficient_matrix.allFinite() ||
        !affine.affine_bias.allFinite()) {
      return failure(SolverStatus::kNumericalError,
                     identity(constraint) + " produced non-finite ZMP rows");
    }

    PreparedZmpConstraint prepared;
    prepared.specification = constraint;
    prepared.geometry = geometry.geometry;
    prepared.physical_constraint = affine;
    assembly.row_count += affine.coefficient_matrix.rows();
    assembly.constraints.push_back(std::move(affine));
    assembly.prepared_constraints.push_back(std::move(prepared));
  }
  return assembly;
}

ZmpConstraintResult validate_zmp_constraint_acceptance(
    const PreparedZmpConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q_solution, const Eigen::VectorXd &dq_next,
    const Eigen::VectorXd &accepted_acceleration,
    ZmpAccelerationDiagnostics *diagnostic) {
  SupportFrameCentroidalRateRows rows;
  try {
    rows = evaluate_rows(robot, prepared.specification, &q_solution,
                         &dq_next);
  } catch (const std::exception &error) {
    return validation_failure(
        SolverStatus::kNumericalError,
        identity(prepared.specification) +
            " accepted ZMP evaluation failed: " + error.what());
  }
  if (accepted_acceleration.size() != robot.nv() ||
      !accepted_acceleration.allFinite() || !rows.matrix.allFinite() ||
      !rows.bias.allFinite() || !rows.com.allFinite() ||
      !rows.gravity.allFinite()) {
    return validation_failure(SolverStatus::kNumericalError,
                              identity(prepared.specification) +
                                  " accepted ZMP state is non-finite");
  }
  double fz = 0.0;
  const Eigen::Vector2d zmp =
      zmp_from_rows(rows, accepted_acceleration, robot.get_total_mass(), &fz);
  bool accepted = std::isfinite(fz) &&
                  fz >= prepared.specification.fz_min -
                            tolerance(fz, prepared.specification.fz_min);
  if (!accepted) {
    fill_diagnostic(prepared, zmp, fz, false, diagnostic);
    return validation_failure(SolverStatus::kNumericalError,
                              identity(prepared.specification) +
                                  " accepted ZMP vertical force is below "
                                  "fz_min");
  }
  accepted = zmp.allFinite() && point_inside(prepared.geometry, zmp);
  fill_diagnostic(prepared, zmp, fz, accepted, diagnostic);
  if (!accepted) {
    return validation_failure(SolverStatus::kNumericalError,
                              identity(prepared.specification) +
                                  " accepted ZMP violates support polygon");
  }
  return {};
}

} // namespace embodik::detail
