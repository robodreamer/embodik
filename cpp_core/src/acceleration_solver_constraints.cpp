#include "acceleration_solver_internal.hpp"

namespace embodik {
namespace acceleration_solver_internal {
namespace {

std::vector<bool> full_axis_mask() {
  return {true, true, true, true, true, true};
}

std::vector<bool> resolved_axis_mask(const std::vector<bool> &axis_mask) {
  return axis_mask.empty() ? full_axis_mask() : axis_mask;
}

bool axis_mask_valid(const std::vector<bool> &axis_mask) {
  return axis_mask.empty() || axis_mask.size() == 6U;
}

bool axis_mask_has_selection(const std::vector<bool> &axis_mask) {
  if (axis_mask.empty()) {
    return true;
  }
  return std::any_of(axis_mask.begin(), axis_mask.end(), [](bool selected) {
    return selected;
  });
}

Eigen::Index selected_axis_count(const std::vector<bool> &axis_mask) {
  Eigen::Index rows = 0;
  for (bool selected : resolved_axis_mask(axis_mask)) {
    if (selected) {
      ++rows;
    }
  }
  return rows;
}

} // namespace

template <typename Constraint>
ConstraintValidation validate_constraint_family(
    const std::vector<Constraint> &constraints, Eigen::Index variable_count,
    std::unordered_set<std::string> *source_ids,
    bool validate_affine_backend_range) {
  ConstraintValidation validation;
  for (const auto &constraint : constraints) {
    const std::string family = constraint_family_name(constraint);
    if (constraint.source_id.empty()) {
      validation.status = SolverStatus::kInvalidInput;
      validation.message = family + " source_id must not be empty";
      return validation;
    }
    if (!source_ids->insert(constraint.source_id).second) {
      validation.status = SolverStatus::kInvalidInput;
      validation.message = "duplicate acceleration constraint source_id '" +
                           constraint.source_id + "'";
      return validation;
    }
    if (constraint.coefficient_matrix.rows() == 0) {
      validation.status = SolverStatus::kInvalidInput;
      validation.message = family + " '" + constraint.source_id +
                           "' must contain at least one row";
      return validation;
    }
    if (constraint.coefficient_matrix.cols() != variable_count) {
      validation.status = SolverStatus::kShapeMismatch;
      validation.message = family + " '" + constraint.source_id +
                           "' matrix must have nv columns";
      return validation;
    }
    const Eigen::Index rows = constraint.coefficient_matrix.rows();
    if (constraint.affine_bias.size() != rows) {
      validation.status = SolverStatus::kShapeMismatch;
      validation.message = family + " '" + constraint.source_id +
                           "' bias must match matrix rows";
      return validation;
    }
    if (constraint.lower_bounds.size() != rows ||
        constraint.upper_bounds.size() != rows) {
      validation.status = SolverStatus::kConstraintBoundsMismatch;
      validation.message = family + " '" + constraint.source_id +
                           "' bounds must match matrix rows";
      return validation;
    }
    if ((!constraint.lower_bound_active.empty() &&
         constraint.lower_bound_active.size() !=
             static_cast<std::size_t>(rows)) ||
        (!constraint.upper_bound_active.empty() &&
         constraint.upper_bound_active.size() !=
             static_cast<std::size_t>(rows))) {
      validation.status = SolverStatus::kShapeMismatch;
      validation.message = family + " '" + constraint.source_id +
                           "' active-side flags must match matrix rows";
      return validation;
    }
    if (!constraint.coefficient_matrix.allFinite() ||
        !constraint.affine_bias.allFinite() ||
        !constraint.lower_bounds.allFinite() ||
        !constraint.upper_bounds.allFinite()) {
      validation.status = SolverStatus::kNonFiniteInput;
      validation.message = family + " '" + constraint.source_id +
                           "' must contain only finite values";
      return validation;
    }
    for (Eigen::Index row = 0; row < rows; ++row) {
      const bool lower_active = lower_side_active(constraint, row);
      const bool upper_active = upper_side_active(constraint, row);
      if (!lower_active && !upper_active) {
        validation.status = SolverStatus::kInvalidInput;
        validation.message = family + " '" + constraint.source_id + "' row " +
                             std::to_string(row) +
                             " has no active bound side";
        return validation;
      }
      if (lower_active && upper_active &&
          constraint.lower_bounds(row) > constraint.upper_bounds(row)) {
        validation.status = SolverStatus::kInvalidInput;
        validation.message = family + " '" + constraint.source_id +
                             "' lower bound exceeds upper bound";
        return validation;
      }
      if (validate_affine_backend_range) {
        if ((lower_active &&
             !checked_shifted_bound(constraint.lower_bounds(row),
                                    constraint.affine_bias(row))) ||
            (upper_active &&
             !checked_shifted_bound(constraint.upper_bounds(row),
                                    constraint.affine_bias(row)))) {
          validation.status = SolverStatus::kInvalidInput;
          validation.message = family + " '" + constraint.source_id +
                               "' shifted bound is not finite";
          return validation;
        }
        double inactive_magnitude = 0.0;
        double inactive_side_bound = 0.0;
        if (!inactive_affine_bound_magnitude(constraint, row,
                                             &inactive_magnitude) ||
            (!lower_active &&
             !checked_add(constraint.affine_bias(row), -inactive_magnitude,
                          &inactive_side_bound)) ||
            (!upper_active &&
             !checked_add(constraint.affine_bias(row), inactive_magnitude,
                          &inactive_side_bound))) {
          validation.status = SolverStatus::kInvalidInput;
          validation.message = family + " '" + constraint.source_id +
                               "' inactive side cannot be represented in the "
                               "finite backend range";
          return validation;
        }
      }
    }
    validation.row_count += rows;
  }
  return validation;
}

ConstraintValidation validate_acceleration_constraints(
    const std::vector<AffineAccelerationConstraint> &affine_constraints,
    const std::vector<FrozenNextVelocityConstraint> &frozen_constraints,
    Eigen::Index variable_count,
    std::unordered_set<std::string> *source_ids) {
  ConstraintValidation validation;
  validation = validate_constraint_family(affine_constraints, variable_count,
                                          source_ids, true);
  if (validation.status != SolverStatus::kSuccess) {
    return validation;
  }
  const Eigen::Index affine_rows = validation.row_count;
  validation = validate_constraint_family(frozen_constraints, variable_count,
                                          source_ids, false);
  validation.row_count += affine_rows;
  return validation;
}

CentroidalMomentumRateBoundsAssembly make_centroidal_momentum_rate_bounds(
    const RobotModel &robot,
    const std::vector<CentroidalMomentumRateBounds> &bounds,
    std::unordered_set<std::string> *source_ids) {
  CentroidalMomentumRateBoundsAssembly assembly;
  if (bounds.empty()) {
    return assembly;
  }
  try {
    assembly.centroidal_matrix = robot.get_centroidal_momentum_matrix();
    assembly.bias = robot.get_centroidal_momentum_matrix_bias();
  } catch (const std::exception &error) {
    assembly.status = SolverStatus::kNumericalError;
    assembly.message =
        std::string("centroidal momentum-rate bounds failed: ") +
        error.what();
    return assembly;
  }
  if (assembly.centroidal_matrix.rows() != 6 ||
      assembly.centroidal_matrix.cols() != robot.nv() ||
      assembly.bias.size() != 6 || !assembly.centroidal_matrix.allFinite() ||
      !assembly.bias.allFinite()) {
    assembly.status = SolverStatus::kNumericalError;
    assembly.message =
        "centroidal momentum-rate bounds produced non-finite dynamics";
    return assembly;
  }

  for (const auto &bound : bounds) {
    const std::string family = constraint_family_name(bound);
    if (bound.source_id.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " source_id must not be empty";
      return assembly;
    }
    if (!source_ids->insert(bound.source_id).second) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = "duplicate acceleration constraint source_id '" +
                         bound.source_id + "'";
      return assembly;
    }
    if (bound.lower_bounds.size() != 6 || bound.upper_bounds.size() != 6) {
      assembly.status = SolverStatus::kConstraintBoundsMismatch;
      assembly.message = family + " '" + bound.source_id +
                         "' bounds must have size 6";
      return assembly;
    }
    if (!bound.lower_bounds.allFinite() || !bound.upper_bounds.allFinite()) {
      assembly.status = SolverStatus::kNonFiniteInput;
      assembly.message = family + " '" + bound.source_id +
                         "' must contain only finite bounds";
      return assembly;
    }
    if (!axis_mask_valid(bound.axis_mask)) {
      assembly.status = SolverStatus::kShapeMismatch;
      assembly.message = family + " '" + bound.source_id +
                         "' axis_mask must have size 6";
      return assembly;
    }
    if (!axis_mask_has_selection(bound.axis_mask)) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " '" + bound.source_id +
                         "' must select at least one axis";
      return assembly;
    }
    if ((!bound.lower_bound_active.empty() &&
         bound.lower_bound_active.size() != 6U) ||
        (!bound.upper_bound_active.empty() &&
         bound.upper_bound_active.size() != 6U)) {
      assembly.status = SolverStatus::kShapeMismatch;
      assembly.message = family + " '" + bound.source_id +
                         "' active-side flags must have size 6";
      return assembly;
    }

    AffineAccelerationConstraint constraint;
    constraint.source_id = bound.source_id;
    const Eigen::Index rows = selected_axis_count(bound.axis_mask);
    constraint.coefficient_matrix.resize(rows, robot.nv());
    constraint.affine_bias.resize(rows);
    constraint.lower_bounds.resize(rows);
    constraint.upper_bounds.resize(rows);
    if (!bound.lower_bound_active.empty()) {
      constraint.lower_bound_active.reserve(static_cast<std::size_t>(rows));
    }
    if (!bound.upper_bound_active.empty()) {
      constraint.upper_bound_active.reserve(static_cast<std::size_t>(rows));
    }
    const auto mask = resolved_axis_mask(bound.axis_mask);
    Eigen::Index cursor = 0;
    for (Eigen::Index axis = 0; axis < 6; ++axis) {
      if (!mask[static_cast<std::size_t>(axis)]) {
        continue;
      }
      const bool lower_active =
          bound.lower_bound_active.empty() ||
          bound.lower_bound_active[static_cast<std::size_t>(axis)];
      const bool upper_active =
          bound.upper_bound_active.empty() ||
          bound.upper_bound_active[static_cast<std::size_t>(axis)];
      if (!lower_active && !upper_active) {
        assembly.status = SolverStatus::kInvalidInput;
        assembly.message = family + " '" + bound.source_id + "' axis " +
                           std::to_string(axis) +
                           " has no active bound side";
        return assembly;
      }
      if (lower_active && upper_active &&
          bound.lower_bounds(axis) > bound.upper_bounds(axis)) {
        assembly.status = SolverStatus::kInvalidInput;
        assembly.message = family + " '" + bound.source_id +
                           "' lower bound exceeds upper bound";
        return assembly;
      }
      if ((lower_active &&
           !checked_shifted_bound(bound.lower_bounds(axis),
                                  assembly.bias(axis))) ||
          (upper_active &&
           !checked_shifted_bound(bound.upper_bounds(axis),
                                  assembly.bias(axis)))) {
        assembly.status = SolverStatus::kInvalidInput;
        assembly.message = family + " '" + bound.source_id +
                           "' shifted bound is not finite";
        return assembly;
      }
      double inactive_magnitude = 0.0;
      double inactive_side_bound = 0.0;
      if (!inactive_bound_magnitude(assembly.centroidal_matrix.row(axis),
                                    &inactive_magnitude) ||
          (!lower_active &&
           !checked_add(assembly.bias(axis), -inactive_magnitude,
                        &inactive_side_bound)) ||
          (!upper_active &&
           !checked_add(assembly.bias(axis), inactive_magnitude,
                        &inactive_side_bound))) {
        assembly.status = SolverStatus::kInvalidInput;
        assembly.message = family + " '" + bound.source_id +
                           "' inactive side cannot be represented in the "
                           "finite backend range";
        return assembly;
      }
      constraint.coefficient_matrix.row(cursor) =
          assembly.centroidal_matrix.row(axis);
      constraint.affine_bias(cursor) = assembly.bias(axis);
      constraint.lower_bounds(cursor) = bound.lower_bounds(axis);
      constraint.upper_bounds(cursor) = bound.upper_bounds(axis);
      if (!bound.lower_bound_active.empty()) {
        constraint.lower_bound_active.push_back(lower_active);
      }
      if (!bound.upper_bound_active.empty()) {
        constraint.upper_bound_active.push_back(upper_active);
      }
      ++cursor;
    }
    assembly.row_count += rows;
    assembly.constraints.push_back(std::move(constraint));
  }
  return assembly;
}

ContactConstraintAssembly make_contact_acceleration_constraints(
    const RobotModel &robot,
    const std::vector<ContactAccelerationConstraint> &contacts,
    std::unordered_set<std::string> *source_ids) {
  ContactConstraintAssembly assembly;
  assembly.constraints.reserve(contacts.size());

  for (const auto &contact : contacts) {
    const std::string family = constraint_family_name(contact);
    if (contact.source_id.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " source_id must not be empty";
      return assembly;
    }
    if (!source_ids->insert(contact.source_id).second) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = "duplicate acceleration constraint source_id '" +
                         contact.source_id + "'";
      return assembly;
    }
    if (contact.frame_name.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " '" + contact.source_id +
                         "' frame_name must not be empty";
      return assembly;
    }
    if (contact.type != ContactType::kPointContact &&
        contact.type != ContactType::kRigidContact) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " '" + contact.source_id +
                         "' has an unsupported contact type";
      return assembly;
    }
    if (!robot.has_frame(contact.frame_name)) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " '" + contact.source_id +
                         "' references an unknown frame '" +
                         contact.frame_name + "'";
      return assembly;
    }

    detail::FrameKinematicDifferential differential;
    try {
      differential =
          detail::evaluate_frame_kinematic_differential(robot,
                                                        contact.frame_name);
    } catch (const std::exception &error) {
      assembly.status = SolverStatus::kNumericalError;
      assembly.message = family + " '" + contact.source_id +
                         "' kinematic differential failed: " + error.what();
      return assembly;
    }
    if (differential.jacobian.cols() != robot.nv() ||
        differential.jacobian.rows() != 6 ||
        differential.affine_bias.size() != 6 ||
        !differential.jacobian.allFinite() ||
        !differential.affine_bias.allFinite()) {
      assembly.status = SolverStatus::kNumericalError;
      assembly.message = family + " '" + contact.source_id +
                         "' produced a non-finite kinematic row";
      return assembly;
    }

    const Eigen::Index rows =
        contact.type == ContactType::kPointContact ? 3 : 6;
    AffineAccelerationConstraint constraint;
    constraint.source_id = contact.source_id;
    constraint.coefficient_matrix =
        differential.jacobian.topRows(rows);
    constraint.affine_bias = differential.affine_bias.head(rows);
    constraint.lower_bounds = Eigen::VectorXd::Zero(rows);
    constraint.upper_bounds = Eigen::VectorXd::Zero(rows);
    assembly.row_count += rows;
    assembly.constraints.push_back(std::move(constraint));
  }

  return assembly;
}

TightPointConstraintAssembly make_tight_point_constraints(
    const RobotModel &robot,
    const std::vector<TightPointAccelerationConstraint> &tight_points,
    double dt, const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    std::unordered_set<std::string> *source_ids) {
  TightPointConstraintAssembly assembly;
  assembly.constraints.reserve(tight_points.size());
  assembly.prepared_constraints.reserve(tight_points.size());

  for (const auto &tight_point : tight_points) {
    const std::string family = constraint_family_name(tight_point);
    if (tight_point.source_id.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " source_id must not be empty";
      return assembly;
    }
    if (!source_ids->insert(tight_point.source_id).second) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = "duplicate acceleration constraint source_id '" +
                         tight_point.source_id + "'";
      return assembly;
    }

    const auto prepared = detail::prepare_tight_point_constraint(
        tight_point, robot, dt, joint_acceleration_lower,
        joint_acceleration_upper);
    if (!prepared.satisfied()) {
      assembly.status = prepared.status;
      assembly.message = prepared.message;
      return assembly;
    }
    assembly.row_count +=
        prepared.prepared->scalar.state_box.physical_constraint
            .coefficient_matrix
            .rows();
    assembly.constraints.push_back(
        prepared.prepared->scalar.state_box.physical_constraint);
    assembly.prepared_constraints.push_back(std::move(*prepared.prepared));
  }

  return assembly;
}

RelativePoseConstraintAssembly make_relative_pose_constraints(
    const RobotModel &robot,
    const std::vector<RelativePoseAccelerationConstraint> &relative_poses,
    double dt, const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    std::unordered_set<std::string> *source_ids) {
  RelativePoseConstraintAssembly assembly;
  assembly.constraints.reserve(relative_poses.size());
  assembly.prepared_constraints.reserve(relative_poses.size());

  for (const auto &relative_pose : relative_poses) {
    const std::string family = constraint_family_name(relative_pose);
    if (relative_pose.source_id.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " source_id must not be empty";
      return assembly;
    }
    if (!source_ids->insert(relative_pose.source_id).second) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = "duplicate acceleration constraint source_id '" +
                         relative_pose.source_id + "'";
      return assembly;
    }

    const auto prepared = detail::prepare_relative_pose_constraint(
        relative_pose, robot, dt, joint_acceleration_lower,
        joint_acceleration_upper);
    if (!prepared.satisfied()) {
      assembly.status = prepared.status;
      assembly.message = prepared.message;
      return assembly;
    }
    assembly.row_count +=
        prepared.prepared->scalar.state_box.physical_constraint
            .coefficient_matrix
            .rows();
    assembly.constraints.push_back(
        prepared.prepared->scalar.state_box.physical_constraint);
    assembly.prepared_constraints.push_back(std::move(*prepared.prepared));
  }

  return assembly;
}

ComSupportPolygonConstraintAssembly make_com_support_polygon_constraints(
    const RobotModel &robot,
    const std::vector<ComSupportPolygonAccelerationConstraint> &polygons,
    double dt, const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    std::unordered_set<std::string> *source_ids) {
  ComSupportPolygonConstraintAssembly assembly;
  assembly.constraints.reserve(polygons.size());
  assembly.prepared_constraints.reserve(polygons.size());

  for (const auto &polygon : polygons) {
    const std::string family = constraint_family_name(polygon);
    if (polygon.source_id.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " source_id must not be empty";
      return assembly;
    }
    if (!source_ids->insert(polygon.source_id).second) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = "duplicate acceleration constraint source_id '" +
                         polygon.source_id + "'";
      return assembly;
    }

    const auto prepared = detail::prepare_com_support_polygon_constraint(
        polygon, robot, dt, joint_acceleration_lower,
        joint_acceleration_upper);
    if (!prepared.satisfied() || !prepared.prepared.has_value()) {
      assembly.status = prepared.status;
      assembly.message = prepared.message;
      return assembly;
    }
    assembly.row_count +=
        prepared.prepared->state_box.physical_constraint.coefficient_matrix
            .rows();
    assembly.constraints.push_back(
        prepared.prepared->state_box.physical_constraint);
    assembly.prepared_constraints.push_back(std::move(*prepared.prepared));
  }

  return assembly;
}

bool insert_lock_indices(const std::vector<int> &indices,
                         const std::string &field_name, Eigen::Index nv,
                         std::unordered_set<int> *seen,
                         SolverStatus *status, std::string *message) {
  std::unordered_set<int> local_seen;
  for (int index : indices) {
    if (index < 0 || index >= nv) {
      *status = SolverStatus::kInvalidInput;
      *message = field_name + " contains out-of-range joint index";
      return false;
    }
    if (!local_seen.insert(index).second) {
      *status = SolverStatus::kInvalidInput;
      *message = field_name + " contains duplicate joint index";
      return false;
    }
    if (!seen->insert(index).second) {
      *status = SolverStatus::kInvalidInput;
      *message = "acceleration lock joint indices overlap across policies";
      return false;
    }
  }
  return true;
}

LockRows build_lock_rows(const AccelerationSolveOptions &options,
                         const Eigen::VectorXd &dq, double dt) {
  LockRows rows;
  const Eigen::Index nv = dq.size();
  std::unordered_set<int> seen;
  if (!insert_lock_indices(options.zero_acceleration_joint_indices,
                           "zero_acceleration_joint_indices", nv, &seen,
                           &rows.status, &rows.message) ||
      !insert_lock_indices(options.zero_next_velocity_joint_indices,
                           "zero_next_velocity_joint_indices", nv, &seen,
                           &rows.status, &rows.message) ||
      !insert_lock_indices(options.fixed_current_position_joint_indices,
                           "fixed_current_position_joint_indices", nv, &seen,
                           &rows.status, &rows.message)) {
    return rows;
  }
  if (seen.empty()) {
    rows.coefficients.resize(0, nv);
    rows.bounds.resize(0);
    return rows;
  }

  const Eigen::Index row_count = static_cast<Eigen::Index>(seen.size());
  rows.coefficients = Eigen::MatrixXd::Zero(row_count, nv);
  rows.bounds.resize(row_count);
  Eigen::Index cursor = 0;
  const auto append = [&](std::vector<int> indices, double numerator_factor,
                          bool divide_by_dt) {
    std::sort(indices.begin(), indices.end());
    for (int index : indices) {
      double target = 0.0;
      double numerator = 0.0;
      if (!checked_product(numerator_factor, dq(index), &numerator) ||
          (divide_by_dt && !checked_quotient(numerator, dt, &target))) {
        rows.status = SolverStatus::kNumericalError;
        rows.message =
            "acceleration lock target cannot be represented finitely";
        return false;
      }
      if (!divide_by_dt) {
        target = numerator;
      }
      rows.coefficients(cursor, index) = 1.0;
      rows.bounds(cursor) = target;
      ++cursor;
    }
    return true;
  };
  if (!append(options.zero_acceleration_joint_indices, 0.0, false) ||
      !append(options.zero_next_velocity_joint_indices, -1.0, true) ||
      !append(options.fixed_current_position_joint_indices, -2.0, true)) {
    return rows;
  }
  return rows;
}

detail::GeneralizedConstraintBlock make_affine_constraint_block(
    const std::vector<AffineAccelerationConstraint> &constraints,
    Eigen::Index variable_count, Eigen::Index row_count) {
  detail::GeneralizedConstraintBlock block;
  block.coefficient_matrix.resize(row_count, variable_count);
  block.affine_bias.resize(row_count);
  block.physical_lower_bounds.resize(row_count);
  block.physical_upper_bounds.resize(row_count);

  Eigen::Index cursor = 0;
  for (const auto &constraint : constraints) {
    const Eigen::Index rows = constraint.coefficient_matrix.rows();
    block.coefficient_matrix.middleRows(cursor, rows) =
        constraint.coefficient_matrix;
    block.affine_bias.segment(cursor, rows) = constraint.affine_bias;
    for (Eigen::Index row = 0; row < rows; ++row) {
      const Eigen::Index output_row = cursor + row;
      double inactive_magnitude = 0.0;
      inactive_affine_bound_magnitude(constraint, row, &inactive_magnitude);
      block.physical_lower_bounds(output_row) =
          lower_side_active(constraint, row)
              ? constraint.lower_bounds(row)
              : constraint.affine_bias(row) - inactive_magnitude;
      block.physical_upper_bounds(output_row) =
          upper_side_active(constraint, row)
              ? constraint.upper_bounds(row)
              : constraint.affine_bias(row) + inactive_magnitude;
    }
    cursor += rows;
  }
  return block;
}

detail::GeneralizedConstraintBlock make_lock_constraint_block(
    const LockRows &locks, Eigen::Index variable_count) {
  detail::GeneralizedConstraintBlock block;
  block.coefficient_matrix = locks.coefficients;
  block.affine_bias = Eigen::VectorXd::Zero(locks.bounds.size());
  block.physical_lower_bounds = locks.bounds;
  block.physical_upper_bounds = locks.bounds;
  if (locks.coefficients.cols() != variable_count) {
    block.coefficient_matrix.resize(0, variable_count);
    block.affine_bias.resize(0);
    block.physical_lower_bounds.resize(0);
    block.physical_upper_bounds.resize(0);
  }
  return block;
}

detail::GeneralizedConstraintBlock make_effort_constraint_block(
    const PreparedEffortConstraint &effort, Eigen::Index variable_count) {
  detail::GeneralizedConstraintBlock block;
  block.coefficient_matrix = effort.mass_matrix;
  block.affine_bias = effort.bias;
  block.physical_lower_bounds = -effort.limits;
  block.physical_upper_bounds = effort.limits;
  if (block.coefficient_matrix.cols() != variable_count) {
    block.coefficient_matrix.resize(0, variable_count);
    block.affine_bias.resize(0);
    block.physical_lower_bounds.resize(0);
    block.physical_upper_bounds.resize(0);
  }
  return block;
}

FrozenTransform transform_frozen_constraints(
    const std::vector<FrozenNextVelocityConstraint> &constraints,
    const Eigen::VectorXd &dq, double dt) {
  FrozenTransform transform;
  transform.transformed_constraints.reserve(constraints.size());
  for (const auto &constraint : constraints) {
    AffineAccelerationConstraint transformed;
    transformed.source_id = constraint.source_id;
    transformed.lower_bounds = constraint.lower_bounds;
    transformed.upper_bounds = constraint.upper_bounds;
    transformed.lower_bound_active = constraint.lower_bound_active;
    transformed.upper_bound_active = constraint.upper_bound_active;
    if (!checked_scaled_matrix(constraint.coefficient_matrix, dt,
                               &transformed.coefficient_matrix)) {
      transform.status = SolverStatus::kNumericalError;
      transform.message = "frozen next-velocity constraint '" +
                          constraint.source_id +
                          "' transformed coefficient is not reliable";
      return transform;
    }
    transformed.affine_bias.resize(constraint.affine_bias.size());
    for (Eigen::Index row = 0; row < constraint.coefficient_matrix.rows();
         ++row) {
      double cdq = 0.0;
      double dt_bias = 0.0;
      if (!checked_dot_row(constraint.coefficient_matrix.row(row), dq, &cdq) ||
          !checked_product(dt, constraint.affine_bias(row), &dt_bias) ||
          !checked_add(cdq, dt_bias, &transformed.affine_bias(row))) {
        transform.status = SolverStatus::kNumericalError;
        transform.message = "frozen next-velocity constraint '" +
                            constraint.source_id +
                            "' transformed bias is not reliable";
        return transform;
      }
      if ((lower_side_active(transformed, row) &&
           !checked_shifted_bound(transformed.lower_bounds(row),
                                  transformed.affine_bias(row))) ||
          (upper_side_active(transformed, row) &&
           !checked_shifted_bound(transformed.upper_bounds(row),
                                  transformed.affine_bias(row)))) {
        transform.status = SolverStatus::kNumericalError;
        transform.message = "frozen next-velocity constraint '" +
                            constraint.source_id +
                            "' transformed bound is not reliable";
        return transform;
      }
      double inactive_magnitude = 0.0;
      double unused = 0.0;
      if (!inactive_affine_bound_magnitude(transformed, row,
                                           &inactive_magnitude) ||
          (!lower_side_active(transformed, row) &&
           !checked_add(transformed.affine_bias(row), -inactive_magnitude,
                        &unused)) ||
          (!upper_side_active(transformed, row) &&
           !checked_add(transformed.affine_bias(row), inactive_magnitude,
                        &unused))) {
        transform.status = SolverStatus::kNumericalError;
        transform.message = "frozen next-velocity constraint '" +
                            constraint.source_id +
                            "' inactive side cannot be represented in the "
                            "finite backend range";
        return transform;
      }
    }
    transform.row_count += transformed.coefficient_matrix.rows();
    transform.transformed_constraints.push_back(std::move(transformed));
  }
  return transform;
}

bool accepted_affine_constraints_are_satisfied(
    const std::vector<AffineAccelerationConstraint> &constraints,
    const Eigen::VectorXd &ddq) {
  for (const auto &constraint : constraints) {
    const Eigen::VectorXd physical =
        constraint.coefficient_matrix * ddq + constraint.affine_bias;
    if (!physical.allFinite()) {
      return false;
    }
    for (Eigen::Index row = 0; row < physical.size(); ++row) {
      const double lower_tolerance =
          std::max(kConstraintTolerance,
                   1e-12 * std::max(std::abs(physical(row)),
                                    std::abs(constraint.lower_bounds(row))));
      const double upper_tolerance =
          std::max(kConstraintTolerance,
                   1e-12 * std::max(std::abs(physical(row)),
                                    std::abs(constraint.upper_bounds(row))));
      if (lower_side_active(constraint, row) &&
          physical(row) < constraint.lower_bounds(row) - lower_tolerance) {
        return false;
      }
      if (upper_side_active(constraint, row) &&
          physical(row) > constraint.upper_bounds(row) + upper_tolerance) {
        return false;
      }
    }
  }
  return true;
}

bool accepted_centroidal_momentum_rate_bounds_are_satisfied(
    const std::vector<AffineAccelerationConstraint> &constraints,
    const Eigen::VectorXd &ddq) {
  constexpr double kCentroidalBoundTolerance = 1e-7;
  for (const auto &constraint : constraints) {
    const Eigen::VectorXd physical =
        constraint.coefficient_matrix * ddq + constraint.affine_bias;
    if (!physical.allFinite()) {
      return false;
    }
    for (Eigen::Index row = 0; row < physical.size(); ++row) {
      if (lower_side_active(constraint, row) &&
          physical(row) < constraint.lower_bounds(row) -
                              kCentroidalBoundTolerance) {
        return false;
      }
      if (upper_side_active(constraint, row) &&
          physical(row) > constraint.upper_bounds(row) +
                              kCentroidalBoundTolerance) {
        return false;
      }
    }
  }
  return true;
}

bool accepted_frozen_constraints_are_satisfied(
    const std::vector<FrozenNextVelocityConstraint> &constraints,
    const Eigen::VectorXd &dq, double dt, const Eigen::VectorXd &ddq) {
  for (const auto &constraint : constraints) {
    for (Eigen::Index row = 0; row < constraint.coefficient_matrix.rows();
         ++row) {
      double cdq = 0.0;
      double cddq = 0.0;
      double row_velocity_delta = 0.0;
      double physical = 0.0;
      if (!checked_dot_row(constraint.coefficient_matrix.row(row), dq, &cdq) ||
          !checked_dot_row(constraint.coefficient_matrix.row(row), ddq,
                           &cddq) ||
          !checked_add(cddq, constraint.affine_bias(row),
                       &row_velocity_delta) ||
          !checked_product(dt, row_velocity_delta, &row_velocity_delta) ||
          !checked_add(cdq, row_velocity_delta, &physical)) {
        return false;
      }
      const double lower_tolerance =
          std::max(kConstraintTolerance,
                   1e-12 * std::max(std::abs(physical),
                                    std::abs(constraint.lower_bounds(row))));
      const double upper_tolerance =
          std::max(kConstraintTolerance,
                   1e-12 * std::max(std::abs(physical),
                                    std::abs(constraint.upper_bounds(row))));
      if (lower_side_active(constraint, row) &&
          physical < constraint.lower_bounds(row) - lower_tolerance) {
        return false;
      }
      if (upper_side_active(constraint, row) &&
          physical > constraint.upper_bounds(row) + upper_tolerance) {
        return false;
      }
    }
  }
  return true;
}

bool accepted_lock_constraints_are_satisfied(
    const AccelerationSolveOptions &options, const Eigen::VectorXd &dq,
    double dt, const Eigen::VectorXd &ddq) {
  for (int index : options.zero_acceleration_joint_indices) {
    if (std::abs(ddq(index)) > kConstraintTolerance) {
      return false;
    }
  }
  for (int index : options.zero_next_velocity_joint_indices) {
    double velocity_delta = 0.0;
    double next_velocity = 0.0;
    if (!checked_product(dt, ddq(index), &velocity_delta) ||
        !checked_add(dq(index), velocity_delta, &next_velocity) ||
        std::abs(next_velocity) > kConstraintTolerance) {
      return false;
    }
  }
  for (int index : options.fixed_current_position_joint_indices) {
    double half_dt = 0.0;
    double half_acceleration_rate = 0.0;
    double average_velocity = 0.0;
    double displacement = 0.0;
    if (!checked_product(0.5, dt, &half_dt) ||
        !checked_product(half_dt, ddq(index), &half_acceleration_rate) ||
        !checked_add(dq(index), half_acceleration_rate, &average_velocity) ||
        !checked_product(dt, average_velocity, &displacement) ||
        std::abs(displacement) > kConstraintTolerance) {
      return false;
    }
  }
  return true;
}


} // namespace acceleration_solver_internal
} // namespace embodik
