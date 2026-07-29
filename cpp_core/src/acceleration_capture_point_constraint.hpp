#pragma once

#include <embodik/acceleration_solver.hpp>

#include "support_polygon_geometry.hpp"

#include <Eigen/Core>

#include <string>
#include <unordered_set>
#include <vector>

namespace embodik::detail {

struct CapturePointConstraintResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

struct PreparedCapturePointConstraint {
  CapturePointAccelerationConstraint specification;
  SupportPolygonGeometry geometry;
  AffineAccelerationConstraint physical_constraint;
  double frozen_omega = 0.0;
};

struct CapturePointConstraintAssembly : CapturePointConstraintResult {
  std::vector<AffineAccelerationConstraint> constraints;
  std::vector<PreparedCapturePointConstraint> prepared_constraints;
  Eigen::Index row_count = 0;
};

CapturePointConstraintAssembly make_capture_point_constraints(
    const RobotModel &robot,
    const std::vector<CapturePointAccelerationConstraint> &constraints,
    double dt, std::unordered_set<std::string> *source_ids);

CapturePointConstraintResult validate_capture_point_constraint_acceptance(
    const PreparedCapturePointConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q_solution, const Eigen::VectorXd &dq_next,
    CapturePointAccelerationDiagnostics *diagnostic = nullptr);

} // namespace embodik::detail
