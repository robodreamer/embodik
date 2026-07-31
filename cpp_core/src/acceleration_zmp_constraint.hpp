#pragma once

#include <embodik/acceleration_solver.hpp>

#include "support_polygon_geometry.hpp"

#include <Eigen/Core>

#include <string>
#include <unordered_set>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif

namespace embodik::detail {

struct ZmpConstraintResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

struct PreparedZmpConstraint {
  ZmpAccelerationConstraint specification;
  SupportPolygonGeometry geometry;
  AffineAccelerationConstraint physical_constraint;
};

struct ZmpConstraintAssembly : ZmpConstraintResult {
  std::vector<AffineAccelerationConstraint> constraints;
  std::vector<PreparedZmpConstraint> prepared_constraints;
  Eigen::Index row_count = 0;
};

ZmpConstraintAssembly make_zmp_constraints(
    const RobotModel &robot,
    const std::vector<ZmpAccelerationConstraint> &constraints,
    std::unordered_set<std::string> *source_ids);

ZmpConstraintResult validate_zmp_constraint_acceptance(
    const PreparedZmpConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q_solution, const Eigen::VectorXd &dq_next,
    const Eigen::VectorXd &accepted_acceleration,
    ZmpAccelerationDiagnostics *diagnostic = nullptr);

} // namespace embodik::detail

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
