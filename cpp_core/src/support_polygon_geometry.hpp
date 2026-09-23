#pragma once

#include <embodik/types.hpp>

#include <Eigen/Core>

#include <string>
#include <vector>

namespace embodik::detail {

struct SupportPolygonGeometryResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

struct SupportPolygonGeometry {
  std::vector<Eigen::Vector2d> hull;
  Eigen::MatrixXd halfspace_normals;
  Eigen::VectorXd halfspace_offsets;
  double char_size = 0.0;
  double inradius = 0.0;
  double proximity_threshold = 0.0;
};

struct SupportPolygonGeometryPreparationResult
    : SupportPolygonGeometryResult {
  SupportPolygonGeometry geometry;
};

SupportPolygonGeometryPreparationResult prepare_support_polygon_geometry(
    const ComSupportPolygonConstraintDefinition &definition,
    const std::string &family_label, const std::string &source_id);

} // namespace embodik::detail
