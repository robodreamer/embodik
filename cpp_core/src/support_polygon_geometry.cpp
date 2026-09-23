#include "support_polygon_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace embodik::detail {
namespace {

constexpr double kGeometryTolerance = 1e-12;

SupportPolygonGeometryPreparationResult failure(SolverStatus status,
                                                std::string message) {
  SupportPolygonGeometryPreparationResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

std::string identity(const std::string &family_label,
                     const std::string &source_id) {
  return family_label + " '" + source_id + "'";
}

double cross2d(const Eigen::Vector2d &origin, const Eigen::Vector2d &a,
               const Eigen::Vector2d &b) {
  return (a.x() - origin.x()) * (b.y() - origin.y()) -
         (a.y() - origin.y()) * (b.x() - origin.x());
}

bool same_point(const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
  return (a - b).norm() <= kGeometryTolerance;
}

std::vector<Eigen::Vector2d>
convex_hull_2d(std::vector<Eigen::Vector2d> points) {
  std::sort(points.begin(), points.end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
              return a.x() < b.x() ||
                     (a.x() == b.x() && a.y() < b.y());
            });
  points.erase(std::unique(points.begin(), points.end(), same_point),
               points.end());
  if (points.size() < 3U) {
    return points;
  }

  std::vector<Eigen::Vector2d> hull;
  hull.reserve(2U * points.size());
  for (const auto &point : points) {
    while (hull.size() >= 2U &&
           cross2d(hull[hull.size() - 2U], hull.back(), point) <=
               kGeometryTolerance) {
      hull.pop_back();
    }
    hull.push_back(point);
  }
  const std::size_t lower_size = hull.size();
  for (auto it = points.rbegin() + 1; it != points.rend(); ++it) {
    while (hull.size() > lower_size &&
           cross2d(hull[hull.size() - 2U], hull.back(), *it) <=
               kGeometryTolerance) {
      hull.pop_back();
    }
    hull.push_back(*it);
  }
  hull.pop_back();
  return hull;
}

double polygon_area(const std::vector<Eigen::Vector2d> &hull) {
  double twice_area = 0.0;
  for (std::size_t index = 0; index < hull.size(); ++index) {
    const auto &a = hull[index];
    const auto &b = hull[(index + 1U) % hull.size()];
    twice_area += a.x() * b.y() - b.x() * a.y();
  }
  return 0.5 * twice_area;
}

Eigen::Vector2d polygon_centroid_average(
    const std::vector<Eigen::Vector2d> &hull) {
  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto &vertex : hull) {
    centroid += vertex;
  }
  return centroid / static_cast<double>(hull.size());
}

double polygon_char_size(const std::vector<Eigen::Vector2d> &hull,
                         const Eigen::Vector2d &centroid) {
  double sum = 0.0;
  for (const auto &vertex : hull) {
    sum += (vertex - centroid).norm();
  }
  return sum / static_cast<double>(hull.size());
}

std::vector<Eigen::Vector2d>
shrink_polygon(const std::vector<Eigen::Vector2d> &hull,
               const Eigen::Vector2d &centroid, double shrink_distance) {
  if (shrink_distance <= 0.0) {
    return hull;
  }
  std::vector<Eigen::Vector2d> shrunk;
  shrunk.reserve(hull.size());
  for (const auto &vertex : hull) {
    const Eigen::Vector2d direction = centroid - vertex;
    const double distance = direction.norm();
    if (distance <= kGeometryTolerance || shrink_distance >= distance) {
      shrunk.push_back(centroid);
    } else {
      shrunk.push_back(vertex + (shrink_distance / distance) * direction);
    }
  }
  return shrunk;
}

bool polygon_to_halfplanes(const std::vector<Eigen::Vector2d> &hull,
                           Eigen::MatrixXd *normals,
                           Eigen::VectorXd *offsets, double *inradius) {
  const Eigen::Index rows = static_cast<Eigen::Index>(hull.size());
  normals->resize(rows, 2);
  offsets->resize(rows);
  const Eigen::Vector2d centroid = polygon_centroid_average(hull);
  double minimum_distance = std::numeric_limits<double>::infinity();

  for (Eigen::Index row = 0; row < rows; ++row) {
    const auto &a = hull[static_cast<std::size_t>(row)];
    const auto &b = hull[static_cast<std::size_t>((row + 1) % rows)];
    const Eigen::Vector2d edge = b - a;
    Eigen::Vector2d normal(edge.y(), -edge.x());
    const double length = normal.norm();
    if (length <= kGeometryTolerance) {
      return false;
    }
    normal /= length;
    double offset = normal.dot(a);
    if (normal.dot(centroid) > offset) {
      normal = -normal;
      offset = -offset;
    }
    const double distance = offset - normal.dot(centroid);
    if (!(distance > kGeometryTolerance) || !std::isfinite(distance)) {
      return false;
    }
    normals->row(row) = normal.transpose();
    (*offsets)(row) = offset;
    minimum_distance = std::min(minimum_distance, distance);
  }
  *inradius = minimum_distance;
  return std::isfinite(*inradius) && *inradius > kGeometryTolerance;
}

} // namespace

SupportPolygonGeometryPreparationResult prepare_support_polygon_geometry(
    const ComSupportPolygonConstraintDefinition &definition,
    const std::string &family_label, const std::string &source_id) {
  if (definition.support_polygon.rows() < 3 ||
      definition.support_polygon.cols() < 2) {
    return failure(SolverStatus::kInvalidInput,
                   identity(family_label, source_id) +
                       " support_polygon must have at least 3 rows and 2 "
                       "columns");
  }
  if (!definition.support_polygon.allFinite() ||
      !std::isfinite(definition.margin) ||
      !std::isfinite(definition.proximity_fraction)) {
    return failure(SolverStatus::kNonFiniteInput,
                   identity(family_label, source_id) +
                       " geometry definition must contain only finite values");
  }
  if (definition.margin < 0.0 || definition.margin > 1.0 ||
      definition.proximity_fraction < 0.0) {
    return failure(SolverStatus::kInvalidInput,
                   identity(family_label, source_id) +
                       " margin must be in [0, 1] and proximity_fraction must "
                       "be non-negative");
  }

  std::vector<Eigen::Vector2d> points;
  points.reserve(static_cast<std::size_t>(definition.support_polygon.rows()));
  for (Eigen::Index row = 0; row < definition.support_polygon.rows(); ++row) {
    points.emplace_back(definition.support_polygon(row, 0),
                        definition.support_polygon(row, 1));
  }

  auto hull = convex_hull_2d(std::move(points));
  if (hull.size() < 3U || std::abs(polygon_area(hull)) <= kGeometryTolerance) {
    return failure(SolverStatus::kInvalidInput,
                   identity(family_label, source_id) +
                       " convex hull is degenerate or collinear");
  }
  if (polygon_area(hull) < 0.0) {
    std::reverse(hull.begin(), hull.end());
  }

  const Eigen::Vector2d centroid = polygon_centroid_average(hull);
  const double char_size = polygon_char_size(hull, centroid);
  if (!(char_size > kGeometryTolerance) || !std::isfinite(char_size)) {
    return failure(SolverStatus::kInvalidInput,
                   identity(family_label, source_id) +
                       " support polygon characteristic size is degenerate");
  }
  const double shrink_distance = definition.margin * char_size;
  if (shrink_distance > 0.0) {
    hull = convex_hull_2d(shrink_polygon(hull, centroid, shrink_distance));
    if (hull.size() < 3U ||
        std::abs(polygon_area(hull)) <= kGeometryTolerance) {
      return failure(SolverStatus::kInvalidInput,
                     identity(family_label, source_id) +
                         " margin collapses the support polygon");
    }
  }

  SupportPolygonGeometryPreparationResult result;
  result.geometry.hull = std::move(hull);
  result.geometry.char_size = char_size;
  if (!polygon_to_halfplanes(result.geometry.hull,
                             &result.geometry.halfspace_normals,
                             &result.geometry.halfspace_offsets,
                             &result.geometry.inradius)) {
    return failure(SolverStatus::kInvalidInput,
                   identity(family_label, source_id) +
                       " half-plane geometry is degenerate");
  }
  result.geometry.proximity_threshold =
      definition.proximity_fraction > 0.0
          ? definition.proximity_fraction * result.geometry.inradius
          : std::numeric_limits<double>::infinity();
  return result;
}

} // namespace embodik::detail
