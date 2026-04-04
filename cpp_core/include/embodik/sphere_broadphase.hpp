#pragma once

#include <Eigen/Core>
#include <vector>

#ifdef PINOCCHIO_WITH_HPP_FCL
#include <pinocchio/multibody/geometry.hpp>
#endif

namespace embodik {

struct BoundingSphere {
  Eigen::Vector3d center;  // offset from parent frame origin
  double radius;
};

class SphereBroadphase {
public:
#ifdef PINOCCHIO_WITH_HPP_FCL
  void build(const pinocchio::GeometryModel &geom_model);
#endif

  double compute_pair_lower_bound(
      std::size_t geom_idx_a, std::size_t geom_idx_b,
      const Eigen::Vector3d &translation_a, const Eigen::Matrix3d &rotation_a,
      const Eigen::Vector3d &translation_b, const Eigen::Matrix3d &rotation_b) const;

  std::size_t size() const { return spheres_.size(); }
  bool is_built() const { return !spheres_.empty(); }
  const BoundingSphere &sphere(std::size_t idx) const { return spheres_[idx]; }

  double safety_margin = 0.01;  // metres, added to cutoff threshold

private:
  std::vector<BoundingSphere> spheres_;
};

}  // namespace embodik
