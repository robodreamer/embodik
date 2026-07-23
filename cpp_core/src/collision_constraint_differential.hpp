#pragma once

#include <embodik/robot_model.hpp>
#include <embodik/types.hpp>

#include <Eigen/Core>

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace embodik::detail {

struct CollisionDifferentialAccess;

enum class CollisionDifferentialProofKind {
  kUnavailable,
  kAnalyticSphereSphere,
};

struct CollisionPairDifferential {
  std::size_t pair_index = 0;
  std::string pair_key;
  std::string object_a;
  std::string object_b;
  Eigen::Vector3d center_a_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d center_b_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d point_a_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d point_b_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_a_to_b_world = Eigen::Vector3d::UnitX();
  Eigen::MatrixXd relative_center_jacobian;
  Eigen::Vector3d relative_center_velocity_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d relative_center_affine_acceleration_world =
      Eigen::Vector3d::Zero();
  double signed_distance = 0.0;
  Eigen::MatrixXd jacobian;
  double rate = 0.0;
  double affine_bias = 0.0;
};

struct BoundedCollisionPairDifferential {
  CollisionPairDifferential nominal;
  double affine_bias_error_bound = 0.0;
  double signed_distance_lower_bound =
      -std::numeric_limits<double>::infinity();
  double signed_distance_upper_bound =
      std::numeric_limits<double>::infinity();
  double rate_lower_bound = -std::numeric_limits<double>::infinity();
  double rate_upper_bound = std::numeric_limits<double>::infinity();
  CollisionDifferentialProofKind proof_kind =
      CollisionDifferentialProofKind::kUnavailable;
};

struct BoundedCollisionPairDifferentialResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::optional<BoundedCollisionPairDifferential> differential;

  bool satisfied() const {
    return status == SolverStatus::kSuccess && differential.has_value();
  }
};

struct BoundedCollisionPairDifferentialsResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<BoundedCollisionPairDifferential> differentials;
  std::size_t pair_evaluations = 0;
  std::size_t active_pairs_considered = 0;

  bool satisfied() const {
    return status == SolverStatus::kSuccess && !differentials.empty();
  }
};

class CollisionDifferentialScratch {
public:
  struct Impl;

  explicit CollisionDifferentialScratch(const RobotModel &robot);
  ~CollisionDifferentialScratch();

  CollisionDifferentialScratch(const CollisionDifferentialScratch &) = delete;
  CollisionDifferentialScratch &
  operator=(const CollisionDifferentialScratch &) = delete;
  CollisionDifferentialScratch(CollisionDifferentialScratch &&) noexcept;
  CollisionDifferentialScratch &
  operator=(CollisionDifferentialScratch &&) noexcept;

private:
  friend struct CollisionDifferentialAccess;
  friend BoundedCollisionPairDifferentialResult
  evaluate_bounded_collision_pair_differential_at_state(
      const RobotModel &, CollisionDifferentialScratch &, std::size_t,
      const Eigen::VectorXd &, const Eigen::VectorXd &);
  friend BoundedCollisionPairDifferentialsResult
  evaluate_bounded_collision_pair_differentials_at_state(
      const RobotModel &, CollisionDifferentialScratch &,
      const std::vector<std::size_t> &, const Eigen::VectorXd &,
      const Eigen::VectorXd &);
  friend BoundedCollisionPairDifferentialsResult
  evaluate_all_active_bounded_collision_pair_differentials_at_state(
      const RobotModel &, CollisionDifferentialScratch &,
      const Eigen::VectorXd &, const Eigen::VectorXd &);

  std::unique_ptr<Impl> impl_;
};

SolverStatus validate_collision_differential_model(const RobotModel &robot,
                                                   std::string *message);

BoundedCollisionPairDifferentialResult
evaluate_bounded_collision_pair_differential_at_state(
    const RobotModel &robot, CollisionDifferentialScratch &scratch,
    std::size_t pair_index, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq);

BoundedCollisionPairDifferentialsResult
evaluate_bounded_collision_pair_differentials_at_state(
    const RobotModel &robot, CollisionDifferentialScratch &scratch,
    const std::vector<std::size_t> &ordered_pair_indices,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq);

/**
 * Evaluate a solver-compiled exact pair set independently of RobotModel's
 * mutable active-pair mask. The caller owns filtering and catalog validation.
 */
BoundedCollisionPairDifferentialsResult
evaluate_selected_bounded_collision_pair_differentials_at_state(
    const RobotModel &robot, CollisionDifferentialScratch &scratch,
    const std::vector<std::size_t> &ordered_pair_indices,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq);

BoundedCollisionPairDifferentialsResult
evaluate_all_active_bounded_collision_pair_differentials_at_state(
    const RobotModel &robot, CollisionDifferentialScratch &scratch,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq);

} // namespace embodik::detail
