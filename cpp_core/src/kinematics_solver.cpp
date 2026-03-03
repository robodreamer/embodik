/**
 * @file kinematics_solver.cpp
 * @brief Implementation of high-level kinematics solver
 */

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <pinocchio/algorithm/geometry.hpp>
#include <stdexcept>
#include <unordered_set>
#ifdef PINOCCHIO_WITH_HPP_FCL
#include <pinocchio/collision/distance.hpp>
#endif
#include <pinocchio/algorithm/joint-configuration.hpp>

#include <embodik/ik_baseline.hpp>
#include <embodik/kinematics_solver.hpp>

namespace embodik {

namespace {
constexpr double kCollisionTolerance = 1e-4;
constexpr double kCollisionUpperDistance = 1e1;
constexpr double kDistanceEpsilon = 1e-9;
constexpr double kCollisionMaxSeparationSpeed = 0.5;
constexpr double kCollisionMaxSeparationSpeedNonPenetration = 0.15;
constexpr double kCollisionPairSwitchHysteresis = 2e-3;
constexpr double kCollisionRepulsionDeadband = 3e-3;
constexpr double kCollisionMinRecoverySpeed = 0.05;
constexpr double kCollisionRecoveryScale = 0.2;
constexpr double kCollisionStuckBand = 3e-3;
constexpr double kCollisionStuckDqNormEps = 1e-6;
constexpr int kCollisionStuckCountThreshold = 10;
} // namespace

KinematicsSolver::KinematicsSolver(std::shared_ptr<RobotModel> robot)
    : robot_(robot) {
  if (!robot_) {
    throw std::invalid_argument("Robot model cannot be null");
  }
}

std::shared_ptr<FrameTask>
KinematicsSolver::add_frame_task(const std::string &name,
                                 const std::string &frame_name,
                                 TaskType task_type) {

  // Check if task already exists
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task = std::make_shared<FrameTask>(name, robot_, frame_name, task_type);
  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

std::shared_ptr<COMTask>
KinematicsSolver::add_com_task(const std::string &name) {
  // Check if task already exists
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task = std::make_shared<COMTask>(name, robot_);
  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

std::shared_ptr<PostureTask>
KinematicsSolver::add_posture_task(const std::string &name,
                                   const std::vector<int> &controlled_joints) {

  // Check if task already exists
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  std::shared_ptr<PostureTask> task;
  if (controlled_joints.empty()) {
    task = std::make_shared<PostureTask>(name, robot_);
  } else {
    task = std::make_shared<PostureTask>(name, robot_, controlled_joints);
  }

  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

std::shared_ptr<JointTask>
KinematicsSolver::add_joint_task(const std::string &name,
                                 const std::string &joint_name,
                                 double target_value) {

  // Check if task already exists
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task =
      std::make_shared<JointTask>(name, robot_, joint_name, target_value);
  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

void KinematicsSolver::remove_task(const std::string &name) {
  auto it = task_map_.find(name);
  if (it != task_map_.end()) {
    auto task = it->second;
    task_map_.erase(it);

    // Remove from tasks vector
    tasks_.erase(std::remove(tasks_.begin(), tasks_.end(), task), tasks_.end());
  }
}

void KinematicsSolver::clear_tasks() {
  tasks_.clear();
  task_map_.clear();
}

std::shared_ptr<Task> KinematicsSolver::get_task(const std::string &name) {
  auto it = task_map_.find(name);
  return (it != task_map_.end()) ? it->second : nullptr;
}

void KinematicsSolver::set_base_position_bounds(const Eigen::Vector3d &lower,
                                                const Eigen::Vector3d &upper) {
  base_position_lower_ = lower;
  base_position_upper_ = upper;
}

void KinematicsSolver::set_base_orientation_bounds(
    const Eigen::Vector3d &lower, const Eigen::Vector3d &upper) {
  base_orientation_lower_ = lower;
  base_orientation_upper_ = upper;
}

void KinematicsSolver::clear_base_bounds() {
  base_position_lower_.reset();
  base_position_upper_.reset();
  base_orientation_lower_.reset();
  base_orientation_upper_.reset();
}

void KinematicsSolver::configure_collision_constraint(
    double min_distance,
    const std::vector<std::pair<std::string, std::string>> &include_pairs,
    const std::vector<std::pair<std::string, std::string>> &exclude_pairs,
    bool nearest_points_all_pairs) {

#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!robot_->has_collision_geometry()) {
    throw std::runtime_error(
        "Collision geometry is not available for this robot model.");
  }

  auto *collision_model_ptr = robot_->collision_model();
  auto *collision_data = robot_->collision_data();

  if (!collision_constraint_.has_value()) {
    collision_constraint_.emplace();
  }

  auto &config = *collision_constraint_;
  config.enabled = true;
  config.min_distance = std::max(0.0, min_distance);
  config.upper_distance = kCollisionUpperDistance;
  config.tolerance = kCollisionTolerance;
  config.nearest_points_all_pairs = nearest_points_all_pairs;
  config.include_pairs.clear();
  config.exclude_pairs.clear();

  for (const auto &pair : include_pairs) {
    config.include_pairs.insert(canonical_pair_key(pair.first, pair.second));
  }

  for (const auto &pair : exclude_pairs) {
    config.exclude_pairs.insert(canonical_pair_key(pair.first, pair.second));
  }

  if (collision_data != nullptr && collision_model_ptr != nullptr) {
    const std::size_t num_pairs = collision_model_ptr->collisionPairs.size();
    if (collision_data->distanceRequests.size() != num_pairs) {
      collision_data->distanceRequests.resize(num_pairs);
      collision_data->distanceResults.resize(num_pairs);
    }
    for (auto &request : collision_data->distanceRequests) {
      request.enable_nearest_points = nearest_points_all_pairs;
      request.enable_signed_distance = true;
    }
    collision_data->activateAllCollisionPairs();

    collision_allowed_pair_mask_.assign(num_pairs, 0);
    for (std::size_t idx = 0; idx < num_pairs; ++idx) {
      const auto &pair = collision_model_ptr->collisionPairs[idx];
      const auto &name_a =
          collision_model_ptr->geometryObjects[pair.first].name;
      const auto &name_b =
          collision_model_ptr->geometryObjects[pair.second].name;
      const auto key = canonical_pair_key(name_a, name_b);
      bool allowed = true;
      if (!config.include_pairs.empty() &&
          config.include_pairs.find(key) == config.include_pairs.end()) {
        allowed = false;
      }
      if (allowed &&
          config.exclude_pairs.find(key) != config.exclude_pairs.end()) {
        allowed = false;
      }
      collision_allowed_pair_mask_[idx] = allowed ? 1 : 0;
      if (!allowed) {
        collision_data->activeCollisionPairs[idx] = false;
      }
    }
    last_collision_constraint_pair_index_.reset();
  }
#else
  (void)min_distance;
  (void)include_pairs;
  (void)exclude_pairs;
  (void)nearest_points_all_pairs;
  throw std::runtime_error("Collision avoidance requires Pinocchio to be built "
                           "with hpp-fcl support.");
#endif
}

std::optional<KinematicsSolver::CollisionDebugInfo>
KinematicsSolver::evaluate_collision_debug(const Eigen::VectorXd &current_q) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!robot_->has_collision_geometry()) {
    return std::nullopt;
  }

  if (current_q.size() > 0) {
    if (current_q.size() != robot_->nq()) {
      throw std::runtime_error(
          "Invalid configuration size for collision evaluation.");
    }
    robot_->update_kinematics(current_q);
  } else {
    robot_->update_kinematics(robot_->get_current_configuration());
  }

  const auto prev_last_collision_debug = last_collision_debug_;
  const auto prev_last_pair_index = last_collision_constraint_pair_index_;
  const int prev_stuck_counter = collision_stuck_counter_;
  const auto prev_stuck_pair_index = collision_stuck_pair_index_;
  const double prev_stuck_last_distance = collision_stuck_last_distance_;

  (void)compute_collision_constraint();
  const auto debug = last_collision_debug_;

  last_collision_debug_ = prev_last_collision_debug;
  last_collision_constraint_pair_index_ = prev_last_pair_index;
  collision_stuck_counter_ = prev_stuck_counter;
  collision_stuck_pair_index_ = prev_stuck_pair_index;
  collision_stuck_last_distance_ = prev_stuck_last_distance;

  return debug;
#else
  (void)current_q;
  return std::nullopt;
#endif
}

void KinematicsSolver::add_collision_constraint(
    const std::vector<std::pair<std::string, std::string>> &link_pairs,
    double min_distance) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  configure_collision_constraint(min_distance, link_pairs, {}, true);
#else
  (void)link_pairs;
  (void)min_distance;
  throw std::runtime_error("Collision avoidance requires Pinocchio to be built "
                           "with hpp-fcl support.");
#endif
}

void KinematicsSolver::clear_collision_constraint() {
  if (collision_constraint_.has_value()) {
    collision_constraint_->enabled = false;
  }
  collision_allowed_pair_mask_.clear();
}

// ============================================================
// CoM support-polygon constraint helpers
// ============================================================
namespace {

// 2D cross product of vectors OA and OB.
static double cross2d(const Eigen::Vector2d &O, const Eigen::Vector2d &A,
                      const Eigen::Vector2d &B) {
  return (A.x() - O.x()) * (B.y() - O.y()) -
         (A.y() - O.y()) * (B.x() - O.x());
}

// Graham scan: returns convex hull vertices in CCW order.
static std::vector<Eigen::Vector2d>
convex_hull_2d(std::vector<Eigen::Vector2d> pts) {
  const int n = static_cast<int>(pts.size());
  if (n < 3)
    return pts;

  std::sort(pts.begin(), pts.end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
              return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
            });
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());

  if (static_cast<int>(pts.size()) < 3)
    return pts;

  std::vector<Eigen::Vector2d> hull;
  hull.reserve(2 * pts.size());

  // Lower hull
  for (const auto &p : pts) {
    while (hull.size() >= 2 &&
           cross2d(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0.0)
      hull.pop_back();
    hull.push_back(p);
  }

  // Upper hull
  const int lower_size = static_cast<int>(hull.size()) + 1;
  for (int i = static_cast<int>(pts.size()) - 2; i >= 0; --i) {
    while (static_cast<int>(hull.size()) >= lower_size &&
           cross2d(hull[hull.size() - 2], hull[hull.size() - 1], pts[i]) <=
               0.0)
      hull.pop_back();
    hull.push_back(pts[i]);
  }
  hull.pop_back();
  return hull;
}

// Build half-plane representation A * x <= b from a CCW convex polygon.
// For each edge (v_i -> v_{i+1}), the outward normal points to the right.
static void polygon_to_halfplanes(const std::vector<Eigen::Vector2d> &hull,
                                  Eigen::MatrixXd &A, Eigen::VectorXd &b) {
  const int n = static_cast<int>(hull.size());
  A.resize(n, 2);
  b.resize(n);
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d &v0 = hull[i];
    const Eigen::Vector2d &v1 = hull[(i + 1) % n];
    Eigen::Vector2d edge = v1 - v0;
    // Outward normal (CCW hull: right side is outside)
    Eigen::Vector2d normal(edge.y(), -edge.x());
    const double len = normal.norm();
    if (len < 1e-12)
      normal = Eigen::Vector2d(1.0, 0.0);
    else
      normal /= len;
    A.row(i) = normal.transpose();
    b(i) = normal.dot(v0);
  }
}

// Shrink polygon vertices toward centroid by fractional margin in [0, 1].
// Uses mean distance from centroid to vertices (char_size) to match the
// optional_wheelbase_viser feasibility check (compute_polygon_characteristic_size).
static std::vector<Eigen::Vector2d>
shrink_polygon(const std::vector<Eigen::Vector2d> &hull, double margin) {
  if (margin <= 0.0 || hull.empty())
    return hull;

  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto &v : hull)
    centroid += v;
  centroid /= static_cast<double>(hull.size());

  double sum_dist = 0.0;
  for (const auto &v : hull)
    sum_dist += (v - centroid).norm();
  const double char_size =
      (hull.size() > 0) ? (sum_dist / static_cast<double>(hull.size())) : 0.0;

  const double shrink_dist =
      std::clamp(margin, 0.0, 1.0) * std::max(0.0, char_size - 1e-9);

  std::vector<Eigen::Vector2d> shrunk;
  shrunk.reserve(hull.size());
  for (const auto &v : hull) {
    Eigen::Vector2d dir = centroid - v;
    const double d = dir.norm();
    if (d < 1e-12)
      shrunk.push_back(v);
    else
      shrunk.push_back(v + (shrink_dist / d) * dir);
  }
  return shrunk;
}

// Minimum perpendicular distance from the polygon centroid to any edge.
// This is the radius of the largest inscribed circle (inradius) and serves
// as the natural distance scale for the polygon.
static double polygon_inradius_2d(const std::vector<Eigen::Vector2d> &hull) {
  if (hull.size() < 3)
    return 0.0;

  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto &v : hull)
    centroid += v;
  centroid /= static_cast<double>(hull.size());

  double min_dist = std::numeric_limits<double>::infinity();
  const int n = static_cast<int>(hull.size());
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d &a = hull[i];
    const Eigen::Vector2d &b = hull[(i + 1) % n];
    const Eigen::Vector2d edge = b - a;
    const double edge_len = edge.norm();
    if (edge_len < 1e-12)
      continue;
    // Outward unit normal (same convention as polygon_to_halfplanes: CCW hull)
    const Eigen::Vector2d normal(edge.y(), -edge.x());
    const double dist = std::fabs((centroid - a).dot(normal) / edge_len);
    min_dist = std::min(min_dist, dist);
  }
  return std::isfinite(min_dist) ? min_dist : 0.0;
}

} // namespace

void KinematicsSolver::configure_com_constraint(
    const Eigen::MatrixXd &vertices_xy, double margin,
    const std::string &frame_name, double com_vel_max, double com_acc_max,
    bool use_acceleration_limits, double proximity_fraction) {
  if (vertices_xy.rows() < 3) {
    throw std::invalid_argument(
        "configure_com_constraint: support polygon must have at least 3 "
        "vertices.");
  }
  if (vertices_xy.cols() < 2) {
    throw std::invalid_argument(
        "configure_com_constraint: vertices must have at least 2 columns (xy).");
  }
  if (frame_name != "world" && !robot_->has_frame(frame_name)) {
    throw std::runtime_error("configure_com_constraint: frame '" + frame_name +
                             "' not found in robot model.");
  }

  // Collect 2D points
  std::vector<Eigen::Vector2d> pts;
  pts.reserve(vertices_xy.rows());
  for (int i = 0; i < vertices_xy.rows(); ++i)
    pts.emplace_back(vertices_xy(i, 0), vertices_xy(i, 1));

  // Compute convex hull (CCW)
  auto hull = convex_hull_2d(pts);
  if (hull.size() < 3) {
    throw std::runtime_error(
        "configure_com_constraint: convex hull has fewer than 3 vertices "
        "(points may be collinear).");
  }

  // Apply inward margin
  if (margin > 0.0)
    hull = shrink_polygon(hull, margin);

  // Build half-plane representation in frame_name
  ComConstraintConfig cfg;
  cfg.enabled = true;
  cfg.vertices_xy = vertices_xy;
  cfg.margin = margin;
  cfg.frame_name = frame_name;
  cfg.com_vel_max = com_vel_max;
  cfg.com_acc_max = com_acc_max;
  cfg.use_acceleration_limits = use_acceleration_limits;
  // Auto-compute proximity threshold from the convex hull inradius so the
  // caller never needs to reason about polygon geometry themselves.
  if (proximity_fraction > 0.0)
    cfg.proximity_threshold = proximity_fraction * polygon_inradius_2d(hull);
  else
    cfg.proximity_threshold = std::numeric_limits<double>::infinity();
  polygon_to_halfplanes(hull, cfg.A, cfg.b);
  com_constraint_ = std::move(cfg);
}

void KinematicsSolver::clear_com_constraint() { com_constraint_.reset(); }

double KinematicsSolver::get_com_proximity_threshold() const {
  if (!com_constraint_.has_value())
    return 0.0;
  return com_constraint_->proximity_threshold;
}

std::optional<KinematicsSolver::ComConstraintResult>
KinematicsSolver::compute_com_constraint() {
  if (!com_constraint_.has_value() || !com_constraint_->enabled)
    return std::nullopt;

  const auto &cfg = *com_constraint_;

  // CoM position (world) and Jacobian (3 x nv)
  const Eigen::Vector3d com_world = robot_->get_com_position();
  const Eigen::MatrixXd J_com = robot_->get_com_jacobian(); // 3 x nv

  // Half-planes in frame_name; transform to world if necessary
  Eigen::MatrixXd A_world = cfg.A; // #hp x 2
  Eigen::VectorXd b_world = cfg.b; // #hp

  if (cfg.frame_name != "world") {
    const auto frame_pose = robot_->get_frame_pose(cfg.frame_name);
    const Eigen::Matrix3d R = frame_pose.rotation();
    const Eigen::Vector3d t = frame_pose.translation();
    const Eigen::Matrix2d R_xy = R.topLeftCorner<2, 2>();

    // x_F = R_xy^T * (x_world_xy - t_xy)
    // A_F * x_F <= b_F
    // => A_F * R_xy^T * x_world_xy <= b_F + A_F * R_xy^T * t_xy
    A_world = cfg.A * R_xy.transpose(); // #hp x 2
    b_world = cfg.b;
    for (int i = 0; i < static_cast<int>(b_world.size()); ++i)
      b_world(i) += A_world.row(i).dot(t.head<2>());
  }

  // Slack per half-plane: positive when CoM is inside polygon.
  const Eigen::VectorXd slack = b_world - A_world * com_world.head<2>();

  // Constraint Jacobian: all half-planes (#hp x nv)
  const Eigen::MatrixXd J_all = A_world * J_com.topRows(2);

  const int n_hp = static_cast<int>(slack.size());
  const double vel_max = cfg.com_vel_max;
  const double acc_max = cfg.com_acc_max;
  const double prox = cfg.proximity_threshold;

  // Anti-chattering: small epsilon dead-zone at the boundary.  When the
  // slack is within [-eps, 0] the CoM is treated as "at the boundary"
  // rather than outside, preventing sign-flip oscillations between
  // frames.  Matches the kMarginEpsilon pattern in
  // calculate_velocity_box_constraint().
  constexpr double kSlackEps = 1e-4; // 0.1 mm

  // All half-plane rows always participate so that vel_max and acceleration
  // limits are enforced everywhere (matching the Spot Flex IK pattern).
  //
  // Three bound layers, from coarsest to tightest:
  //   1. vel_max              — always active, caps speed in every direction.
  //   2. sqrt(2*acc*slack_c)  — always active when use_acceleration_limits is
  //      set; starts tapering velocity well before the boundary, ensuring
  //      the CoM can decelerate smoothly (bounded tipping energy).
  //   3. slack_c/dt           — only active when slack < proximity_threshold;
  //      the hard position-based limit that prevents overshooting the
  //      boundary in a single time step.
  //
  // slack_c = max(0, slack): clamped to non-negative so that position and
  // acceleration terms never flip sign (same as the joint-limit pattern in
  // calculate_velocity_box_constraint).  When the CoM is slightly outside
  // (slack < 0 but > -eps), slack_c = 0 produces upper = 0: the solver
  // stops outward motion without commanding a recovery kick that would
  // cause chattering.
  Eigen::VectorXd lower(n_hp), upper(n_hp);

  for (int i = 0; i < n_hp; ++i) {
    const double m = slack(i);
    const double m_clamped = std::max(0.0, m);

    // Start with the velocity cap (always active)
    double ub = vel_max;

    // Acceleration bound (always active): smoothly reduce approach speed
    // as the CoM gets closer to the boundary.
    if (cfg.use_acceleration_limits) {
      ub = std::min(ub, std::sqrt(2.0 * acc_max * m_clamped));
    }

    // Position-based limit (near boundary only): prevents overshooting
    // the boundary in a single time step.
    if (m < prox) {
      ub = std::min(ub, m_clamped / dt_);
    }

    upper(i) = ub;

    // Lower bound: allow full velocity away from boundary, but if the
    // CoM is significantly outside (slack < -eps) also let the solver
    // push it back inward without over-restricting.
    if (m < -kSlackEps) {
      // Outside by more than epsilon: relax the upper bound so the QP
      // can push the CoM back toward the interior.
      upper(i) = vel_max;
    }

    lower(i) = -vel_max;
  }

  ComConstraintResult result;
  result.jacobian = J_all;
  result.lower_bounds = lower;
  result.upper_bounds = upper;
  return result;
}

std::string KinematicsSolver::canonical_pair_key(const std::string &a,
                                                 const std::string &b) const {
  if (a <= b) {
    return a + "|" + b;
  }
  return b + "|" + a;
}

bool KinematicsSolver::collision_pair_allowed(const std::string &a,
                                              const std::string &b) const {
  if (!collision_constraint_.has_value() || !collision_constraint_->enabled) {
    return false;
  }

  const auto key = canonical_pair_key(a, b);
  const auto &config = *collision_constraint_;

  if (!config.include_pairs.empty() &&
      config.include_pairs.find(key) == config.include_pairs.end()) {
    return false;
  }

  if (config.exclude_pairs.find(key) != config.exclude_pairs.end()) {
    return false;
  }

  return true;
}

std::vector<std::pair<std::string, std::string>>
KinematicsSolver::get_active_collision_pairs() const {
  std::vector<std::pair<std::string, std::string>> result;
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!robot_->has_collision_geometry()) {
    return result;
  }

  const auto *collision_model_ptr = robot_->collision_model();
  if (collision_model_ptr == nullptr) {
    return result;
  }

  if (!collision_constraint_.has_value() || !collision_constraint_->enabled) {
    return result;
  }

  const auto &pairs = collision_model_ptr->collisionPairs;
  if (collision_allowed_pair_mask_.size() == pairs.size()) {
    for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
      if (!collision_allowed_pair_mask_[idx]) {
        continue;
      }
      const auto &pair = pairs[idx];
      const auto &name_a =
          collision_model_ptr->geometryObjects[pair.first].name;
      const auto &name_b =
          collision_model_ptr->geometryObjects[pair.second].name;
      result.emplace_back(name_a, name_b);
    }
  } else {
    for (const auto &pair : pairs) {
      const auto &name_a =
          collision_model_ptr->geometryObjects[pair.first].name;
      const auto &name_b =
          collision_model_ptr->geometryObjects[pair.second].name;
      if (collision_pair_allowed(name_a, name_b)) {
        result.emplace_back(name_a, name_b);
      }
    }
  }
#endif
  return result;
}

std::optional<KinematicsSolver::CollisionConstraintResult>
KinematicsSolver::compute_collision_constraint() {
#ifdef PINOCCHIO_WITH_HPP_FCL
  last_collision_debug_.reset();
  if (!robot_->has_collision_geometry()) {
    return std::nullopt;
  }

  auto *collision_model = robot_->collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return std::nullopt;
  }

  bool constraint_active =
      collision_constraint_.has_value() && collision_constraint_->enabled;
  const bool nearest_points_all_pairs =
      collision_constraint_.has_value() &&
      collision_constraint_->nearest_points_all_pairs;

  // Update collision placements and compute distances for all pairs
  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);
  const auto &pairs = collision_model->collisionPairs;

  double best_distance_allowed = std::numeric_limits<double>::infinity();
  std::optional<std::size_t> best_index_allowed;

  double best_distance_debug = std::numeric_limits<double>::infinity();
  std::optional<std::size_t> best_index_debug;

  auto ensure_nearest_points_for_pair = [&](std::size_t idx) {
    if (nearest_points_all_pairs) {
      return;
    }
    if (idx >= collision_data->distanceRequests.size()) {
      return;
    }
    auto &req = collision_data->distanceRequests[idx];
    if (req.enable_nearest_points) {
      return;
    }
    req.enable_nearest_points = true;
    pinocchio::computeDistance(*collision_model, *collision_data, idx);
    req.enable_nearest_points = false;
  };

  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    // Honor Pinocchio's active-pair mask *before* distance computation.
    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[idx]) {
      continue;
    }
    pinocchio::computeDistance(*collision_model, *collision_data, idx);

    const auto &pair = pairs[idx];
    const auto &object_a = collision_model->geometryObjects[pair.first];
    const auto &object_b = collision_model->geometryObjects[pair.second];
    const auto &distance_result = collision_data->distanceResults[idx];

    double distance = distance_result.min_distance;
    if (!std::isfinite(distance)) {
      continue;
    }

    if (distance < best_distance_debug) {
      best_distance_debug = distance;
      best_index_debug = idx;
    }

    if (!constraint_active) {
      continue;
    }

    // Only consider include/exclude-allowed pairs for the constraint.
    // (We still compute debug info over the closest active pair overall.)
    if (!collision_allowed_pair_mask_.empty() &&
        idx < collision_allowed_pair_mask_.size() &&
        !collision_allowed_pair_mask_[idx]) {
      continue;
    }

    if (distance < best_distance_allowed) {
      best_distance_allowed = distance;
      best_index_allowed = idx;
    }
  }

  // Apply hysteresis to reduce rapid pair switching when multiple pairs have
  // similar minimum distance.
  if (constraint_active && best_index_allowed.has_value()) {
    if (last_collision_constraint_pair_index_.has_value()) {
      const std::size_t prev_idx = *last_collision_constraint_pair_index_;
      if (prev_idx < pairs.size()) {
        const auto &res_prev = collision_data->distanceResults[prev_idx];
        const double prev_dist = res_prev.min_distance;
        const bool prev_active = collision_data->activeCollisionPairs.empty() ||
                                 collision_data->activeCollisionPairs[prev_idx];
        const bool prev_allowed =
            collision_allowed_pair_mask_.empty() ||
            (prev_idx < collision_allowed_pair_mask_.size() &&
             collision_allowed_pair_mask_[prev_idx]);
        if (prev_active && prev_allowed && std::isfinite(prev_dist)) {
          // Keep previous pair unless it's meaningfully worse than the new
          // best.
          if (prev_dist <=
              best_distance_allowed + kCollisionPairSwitchHysteresis) {
            best_index_allowed = prev_idx;
            best_distance_allowed = prev_dist;
          }
        }
      }
    }
    last_collision_constraint_pair_index_ = best_index_allowed;
  } else {
    last_collision_constraint_pair_index_.reset();
  }

  // For debug visualization/logging, prefer the constrained pair when enabled;
  // otherwise show the globally closest active pair.
  std::optional<std::size_t> debug_index_to_use = best_index_debug;
  if (constraint_active && best_index_allowed.has_value()) {
    debug_index_to_use = best_index_allowed;
  }

  if (debug_index_to_use.has_value()) {
    ensure_nearest_points_for_pair(*debug_index_to_use);
    const auto &pair_debug = pairs[*debug_index_to_use];
    const auto &obj_da = collision_model->geometryObjects[pair_debug.first];
    const auto &obj_db = collision_model->geometryObjects[pair_debug.second];
    const auto &res_debug =
        collision_data->distanceResults[*debug_index_to_use];

    CollisionDebugInfo debug_info;
    debug_info.object_a = obj_da.name;
    debug_info.object_b = obj_db.name;
    debug_info.distance = res_debug.min_distance;
    debug_info.point_a_world = res_debug.nearest_points[0].cast<double>();
    debug_info.point_b_world = res_debug.nearest_points[1].cast<double>();
    last_collision_debug_ = debug_info;
  } else {
    last_collision_debug_.reset();
  }

  if (!constraint_active || !best_index_allowed.has_value()) {
    return std::nullopt;
  }

  ensure_nearest_points_for_pair(*best_index_allowed);
  const auto &pair = pairs[*best_index_allowed];
  const auto &object_a = collision_model->geometryObjects[pair.first];
  const auto &object_b = collision_model->geometryObjects[pair.second];
  const auto &distance_result =
      collision_data->distanceResults[*best_index_allowed];

  Eigen::Vector3d p1_world = distance_result.nearest_points[0].cast<double>();
  Eigen::Vector3d p2_world = distance_result.nearest_points[1].cast<double>();

  Eigen::Vector3d distance_vector = p1_world - p2_world;
  double distance_norm = distance_vector.norm();
  Eigen::Vector3d normal = Eigen::Vector3d::UnitX();
  if (distance_norm > kDistanceEpsilon) {
    normal = distance_vector / distance_norm;
  }

  const auto frame_a_id = object_a.parentFrame;
  const auto frame_b_id = object_b.parentFrame;
  const auto &frame_a = robot_->model().frames[frame_a_id];
  const auto &frame_b = robot_->model().frames[frame_b_id];

  const auto &transform_a = robot_->data().oMf[frame_a_id];
  const auto &transform_b = robot_->data().oMf[frame_b_id];

  Eigen::Vector3d p1_local = transform_a.rotation().transpose() *
                             (p1_world - transform_a.translation());
  Eigen::Vector3d p2_local = transform_b.rotation().transpose() *
                             (p2_world - transform_b.translation());

  Eigen::Matrix<double, 3, Eigen::Dynamic> jacobian_a =
      robot_->get_point_jacobian(frame_a.name, p1_local);
  Eigen::Matrix<double, 3, Eigen::Dynamic> jacobian_b =
      robot_->get_point_jacobian(frame_b.name, p2_local);

  Eigen::Matrix3d rotation_to_x = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d rotation_from_negative = Eigen::Matrix3d::Identity();
  if (distance_norm > kDistanceEpsilon) {
    rotation_to_x =
        Eigen::Quaterniond::FromTwoVectors(normal, Eigen::Vector3d::UnitX())
            .toRotationMatrix();
    rotation_from_negative =
        Eigen::Quaterniond::FromTwoVectors(-normal, Eigen::Vector3d::UnitX())
            .toRotationMatrix();
  }

  Eigen::RowVectorXd row_a = (rotation_to_x * jacobian_a).row(0);
  Eigen::RowVectorXd row_b = (rotation_from_negative * jacobian_b).row(0);

  CollisionConstraintResult result;
  // Use a single constraint on the *relative* separating velocity:
  //   normalᵀ (v_a - v_b) >= lower_bound
  // Using two separate rows can over-constrain the QP.
  result.jacobian.resize(1, robot_->nv());
  result.jacobian.row(0) = row_a + row_b;

  double dt = std::max(dt_, 1e-6);
  const auto &config = *collision_constraint_;
  const double signed_distance = distance_result.min_distance;

  // Detect a "stuck" condition: if we are non-penetrating but significantly
  // inside min_distance for multiple consecutive cycles AND the previous dq was
  // near-zero, force a stronger recovery push.
  auto compute_stuck_active = [&]() -> bool {
    const std::size_t pair_idx = *best_index_allowed;
    const bool deep_non_penetration =
        (signed_distance >= 0.0) &&
        (signed_distance < (config.min_distance - kCollisionStuckBand));
    const bool dq_small = (last_solution_dq_norm_ < kCollisionStuckDqNormEps);
    if (!(deep_non_penetration && dq_small)) {
      collision_stuck_counter_ = 0;
      collision_stuck_pair_index_.reset();
      collision_stuck_last_distance_ = std::numeric_limits<double>::infinity();
      return false;
    }

    const bool same_pair = collision_stuck_pair_index_.has_value() &&
                           (*collision_stuck_pair_index_ == pair_idx);
    const bool not_improving =
        std::isfinite(collision_stuck_last_distance_) &&
        (signed_distance <= (collision_stuck_last_distance_ + 1e-6));
    if (same_pair && not_improving) {
      collision_stuck_counter_ += 1;
    } else {
      collision_stuck_counter_ = 1;
      collision_stuck_pair_index_ = pair_idx;
    }
    collision_stuck_last_distance_ = signed_distance;
    return collision_stuck_counter_ >= kCollisionStuckCountThreshold;
  };
  const bool stuck_active = compute_stuck_active();
  // Deadband near the boundary to reduce "fighting" jitter:
  // - outside the band (d >= min + band): allow approach but limit it (velocity
  // damper)
  // - inside the band (min - band <= d < min + band): prevent decreasing
  // distance (no approach)
  // - deeper penetration (d < min - band): apply repulsion (capped)
  double lower_bound = 0.0;
  if (signed_distance >= (config.min_distance + kCollisionRepulsionDeadband)) {
    // Outside: limit approach speed (negative lower_bound is allowed).
    lower_bound =
        (config.min_distance + config.tolerance - signed_distance) / dt;
  } else if (signed_distance >= config.min_distance) {
    lower_bound = 0.0;
  } else {
    // Below min_distance: ensure we can recover (avoid getting stuck).
    // - if only slightly inside (within deadband), apply a gentle, scaled push
    // - if in deeper violation (or true penetration), apply stronger push
    // (capped)
    const double desired =
        (config.min_distance + config.tolerance - signed_distance) / dt;

    if (signed_distance >=
            (config.min_distance - kCollisionRepulsionDeadband) &&
        signed_distance >= 0.0) {
      // Slightly inside, not penetrating: avoid oscillations; do not force
      // strong repulsion. However, setting lower_bound=0 here can allow the
      // solver to get "stuck" slightly inside min_distance with dq≈0. Apply a
      // *very* gentle recovery push so we slowly return to the boundary while
      // keeping the system stable.
      const double gentle_scale = kCollisionRecoveryScale * 0.05;
      lower_bound = std::min(kCollisionMaxSeparationSpeedNonPenetration,
                             std::max(0.0, desired * gentle_scale));
    } else {
      // Deeper violation or penetration: recover more assertively, still
      // capped.
      if (signed_distance >= 0.0) {
        // Not penetrating: keep recovery gentle to avoid numerical issues.
        lower_bound =
            std::min(kCollisionMaxSeparationSpeedNonPenetration,
                     std::max(0.0, desired * kCollisionRecoveryScale));
      } else {
        // Penetration: enforce a minimum recovery speed, still capped.
        lower_bound = std::min(kCollisionMaxSeparationSpeed, desired);
        if (lower_bound < kCollisionMinRecoverySpeed) {
          lower_bound = kCollisionMinRecoverySpeed;
        }
      }
    }
  }
  // If we're stuck (deep inside min_distance but non-penetrating), enforce a
  // stronger non-penetration recovery. This remains capped to preserve
  // numerical stability.
  if (stuck_active && signed_distance >= 0.0) {
    const double desired =
        (config.min_distance + config.tolerance - signed_distance) / dt;
    const double strong =
        std::min(kCollisionMaxSeparationSpeedNonPenetration,
                 std::max(kCollisionMinRecoverySpeed,
                          std::max(0.0, desired * kCollisionRecoveryScale)));
    if (strong > lower_bound) {
      lower_bound = strong;
    }
  }
  const double upper_bound =
      (config.upper_distance - config.tolerance + signed_distance) / dt;

  result.lower_bounds = Eigen::VectorXd::Constant(1, lower_bound);
  result.upper_bounds = Eigen::VectorXd::Constant(1, upper_bound);
  result.distance = signed_distance;
  result.object_a = object_a.name;
  result.object_b = object_b.name;
  result.point_a_world = p1_world;
  result.point_b_world = p2_world;

  return result;
#else
  last_collision_debug_.reset();
  return std::nullopt;
#endif
}

std::pair<double, double> KinematicsSolver::calculate_velocity_box_constraint(
    double position_margin_lower, double position_margin_upper,
    double velocity_limit, double acceleration_limit, double dt) const {
  constexpr double kMarginEpsilon = 1e-4;
  const double raw_margin_lower = position_margin_lower;
  const double raw_margin_upper = position_margin_upper;
  const bool outside_lower = raw_margin_lower < -kMarginEpsilon;
  const bool outside_upper = raw_margin_upper < -kMarginEpsilon;

  if (outside_lower && outside_upper) {
    return std::make_pair(-velocity_limit, velocity_limit);
  }

  // Clamp margins to be non-negative for nominal bounds.
  position_margin_lower = std::max(0.0, position_margin_lower);
  position_margin_upper = std::max(0.0, position_margin_upper);

  // Calculate velocity limits based on position margins
  // Computed as: min of position/dt, vel_max, and sqrt(2*accel*margin)
  double vel_from_pos_lower = -position_margin_lower / dt;
  double vel_from_pos_upper = position_margin_upper / dt;

  double vel_from_accel_lower =
      -std::sqrt(2 * acceleration_limit * position_margin_lower);
  double vel_from_accel_upper =
      std::sqrt(2 * acceleration_limit * position_margin_upper);

  // Take most restrictive limits
  double lower_limit =
      std::max({vel_from_pos_lower, -velocity_limit, vel_from_accel_lower});
  double upper_limit =
      std::min({vel_from_pos_upper, velocity_limit, vel_from_accel_upper});

  if (outside_lower) {
    const double violation = -raw_margin_lower;
    const double recovery_min = (limit_recovery_gain_ * violation) / dt;
    const double recovery_lower = std::min(recovery_min, velocity_limit);
    if (recovery_lower > lower_limit) {
      lower_limit = recovery_lower;
    }
  } else if (outside_upper) {
    const double violation = -raw_margin_upper;
    const double recovery_max = -(limit_recovery_gain_ * violation) / dt;
    const double recovery_upper = std::max(recovery_max, -velocity_limit);
    if (recovery_upper < upper_limit) {
      upper_limit = recovery_upper;
    }
  }

  if (lower_limit > upper_limit) {
    const double midpoint = 0.5 * (lower_limit + upper_limit);
    lower_limit = midpoint;
    upper_limit = midpoint;
  }

  return std::make_pair(lower_limit, upper_limit);
}

void KinematicsSolver::sort_tasks_by_priority() {
  std::stable_sort(
      tasks_.begin(), tasks_.end(),
      [](const std::shared_ptr<Task> &a, const std::shared_ptr<Task> &b) {
        return a->getPriority() < b->getPriority();
      });
}

VelocitySolverResult
KinematicsSolver::solve_velocity(const Eigen::VectorXd &current_q,
                                 bool apply_limits) {
  const bool timing = timing_breakdown_enabled_;
  auto get_elapsed_ms = [](auto start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count() /
           1000.0;
  };

  VelocitySolverResult result;
  // Timing fields default to 0.0; only populate when timing is enabled.

  // Use provided configuration or robot's current
  if (current_q.size() > 0) {
    if (current_q.size() != robot_->nq()) {
      result.status = SolverStatus::kInvalidInput;
      return result;
    }
    if (timing) {
      auto t_kin_start = std::chrono::high_resolution_clock::now();
      robot_->update_kinematics(current_q);
      result.pinocchio_kinematics_time_ms = get_elapsed_ms(t_kin_start);
    } else {
      robot_->update_kinematics(current_q);
    }
  } else {
    if (timing) {
      auto t_kin_start = std::chrono::high_resolution_clock::now();
      robot_->update_kinematics(robot_->get_current_configuration());
      result.pinocchio_kinematics_time_ms = get_elapsed_ms(t_kin_start);
    } else {
      robot_->update_kinematics(robot_->get_current_configuration());
    }
  }

  // Sort tasks by priority
  sort_tasks_by_priority();

  // Update all tasks with current robot state
  if (timing) {
    auto t_task_start = std::chrono::high_resolution_clock::now();
    for (auto &task : tasks_) {
      if (task->isActive()) {
        task->update(*robot_);
      }
    }
    result.task_update_time_ms = get_elapsed_ms(t_task_start);
  } else {
    for (auto &task : tasks_) {
      if (task->isActive()) {
        task->update(*robot_);
      }
    }
  }

  // Collect active tasks: group by priority for order-invariant behavior.
  std::vector<Eigen::VectorXd> goals;
  std::vector<Eigen::MatrixXd> jacobians;
  std::unordered_set<int> excluded_union;

  int current_priority = std::numeric_limits<int>::min();
  std::vector<Eigen::VectorXd> group_goals;
  std::vector<Eigen::MatrixXd> group_jacobians;
  group_goals.reserve(tasks_.size());
  group_jacobians.reserve(tasks_.size());

  auto flush_group = [&]() {
    if (group_goals.empty()) {
      return;
    }
    int total_rows = 0;
    for (const auto &g : group_goals) {
      total_rows += static_cast<int>(g.rows());
    }
    Eigen::VectorXd combined_goal(total_rows);
    Eigen::MatrixXd combined_jac(total_rows, robot_->nv());
    combined_goal.setZero();
    combined_jac.setZero();
    int offset = 0;
    for (size_t i = 0; i < group_goals.size(); ++i) {
      const auto &g = group_goals[i];
      const auto &J = group_jacobians[i];
      if (g.rows() > 0) {
        combined_goal.segment(offset, g.rows()) = g;
        combined_jac.block(offset, 0, J.rows(), robot_->nv()) = J;
        offset += static_cast<int>(g.rows());
      }
    }
    goals.push_back(std::move(combined_goal));
    jacobians.push_back(std::move(combined_jac));
    group_goals.clear();
    group_jacobians.clear();
  };

  for (const auto &task : tasks_) {
    if (!task->isActive()) {
      continue;
    }
    for (int idx : task->get_excluded_joint_indices()) {
      if (idx >= 0 && idx < robot_->nv()) {
        excluded_union.insert(idx);
      }
    }
    const int prio = task->getPriority();
    if (group_goals.empty()) {
      current_priority = prio;
    } else if (prio != current_priority) {
      flush_group();
      current_priority = prio;
    }
    group_goals.push_back(task->getVelocity());
    group_jacobians.push_back(task->getJacobian());
  }
  flush_group();

  // If no active tasks, return early
  if (goals.empty()) {
    result.status = SolverStatus::kSuccess;
    result.solution.resize(robot_->nv(), 0.0);
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.limits_applied = false;
    last_solution_dq_norm_ = 0.0;
    return result;
  }

  // Collision constraints are expensive when collision geometry exists.
  std::optional<CollisionConstraintResult> collision_constraint_result =
      std::nullopt;
  if (collision_constraint_.has_value() && collision_constraint_->enabled) {
    if (timing) {
      auto t_collision_start = std::chrono::high_resolution_clock::now();
      collision_constraint_result = compute_collision_constraint();
      result.collision_constraint_time_ms = get_elapsed_ms(t_collision_start);
    } else {
      collision_constraint_result = compute_collision_constraint();
    }
  }
  if (collision_constraint_result.has_value() && !excluded_union.empty()) {
    for (int idx : excluded_union) {
      collision_constraint_result->jacobian.col(idx).setZero();
    }
  }

  // CoM support-polygon constraint
  std::optional<ComConstraintResult> com_constraint_result = std::nullopt;
  if (com_constraint_.has_value() && com_constraint_->enabled) {
    com_constraint_result = compute_com_constraint();
  }
  if (com_constraint_result.has_value() && !excluded_union.empty()) {
    for (int idx : excluded_union)
      com_constraint_result->jacobian.col(idx).setZero();
  }

  // Build constraint matrix
  std::optional<std::chrono::high_resolution_clock::time_point>
      t_constraint_start;
  if (timing) {
    t_constraint_start = std::chrono::high_resolution_clock::now();
  }
  Eigen::MatrixXd C;
  Eigen::VectorXd c_lower, c_upper;

  // For floating-base robots, we need to handle base and joint constraints
  // separately
  int num_constraints = robot_->nv(); // Velocity constraints

  // Add position-based velocity constraints if enabled
  if (apply_limits && use_position_limits_) {
    num_constraints += robot_->nv(); // Position constraints
  }

  if (collision_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(collision_constraint_result->jacobian.rows());
  }

  if (com_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(com_constraint_result->jacobian.rows());
  }

  C = Eigen::MatrixXd::Zero(num_constraints, robot_->nv());
  c_lower = Eigen::VectorXd::Zero(num_constraints);
  c_upper = Eigen::VectorXd::Zero(num_constraints);

  int constraint_idx = 0;

  // Joint velocity constraints
  C.block(constraint_idx, 0, robot_->nv(), robot_->nv()) =
      Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());

  if (apply_limits && use_velocity_limits_) {
    auto vel_limits = robot_->get_velocity_limits();
    c_lower.segment(constraint_idx, robot_->nv()) = -vel_limits;
    c_upper.segment(constraint_idx, robot_->nv()) = vel_limits;
  } else {
    c_lower.segment(constraint_idx, robot_->nv()).setConstant(-1e10);
    c_upper.segment(constraint_idx, robot_->nv()).setConstant(1e10);
  }
  constraint_idx += robot_->nv();

  // Position-based velocity constraints
  if (apply_limits && use_position_limits_) {
    auto [q_min, q_max] = robot_->get_joint_limits();
    Eigen::VectorXd q_current = robot_->get_current_configuration();
    auto vel_limits = robot_->get_velocity_limits();

    // Get acceleration limits from robot model
    Eigen::VectorXd accel_limits = robot_->get_acceleration_limits();

    // Map each velocity DoF index to the corresponding configuration DoF index.
    // This is required when nq != nv (e.g. continuous joints represented as
    // [cos(theta), sin(theta)] in q but 1 DoF in v).
    std::vector<int> velocity_to_config_index(robot_->nv(), -1);
    const auto joint_names = robot_->get_joint_names();
    for (const auto &joint_name : joint_names) {
      const int v_idx = robot_->get_joint_velocity_index(joint_name);
      const int v_size = robot_->get_joint_velocity_size(joint_name);
      const int q_idx = robot_->get_joint_config_index(joint_name);
      const int q_size = robot_->get_joint_config_size(joint_name);

      if (v_size <= 0 || v_idx < 0 || q_idx < 0) {
        continue;
      }

      // 1-DoF joints with non-scalar q representation (e.g. continuous) do not
      // admit simple scalar position bounds in q-space. Keep them unconstrained
      // by position limits here; velocity limits still apply.
      if (v_size == 1 && q_size != 1) {
        continue;
      }

      const int dims = std::min(v_size, q_size);
      for (int k = 0; k < dims; ++k) {
        const int vk = v_idx + k;
        const int qk = q_idx + k;
        if (vk >= 0 && vk < robot_->nv() && qk >= 0 && qk < q_current.size() &&
            qk < q_min.size() && qk < q_max.size()) {
          velocity_to_config_index[vk] = qk;
        }
      }
    }

    // For each joint, compute maximum velocity to stay within position limits
    C.block(constraint_idx, 0, robot_->nv(), robot_->nv()) =
        Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());

    // Consider position, velocity, and
    // acceleration constraints
    // 1 mrad margin to absorb QP tolerance and numerical drift (violations
    // were ~0.4–0.9 mrad with 0.1 mrad margin).
    constexpr double margin_limit = 1e-3;

    if (robot_->is_floating_base()) {
      // Handle floating-base constraints (first 6 DoFs)
      // Get current base pose error if bounds are set
      if (base_position_lower_.has_value() ||
          base_position_upper_.has_value() ||
          base_orientation_lower_.has_value() ||
          base_orientation_upper_.has_value()) {

        // Get current base position
        Eigen::Vector3d base_pos = q_current.head<3>();

        // Position constraints (first 3 DoFs)
        for (int i = 0; i < 3; ++i) {
          if (base_position_lower_.has_value() &&
              base_position_upper_.has_value()) {
            double lower_margin =
                base_pos[i] - base_position_lower_.value()[i] - margin_limit;
            double upper_margin =
                base_position_upper_.value()[i] - base_pos[i] - margin_limit;

            auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
                lower_margin, upper_margin, vel_limits[i], accel_limits[i],
                dt_);

            c_lower(constraint_idx + i) = lower_limit;
            c_upper(constraint_idx + i) = upper_limit;
          } else {
            c_lower(constraint_idx + i) = -1e10;
            c_upper(constraint_idx + i) = 1e10;
          }
        }

        // Orientation constraints (next 3 DoFs)
        for (int i = 3; i < 6; ++i) {
          if (base_orientation_lower_.has_value() &&
              base_orientation_upper_.has_value()) {
            // For orientation, we work directly in velocity space
            c_lower(constraint_idx + i) = std::max(
                base_orientation_lower_.value()[i - 3], -vel_limits[i]);
            c_upper(constraint_idx + i) =
                std::min(base_orientation_upper_.value()[i - 3], vel_limits[i]);
          } else {
            c_lower(constraint_idx + i) = -1e10;
            c_upper(constraint_idx + i) = 1e10;
          }
        }
      } else {
        // No bounds set, use unlimited
        for (int i = 0; i < 6; ++i) {
          c_lower(constraint_idx + i) = -1e10;
          c_upper(constraint_idx + i) = 1e10;
        }
      }

      // Handle joint constraints (remaining DoFs)
      for (int i = 6; i < robot_->nv(); ++i) {
        const int q_idx = velocity_to_config_index[i];
        if (q_idx < 0 || q_idx >= q_current.size() || q_idx >= q_min.size() ||
            q_idx >= q_max.size()) {
          c_lower(constraint_idx + i) = -1e10;
          c_upper(constraint_idx + i) = 1e10;
          continue;
        }

        if (!std::isfinite(q_min[q_idx]) || !std::isfinite(q_max[q_idx])) {
          c_lower(constraint_idx + i) = -1e10;
          c_upper(constraint_idx + i) = 1e10;
          continue;
        }

        // Calculate margins to limits
        double lower_margin = q_current[q_idx] - q_min[q_idx] - margin_limit;
        double upper_margin = q_max[q_idx] - q_current[q_idx] - margin_limit;

        auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
            lower_margin, upper_margin, vel_limits[i], accel_limits[i], dt_);

        c_lower(constraint_idx + i) = lower_limit;
        c_upper(constraint_idx + i) = upper_limit;
      }
    } else {
      // Fixed-base robot: q-v mapping is only direct for joints with nqs == nvs.
      // Continuous joints (nqs=2, nvs=1) require explicit index mapping.
      for (int i = 0; i < robot_->nv(); ++i) {
        const int q_idx = velocity_to_config_index[i];
        if (q_idx < 0 || q_idx >= q_current.size() || q_idx >= q_min.size() ||
            q_idx >= q_max.size()) {
          c_lower(constraint_idx + i) = -1e10;
          c_upper(constraint_idx + i) = 1e10;
          continue;
        }

        if (!std::isfinite(q_min[q_idx]) || !std::isfinite(q_max[q_idx])) {
          c_lower(constraint_idx + i) = -1e10;
          c_upper(constraint_idx + i) = 1e10;
          continue;
        }

        // Calculate margins to limits
        double lower_margin = q_current[q_idx] - q_min[q_idx] - margin_limit;
        double upper_margin = q_max[q_idx] - q_current[q_idx] - margin_limit;

        auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
            lower_margin, upper_margin, vel_limits[i], accel_limits[i], dt_);

        c_lower(constraint_idx + i) = lower_limit;
        c_upper(constraint_idx + i) = upper_limit;
      }
    }
    constraint_idx += robot_->nv();
  }

  if (collision_constraint_result.has_value()) {
    const auto &collision = collision_constraint_result.value();
    int rows = static_cast<int>(collision.jacobian.rows());
    C.block(constraint_idx, 0, rows, robot_->nv()) = collision.jacobian;
    c_lower.segment(constraint_idx, rows) = collision.lower_bounds;
    c_upper.segment(constraint_idx, rows) = collision.upper_bounds;
    constraint_idx += rows;
  }

  if (com_constraint_result.has_value()) {
    const auto &com = com_constraint_result.value();
    int rows = static_cast<int>(com.jacobian.rows());
    C.block(constraint_idx, 0, rows, robot_->nv()) = com.jacobian;
    c_lower.segment(constraint_idx, rows) = com.lower_bounds;
    c_upper.segment(constraint_idx, rows) = com.upper_bounds;
    constraint_idx += rows;
  }

  if (timing && t_constraint_start.has_value()) {
    result.constraint_setup_time_ms = get_elapsed_ms(*t_constraint_start);
  }

  // Configure solver
  VelocitySolverConfig config;
  config.epsilon = solver_tolerance_;
  config.precision_threshold = tight_tolerance_;
  config.iteration_limit = max_iterations_;
  config.magnitude_limit = norm_threshold_;
  config.stall_detection_count = max_zero_scale_iterations_;
  config.regularization_config.epsilon = solver_tolerance_;
  config.regularization_config.regularization_factor = damping_;

  // Call the backend solver
  auto backend_result = computeMultiObjectiveVelocitySolutionEigen(
      goals, jacobians, C, c_lower, c_upper, config);

  // Create velocity-specific result
  result.status = backend_result.status;
  result.solution = backend_result.solution;
  result.computation_time_ms = backend_result.computation_time_ms;
  result.solver_computation_time_ms = backend_result.computation_time_ms;
  result.iterations = backend_result.iterations;
  result.final_error = backend_result.final_error;
  result.task_scales = backend_result.task_scales;
  result.task_errors = backend_result.task_errors;
  result.limits_applied = apply_limits;

  // Convert solution to Eigen vector
  if (!result.solution.empty()) {
    // Clamp velocity solution to constraint bounds to avoid post-integration
    // limit violations from QP tolerance.
    if (apply_limits && c_lower.size() >= robot_->nv()) {
      Eigen::Map<Eigen::VectorXd> dq(result.solution.data(),
                                     result.solution.size());
      for (int i = 0; i < robot_->nv(); ++i) {
        double lower = c_lower[i];
        double upper = c_upper[i];
        if (use_position_limits_ &&
            static_cast<int>(c_lower.size()) >= 2 * robot_->nv()) {
          lower = std::max(lower, c_lower[robot_->nv() + i]);
          upper = std::min(upper, c_upper[robot_->nv() + i]);
        }
        dq[i] = std::clamp(dq[i], lower, upper);
      }
    }

    result.joint_velocities = Eigen::Map<const Eigen::VectorXd>(
        result.solution.data(), result.solution.size());
    last_solution_dq_norm_ = result.joint_velocities.norm();

    // Identify saturated joints based on constraint bounds
    if (apply_limits && c_lower.size() >= robot_->nv()) {
      double tolerance = 0.01; // 1% tolerance

      // Check joint velocities against their constraint bounds directly
      // The first robot_->nv() constraints are typically joint velocity
      // constraints
      for (int i = 0; i < robot_->nv(); ++i) {
        double joint_vel = result.joint_velocities[i];

        // Check if joint velocity is near its constraint bounds
        if (joint_vel <= c_lower[i] + tolerance ||
            joint_vel >= c_upper[i] - tolerance) {
          result.saturated_joints.push_back(i);
        }
      }
    }
  } else {
    last_solution_dq_norm_ = 0.0;
  }

  return result;
}

PositionIKResult KinematicsSolver::solve_position(
    const Eigen::VectorXd &seed_q, const Eigen::Matrix4d &target_pose,
    const std::string &frame_name, const PositionIKOptions &options) {

  PositionIKResult result;
  std::vector<double> position_trace;
  std::vector<double> orientation_trace;
  position_trace.reserve(options.max_iterations);
  orientation_trace.reserve(options.max_iterations);
  double prev_combined_error = std::numeric_limits<double>::infinity();
  int stagnation_iters = 0;

  // Validate input
  if (seed_q.size() != robot_->nq()) {
    result.status = SolverStatus::kInvalidInput;
    return result;
  }

  // Create frame task (using the proper constructor)
  auto frame_task = std::make_shared<FrameTask>(
      "position_ik_task", robot_, frame_name, TaskType::FRAME_POSE);
  if (!options.excluded_joint_indices.empty()) {
    frame_task->set_excluded_joint_indices(options.excluded_joint_indices);
  }

  // Create nullspace task if bias is provided
  std::shared_ptr<PostureTask> posture_task = nullptr;
  if (options.nullspace_bias.has_value()) {
    // Create posture task with specific joints if provided
    if (!options.nullspace_active_joints.empty()) {
      posture_task = std::make_shared<PostureTask>(
          "nullspace_task", robot_, options.nullspace_active_joints);
    } else {
      posture_task = std::make_shared<PostureTask>("nullspace_task", robot_);
    }

    posture_task->setTargetConfiguration(options.nullspace_bias.value());
    posture_task->setWeight(options.nullspace_gain);
    posture_task->setPriority(1); // Lower priority than main task
    if (!options.excluded_joint_indices.empty()) {
      posture_task->set_excluded_joint_indices(options.excluded_joint_indices);
    }
  }

  // Set robot to seed configuration
  Eigen::VectorXd q_current = seed_q;
  robot_->update_configuration(q_current);

  // Extract position and orientation from target pose
  Eigen::Vector3d target_position = target_pose.block<3, 1>(0, 3);
  Eigen::Matrix3d target_rotation = target_pose.block<3, 3>(0, 0);

  // Set task targets
  frame_task->setTargetPosition(target_position);
  frame_task->setTargetOrientation(target_rotation);
  frame_task->setWeight(10.0); // High weight for position IK
  frame_task->setPriority(0);  // Highest priority

  // Iterative solver loop
  int iter = 0;
  bool converged = false;

  while (iter < options.max_iterations && !converged) {
    // Update task with current robot state
    frame_task->update(*robot_);

    // Get current error
    Eigen::VectorXd error = frame_task->getError();
    double pos_error = error.head(3).norm();
    double ori_error = error.tail(3).norm();
    position_trace.push_back(pos_error);
    orientation_trace.push_back(ori_error);

    if (position_ik_debug_) {
      std::cout << "[embodiK][IKDebug] iter " << iter
                << " pos_err=" << pos_error << " ori_err=" << ori_error
                << std::endl;
    }

    // Check convergence
    if (pos_error < options.position_tolerance &&
        ori_error < options.orientation_tolerance) {
      converged = true;
      break;
    }

    double combined_error = pos_error + ori_error;
    if (combined_error < prev_combined_error - options.stagnation_tolerance) {
      prev_combined_error = combined_error;
      stagnation_iters = 0;
    } else {
      stagnation_iters++;
      if (stagnation_iters >= options.stagnation_iterations) {
        if (position_ik_debug_) {
          std::cout << "[embodiK][IKDebug] Stagnation detected after "
                    << stagnation_iters << " iterations; aborting."
                    << std::endl;
        }
        result.status = SolverStatus::kNumericalError;
        break;
      }
    }

    // Prepare tasks for solving
    std::vector<Eigen::VectorXd> goals;
    std::vector<Eigen::MatrixXd> jacobians;

    // Primary task: end-effector position/orientation
    Eigen::VectorXd v_desired = frame_task->getVelocity();

    // Apply step size limits if needed
    if (options.max_linear_step > 0 || options.max_angular_step > 0) {
      double linear_vel = v_desired.head(3).norm();
      double angular_vel = v_desired.tail(3).norm();

      double scale = 1.0;
      if (linear_vel > options.max_linear_step / options.dt) {
        scale = std::min(scale,
                         (options.max_linear_step / options.dt) / linear_vel);
      }
      if (angular_vel > options.max_angular_step / options.dt) {
        scale = std::min(scale,
                         (options.max_angular_step / options.dt) / angular_vel);
      }
      v_desired *= scale;
    }

    goals.push_back(v_desired);
    jacobians.push_back(frame_task->getJacobian());

    // Secondary task: nullspace bias (if provided)
    if (posture_task) {
      posture_task->update(*robot_);
      goals.push_back(posture_task->getVelocity());
      jacobians.push_back(posture_task->getJacobian());
    }

    // Compute collision constraint if enabled
    std::optional<CollisionConstraintResult> collision_constraint_result =
        std::nullopt;
    if (collision_constraint_.has_value() && collision_constraint_->enabled) {
      collision_constraint_result = compute_collision_constraint();
    }
    if (collision_constraint_result.has_value() &&
        !options.excluded_joint_indices.empty()) {
      for (int idx : options.excluded_joint_indices) {
        if (idx >= 0 && idx < collision_constraint_result->jacobian.cols()) {
          collision_constraint_result->jacobian.col(idx).setZero();
        }
      }
    }

    // Build constraint matrices for velocity limits + collision
    int num_constraints = robot_->nv();
    if (collision_constraint_result.has_value()) {
      num_constraints +=
          static_cast<int>(collision_constraint_result->jacobian.rows());
    }

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(num_constraints, robot_->nv());
    Eigen::VectorXd c_lower = Eigen::VectorXd::Constant(num_constraints, -1e10);
    Eigen::VectorXd c_upper = Eigen::VectorXd::Constant(num_constraints, 1e10);

    // Joint velocity constraints (identity block)
    C.block(0, 0, robot_->nv(), robot_->nv()) =
        Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());

    if (use_position_limits_ || use_velocity_limits_) {
      // Apply velocity and position-based limits
      auto vel_limits = robot_->get_velocity_limits();
      auto accel_limits = robot_->get_acceleration_limits();
      auto [q_min, q_max] = robot_->get_joint_limits();

      for (int i = 0; i < robot_->nv(); ++i) {
        double lower_margin =
            q_current[i] - q_min[i] - 0.02; // Use small margin
        double upper_margin = q_max[i] - q_current[i] - 0.02;

        auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
            lower_margin, upper_margin, vel_limits[i], accel_limits[i],
            options.dt);

        c_lower[i] = lower_limit;
        c_upper[i] = upper_limit;
      }
    }

    // Add collision constraint rows if enabled
    if (collision_constraint_result.has_value()) {
      int collision_rows =
          static_cast<int>(collision_constraint_result->jacobian.rows());
      int constraint_idx = robot_->nv();
      C.block(constraint_idx, 0, collision_rows, robot_->nv()) =
          collision_constraint_result->jacobian;
      c_lower.segment(constraint_idx, collision_rows) =
          collision_constraint_result->lower_bounds;
      c_upper.segment(constraint_idx, collision_rows) =
          collision_constraint_result->upper_bounds;
    }

    // Prepare solver configuration
    VelocitySolverConfig config;
    config.epsilon = solver_tolerance_;
    config.precision_threshold = tight_tolerance_;
    config.iteration_limit = max_iterations_;
    config.magnitude_limit = norm_threshold_;
    config.stall_detection_count = max_zero_scale_iterations_;
    config.regularization_config.epsilon = solver_tolerance_;
    config.regularization_config.regularization_factor = damping_;

    // Call the backend solver
    auto vel_result = computeMultiObjectiveVelocitySolutionEigen(
        goals, jacobians, C, c_lower, c_upper, config);

    if (vel_result.status != SolverStatus::kSuccess) {
      result.status = vel_result.status;
      if (position_ik_debug_) {
        std::cout << "[embodiK][IKDebug] velocity solver failure at iter "
                  << iter << " status=" << static_cast<int>(vel_result.status)
                  << std::endl;
      }
      break;
    }

    // Integrate velocities using Lie-group-aware integration
    // (handles quaternion/SO3 joints correctly for floating-base robots)
    Eigen::VectorXd dq = Eigen::Map<const Eigen::VectorXd>(
        vel_result.solution.data(), vel_result.solution.size());
    q_current =
        pinocchio::integrate(robot_->model(), q_current, options.dt * dq);

    // Update robot configuration
    robot_->update_configuration(q_current);

    iter++;
  }

  // Fill result
  result.q_solution = q_current;
  result.achieved_pose = robot_->get_frame_pose(frame_name);

  // Calculate final errors
  frame_task->update(*robot_);
  Eigen::VectorXd final_error = frame_task->getError();
  result.position_error = final_error.head(3).norm();
  result.orientation_error = final_error.tail(3).norm();
  result.iterations_used = iter;
  result.position_error_trace = std::move(position_trace);
  result.orientation_error_trace = std::move(orientation_trace);

  const bool within_tolerance =
      (result.position_error <= options.position_tolerance) &&
      (result.orientation_error <= options.orientation_tolerance);

  if (converged || within_tolerance) {
    result.status = SolverStatus::kSuccess;
  } else if (result.status == SolverStatus::kInvalidInput ||
             result.status == SolverStatus::kSuccess) {
    // If the solver stopped for any other reason (max iterations, stagnation),
    // report numerical error.
    result.status = SolverStatus::kNumericalError;
  }

  if (position_ik_debug_) {
    std::cout << "[embodiK][IKDebug] solve_position finished with status="
              << static_cast<int>(result.status) << " iterations=" << iter
              << " final_pos_err=" << result.position_error
              << " final_ori_err=" << result.orientation_error << std::endl;
  }

  return result;
}

PositionIKResult KinematicsSolver::solve_position_in_tcp(
    const Eigen::VectorXd &seed_q, const Eigen::Matrix4d &relative_target,
    const std::string &frame_name, const PositionIKOptions &options) {

  // Set robot to seed configuration
  robot_->update_configuration(seed_q);

  // Get current TCP pose
  Eigen::Matrix4d current_tcp_pose = robot_->get_frame_pose(frame_name);

  // Calculate target in base frame
  Eigen::Matrix4d target_pose = current_tcp_pose * relative_target;

  // Use regular position IK solver
  return solve_position(seed_q, target_pose, frame_name, options);
}

} // namespace embodik
