/**
 * @file kinematics_solver.cpp
 * @brief Implementation of high-level kinematics solver
 */

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <pinocchio/algorithm/geometry.hpp>
#include <stdexcept>
#ifdef PINOCCHIO_WITH_HPP_FCL
#include <pinocchio/collision/distance.hpp>
#endif
#include <embodik/ik_baseline.hpp>
#include <embodik/kinematics_solver.hpp>

namespace embodik {

namespace {
constexpr double kCollisionTolerance = 1e-4;
constexpr double kCollisionUpperDistance = 1e1;
constexpr double kDistanceEpsilon = 1e-9;
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
    const std::vector<std::pair<std::string, std::string>> &exclude_pairs) {

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
  config.include_pairs.clear();
  config.exclude_pairs.clear();

  for (const auto &pair : include_pairs) {
    config.include_pairs.insert(canonical_pair_key(pair.first, pair.second));
  }

  for (const auto &pair : exclude_pairs) {
    config.exclude_pairs.insert(canonical_pair_key(pair.first, pair.second));
  }

  if (collision_data != nullptr && collision_model_ptr != nullptr) {
    if (collision_data->distanceRequests.size() !=
        collision_model_ptr->collisionPairs.size()) {
      collision_data->distanceRequests.resize(
          collision_model_ptr->collisionPairs.size());
      collision_data->distanceResults.resize(
          collision_model_ptr->collisionPairs.size());
    }
    for (auto &request : collision_data->distanceRequests) {
      request.enable_nearest_points = true;
      request.enable_signed_distance = true;
    }
    collision_data->activateAllCollisionPairs();
    for (std::size_t idx = 0; idx < collision_model_ptr->collisionPairs.size();
         ++idx) {
      const auto &pair = collision_model_ptr->collisionPairs[idx];
      const auto &name_a =
          collision_model_ptr->geometryObjects[pair.first].name;
      const auto &name_b =
          collision_model_ptr->geometryObjects[pair.second].name;
      if (!collision_pair_allowed(name_a, name_b)) {
        collision_data->activeCollisionPairs[idx] = false;
      }
    }
  }
#else
  (void)min_distance;
  (void)include_pairs;
  (void)exclude_pairs;
  throw std::runtime_error("Collision avoidance requires Pinocchio to be built "
                           "with hpp-fcl support.");
#endif
}

void KinematicsSolver::add_collision_constraint(
    const std::vector<std::pair<std::string, std::string>> &link_pairs,
    double min_distance) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  configure_collision_constraint(min_distance, link_pairs, {});
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

  for (const auto &pair : collision_model_ptr->collisionPairs) {
    const auto &name_a = collision_model_ptr->geometryObjects[pair.first].name;
    const auto &name_b = collision_model_ptr->geometryObjects[pair.second].name;
    if (collision_pair_allowed(name_a, name_b)) {
      result.emplace_back(name_a, name_b);
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

  // Update collision placements and compute distances for all pairs
  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);
  const auto &pairs = collision_model->collisionPairs;

  double best_distance_allowed = std::numeric_limits<double>::infinity();
  std::optional<std::size_t> best_index_allowed;

  double best_distance_debug = std::numeric_limits<double>::infinity();
  std::optional<std::size_t> best_index_debug;

  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    pinocchio::computeDistance(*collision_model, *collision_data, idx);

    const auto &pair = pairs[idx];
    const auto &object_a = collision_model->geometryObjects[pair.first];
    const auto &object_b = collision_model->geometryObjects[pair.second];
    const auto &distance_result = collision_data->distanceResults[idx];

    double distance = distance_result.min_distance;
    if (!std::isfinite(distance)) {
      continue;
    }

    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[idx]) {
      continue;
    }

    if (distance < best_distance_debug) {
      best_distance_debug = distance;
      best_index_debug = idx;
    }

    if (!constraint_active) {
      continue;
    }

    if (distance < best_distance_allowed) {
      best_distance_allowed = distance;
      best_index_allowed = idx;
    }
  }

  if (best_index_debug.has_value()) {
    const auto &pair_debug = pairs[*best_index_debug];
    const auto &obj_da = collision_model->geometryObjects[pair_debug.first];
    const auto &obj_db = collision_model->geometryObjects[pair_debug.second];
    const auto &res_debug = collision_data->distanceResults[*best_index_debug];

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
  result.jacobian.resize(2, robot_->nv());
  result.jacobian.row(0) = row_a;
  result.jacobian.row(1) = row_b;

  double dt = std::max(dt_, 1e-6);
  const auto &config = *collision_constraint_;
  double lower_bound =
      (config.min_distance + config.tolerance - distance_norm) / dt;
  double upper_bound =
      (config.upper_distance - config.tolerance + distance_norm) / dt;

  result.lower_bounds =
      Eigen::VectorXd::Constant(2, std::min(lower_bound, 0.0));
  result.upper_bounds =
      Eigen::VectorXd::Constant(2, std::max(upper_bound, 0.0));
  result.distance = distance_norm;
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

  // Clamp margins to be non-negative
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

  // Use provided configuration or robot's current
  if (current_q.size() > 0) {
    if (current_q.size() != robot_->nq()) {
      VelocitySolverResult result;
      result.status = SolverStatus::kInvalidInput;
      return result;
    }
    robot_->update_kinematics(current_q);
  } else {
    robot_->update_kinematics(robot_->get_current_configuration());
  }

  // Sort tasks by priority
  sort_tasks_by_priority();

  // Update all tasks with current robot state
  for (auto &task : tasks_) {
    if (task->isActive()) {
      task->update(*robot_);
    }
  }

  // Collect active tasks
  std::vector<Eigen::VectorXd> goals;
  std::vector<Eigen::MatrixXd> jacobians;

  for (const auto &task : tasks_) {
    if (!task->isActive())
      continue;

    goals.push_back(task->getVelocity());
    jacobians.push_back(task->getJacobian());
  }

  // If no active tasks, return early
  if (goals.empty()) {
    VelocitySolverResult result;
    result.status = SolverStatus::kSuccess;
    result.solution.resize(robot_->nv(), 0.0);
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.limits_applied = false;
    return result;
  }

  auto collision_constraint_result = compute_collision_constraint();

  // Build constraint matrix
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

    // For each joint, compute maximum velocity to stay within position limits
    C.block(constraint_idx, 0, robot_->nv(), robot_->nv()) =
        Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());

    // Consider position, velocity, and
    // acceleration constraints
    double margin_limit = 1e-4; // Small margin from exact limits

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
        int q_idx = i + 1; // Account for quaternion (q has 7 for base, v has 6)

        // Calculate margins to limits
        double lower_margin = q_current[q_idx] - q_min[q_idx] - margin_limit;
        double upper_margin = q_max[q_idx] - q_current[q_idx] - margin_limit;

        auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
            lower_margin, upper_margin, vel_limits[i], accel_limits[i], dt_);

        c_lower(constraint_idx + i) = lower_limit;
        c_upper(constraint_idx + i) = upper_limit;
      }
    } else {
      // Fixed-base robot: direct mapping between q and v indices
      for (int i = 0; i < robot_->nv(); ++i) {
        // Calculate margins to limits
        double lower_margin = q_current[i] - q_min[i] - margin_limit;
        double upper_margin = q_max[i] - q_current[i] - margin_limit;

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
  VelocitySolverResult result;
  result.status = backend_result.status;
  result.solution = backend_result.solution;
  result.computation_time_ms = backend_result.computation_time_ms;
  result.iterations = backend_result.iterations;
  result.final_error = backend_result.final_error;
  result.task_scales = backend_result.task_scales;
  result.task_errors = backend_result.task_errors;
  result.limits_applied = apply_limits;

  // Convert solution to Eigen vector
  if (!result.solution.empty()) {
    result.joint_velocities = Eigen::Map<const Eigen::VectorXd>(
        result.solution.data(), result.solution.size());

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

    // Build constraint matrices for velocity limits
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());
    Eigen::VectorXd c_lower = Eigen::VectorXd::Constant(robot_->nv(), -1e10);
    Eigen::VectorXd c_upper = Eigen::VectorXd::Constant(robot_->nv(), 1e10);

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

    // Integrate velocities
    Eigen::VectorXd dq = Eigen::Map<const Eigen::VectorXd>(
        vel_result.solution.data(), vel_result.solution.size());
    q_current += options.dt * dq;

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
