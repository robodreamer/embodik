#include <embodik/cartesian_corridor_recovery.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <pinocchio/spatial/explog.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#ifdef PINOCCHIO_WITH_HPP_FCL
#include <pinocchio/collision/distance.hpp>
#endif

namespace embodik {
namespace {
std::string pair_key(const std::string &a, const std::string &b) {
  return a < b ? a + '\0' + b : b + '\0' + a;
}
struct RestoreConfiguration {
  std::shared_ptr<RobotModel> robot;
  Eigen::VectorXd q;
  explicit RestoreConfiguration(std::shared_ptr<RobotModel> model)
      : robot(std::move(model)), q(robot->get_current_configuration()) {}
  ~RestoreConfiguration() { robot->update_configuration(q); }
};
struct SearchGuard {
  std::atomic<unsigned int> &count;
  explicit SearchGuard(std::atomic<unsigned int> &value) : count(value) { ++count; }
  ~SearchGuard() { --count; }
};
struct ClearRecoveryRows {
  std::shared_ptr<KinematicsSolver> solver;
  ~ClearRecoveryRows() { if (solver) solver->clear_linear_velocity_constraints(); }
};
} // namespace

CartesianCorridorRecovery::CartesianCorridorRecovery(
    std::shared_ptr<KinematicsSolver> solver, std::vector<std::string> frames,
    CartesianCorridorOptions options)
    : solver_(std::move(solver)), frames_(std::move(frames)), options_(std::move(options)) {
  if (!solver_) throw std::invalid_argument("A configured native solver is required");
  robot_ = solver_->robot();
  if (options_.guide_mode != TaskSolveMode::kScale && options_.guide_mode != TaskSolveMode::kMinError)
    throw std::invalid_argument("Guide mode must be SCALE or MIN_ERROR");
  if (options_.budget_velocity_constraints && solver_->get_linear_velocity_constraint_rows())
    throw std::invalid_argument("Budget-aware recovery requires exclusive ownership of user velocity rows");
  if (solver_->get_acceleration_limits_enabled())
    throw std::invalid_argument("Acceleration-history control requires a separate recovery adapter");
  if (robot_->is_floating_base() || robot_->nq() != robot_->nv())
    throw std::invalid_argument("Fixed-base scalar configuration required");
  if (solver_->tasks().size() != 1 ||
      !(posture_ = std::dynamic_pointer_cast<PostureTask>(solver_->tasks().front())))
    throw std::invalid_argument("Recovery solver must contain exactly one PostureTask");
  if (frames_.size() != 2 || frames_[0] == frames_[1])
    throw std::invalid_argument("Two distinct tool frames required");
  for (const auto &frame : frames_) (void)robot_->get_frame_pose(frame);
  if (!std::isfinite(options_.dt) || options_.dt <= 0.0 ||
      !std::isfinite(options_.guide_gain) || options_.guide_gain <= 0.0 ||
      options_.pose_caps.rows() != 2 || options_.pose_caps.cols() != 2 ||
      options_.terminal_tolerances.rows() != 2 || options_.terminal_tolerances.cols() != 2 ||
      !options_.pose_caps.allFinite() || !options_.terminal_tolerances.allFinite() ||
      (options_.terminal_tolerances.array() <= 0.0).any() ||
      (options_.pose_caps.array() < options_.terminal_tolerances.array()).any())
    throw std::invalid_argument("Finite positive timestep, guide gain and per-hand pose budgets required");
  if (options_.support_halfplanes.size() &&
      (options_.support_halfplanes.cols() != 2 ||
       options_.support_halfplanes.rows() != options_.support_upper.size() ||
       !options_.support_halfplanes.allFinite() || !options_.support_upper.allFinite() ||
       options_.support_frame.empty()))
    throw std::invalid_argument("Finite root-fixed support halfplanes required");
  if (!options_.support_halfplanes.size() && options_.support_upper.size())
    throw std::invalid_argument("Support bounds require halfplanes");
  if (options_.collision_floors.size() != static_cast<Eigen::Index>(options_.collision_floor_pairs.size()) ||
      !options_.collision_floors.allFinite())
    throw std::invalid_argument("One finite floor per exact geometry pair required");
  for (size_t i = 0; i < options_.collision_floor_pairs.size(); ++i) {
    const auto &p = options_.collision_floor_pairs[i];
    if (p.geometry_a.empty() || p.geometry_b.empty() || p.geometry_a == p.geometry_b ||
        !collision_floors_.emplace(pair_key(p.geometry_a, p.geometry_b), options_.collision_floors(i)).second)
      throw std::invalid_argument("Unique exact collision-floor pairs required");
  }
  std::tie(lower_, upper_) = robot_->get_joint_limits();
  velocity_limits_ = robot_->get_velocity_limits();
  if (!lower_.allFinite() || !upper_.allFinite() || !velocity_limits_.allFinite() ||
      (lower_.array() >= upper_.array()).any() || (velocity_limits_.array() <= 0).any())
    throw std::invalid_argument("Finite ordered scalar joint bounds required");
  solver_->set_dt(options_.dt);
  solver_->enable_position_limits(true);
  solver_->enable_velocity_limits(true);
  auto config = solver_->runtime_config();
  config.weighted_fallback_enabled = false;
  solver_->configure_runtime(config);
  posture_->setAllowMinErrorFallback(false);
}

bool CartesianCorridorRecovery::valid_q(const Eigen::VectorXd &q) const {
  return q.size() == lower_.size() && q.allFinite() &&
         (q.array() >= lower_.array() - 1e-9).all() &&
         (q.array() <= upper_.array() + 1e-9).all();
}

void CartesianCorridorRecovery::validate_targets(const std::vector<Eigen::Matrix4d> &targets) const {
  if (targets.size() != 2) throw std::invalid_argument("Two original Cartesian poses required");
  for (const auto &pose : targets) {
    const Eigen::Matrix3d rotation = pose.topLeftCorner<3, 3>();
    if (!pose.allFinite() || (pose.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() > 1e-9 ||
        (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() > 1e-8 ||
        std::abs(rotation.determinant() - 1.0) > 1e-8)
      throw std::invalid_argument("Finite rigid Cartesian target transforms required");
  }
}

Eigen::MatrixXd CartesianCorridorRecovery::pose_errors(const Eigen::VectorXd &q) {
  RestoreConfiguration restore(robot_);
  robot_->update_configuration(q);
  Eigen::MatrixXd errors(2, 2);
  for (size_t i = 0; i < 2; ++i) {
    const auto pose = robot_->get_frame_pose(frames_[i]);
    errors(i, 0) = (pose.translation() - targets_[i].topRightCorner<3, 1>()).norm();
    const Eigen::Matrix3d relative = targets_[i].topLeftCorner<3, 3>() * pose.rotation().transpose();
    errors(i, 1) = std::acos(std::clamp((relative.trace() - 1.0) * 0.5, -1.0, 1.0));
  }
  return errors;
}

bool CartesianCorridorRecovery::within_caps(const Eigen::MatrixXd &errors) const {
  return errors.allFinite() && (errors.array() <= options_.pose_caps.array() + 1e-12).all();
}

bool CartesianCorridorRecovery::physically_safe(const Eigen::VectorXd &q) {
  RestoreConfiguration restore(robot_);
  robot_->update_configuration(q);
  if (options_.support_halfplanes.size()) {
    const auto frame = robot_->get_frame_pose(options_.support_frame);
    const Eigen::Vector3d com = frame.rotation().transpose() * (robot_->get_com_position() - frame.translation());
    if (!com.allFinite() || (options_.support_halfplanes * com.head<2>() - options_.support_upper).maxCoeff() > 1e-9)
      return false;
  }
  const auto pairs = solver_->evaluate_collision_pair_distances(q);
  if (!pairs) return collision_floors_.empty();
  if (pairs->size() != collision_floors_.size()) return false;
  for (const auto &p : *pairs) {
    const auto floor = collision_floors_.find(pair_key(p.geometry_a, p.geometry_b));
    if (floor == collision_floors_.end() || !std::isfinite(p.distance) || p.distance < floor->second - 1e-9)
      return false;
  }
  return true;
}

void CartesianCorridorRecovery::interrupt(const std::string &reason) {
  phase_ = CartesianRecoveryPhase::kInterrupted;
  reason_ = reason;
  next_backoff_ = 0;
}

CartesianGuideResult CartesianCorridorRecovery::find_guide(
    const std::vector<Eigen::VectorXd> &seeds,
    const std::vector<Eigen::Matrix4d> &targets, unsigned int max_iterations) {
  if (phase_ == CartesianRecoveryPhase::kActive)
    throw std::invalid_argument("Interrupt an active recovery before generating a new guide");
  if (seeds.empty() || max_iterations == 0 || max_iterations > 10000)
    throw std::invalid_argument("Nonempty seeds and an iteration budget in [1,10000] required");
  validate_targets(targets);
  for (const auto &seed : seeds)
    if (!valid_q(seed)) throw std::invalid_argument("Finite physically bounded search seeds required");
  const auto begun = std::chrono::steady_clock::now();
  RestoreConfiguration restore(robot_);
  CartesianGuideResult out;
  const auto search_generation = search_generation_.load();
  SearchGuard running(active_searches_);
  // A disposable projection solver shares model geometry, not live solver
  // policies or tasks. Search steps use trust boxes instead of hardware speed
  // limits; every executed corridor step still uses the original solver.
  std::shared_ptr<KinematicsSolver> projection_solver;
  std::shared_ptr<PostureTask> projection_task;
  if (options_.endpoint_safety_steering) {
    projection_solver = std::make_shared<KinematicsSolver>(robot_);
    projection_solver->set_dt(1.0);
    projection_solver->enable_position_limits(false);
    projection_solver->enable_velocity_limits(false);
    auto config = projection_solver->runtime_config();
    config.weighted_fallback_enabled = false;
    projection_solver->configure_runtime(config);
    projection_task = projection_solver->add_posture_task("endpoint_projection");
    projection_task->setSolveMode(TaskSolveMode::kMinError);
    projection_task->setAllowMinErrorFallback(false);
  }
#ifdef PINOCCHIO_WITH_HPP_FCL
  const auto *geometry = static_cast<const RobotModel &>(*robot_).collision_model();
  std::unique_ptr<pinocchio::GeometryData> query_data;
  std::unordered_map<std::string, size_t> pair_indices;
  if (options_.endpoint_safety_steering && !collision_floors_.empty() && geometry) {
    query_data = std::make_unique<pinocchio::GeometryData>(*geometry);
    for (auto &request : query_data->distanceRequests) {
      request.gjk_tolerance = 1e-12;
      request.enable_nearest_points = true;
    }
    for (size_t i = 0; i < geometry->collisionPairs.size(); ++i) {
      const auto &pair = geometry->collisionPairs[i];
      pair_indices.emplace(pair_key(geometry->geometryObjects[pair.first].name,
                                    geometry->geometryObjects[pair.second].name), i);
    }
  }
#endif
  auto project_direction = [&](const Eigen::VectorXd &point, Eigen::VectorXd &direction) {
    if (!projection_solver) return true;
    const auto observed = solver_->evaluate_collision_pair_distances(point);
    if ((!observed && !collision_floors_.empty()) ||
        (observed && observed->size() != collision_floors_.size())) return false;
    std::vector<CollisionPairDistance> selected;
    if (observed) {
      selected = *observed;
      for (const auto &pair : selected)
        if (!collision_floors_.count(pair_key(pair.geometry_a, pair.geometry_b))) return false;
      std::stable_sort(selected.begin(), selected.end(), [&](const auto &a, const auto &b) {
        return a.distance - collision_floors_.at(pair_key(a.geometry_a, a.geometry_b)) <
               b.distance - collision_floors_.at(pair_key(b.geometry_a, b.geometry_b));
      });
      if (selected.size() > 8) selected.resize(8);
    }
    robot_->update_configuration(point);
    const Eigen::Index n = point.size(), support_rows = options_.support_halfplanes.rows();
    Eigen::MatrixXd rows = Eigen::MatrixXd::Zero(n + selected.size() + support_rows, n);
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(rows.rows(), -1e6);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(rows.rows(), 1e6);
    rows.topRows(n).setIdentity();
    lower.head(n) = (lower_ - point).cwiseMax(-0.25);
    upper.head(n) = (upper_ - point).cwiseMin(0.25);
#ifdef PINOCCHIO_WITH_HPP_FCL
    if (!selected.empty()) {
      if (!query_data) return false;
      pinocchio::computeJointJacobians(robot_->model(), robot_->data(), point);
      pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(), *geometry, *query_data);
      auto point_jacobian = [&](pinocchio::JointIndex joint, const Eigen::Vector3d &position) {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(3, n);
        if (joint == 0) return result;
        Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian(6, n);
        jacobian.setZero();
        pinocchio::getJointJacobian(robot_->model(), robot_->data(), joint,
                                    pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
        const Eigen::Vector3d offset = position - robot_->data().oMi[joint].translation();
        result = jacobian.topRows<3>();
        for (Eigen::Index col = 0; col < n; ++col)
          result.col(col) += jacobian.bottomRows<3>().col(col).cross(offset);
        return result;
      };
      for (size_t i = 0; i < selected.size(); ++i) {
        const auto &record = selected[i];
        const auto key = pair_key(record.geometry_a, record.geometry_b);
        const auto index = pair_indices.find(key);
        if (index == pair_indices.end()) return false;
        pinocchio::computeDistance(*geometry, *query_data, index->second);
        const auto &distance = query_data->distanceResults[index->second];
        const Eigen::Vector3d first = distance.nearest_points[0].cast<double>();
        const Eigen::Vector3d second = distance.nearest_points[1].cast<double>();
        const Eigen::Vector3d delta = second - first;
        if (!(distance.min_distance > 0) || !delta.allFinite() || delta.norm() <= 1e-12 ||
            std::abs(distance.min_distance - record.distance) > 1e-6) return false;
        const auto &pair = geometry->collisionPairs[index->second];
        rows.row(n + i) = delta.normalized().transpose() *
            (point_jacobian(geometry->geometryObjects[pair.second].parentJoint, second) -
             point_jacobian(geometry->geometryObjects[pair.first].parentJoint, first));
        lower(n + i) = 0.8 * (collision_floors_.at(key) + 1e-4 - record.distance);
      }
    }
#else
    if (!selected.empty()) return false;
#endif
    if (support_rows) {
      // The options explicitly describe a root-fixed support frame.
      if (robot_->get_frame_jacobian(options_.support_frame).norm() > 1e-12) return false;
      const auto frame = robot_->get_frame_pose(options_.support_frame);
      const Eigen::Vector3d com = frame.rotation().transpose() * (robot_->get_com_position() - frame.translation());
      const Eigen::MatrixXd jacobian = frame.rotation().transpose() * robot_->get_com_jacobian();
      rows.bottomRows(support_rows) = options_.support_halfplanes * jacobian.topRows(2);
      upper.tail(support_rows) = 0.8 * (options_.support_upper - options_.support_halfplanes * com.head<2>());
    }
    if (!rows.allFinite() || !lower.allFinite() || !upper.allFinite()) return false;
    projection_solver->set_linear_velocity_constraints(rows, lower, upper);
    projection_task->setTargetVelocity(direction);
    const auto result = projection_solver->solve_velocity(point, true, false);
    if (result.status != SolverStatus::kSuccess || !result.joint_velocities.allFinite()) return false;
    direction = result.joint_velocities;
    return (rows * direction - upper).maxCoeff() <= 1e-8 &&
           (lower - rows * direction).maxCoeff() <= 1e-8;
  };
  auto evaluate = [&](const Eigen::VectorXd &q, Eigen::MatrixXd *jacobian) {
    robot_->update_configuration(q);
    Eigen::VectorXd residual(12);
    if (jacobian) jacobian->resize(12, q.size());
    for (size_t i = 0; i < 2; ++i) {
      const auto pose = robot_->get_frame_pose(frames_[i]);
      const Eigen::Matrix3d relative = targets[i].topLeftCorner<3, 3>() * pose.rotation().transpose();
      const double position_weight = 1.0 / options_.terminal_tolerances(i, 0);
      const double orientation_weight = 1.0 / options_.terminal_tolerances(i, 1);
      residual.segment<3>(6 * i) = position_weight * (targets[i].topRightCorner<3, 1>() - pose.translation());
      residual.segment<3>(6 * i + 3) = orientation_weight * pinocchio::log3(relative);
      if (jacobian) {
        const auto physical = robot_->get_frame_jacobian(frames_[i]);
        Eigen::Matrix3d log_derivative;
        pinocchio::Jlog3(relative, log_derivative);
        jacobian->block(6 * i, 0, 3, q.size()) = position_weight * physical.topRows(3);
        jacobian->block(6 * i + 3, 0, 3, q.size()) = orientation_weight * log_derivative * physical.bottomRows(3);
      }
    }
    return residual;
  };
  for (size_t seed_index = 0; seed_index < seeds.size(); ++seed_index) {
    Eigen::VectorXd point = seeds[seed_index];
    double damping = 1e-3;
    for (unsigned int iteration = 0; iteration < max_iterations; ++iteration) {
      if (search_generation_.load() != search_generation) { out.cancelled = true; break; }
      ++out.total_iterations;
      Eigen::MatrixXd jacobian;
      const auto residual = evaluate(point, &jacobian);
      Eigen::MatrixXd errors(2, 2);
      for (int i = 0; i < 2; ++i) {
        errors(i, 0) = residual.segment<3>(6 * i).norm() * options_.terminal_tolerances(i, 0);
        errors(i, 1) = residual.segment<3>(6 * i + 3).norm() * options_.terminal_tolerances(i, 1);
      }
      if ((errors.array() <= options_.terminal_tolerances.array()).all()) {
        // Only endpoint feasibility is certified here; execution still needs
        // the corridor's per-tick physical and Cartesian acceptance gates.
        if (physically_safe(point)) {
          out.found = true; out.q_guide = point; out.pose_errors = errors;
          out.seed_index = seed_index; out.iterations = iteration + 1;
        }
        break;
      }
      const Eigen::VectorXd gradient = jacobian.transpose() * residual;
      for (Eigen::Index col = 0; col < point.size(); ++col) {
        if ((point(col) <= lower_(col) + 1e-10 && gradient(col) < 0.0) ||
            (point(col) >= upper_(col) - 1e-10 && gradient(col) > 0.0))
          jacobian.col(col).setZero();
      }
      Eigen::MatrixXd normal = jacobian.transpose() * jacobian;
      normal.diagonal().array() += damping;
      const Eigen::VectorXd rhs = jacobian.transpose() * residual;
      Eigen::VectorXd direction = normal.ldlt().solve(rhs);
      if (!direction.allFinite() || direction.norm() < 1e-12) break;
      const double largest = direction.lpNorm<Eigen::Infinity>();
      if (largest > 0.25) direction *= 0.25 / largest;
      if (!project_direction(point, direction)) break;
      bool improved = false;
      for (int backoff = 0; backoff < 16; ++backoff) {
        const Eigen::VectorXd candidate = (point + std::pow(0.5, backoff) * direction)
            .cwiseMax(lower_).cwiseMin(upper_);
        const auto trial = evaluate(candidate, nullptr);
        if (trial.allFinite() && trial.squaredNorm() < residual.squaredNorm() - 1e-12) {
          point = candidate; damping = std::max(1e-8, damping * 0.5);
          improved = true; break;
        }
      }
      if (!improved) {
        damping *= 10.0;
        if (damping > 1e10) break;
      }
    }
    if (out.found || out.cancelled) break;
  }
  out.computation_time_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begun).count();
  return out;
}

CartesianGuideResult CartesianCorridorRecovery::find_corridor_guide(
    const Eigen::VectorXd &q, const std::vector<Eigen::VectorXd> &seeds,
    const std::vector<Eigen::Matrix4d> &targets, unsigned int max_iterations,
    unsigned int max_steps) {
  if (phase_ == CartesianRecoveryPhase::kActive)
    throw std::invalid_argument("Interrupt an active recovery before searching a corridor");
  if (!valid_q(q) || max_steps == 0 || max_steps > 10000 || seeds.empty())
    throw std::invalid_argument("Bounded current state, seeds and rollout budget required");
  const auto begun = std::chrono::steady_clock::now();
  RestoreConfiguration restore(robot_);
  CartesianGuideResult out;
  const auto search_generation = search_generation_.load();
  SearchGuard running(active_searches_);
  for (size_t i = 0; i < seeds.size(); ++i) {
    if (search_generation_.load() != search_generation) { out.cancelled = true; break; }
    auto guide = find_guide({seeds[i]}, targets, max_iterations);
    out.total_iterations += guide.total_iterations;
    if (guide.cancelled) { out.cancelled = true; break; }
    if (!guide.found) continue;
    CartesianCorridorRecovery trial(solver_, frames_, options_);
    try {
      trial.start(q, guide.q_guide, targets, true);
    } catch (const std::invalid_argument &) {
      continue;
    }
    Eigen::VectorXd surrogate = q;
    unsigned int accepted = 0;
    for (unsigned int step = 0; step < max_steps; ++step) {
      if (search_generation_.load() != search_generation) { out.cancelled = true; break; }
      auto result = trial.step(surrogate, targets, true);
      if (result.accepted) { surrogate = result.q_next; ++accepted; }
      if (result.phase == CartesianRecoveryPhase::kSettled) {
        const auto total = out.total_iterations;
        out = std::move(guide);
        out.seed_index = i;
        out.total_iterations = total;
        out.corridor_verified = true;
        out.rollout_steps = accepted;
        break;
      }
      if (!result.accepted) break;
    }
    if (out.corridor_verified || out.cancelled) break;
  }
  out.computation_time_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begun).count();
  return out;
}

void CartesianCorridorRecovery::start(const Eigen::VectorXd &q, const Eigen::VectorXd &guide,
                                     const std::vector<Eigen::Matrix4d> &targets, bool both_hands_free) {
  interrupt("activation_invalidated_previous_guide");
  if (!both_hands_free) throw std::invalid_argument("Both hands must be explicitly free");
  validate_targets(targets);
  if (!valid_q(q) || !valid_q(guide)) throw std::invalid_argument("Physical current and guide bounds required");
  targets_ = targets;
  if ((!options_.allow_pose_zone_reentry && !within_caps(pose_errors(q))) || !physically_safe(q))
    throw std::invalid_argument("Initial pose and physical safety budgets violated");
  expected_q_ = q;
  guide_ = guide;
  posture_->setTargetConfiguration(guide);
  phase_ = CartesianRecoveryPhase::kActive;
  reason_ = "guide_activated";
  accepted_steps_ = 0;
}

CartesianCorridorResult CartesianCorridorRecovery::step(
    const Eigen::VectorXd &q, const std::vector<Eigen::Matrix4d> &targets, bool both_hands_free) {
  const auto begun = std::chrono::steady_clock::now();
  CartesianCorridorResult out;
  auto finish = [&]() {
    out.phase = phase_;
    out.reason = reason_;
    out.computation_time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begun).count();
    return out;
  };
  if (phase_ != CartesianRecoveryPhase::kActive) return finish();
  if (solver_->get_acceleration_limits_enabled() || solver_->tasks().size() != 1 ||
      (options_.budget_velocity_constraints && solver_->get_linear_velocity_constraint_rows()) ||
      solver_->tasks().front() != posture_ || solver_->runtime_config().weighted_fallback_enabled) {
    interrupt("solver_policy_changed"); return finish();
  }
  if (!both_hands_free) { interrupt("hand_protection_changed"); return finish(); }
  if (targets.size() != targets_.size()) { interrupt("target_changed"); return finish(); }
  for (size_t i = 0; i < targets.size(); ++i)
    if (!(targets[i].array() == targets_[i].array()).all()) { interrupt("target_changed"); return finish(); }
  if (!valid_q(q) || (!options_.allow_measured_state_updates &&
                     (q - expected_q_).lpNorm<Eigen::Infinity>() > 1e-12)) {
    interrupt("input_state_changed"); return finish();
  }
  out.pose_errors = pose_errors(q);
  // For the fixed robot geometry and immutable endpoint envelope, the exact
  // expected state was certified at activation or the preceding accepted step.
  // Recheck even a tiny caller-side perturbation; certify every new candidate.
  const bool reuse_accepted_state = (q.array() == expected_q_.array()).all();
  if ((!options_.allow_pose_zone_reentry && !within_caps(out.pose_errors)) ||
      (!reuse_accepted_state && !physically_safe(q))) {
    interrupt("current_safety_budget_violated"); return finish();
  }
  if ((out.pose_errors.array() <= options_.terminal_tolerances.array()).all()) {
    phase_ = CartesianRecoveryPhase::kSettled; reason_ = "original_targets_reached"; return finish();
  }
  Eigen::VectorXd difference = guide_ - q;
  const Eigen::MatrixXd step_caps = options_.allow_pose_zone_reentry
      ? options_.pose_caps.cwiseMax(out.pose_errors).eval() : options_.pose_caps;
  ClearRecoveryRows clear_rows;
  if (options_.budget_velocity_constraints) {
    RestoreConfiguration restore(robot_);
    robot_->update_configuration(q);
    Eigen::MatrixXd rows(4, q.size());
    Eigen::VectorXd bounds(4);
    Eigen::Index count = 0;
    for (size_t i = 0; i < 2; ++i) {
      const auto pose = robot_->get_frame_pose(frames_[i]);
      const auto jacobian = robot_->get_frame_jacobian(frames_[i]);
      const Eigen::Vector3d position_error = pose.translation() - targets_[i].topRightCorner<3, 1>();
      const Eigen::Matrix3d relative = targets_[i].topLeftCorner<3, 3>() * pose.rotation().transpose();
      const Eigen::Vector3d rotation_error = pinocchio::log3(relative);
      Eigen::Matrix3d log_derivative;
      pinocchio::Jlog3(relative, log_derivative);
      for (int component = 0; component < 2; ++component) {
        const double error = component == 0 ? position_error.norm() : rotation_error.norm();
        if (error < 1e-8) continue;
        if (component == 0)
          rows.row(count) = position_error.transpose() * jacobian.topRows(3) / error;
        else
          rows.row(count) = -rotation_error.transpose() * log_derivative * jacobian.bottomRows(3) / error;
        // Reserve a small inward buffer for curvature at a zone boundary.
        // These linearized rows steer proposals; exact endpoint gates remain
        // authoritative and may back off/reject nonlinear violations.
        bounds(count) = 0.8 * (0.999 * step_caps(i, component) - error) / options_.dt;
        ++count;
      }
    }
    clear_rows.solver = solver_;
    const Eigen::VectorXd lower_bounds = -(rows.topRows(count).cwiseAbs() * velocity_limits_);
    solver_->set_linear_velocity_constraints(rows.topRows(count),
        lower_bounds, bounds.head(count));
  }
  for (Eigen::Index i = 0; i < difference.size(); ++i)
    if (std::abs(difference(i)) <= 1e-4) difference(i) = 0.0;
  const auto alternate_mode = options_.guide_mode == TaskSolveMode::kScale
      ? TaskSolveMode::kMinError : TaskSolveMode::kScale;
  for (auto mode : {options_.guide_mode, alternate_mode}) {
    posture_->setSolveMode(mode);
    const int first_backoff = options_.warm_start_guide_backoff ? next_backoff_ : 0;
    for (int trial = 0; trial < 8; ++trial) {
      // Visit every original gain once. If the remembered smaller gains fail,
      // larger gains are still tried before declaring this mode blocked.
      const int backoff = (first_backoff + trial) % 8;
      posture_->setTargetVelocity((options_.guide_gain / std::pow(2.0, backoff)) * difference);
      auto result = solver_->solve_velocity(q, true, false);
      ++out.attempts;
      const Eigen::VectorXd velocity = Eigen::Map<const Eigen::VectorXd>(result.solution.data(), result.solution.size());
      if (result.status != SolverStatus::kSuccess || velocity.size() != q.size() ||
          !velocity.allFinite() || (velocity.array().abs() > velocity_limits_.array() + 1e-8).any()) continue;
      const Eigen::VectorXd candidate = q + options_.dt * velocity;
      if (!valid_q(candidate)) continue;
      const auto errors = pose_errors(candidate);
      if (!errors.allFinite() || (errors.array() > step_caps.array() + 1e-12).any() || !physically_safe(candidate) ||
          (guide_ - candidate).norm() >= (guide_ - q).norm() - 1e-10) continue;
      out.accepted = true; out.q_next = candidate; out.velocity = velocity;
      out.accepted_backoff = backoff;
      next_backoff_ = std::max(0, backoff - 1);
      out.mode_requested = mode;
      out.pose_errors = errors; out.solver_result = std::move(result);
      expected_q_ = candidate; ++accepted_steps_;
      reason_ = within_caps(out.pose_errors) ? "guide_activated" : "pose_zone_reentry";
      if ((errors.array() <= options_.terminal_tolerances.array()).all()) {
        phase_ = CartesianRecoveryPhase::kSettled; reason_ = "original_targets_reached";
      }
      return finish();
    }
  }
  phase_ = CartesianRecoveryPhase::kBlocked; reason_ = "no_admissible_corridor_step";
  return finish();
}

} // namespace embodik
