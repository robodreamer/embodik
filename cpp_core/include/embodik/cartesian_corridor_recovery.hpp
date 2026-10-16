#pragma once

#include <embodik/kinematics_solver.hpp>
#include <memory>
#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

namespace embodik {

enum class CartesianRecoveryPhase { kIdle, kActive, kSettled, kInterrupted, kBlocked };

/// Explicit endpoint envelope for a fixed-base, both-hands-free velocity trial.
struct CartesianCorridorOptions {
  double dt = 0.02;
  double guide_gain = 3.0;
  TaskSolveMode guide_mode = TaskSolveMode::kScale;
  bool budget_velocity_constraints = false;
  /// Steer hypothetical Newton steps with collision/support linearizations.
  /// Final exact endpoint and execution checks remain authoritative.
  bool endpoint_safety_steering = false;
  /// Revalidate measured state changes instead of interrupting on tracking error.
  bool allow_measured_state_updates = false;
  /// Already-outside free-hand errors may decrease back into original zones.
  bool allow_pose_zone_reentry = false;
  /// Try one gain level above the previous accepted backoff first, retaining
  /// all original candidates and exact acceptance checks. Disabled by default.
  bool warm_start_guide_backoff = false;
  Eigen::MatrixXd pose_caps;
  Eigen::MatrixXd terminal_tolerances;
  Eigen::MatrixXd support_halfplanes;
  Eigen::VectorXd support_upper;
  std::string support_frame;
  std::vector<CollisionGeometryPair> collision_floor_pairs;
  Eigen::VectorXd collision_floors;
};

struct CartesianCorridorResult {
  bool accepted = false;
  CartesianRecoveryPhase phase = CartesianRecoveryPhase::kIdle;
  std::string reason;
  Eigen::VectorXd q_next;
  Eigen::VectorXd velocity;
  Eigen::MatrixXd pose_errors;
  TaskSolveMode mode_requested = TaskSolveMode::kScale;
  VelocitySolverResult solver_result;
  double computation_time_ms = 0.0;
  unsigned int attempts = 0;
  int accepted_backoff = -1;
};

/// A hypothetical endpoint, never an executable configuration command.
struct CartesianGuideResult {
  bool found = false;
  Eigen::VectorXd q_guide;
  Eigen::MatrixXd pose_errors;
  unsigned int seed_index = 0;
  unsigned int iterations = 0;
  unsigned int total_iterations = 0;
  bool corridor_verified = false;
  unsigned int rollout_steps = 0;
  bool cancelled = false;
  double computation_time_ms = 0.0;
};

/// Native lifecycle, velocity stepping and endpoint validation for a guide.
/// The supplied solver must contain only one PostureTask, with its physical
/// constraints already configured. Guide search uses hypothetical states;
/// acceleration/contact control and swept-collision certification are unsupported.
class CartesianCorridorRecovery {
public:
  CartesianCorridorRecovery(std::shared_ptr<KinematicsSolver> solver,
                           std::vector<std::string> frames,
                           CartesianCorridorOptions options);
  void start(const Eigen::VectorXd &q, const Eigen::VectorXd &guide,
             const std::vector<Eigen::Matrix4d> &targets, bool both_hands_free);
  CartesianGuideResult find_guide(const std::vector<Eigen::VectorXd> &seeds,
                                 const std::vector<Eigen::Matrix4d> &targets,
                                 unsigned int max_iterations = 300);
  CartesianGuideResult find_corridor_guide(const Eigen::VectorXd &q,
      const std::vector<Eigen::VectorXd> &seeds,
      const std::vector<Eigen::Matrix4d> &targets,
      unsigned int max_iterations = 300, unsigned int max_steps = 600);
  CartesianCorridorResult step(const Eigen::VectorXd &q,
                              const std::vector<Eigen::Matrix4d> &targets,
                              bool both_hands_free);
  void interrupt(const std::string &reason);
  void cancel_search() { search_generation_.fetch_add(1); }
  bool search_running() const { return active_searches_.load() != 0; }
  CartesianRecoveryPhase phase() const { return phase_; }
  const std::string &reason() const { return reason_; }
  unsigned int accepted_steps() const { return accepted_steps_; }

private:
  bool valid_q(const Eigen::VectorXd &q) const;
  void validate_targets(const std::vector<Eigen::Matrix4d> &targets) const;
  Eigen::MatrixXd pose_errors(const Eigen::VectorXd &q);
  bool within_caps(const Eigen::MatrixXd &errors) const;
  bool physically_safe(const Eigen::VectorXd &q);
  std::shared_ptr<KinematicsSolver> solver_;
  std::shared_ptr<RobotModel> robot_;
  std::shared_ptr<PostureTask> posture_;
  std::vector<std::string> frames_;
  CartesianCorridorOptions options_;
  Eigen::VectorXd lower_, upper_, velocity_limits_, expected_q_, guide_;
  std::vector<Eigen::Matrix4d> targets_;
  std::unordered_map<std::string, double> collision_floors_;
  CartesianRecoveryPhase phase_ = CartesianRecoveryPhase::kIdle;
  std::string reason_ = "not_started";
  unsigned int accepted_steps_ = 0;
  int next_backoff_ = 0;
  std::atomic<unsigned long long> search_generation_{0};
  std::atomic<unsigned int> active_searches_{0};
};

} // namespace embodik
