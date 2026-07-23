/**
 * @file acceleration_solver.hpp
 * @brief Fixed-base acceleration-level solver API.
 */

#pragma once

#include <embodik/robot_model.hpp>
#include <embodik/tasks.hpp>
#include <embodik/types.hpp>

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace embodik {

class KinematicsSolver;

/**
 * @brief Named caller-owned physical affine acceleration constraint.
 *
 * Each row enforces:
 *   lower_bounds <= coefficient_matrix * ddq + affine_bias <= upper_bounds
 *
 * Empty active-side vectors mean every side is active. Otherwise each vector
 * must have one flag per row, and every row must keep at least one active side.
 * The record is per-call only; the solver never stores it across solve() calls.
 */
struct AffineAccelerationConstraint {
  std::string source_id;
  Eigen::MatrixXd coefficient_matrix;
  Eigen::VectorXd affine_bias;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
  std::vector<bool> lower_bound_active;
  std::vector<bool> upper_bound_active;
};

/**
 * @brief Named caller-owned frozen next-velocity affine constraint.
 *
 * Each row enforces:
 *   lower_bounds <= C * dq + dt * (C * ddq + affine_bias) <= upper_bounds
 *
 * The coefficient matrix is frozen for this single solve call. The solver does
 * not infer or apply any derivative of C.
 */
struct FrozenNextVelocityConstraint {
  std::string source_id;
  Eigen::MatrixXd coefficient_matrix;
  Eigen::VectorXd affine_bias;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
  std::vector<bool> lower_bound_active;
  std::vector<bool> upper_bound_active;
};

/**
 * @brief Named caller-owned physical task acceleration bounds.
 *
 * The record names one active registered task and enforces the physical rows:
 *   lower_bounds <= J_physical * ddq + Jdot_dq <= upper_bounds
 *
 * Empty active-side vectors mean every side is active. Otherwise each vector
 * must have one flag per task row, and every row must keep at least one active
 * side. Task-local excluded joint columns are ignored for these hard physical
 * rows; exclusions still apply to the soft task objective.
 */
struct TaskAccelerationBounds {
  std::string source_id;
  std::string task_name;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
  std::vector<bool> lower_bound_active;
  std::vector<bool> upper_bound_active;
};

/**
 * @brief Optional diagonal metric/reference for generalized acceleration
 * allocation.
 *
 * A positive diagonal metric H and physical reference r reparameterize the
 * backend solve as ddq = r + S*z, with S proportional to H^-1/2.
 */
struct GeneralizedAccelerationAllocation {
  Eigen::VectorXd metric_diagonal;
  Eigen::VectorXd reference_acceleration;
};

/**
 * @brief Optional fixed-base inverse-dynamics effort envelope.
 *
 * When present, the solver enforces:
 *   -effective_limits <= M(q) * ddq + h(q, dq) <= effective_limits
 *
 * Empty limits_override means the RobotModel effort metadata is used.
 */
struct EffortConstraintOptions {
  std::optional<Eigen::VectorXd> limits_override;
  double margin_fraction = 0.0;
};

/**
 * @brief Per-call fixed-base contact acceleration equality.
 *
 * Point contacts enforce zero translational acceleration of the named frame:
 *   J_linear(q) * ddq + Jdot_linear(q, dq) * dq = 0
 *
 * Rigid contacts enforce the full LOCAL_WORLD_ALIGNED 6D frame acceleration.
 * Dynamic contact forces are intentionally outside this API surface.
 */
struct ContactAccelerationConstraint {
  std::string source_id;
  std::string frame_name;
  ContactType type = ContactType::kRigidContact;
};

/**
 * @brief Per-axis derivative policy for acceleration-level geometric rows.
 *
 * Braking accelerations are positive caller-declared physical envelope
 * magnitudes used for one-tick endpoint viability. They may be asymmetric, but
 * must fit inside the symmetric acceleration authority for the same axis.
 */
struct GeometricConstraintAccelerationPolicy {
  Eigen::VectorXd rate_limits = Eigen::VectorXd::Ones(3);
  Eigen::VectorXd acceleration_limits = Eigen::VectorXd::Ones(3);
  Eigen::VectorXd lower_braking_accelerations = Eigen::VectorXd::Ones(3);
  Eigen::VectorXd upper_braking_accelerations = Eigen::VectorXd::Ones(3);
};

enum class ComSupportPolygonOutsidePolicy {
  kReject,
  kRecoverNonWorsening,
};

/**
 * @brief Derivative policy for acceleration-level CoM support polygons.
 *
 * Bounds apply along each canonical support-polygon half-plane in physical
 * outward-positive coordinates. Defaults mirror the velocity CoM prototype:
 * 0.4 m/s rate, 0.1 m/s^2 acceleration/braking, 0.1 mm boundary dead-zone.
 */
struct ComSupportPolygonAccelerationPolicy {
  double rate_limit = 0.4;
  double acceleration_limit = 0.1;
  double braking_acceleration = 0.1;
  double boundary_epsilon = 1e-4;
  double outside_recovery_scale = 0.2;
  double outside_min_recovery_speed = 0.01;
  ComSupportPolygonOutsidePolicy outside_policy =
      ComSupportPolygonOutsidePolicy::kRecoverNonWorsening;
};

/**
 * @brief Named caller-owned acceleration-level CoM support-polygon constraint.
 *
 * The support polygon uses the shared definition and is only supported in
 * world or structurally root-fixed frames. Moving support frames are rejected
 * explicitly; this constraint is kinematic and does not claim dynamic balance
 * or contact-force feasibility.
 */
struct ComSupportPolygonAccelerationConstraint {
  std::string source_id;
  ComSupportPolygonConstraintDefinition definition;
  ComSupportPolygonAccelerationPolicy policy;
};

/**
 * @brief Named caller-owned acceleration-level tight point constraint.
 *
 * The physical coordinate is the named frame origin in world coordinates minus
 * definition.target_point. Active translation axes enforce bounded state,
 * rate, physical acceleration, continuous-path, and endpoint braking viability.
 */
struct TightPointAccelerationConstraint {
  std::string source_id;
  TightPointConstraintDefinition definition;
  GeometricConstraintAccelerationPolicy policy;
};

/**
 * @brief Named caller-owned acceleration-level tight frame pose constraint.
 *
 * The physical coordinate is:
 *   [world translation - target translation,
 *    log3(target_R^T * frame_R)]
 *
 * Active axes enforce symmetric position/orientation boxes, rate, physical
 * acceleration, continuous-path, and endpoint braking viability. Translation
 * only masks do not evaluate the SO(3) chart.
 */
struct TightFramePoseAccelerationConstraint {
  std::string source_id;
  TightFramePoseConstraintDefinition definition;
  GeometricConstraintAccelerationPolicy policy;
};

/**
 * @brief Named caller-owned two-moving-frame relative pose acceleration bounds.
 *
 * The physical coordinate is exactly:
 *   [translation(T_a^-1 * T_b) expressed in frame A,
 *    log3(R_a^T * R_b)]
 *
 * lower_bounds and upper_bounds are explicit bounds around this physical
 * relative pose. The acceleration solver never captures, recenters, or manages
 * any relative reference lifecycle. Translation-only masks do not evaluate the
 * SO(3) chart.
 */
struct RelativePoseAccelerationConstraint {
  std::string source_id;
  RelativePoseConstraintDefinition definition;
  GeometricConstraintAccelerationPolicy policy;
};

/**
 * @brief Named caller-owned torso pose acceleration bounds.
 *
 * reference_pose is required and is never captured or recentered by the
 * acceleration solver. Coordinates are:
 *   [world translation - reference translation,
 *    log3(reference_R^T * frame_R)]
 */
struct TorsoPoseBoundAccelerationConstraint {
  std::string source_id;
  TorsoPoseBoundDefinition definition;
  GeometricConstraintAccelerationPolicy policy;
};

struct AccelerationSolveOptions {
  std::optional<Eigen::VectorXd> acceleration_limits_override;
  bool apply_velocity_limits = true;
  bool apply_position_limits = true;
  std::optional<GeneralizedAccelerationAllocation>
      generalized_acceleration_allocation;
  std::optional<EffortConstraintOptions> effort_constraints;
  std::vector<AffineAccelerationConstraint> affine_constraints;
  std::vector<FrozenNextVelocityConstraint> frozen_next_velocity_constraints;
  std::vector<TaskAccelerationBounds> task_acceleration_bounds;
  std::vector<ContactAccelerationConstraint> contact_acceleration_constraints;
  std::vector<TightPointAccelerationConstraint> tight_point_constraints;
  std::vector<TightFramePoseAccelerationConstraint>
      tight_frame_pose_constraints;
  std::vector<RelativePoseAccelerationConstraint> relative_pose_constraints;
  std::vector<TorsoPoseBoundAccelerationConstraint>
      torso_pose_bound_constraints;
  std::vector<ComSupportPolygonAccelerationConstraint>
      com_support_polygon_constraints;
  std::vector<int> zero_acceleration_joint_indices;
  std::vector<int> zero_next_velocity_joint_indices;
  std::vector<int> fixed_current_position_joint_indices;
};

/**
 * @brief Sampling policy for velocity-row collision compatibility.
 *
 * One substep validates only the integrated endpoint and is the low-overhead
 * default. Values greater than one also validate uniformly spaced interior
 * samples. No setting provides a continuous swept-path certificate;
 * `collision_step_certified` remains false for this adapter.
 */
struct VelocityCollisionLiftOptions {
  int validation_substeps = 1;
};

enum class CollisionConstraintOutsidePolicy {
  kReject,
  kRecoverNonWorsening,
};

/**
 * @brief One-sided acceleration policy for native signed-distance rows.
 *
 * Positive coordinate rate and acceleration increase sphere separation.
 * Native continuous certification is initially limited to exact-name
 * sphere-sphere catalogs on fixed-base all-prismatic scalar models.
 */
struct CollisionConstraintAccelerationPolicy {
  bool proximity_activation_enabled = true;
  double activation_margin = 0.05;
  double maximum_approach_rate = 0.4;
  double maximum_inward_acceleration = 0.1;
  double minimum_braking_acceleration = 0.1;
  CollisionConstraintOutsidePolicy outside_policy =
      CollisionConstraintOutsidePolicy::kRecoverNonWorsening;
  double recovery_scale = 0.2;
  double minimum_recovery_rate = 0.01;
  double maximum_recovery_rate = 0.15;
};

enum class AccelerationCollisionRegime {
  kUnknown,
  kStrictInterior,
  kExactFloor,
  kBelowLimitRecovery,
};

struct AccelerationTaskReference {
  Eigen::VectorXd desired_velocity;
  Eigen::VectorXd desired_acceleration;
  double proportional_gain = 1.0;
  double derivative_gain = 1.0;
};

struct AccelerationTaskDiagnostics {
  std::string task_name;
  Eigen::VectorXd reference_acceleration;
  Eigen::VectorXd jacobian_bias;
  Eigen::VectorXd achieved_acceleration;
  Eigen::VectorXd residual;
  double scale = 1.0;
  TaskSolveMode effective_mode = TaskSolveMode::kScale;
  bool used_min_error_fallback = false;
};

struct AccelerationAllocationDiagnostics {
  bool applied = false;
  Eigen::VectorXd physical_metric_diagonal;
  Eigen::VectorXd reference_acceleration;
  Eigen::VectorXd weighted_physical_residual;
  double objective_value = 0.0;
};

struct AccelerationAnalyticCollisionPairDiagnostics {
  std::size_t pair_index = 0;
  std::string pair_key;
  std::string object_a;
  std::string object_b;
  AccelerationCollisionRegime regime =
      AccelerationCollisionRegime::kUnknown;
  double minimum_distance = 0.0;
  double current_signed_distance =
      std::numeric_limits<double>::quiet_NaN();
  double current_signed_distance_lower_bound =
      std::numeric_limits<double>::quiet_NaN();
  double current_signed_distance_upper_bound =
      std::numeric_limits<double>::quiet_NaN();
  double current_rate_lower_bound =
      std::numeric_limits<double>::quiet_NaN();
  double current_rate_upper_bound =
      std::numeric_limits<double>::quiet_NaN();
  double endpoint_signed_distance_lower_bound =
      std::numeric_limits<double>::quiet_NaN();
  double lowest_path_signed_distance_lower_bound =
      std::numeric_limits<double>::quiet_NaN();
  bool state_rate_shaping_active = false;
  bool braking_witness_required = false;
  bool step_certified = false;
};

struct AccelerationSolverResult : public SolverResult {
  Eigen::VectorXd joint_accelerations;
  Eigen::VectorXd joint_velocities_next;
  Eigen::VectorXd q_solution;

  /// True once the compatible acceleration box participated in the solve,
  /// including backend attempts that return no executable motion.
  bool acceleration_limits_applied = false;
  std::vector<int> saturated_acceleration_indices;
  std::vector<int> saturated_velocity_indices;
  std::vector<int> saturated_position_indices;
  Eigen::VectorXd predicted_torques;
  bool effort_limits_applied = false;
  std::vector<int> saturated_effort_indices;
  std::vector<AccelerationTaskDiagnostics> task_diagnostics;
  AccelerationAllocationDiagnostics allocation_diagnostics;
  bool velocity_collision_lift_applied = false;
  bool collision_endpoint_validated = false;
  bool collision_step_certified = false;
  std::uint64_t collision_validation_samples = 0;
  std::uint64_t collision_validation_allowed_pairs = 0;
  std::uint64_t collision_validation_pairs_checked = 0;
  std::uint64_t collision_validation_exact_queries = 0;
  std::uint64_t collision_lift_pairs_considered = 0;
  std::uint64_t collision_lift_row_pairs = 0;
  std::uint64_t collision_lift_row_exact_queries = 0;
  bool native_collision_constraint_applied = false;
  std::vector<AccelerationAnalyticCollisionPairDiagnostics>
      native_collision_diagnostics;
  std::uint64_t native_collision_pair_evaluations = 0;
  std::uint64_t native_collision_path_visited_nodes = 0;
  std::uint64_t native_collision_certified_intervals = 0;
};

struct AccelerationSolverCapabilities {
  bool supports_fixed_base_scalar_joints = true;
  bool supports_floating_base = false;
  bool supports_scale_elastic = false;
#ifdef PINOCCHIO_WITH_HPP_FCL
  // True when at least one native collision family is available. Inspect the
  // precise capability below and configuration contract for its scope.
  bool supports_collision_constraints = true;
  bool supports_analytic_sphere_collision_constraints = true;
#else
  bool supports_collision_constraints = false;
  bool supports_analytic_sphere_collision_constraints = false;
#endif
  bool supports_effort_constraints = true;
  bool supports_fixed_base_contact_kinematics = true;
  bool supports_tight_point_constraints = true;
  bool supports_tight_frame_pose_constraints = true;
  bool supports_relative_pose_constraints = true;
  bool supports_torso_pose_bound_constraints = true;
  bool supports_com_support_polygon_constraints = true;
#ifdef PINOCCHIO_WITH_HPP_FCL
  bool supports_velocity_collision_lift = true;
#else
  bool supports_velocity_collision_lift = false;
#endif
  bool supports_dynamic_contact = false;
};

/**
 * @brief Native acceleration-level eSNS solver for fixed-base scalar joints.
 *
 * This initial surface consumes explicit q, dq, and dt on every solve. It does
 * not store or infer previous command state. Unsupported model/task families
 * fail closed instead of falling back to velocity-level behavior. The API is
 * not a hard real-time controller and makes no allocation-free or bounded-time
 * execution guarantee.
 */
class AccelerationSolver {
public:
  explicit AccelerationSolver(std::shared_ptr<RobotModel> robot);

  AccelerationSolver(const AccelerationSolver &) = delete;
  AccelerationSolver &operator=(const AccelerationSolver &) = delete;
  AccelerationSolver(AccelerationSolver &&) noexcept = default;
  AccelerationSolver &operator=(AccelerationSolver &&) noexcept = default;

  static AccelerationSolverCapabilities capabilities();

  std::shared_ptr<FrameTask>
  add_frame_task(const std::string &name, const std::string &frame_name,
                 TaskType task_type = TaskType::FRAME_POSE);
  std::shared_ptr<COMTask> add_com_task(const std::string &name);
  std::shared_ptr<PostureTask>
  add_posture_task(const std::string &name,
                   const std::vector<int> &controlled_joints = {});
  std::shared_ptr<JointTask>
  add_joint_task(const std::string &name, const std::string &joint_name,
                 double target_value = 0.0);

  void set_task_reference(const std::string &name,
                          const AccelerationTaskReference &reference);

  std::shared_ptr<Task> get_task(const std::string &name) const;
  void remove_task(const std::string &name);
  void clear_tasks();

  /**
   * @brief Configure native continuously certified analytic collision.
   *
   * The definition uses exact collision-geometry names. Configuration is
   * atomic and rejects unsupported geometry or topology. The initial native
   * proof supports complete sphere-sphere catalogs on fixed-base
   * all-prismatic scalar models; other catalogs remain available through the
   * non-certifying velocity-collision compatibility adapter.
   *
   * While this native mode is configured, solve() accepts joint position,
   * velocity, and acceleration boxes, generalized allocation, and soft tasks.
   * Other per-call hard families are rejected until they provide matching
   * predicted-state certificate hooks.
   */
  void configure_collision_constraint(
      const CollisionConstraintDefinition &definition,
      const CollisionConstraintAccelerationPolicy &policy);
  void clear_collision_constraint();
  bool has_collision_constraint() const;
  /// Returns the default distance, not any pair-specific override.
  double get_collision_min_distance() const;
  std::optional<CollisionConstraintDefinition>
  get_collision_constraint_definition() const;
  std::optional<CollisionConstraintAccelerationPolicy>
  get_collision_constraint_policy() const;
  std::vector<CollisionGeometryPair> get_active_collision_pairs() const;

  AccelerationSolverResult
  solve(const Eigen::VectorXd &q, const Eigen::VectorXd &dq, double dt,
        const AccelerationSolveOptions &options = AccelerationSolveOptions{});

  /**
   * @brief Reuse a configured velocity solver's collision rows and pair policy.
   *
   * The frozen next-velocity rows are lifted into the acceleration solve, then
   * every allowed collision pair is checked exactly at the requested samples.
   * This compatibility adapter fails closed but does not certify the continuous
   * path between samples.
   */
  AccelerationSolverResult solve_with_velocity_collision(
      KinematicsSolver &collision_solver, const Eigen::VectorXd &q,
      const Eigen::VectorXd &dq, double dt,
      const AccelerationSolveOptions &options = AccelerationSolveOptions{},
      const VelocityCollisionLiftOptions &lift_options =
          VelocityCollisionLiftOptions{});

private:
  void ensure_unique_task_name(const std::string &name) const;

  std::shared_ptr<RobotModel> robot_;
  std::vector<std::shared_ptr<Task>> tasks_;
  std::unordered_map<std::string, std::shared_ptr<Task>> task_map_;
  std::unordered_map<std::string, AccelerationTaskReference> task_references_;
  std::optional<CollisionConstraintDefinition> native_collision_definition_;
  std::optional<CollisionConstraintAccelerationPolicy>
      native_collision_policy_;
  std::vector<CollisionGeometryPair> native_collision_active_pairs_;
  std::vector<std::size_t> native_collision_pair_indices_;
  std::vector<double> native_collision_minimum_distances_;
};

} // namespace embodik
