/**
 * @file tasks.hpp
 * @brief Task definitions for EmbodiK solver
 *
 * This file defines various task types used in the IK solver:
 * - FrameTask: Track position/orientation of robot frames
 * - COMTask: Control center of mass position
 * - PostureTask: Joint space regularization
 */

#pragma once

#include <Eigen/Dense>
#include <embodik/types.hpp>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace embodik {

// Type definitions for convenience
using Matrix6Xd = Eigen::Matrix<double, 6, Eigen::Dynamic>;

// Forward declarations
class RobotModel;
class KinematicsSolver;

/**
 * @brief Task types enumeration
 */
enum class TaskType {
    FRAME_POSITION,      // Position only (3 DOF)
    FRAME_ORIENTATION,   // Orientation only (3 DOF)
    FRAME_POSE,         // Position + Orientation (6 DOF)
    COM,                // Center of mass (3 DOF)
    POSTURE,            // Joint regularization (n DOF)
    JOINT               // Single joint target (1 DOF)
};

/**
 * @brief Base class for all IK tasks
 *
 * Tasks compute:
 * - Error vector (desired - current)
 * - Jacobian matrix
 * - Task velocity (derivative of error)
 */
class Task {
public:
    /**
     * @brief Constructor
     * @param name Task name for identification
     * @param priority Task priority (0 = highest)
     * @param weight Task weight/gain
     */
    Task(const std::string& name, int priority = 0, double weight = 1.0)
        : name_(name), priority_(priority), weight_(weight) {}

    virtual ~Task() = default;

    /**
     * @brief Update task computations based on current robot state
     * @param model Robot model with current configuration
     */
    virtual void update(const RobotModel& model) = 0;

    /**
     * @brief Get task error vector (desired - current)
     * @return Error vector
     */
    virtual Eigen::VectorXd getError() const = 0;

    /**
     * @brief Get task Jacobian matrix
     * @return Jacobian matrix (task_dim x nv)
     */
    virtual Eigen::MatrixXd getJacobian() const = 0;

    /**
     * @brief Set excluded joint indices (velocity space indices)
     * @param excluded_indices Vector of velocity space indices to exclude from Jacobian
     *
     * Excluded joints will have their Jacobian columns zeroed out, effectively
     * preventing those joints from contributing to the task solution.
     * This is useful for two-stage IK where different stages control different joint groups.
     */
    virtual void set_excluded_joint_indices(const std::vector<int>& excluded_indices) {
        excluded_joint_indices_ = excluded_indices;
    }

    /**
     * @brief Clear excluded joint indices
     */
    virtual void clear_excluded_joint_indices() {
        excluded_joint_indices_.clear();
    }

    /**
     * @brief Get excluded joint indices
     * @return Vector of excluded velocity space indices
     */
    const std::vector<int>& get_excluded_joint_indices() const {
        return excluded_joint_indices_;
    }

    /**
     * @brief Get task velocity (proportional feedback toward the target)
     * @return Task velocity vector
     */
    virtual Eigen::VectorXd getVelocity() const {
        if (target_velocity_.has_value()) {
            // Direct velocity mode (e.g. solve_position_step): magnitude is not
            // scaled by weight_; use weight_=0 or clearTargetVelocity() to drop
            // a task from contributing via this path.
            return target_velocity_.value();
        }
        // Drive the task toward the target using proportional feedback.
        return weight_ * getError();
    }

    /**
     * @brief Set target velocity directly
     * @param velocity Target velocity vector
     */
    virtual void setTargetVelocity(const Eigen::VectorXd& velocity) {
        if (!target_velocity_.has_value() ||
            !target_velocity_->isApprox(velocity, 0.0)) {
            markContinuityStateChanged();
        }
        target_velocity_ = velocity;
    }

    /**
     * @brief Clear target velocity (revert to error-based velocity)
     */
    virtual void clearTargetVelocity() {
        if (target_velocity_.has_value()) {
            markContinuityStateChanged();
        }
        target_velocity_.reset();
    }

    /**
     * @brief Get task dimension
     * @return Number of degrees of freedom for this task
     */
    virtual int getDimension() const = 0;

    /**
     * @brief Get task type
     * @return Type of this task
     */
    virtual TaskType getType() const = 0;

    /**
     * @brief Whether near-limit Jacobian clamping should use the task goal.
     */
    virtual bool usesGoalDirectedLimitClamp() const { return false; }

    // Getters and setters
    const std::string& getName() const { return name_; }
    int getPriority() const { return priority_; }
    double getWeight() const { return weight_; }
    bool isActive() const { return active_; }

    void setPriority(int priority) { priority_ = priority; }
    void setWeight(double weight) { weight_ = weight; }
    void setActive(bool active) { active_ = active; }
    TaskSolveMode getSolveMode() const { return solve_mode_; }
    void setSolveMode(TaskSolveMode mode) { solve_mode_ = mode; }
    bool getAllowMinErrorFallback() const { return allow_min_error_fallback_; }
    void setAllowMinErrorFallback(bool allow) { allow_min_error_fallback_ = allow; }
    TaskSolveMode getLastEffectiveMode() const { return last_effective_mode_; }
    void setLastEffectiveMode(TaskSolveMode mode) { last_effective_mode_ = mode; }
    bool getUsedMinErrorFallback() const { return used_min_error_fallback_; }
    void setUsedMinErrorFallback(bool used) { used_min_error_fallback_ = used; }
    std::uint64_t getContinuityRevision() const { return continuity_revision_; }
    std::uint64_t getContinuityTargetRevision() const {
        return continuity_target_revision_;
    }

protected:
    void markContinuityStateChanged() { ++continuity_revision_; }
    void markContinuityTargetChanged() { ++continuity_target_revision_; }

    std::string name_;
    int priority_;
    double weight_;
    bool active_ = true;
    TaskSolveMode solve_mode_ = TaskSolveMode::kScale;
    bool allow_min_error_fallback_ = false;
    TaskSolveMode last_effective_mode_ = TaskSolveMode::kScale;
    bool used_min_error_fallback_ = false;
    mutable std::optional<Eigen::VectorXd> target_velocity_;  // Direct velocity specification
    std::vector<int> excluded_joint_indices_;  // Velocity space indices to exclude from Jacobian
    std::uint64_t continuity_revision_ = 0;
    std::uint64_t continuity_target_revision_ = 0;

private:
    friend class KinematicsSolver;

    void setPositionStepTargetVelocity(const Eigen::VectorXd& velocity) {
        target_velocity_ = velocity;
    }

    void clearPositionStepTargetVelocity() {
        target_velocity_.reset();
    }
};

/**
 * @brief Task for tracking frame position and/or orientation
 */
class FrameTask : public Task {
public:
    /**
     * @brief Constructor for frame task
     * @param name Task name
     * @param model Robot model
     * @param frame_name Name of the frame to track
     * @param task_type Type of frame task (position/orientation/pose)
     * @param priority Task priority
     * @param weight Task weight
     */
    FrameTask(const std::string& name,
              std::shared_ptr<RobotModel> model,
              const std::string& frame_name,
              TaskType task_type = TaskType::FRAME_POSE,
              int priority = 0,
              double weight = 1.0);

    /**
     * @brief Set desired position
     * @param position Desired 3D position
     */
    void setTargetPosition(const Eigen::Vector3d& position);

    /**
     * @brief Set desired orientation
     * @param rotation Desired rotation matrix
     */
    void setTargetOrientation(const Eigen::Matrix3d& rotation);

    /**
     * @brief Set desired pose (position + orientation)
     * @param position Desired 3D position
     * @param rotation Desired rotation matrix
     */
    void setTargetPose(const Eigen::Vector3d& position, const Eigen::Matrix3d& rotation);

    /**
     * @brief Set target linear velocity (3D)
     * @param velocity Target linear velocity
     */
    void setTargetPositionVelocity(const Eigen::Vector3d& velocity);

    /**
     * @brief Set target angular velocity (3D)
     * @param omega Target angular velocity
     */
    void setTargetAngularVelocity(const Eigen::Vector3d& omega);

    /**
     * @brief Set full spatial velocity (6D: linear + angular)
     * @param velocity Target velocity (linear[3] + angular[3])
     */
    void setTargetVelocity(const Eigen::VectorXd& velocity) override;

    /**
     * @brief Set position mask (which axes to control)
     * @param mask 3D boolean mask (true = control axis)
     */
    void setPositionMask(const Eigen::Vector3d& mask) {
        if (!position_mask_.isApprox(mask, 0.0)) {
            position_mask_ = mask;
            markContinuityStateChanged();
            invalidateCache();
        }
    }

    /**
     * @brief Set orientation mask (which axes to control)
     * @param mask 3D boolean mask (true = control axis)
     */
    void setOrientationMask(const Eigen::Vector3d& mask) {
        if (!orientation_mask_.isApprox(mask, 0.0)) {
            orientation_mask_ = mask;
            markContinuityStateChanged();
            invalidateCache();
        }
    }

    /**
     * @brief Override to invalidate cache when exclusion changes
     */
    void set_excluded_joint_indices(const std::vector<int>& excluded_indices) override {
        Task::set_excluded_joint_indices(excluded_indices);
        invalidateCache();
    }

    /**
     * @brief Override to invalidate cache when exclusion changes
     */
    void clear_excluded_joint_indices() override {
        Task::clear_excluded_joint_indices();
        invalidateCache();
    }

    // Implement base class methods
    void update(const RobotModel& model) override;
    Eigen::VectorXd getError() const override;
    Eigen::MatrixXd getJacobian() const override;
    int getDimension() const override;
    TaskType getType() const override { return task_type_; }

    // Getters for current state
    const std::string& getFrameName() const { return frame_name_; }
    const Eigen::Vector3d& getCurrentPosition() const { return current_position_; }
    const Eigen::Matrix3d& getCurrentOrientation() const { return current_orientation_; }

private:
    std::shared_ptr<RobotModel> model_;
    std::string frame_name_;
    TaskType task_type_;

    // Target values
    std::optional<Eigen::Vector3d> target_position_;
    std::optional<Eigen::Matrix3d> target_orientation_;

    // Current values (updated in update())
    Eigen::Vector3d current_position_;
    Eigen::Matrix3d current_orientation_;

    // Jacobians (updated in update())
    Eigen::MatrixXd position_jacobian_;
    Eigen::MatrixXd orientation_jacobian_;

    // Masks for selective control
    Eigen::Vector3d position_mask_ = Eigen::Vector3d::Ones();
    Eigen::Vector3d orientation_mask_ = Eigen::Vector3d::Ones();

    // Cached values
    mutable Eigen::VectorXd error_cache_;
    mutable Eigen::MatrixXd jacobian_cache_;
    mutable bool cache_valid_ = false;

    void invalidateCache() { cache_valid_ = false; }
    Eigen::Vector3d computeOrientationError(const Eigen::Matrix3d& R_current,
                                           const Eigen::Matrix3d& R_desired) const;
};

/**
 * @brief Task for controlling center of mass position
 */
class COMTask : public Task {
public:
    /**
     * @brief Constructor for COM task
     * @param name Task name
     * @param model Robot model
     * @param priority Task priority
     * @param weight Task weight
     */
    COMTask(const std::string& name,
            std::shared_ptr<RobotModel> model,
            int priority = 0,
            double weight = 1.0);

    /**
     * @brief Set desired COM position
     * @param position Desired 3D COM position
     */
    void setTargetPosition(const Eigen::Vector3d& position);

    /**
     * @brief Set position mask (which axes to control)
     * @param mask 3D boolean mask (true = control axis)
     */
    void setPositionMask(const Eigen::Vector3d& mask) {
        if (!position_mask_.isApprox(mask, 0.0)) {
            position_mask_ = mask;
            markContinuityStateChanged();
        }
    }

    // Implement base class methods
    void update(const RobotModel& model) override;
    Eigen::VectorXd getError() const override;
    Eigen::MatrixXd getJacobian() const override;
    int getDimension() const override;
    TaskType getType() const override { return TaskType::COM; }

    // Getter for current COM position
    const Eigen::Vector3d& getCurrentPosition() const { return current_position_; }

private:
    std::shared_ptr<RobotModel> model_;

    // Target and current values
    Eigen::Vector3d target_position_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d current_position_;

    // Jacobian (updated in update())
    Eigen::MatrixXd com_jacobian_;

    // Mask for selective control
    Eigen::Vector3d position_mask_ = Eigen::Vector3d::Ones();
};

/**
 * @brief Task for joint space regularization
 */
class PostureTask : public Task {
public:
    /**
     * @brief Constructor for posture task
     * @param name Task name
     * @param model Robot model
     * @param priority Task priority
     * @param weight Task weight
     */
    PostureTask(const std::string& name,
                std::shared_ptr<RobotModel> model,
                int priority = 10,  // Low priority by default
                double weight = 0.1);  // Small weight by default

    /**
     * @brief Constructor with specific joint indices
     * @param name Task name
     * @param model Robot model
     * @param controlled_joint_indices Indices of joints to control
     * @param priority Task priority
     * @param weight Task weight
     */
    PostureTask(const std::string& name,
                std::shared_ptr<RobotModel> model,
                const std::vector<int>& controlled_joint_indices,
                int priority = 10,
                double weight = 0.1);

    /**
     * @brief Set target joint configuration
     * @param q_target Target joint configuration
     */
    void setTargetConfiguration(const Eigen::VectorXd& q_target);

    /**
     * @brief Update a moving regularization reference without declaring a new
     * posture command.
     * @param q_reference Reference joint configuration
     */
    void setReferenceConfiguration(const Eigen::VectorXd& q_reference);

    /**
     * @brief Set target values for controlled joints only
     * @param target_values Values for controlled joints (size must match controlled_joint_indices)
     */
    void setControlledJointTargets(const Eigen::VectorXd& target_values);

    /**
     * @brief Set joint mask (which joints to control)
     * @param mask Boolean mask (true = control joint)
     */
    void setJointMask(const Eigen::VectorXd& mask) {
        if (joint_mask_.size() != mask.size() || !joint_mask_.isApprox(mask, 0.0)) {
            joint_mask_ = mask;
            markContinuityStateChanged();
        }
    }

    /**
     * @brief Set controlled joint indices
     * @param indices Vector of joint indices to control
     */
    void setControlledJointIndices(const std::vector<int>& indices);

    /**
     * @brief Set per-joint weights
     * @param weights Weight for each joint
     */
    void setJointWeights(const Eigen::VectorXd& weights) {
        if (joint_weights_.size() != weights.size() ||
            !joint_weights_.isApprox(weights, 0.0)) {
            joint_weights_ = weights;
            markContinuityStateChanged();
        }
    }

    /**
     * @brief Set weights for controlled joints only
     * @param weights Weights for controlled joints (size must match controlled_joint_indices)
     */
    void setControlledJointWeights(const Eigen::VectorXd& weights);

    // Implement base class methods
    void update(const RobotModel& model) override;
    Eigen::VectorXd getError() const override;
    Eigen::MatrixXd getJacobian() const override;
    int getDimension() const override;
    TaskType getType() const override { return TaskType::POSTURE; }

    // Getters
    const std::vector<int>& getControlledJointIndices() const { return controlled_joint_indices_; }

private:
    std::shared_ptr<RobotModel> model_;

    // Target configuration
    Eigen::VectorXd q_target_;

    // Current configuration
    Eigen::VectorXd q_current_;

    // Controlled joint indices (empty means all joints)
    std::vector<int> controlled_joint_indices_;

    // Masks and weights
    Eigen::VectorXd joint_mask_;
    Eigen::VectorXd joint_weights_;

    // Projection matrix for controlled joints
    Eigen::MatrixXd projection_matrix_;

    // Jacobian (identity or projected for configuration space tasks)
    Eigen::MatrixXd jacobian_;

    void updateProjectionMatrix();
};

/**
 * @brief Joint-space task that ascends a frame manipulability score.
 *
 * Computes the analytic gradient of
 * 0.5 * log det(J J^T + epsilon^2 I) for a selected LOCAL frame Jacobian
 * block, then smoothly bounds the gradient magnitude for singularity-safe use
 * as a velocity task.
 */
class ManipulabilityTask : public Task {
public:
    ManipulabilityTask(const std::string& name,
                       std::shared_ptr<RobotModel> model,
                       const std::string& frame_name,
                       TaskType frame_task_type = TaskType::FRAME_POSITION,
                       int priority = 10,
                       double weight = 1.0);

    /**
     * @brief Set controlled velocity-space indices. Empty controls all nv.
     * @param indices Velocity-space indices to control
     */
    void setControlledJointIndices(const std::vector<int>& indices);

    /**
     * @brief Set positive determinant regularization epsilon.
     */
    void setRegularization(double regularization);

    /**
     * @brief Penalize normalized proximity to scalar joint limits.
     *
     * The frame-manipulability gradient is first projected onto the
     * first-order non-worsening half-space of the joint-limit metric, then the
     * inward penalty is applied. This prevents the two objectives from trading
     * away hard-limit recovery when their gradients oppose each other.
     *
     * A zero penalty preserves the frame-only manipulability objective.
     */
    void setJointLimitPenalty(double penalty, double epsilon = 0.04);

    void set_excluded_joint_indices(
        const std::vector<int>& excluded_indices) override;
    void clear_excluded_joint_indices() override;

    void update(const RobotModel& model) override;
    Eigen::VectorXd getError() const override;
    Eigen::MatrixXd getJacobian() const override;
    int getDimension() const override;
    TaskType getType() const override { return TaskType::POSTURE; }
    bool usesGoalDirectedLimitClamp() const override { return true; }

    const std::string& getFrameName() const { return frame_name_; }
    TaskType getFrameTaskType() const { return frame_task_type_; }
    double getScore() const { return score_; }
    double getRegularization() const { return regularization_; }
    double getJointLimitPenalty() const { return joint_limit_penalty_; }
    double getJointLimitEpsilon() const { return joint_limit_epsilon_; }
    const std::vector<int>& getControlledJointIndices() const {
        return controlled_joint_indices_;
    }

private:
    std::shared_ptr<RobotModel> model_;
    std::string frame_name_;
    TaskType frame_task_type_;
    std::vector<int> controlled_joint_indices_;
    double regularization_ = 1e-6;
    double joint_limit_penalty_ = 0.0;
    double joint_limit_epsilon_ = 0.04;
    double score_ = 0.0;
    Eigen::VectorXd bounded_gradient_;
    Eigen::MatrixXd jacobian_;
    std::vector<int> velocity_to_config_index_;

    std::vector<int> taskVelocityIndices() const;
    std::vector<int> metricVelocityIndices() const;
    Eigen::MatrixXd selectTaskRows(const Eigen::MatrixXd& spatial) const;
    void rebuildVelocityToConfigIndex();
    void updateJacobian();
};

/**
 * @brief Smooth joint-space objective that moves scalar joints away from limits.
 *
 * Each controlled joint contributes a signed cubic-smoothstep activation inside
 * a configurable proximity margin. The task is exactly zero outside the margin
 * and remains subject to the solver's hard position and velocity constraints.
 */
class JointLimitAvoidanceTask : public Task {
public:
    JointLimitAvoidanceTask(
        const std::string& name,
        std::shared_ptr<RobotModel> model,
        const std::vector<int>& controlled_joint_indices = {},
        int priority = 10,
        double weight = 0.01);

    void setControlledJointIndices(const std::vector<int>& indices);
    void setActivationMargin(double activation_margin);

    void set_excluded_joint_indices(
        const std::vector<int>& excluded_indices) override;
    void clear_excluded_joint_indices() override;

    void update(const RobotModel& model) override;
    Eigen::VectorXd getError() const override;
    Eigen::MatrixXd getJacobian() const override;
    int getDimension() const override;
    TaskType getType() const override { return TaskType::POSTURE; }
    bool usesGoalDirectedLimitClamp() const override { return true; }

    double getActivationMargin() const { return activation_margin_; }
    const std::vector<int>& getControlledJointIndices() const {
        return controlled_joint_indices_;
    }

private:
    std::shared_ptr<RobotModel> model_;
    std::vector<int> controlled_joint_indices_;
    std::vector<int> velocity_to_config_index_;
    double activation_margin_ = 0.05;
    Eigen::VectorXd avoidance_velocity_;
    Eigen::MatrixXd jacobian_;

    std::vector<int> taskVelocityIndices() const;
    void rebuildVelocityToConfigIndex();
    void updateJacobian();
};

/**
 * @brief Task for controlling a single joint
 */
class JointTask : public Task {
public:
    /**
     * @brief Constructor for joint task
     * @param name Task name
     * @param model Robot model
     * @param joint_name Name of the joint to control
     * @param target_value Target joint value
     * @param priority Task priority
     * @param weight Task weight
     */
    JointTask(const std::string& name,
              std::shared_ptr<RobotModel> model,
              const std::string& joint_name,
              double target_value = 0.0,
              int priority = 0,
              double weight = 1.0);

    /**
     * @brief Constructor using joint index
     * @param name Task name
     * @param model Robot model
     * @param joint_index Index of the joint to control
     * @param target_value Target joint value
     * @param priority Task priority
     * @param weight Task weight
     */
    JointTask(const std::string& name,
              std::shared_ptr<RobotModel> model,
              int joint_index,
              double target_value = 0.0,
              int priority = 0,
              double weight = 1.0);

    /**
     * @brief Set target joint value
     * @param value Target value in radians
     */
    void setTargetValue(double value) {
        if (target_value_ != value) {
            target_value_ = value;
            markContinuityStateChanged();
        }
    }

    // Implement base class methods
    void update(const RobotModel& model) override;
    Eigen::VectorXd getError() const override;
    Eigen::MatrixXd getJacobian() const override;
    int getDimension() const override { return 1; }
    TaskType getType() const override { return TaskType::JOINT; }

private:
    std::shared_ptr<RobotModel> model_;
    int joint_index_;
    double target_value_;
    double current_value_;

    // Jacobian (single row with 1 at joint index)
    Eigen::MatrixXd jacobian_;
};

/**
 * @brief Task for controlling multiple specific joints
 */
class MultiJointTask : public Task {
public:
    /**
     * @brief Constructor for multi-joint task
     * @param name Task name
     * @param model Robot model
     * @param joint_indices Indices of joints to control
     * @param target_values Target values for each joint
     * @param priority Task priority
     * @param weight Task weight
     */
    MultiJointTask(const std::string& name,
                   std::shared_ptr<RobotModel> model,
                   const std::vector<int>& joint_indices,
                   const Eigen::VectorXd& target_values = Eigen::VectorXd(),
                   int priority = 0,
                   double weight = 1.0);

    /**
     * @brief Constructor using joint names
     * @param name Task name
     * @param model Robot model
     * @param joint_names Names of joints to control
     * @param target_values Target values for each joint
     * @param priority Task priority
     * @param weight Task weight
     */
    MultiJointTask(const std::string& name,
                   std::shared_ptr<RobotModel> model,
                   const std::vector<std::string>& joint_names,
                   const Eigen::VectorXd& target_values = Eigen::VectorXd(),
                   int priority = 0,
                   double weight = 1.0);

    /**
     * @brief Set target values for all controlled joints
     * @param values Target values (size must match number of controlled joints)
     */
    void setTargetValues(const Eigen::VectorXd& values);

    /**
     * @brief Set target value for a specific controlled joint
     * @param idx Index in controlled joints list (not global joint index)
     * @param value Target value
     */
    void setTargetValue(int idx, double value);

    /**
     * @brief Set per-joint weights
     * @param weights Weights for each controlled joint
     */
    void setJointWeights(const Eigen::VectorXd& weights);

    // Implement base class methods
    void update(const RobotModel& model) override;
    Eigen::VectorXd getError() const override;
    Eigen::MatrixXd getJacobian() const override;
    int getDimension() const override { return static_cast<int>(joint_indices_.size()); }
    TaskType getType() const override { return TaskType::JOINT; }

    // Getters
    const std::vector<int>& getJointIndices() const { return joint_indices_; }
    const Eigen::VectorXd& getTargetValues() const { return target_values_; }

private:
    std::shared_ptr<RobotModel> model_;
    std::vector<int> joint_indices_;
    Eigen::VectorXd target_values_;
    Eigen::VectorXd current_values_;
    Eigen::VectorXd joint_weights_;

    // Jacobian (sparse with 1s at controlled joint indices)
    Eigen::MatrixXd jacobian_;
};

/**
 * @brief Task for tracking the relative pose between two frames
 *
 * Drives T_a^{-1} * T_b toward a target relative pose.
 * Supports per-axis masking to free individual DOFs.
 */
class RelativeFrameTask : public Task {
public:
    RelativeFrameTask(const std::string& name,
                      std::shared_ptr<RobotModel> model,
                      const std::string& frame_a,
                      const std::string& frame_b,
                      int priority = 0,
                      double weight = 1.0);

    void setTargetPose(const Eigen::Vector3d& position,
                       const Eigen::Matrix3d& rotation);

    void setPositionMask(const Eigen::Vector3d& mask) {
        if (!position_mask_.isApprox(mask, 0.0)) {
            position_mask_ = mask;
            markContinuityStateChanged();
        }
    }
    void setOrientationMask(const Eigen::Vector3d& mask) {
        if (!orientation_mask_.isApprox(mask, 0.0)) {
            orientation_mask_ = mask;
            markContinuityStateChanged();
        }
    }

    /**
     * @brief Capture the current relative pose as the target
     *
     * Call after update() to snapshot the current T_a^{-1} * T_b.
     */
    void captureCurrentAsTarget();

    const Eigen::Vector3d& getCurrentPosition() const { return current_rel_position_; }
    const Eigen::Matrix3d& getCurrentOrientation() const { return current_rel_orientation_; }

    const std::string& getFrameA() const { return frame_a_; }
    const std::string& getFrameB() const { return frame_b_; }

    void update(const RobotModel& model) override;
    Eigen::VectorXd getError() const override;
    Eigen::MatrixXd getJacobian() const override;
    int getDimension() const override { return 6; }
    TaskType getType() const override { return TaskType::FRAME_POSE; }

private:
    std::shared_ptr<RobotModel> model_;
    std::string frame_a_, frame_b_;

    std::optional<Eigen::Vector3d> target_position_;
    std::optional<Eigen::Matrix3d> target_orientation_;

    Eigen::Vector3d current_rel_position_;
    Eigen::Matrix3d current_rel_orientation_;

    Eigen::MatrixXd relative_jacobian_;

    Eigen::Vector3d position_mask_ = Eigen::Vector3d::Ones();
    Eigen::Vector3d orientation_mask_ = Eigen::Vector3d::Ones();
};

/**
 * @brief Task for tracking the absolute (object-centric) pose of two frames
 *
 * The absolute frame is a weighted interpolation of two end-effector frames,
 * parameterized by alpha (coordination ratio). Supports virtual-tip offsets.
 */
class AbsoluteFrameTask : public Task {
public:
    AbsoluteFrameTask(const std::string& name,
                      std::shared_ptr<RobotModel> model,
                      const std::string& frame_a,
                      const std::string& frame_b,
                      double alpha = 0.5,
                      int priority = 0,
                      double weight = 1.0);

    void setTargetPose(const Eigen::Vector3d& position,
                       const Eigen::Matrix3d& rotation);

    void setAlpha(double alpha) {
        if (alpha_ != alpha) {
            alpha_ = alpha;
            markContinuityStateChanged();
        }
    }
    double getAlpha() const { return alpha_; }

    void setPositionMask(const Eigen::Vector3d& mask) {
        if (!position_mask_.isApprox(mask, 0.0)) {
            position_mask_ = mask;
            markContinuityStateChanged();
        }
    }
    void setOrientationMask(const Eigen::Vector3d& mask) {
        if (!orientation_mask_.isApprox(mask, 0.0)) {
            orientation_mask_ = mask;
            markContinuityStateChanged();
        }
    }

    /**
     * @brief Set virtual TCP offsets applied to each frame before ECTS computation
     *
     * T_virtual_a = T_frame_a * offset_a, T_virtual_b = T_frame_b * offset_b
     */
    void setTcpOffsets(const Eigen::Matrix4d& offset_a,
                       const Eigen::Matrix4d& offset_b);

    /**
     * @brief Auto-compute TCP offsets so both virtual TCPs coincide at object_frame
     */
    void setObjectCenterFrame(const Eigen::Matrix4d& object_frame);

    /**
     * @brief Diagnostic structure for inconsistent left/right grasp targets.
     */
    struct GraspDivergenceDiagnostic {
        double linear_m = 0.0;
        double angular_rad = 0.0;
    };

    /**
     * @brief Calibrate rigid arm-to-object offsets at the current task pose.
     *
     * The task must have been updated at least once so current_position and
     * current_orientation represent the calibration object frame.
     */
    void calibrate_grasp_offsets(const pinocchio::SE3& T_L_FK,
                                 const pinocchio::SE3& T_R_FK);
    void calibrate_grasp_offsets(const Eigen::Matrix4d& T_L_FK,
                                 const Eigen::Matrix4d& T_R_FK);

    /**
     * @brief Set this absolute task target from calibrated per-arm targets.
     *
     * Each arm target implies an object pose through the calibrated rigid
     * offsets. Consistent arm targets recover the same object pose; divergent
     * targets are blended through compute_absolute_frame(..., 0.5) and reported
     * through get_grasp_divergence().
     */
    void set_target_from_arm_targets(const pinocchio::SE3& T_L_target,
                                     const pinocchio::SE3& T_R_target);
    void set_target_from_arm_targets(const Eigen::Matrix4d& T_L_target,
                                     const Eigen::Matrix4d& T_R_target);

    const GraspDivergenceDiagnostic& get_grasp_divergence() const {
        return last_grasp_divergence_;
    }
    bool grasp_offsets_calibrated() const { return grasp_offsets_calibrated_; }

    const Eigen::Vector3d& getCurrentPosition() const { return current_abs_position_; }
    const Eigen::Matrix3d& getCurrentOrientation() const { return current_abs_orientation_; }

    const std::string& getFrameA() const { return frame_a_; }
    const std::string& getFrameB() const { return frame_b_; }

    void update(const RobotModel& model) override;
    Eigen::VectorXd getError() const override;
    Eigen::MatrixXd getJacobian() const override;
    int getDimension() const override { return 6; }
    TaskType getType() const override { return TaskType::FRAME_POSE; }

private:
    std::shared_ptr<RobotModel> model_;
    std::string frame_a_, frame_b_;
    double alpha_;

    std::optional<Eigen::Vector3d> target_position_;
    std::optional<Eigen::Matrix3d> target_orientation_;

    Eigen::Vector3d current_abs_position_;
    Eigen::Matrix3d current_abs_orientation_;

    Eigen::MatrixXd absolute_jacobian_;

    pinocchio::SE3 offset_a_ = pinocchio::SE3::Identity();
    pinocchio::SE3 offset_b_ = pinocchio::SE3::Identity();
    pinocchio::SE3 L_in_obj_ = pinocchio::SE3::Identity();
    pinocchio::SE3 R_in_obj_ = pinocchio::SE3::Identity();
    pinocchio::SE3 inv_L_in_obj_ = pinocchio::SE3::Identity();
    pinocchio::SE3 inv_R_in_obj_ = pinocchio::SE3::Identity();
    bool current_abs_pose_valid_ = false;
    bool grasp_offsets_calibrated_ = false;
    GraspDivergenceDiagnostic last_grasp_divergence_;

    Eigen::Vector3d position_mask_ = Eigen::Vector3d::Ones();
    Eigen::Vector3d orientation_mask_ = Eigen::Vector3d::Ones();
};

} // namespace embodik
