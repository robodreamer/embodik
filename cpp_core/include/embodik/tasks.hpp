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
#include <memory>
#include <string>
#include <optional>

namespace embodik {

// Type definitions for convenience
using Matrix6Xd = Eigen::Matrix<double, 6, Eigen::Dynamic>;

// Forward declarations
class RobotModel;

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
        target_velocity_ = velocity;
    }

    /**
     * @brief Clear target velocity (revert to error-based velocity)
     */
    virtual void clearTargetVelocity() {
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

    // Getters and setters
    const std::string& getName() const { return name_; }
    int getPriority() const { return priority_; }
    double getWeight() const { return weight_; }
    bool isActive() const { return active_; }

    void setPriority(int priority) { priority_ = priority; }
    void setWeight(double weight) { weight_ = weight; }
    void setActive(bool active) { active_ = active; }

protected:
    std::string name_;
    int priority_;
    double weight_;
    bool active_ = true;
    mutable std::optional<Eigen::VectorXd> target_velocity_;  // Direct velocity specification
    std::vector<int> excluded_joint_indices_;  // Velocity space indices to exclude from Jacobian
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
    void setPositionMask(const Eigen::Vector3d& mask) { position_mask_ = mask; }

    /**
     * @brief Set orientation mask (which axes to control)
     * @param mask 3D boolean mask (true = control axis)
     */
    void setOrientationMask(const Eigen::Vector3d& mask) {
        orientation_mask_ = mask;
        invalidateCache();
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
    void setPositionMask(const Eigen::Vector3d& mask) { position_mask_ = mask; }

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
     * @brief Set target values for controlled joints only
     * @param target_values Values for controlled joints (size must match controlled_joint_indices)
     */
    void setControlledJointTargets(const Eigen::VectorXd& target_values);

    /**
     * @brief Set joint mask (which joints to control)
     * @param mask Boolean mask (true = control joint)
     */
    void setJointMask(const Eigen::VectorXd& mask) { joint_mask_ = mask; }

    /**
     * @brief Set controlled joint indices
     * @param indices Vector of joint indices to control
     */
    void setControlledJointIndices(const std::vector<int>& indices);

    /**
     * @brief Set per-joint weights
     * @param weights Weight for each joint
     */
    void setJointWeights(const Eigen::VectorXd& weights) { joint_weights_ = weights; }

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
    void setTargetValue(double value) { target_value_ = value; }

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

} // namespace embodik
