/*
 * Copyright 2025-2026 Andy Park <andypark.purdue@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace embodik {

/**
 * @brief Wrapper class for Pinocchio robot model providing kinematics
 * operations
 *
 * This class encapsulates a Pinocchio model and provides convenient methods
 * for forward kinematics, Jacobian computation, and center of mass
 * calculations.
 */
class RobotModel {
public:
  using SE3 = pinocchio::SE3;
  using Model = pinocchio::Model;
  using Data = pinocchio::Data;
  using FrameIndex = pinocchio::FrameIndex;
  using JointIndex = pinocchio::JointIndex;

  /**
   * @brief Construct robot model from URDF file
   * @param urdf_path Path to URDF file
   * @param floating_base Whether robot has floating base (default: false)
   * @throws std::runtime_error if URDF loading fails
   */
  explicit RobotModel(const std::string &urdf_path, bool floating_base = false);

  /**
   * @brief Construct reduced robot model with only actuated joints.
   *
   * Loads the full URDF, then builds a reduced model by locking all joints
   * not in actuated_joint_names at their neutral configuration. The resulting
   * model's nq/nv match the actuated joint count, eliminating index mapping
   * when integrating with external systems that use reduced
   * configurations.
   *
   * @param urdf_path Path to URDF file
   * @param actuated_joint_names Names of joints to keep actuated (all others
   *        are locked at neutral)
   * @param floating_base Whether robot has floating base (default: false)
   * @throws std::runtime_error if URDF loading fails or any actuated joint
   *         name is not found in the model
   */
  RobotModel(const std::string &urdf_path,
             const std::vector<std::string> &actuated_joint_names,
             bool floating_base = false);

  /**
   * @brief Create robot model from XACRO file
   * @param xacro_path Path to XACRO file
   * @param floating_base Whether robot has floating base
   * @return Unique pointer to created RobotModel
   * @throws std::runtime_error if XACRO processing or URDF loading fails
   */
  static std::unique_ptr<RobotModel> from_xacro(const std::string &xacro_path,
                                                bool floating_base = false);

  /**
   * @brief Update robot configuration and compute forward kinematics
   * @param q Joint configuration vector (size must match model.nq)
   */
  void update_configuration(const Eigen::VectorXd &q);

  /**
   * @brief Update robot configuration and velocity, compute forward kinematics
   * @param q Joint configuration vector
   * @param v Joint velocity vector (size must match model.nv)
   */
  void update_kinematics(const Eigen::VectorXd &q,
                         const Eigen::VectorXd &v = Eigen::VectorXd());

  /**
   * @brief Get pose of specified frame
   * @param frame_name Name of the frame
   * @return SE3 transformation from world to frame
   * @throws std::runtime_error if frame not found
   */
  SE3 get_frame_pose(const std::string &frame_name) const;

  /**
   * @brief Get Jacobian of specified frame
   * @param frame_name Name of the frame
   * @param ref Reference frame for Jacobian computation
   * @return 6xN Jacobian matrix
   * @throws std::runtime_error if frame not found
   */
  Eigen::Matrix<double, 6, Eigen::Dynamic> get_frame_jacobian(
      const std::string &frame_name,
      pinocchio::ReferenceFrame ref = pinocchio::LOCAL_WORLD_ALIGNED) const;

  /**
   * @brief Get the Jacobian of a point expressed in a frame's local
   * coordinates.
   * @param frame_name Name of the frame that contains the point.
   * @param local_point Position of the point expressed in the frame
   * coordinates.
   * @return 3xN Jacobian of the point linear velocity expressed in world frame.
   */
  Eigen::Matrix<double, 3, Eigen::Dynamic>
  get_point_jacobian(const std::string &frame_name,
                     const Eigen::Vector3d &local_point) const;

  /**
   * @brief Get current center of mass position
   * @return 3D position of center of mass
   */
  Eigen::Vector3d get_com_position() const;

  /**
   * @brief Get center of mass velocity
   * @return 3D velocity of center of mass
   * @note Requires update_kinematics() to be called with velocity
   */
  Eigen::Vector3d get_com_velocity() const;

  /**
   * @brief Get center of mass Jacobian
   * @return 3xN Jacobian matrix for center of mass
   */
  Eigen::Matrix<double, 3, Eigen::Dynamic> get_com_jacobian() const;

  /**
   * @brief Get list of all frame names
   * @return Vector of frame names
   */
  std::vector<std::string> get_frame_names() const;

  /**
   * @brief Get list of all joint names
   * @return Vector of joint names
   */
  std::vector<std::string> get_joint_names() const;

  /**
   * @brief Get joint position limits
   * @return Pair of vectors (lower_limits, upper_limits)
   */
  std::pair<Eigen::VectorXd, Eigen::VectorXd> get_joint_limits() const;

  /**
   * @brief Overwrite joint position limits
   * @param lower Vector of new lower bounds (size must match nq)
   * @param upper Vector of new upper bounds (size must match nq)
   * @throws std::runtime_error if sizes mismatch or lower > upper
   */
  void set_joint_limits(const Eigen::VectorXd &lower,
                        const Eigen::VectorXd &upper);

  /**
   * @brief Get joint velocity limits
   * @return Vector of velocity limits
   */
  Eigen::VectorXd get_velocity_limits() const;

  /**
   * @brief Get joint acceleration limits
   * @return Vector of acceleration limits (if not available, returns high
   * default values)
   */
  Eigen::VectorXd get_acceleration_limits() const;

  /**
   * @brief Set custom joint acceleration limits
   * @param accel_limits Vector of acceleration limits (size must match nv)
   */
  void set_acceleration_limits(const Eigen::VectorXd &accel_limits);

  /**
   * @brief Get joint effort limits
   * @return Vector of effort limits
   */
  Eigen::VectorXd get_effort_limits() const;

  /**
   * @brief Check if frame exists
   * @param frame_name Name of the frame
   * @return true if frame exists
   */
  bool has_frame(const std::string &frame_name) const;

  /**
   * @brief Check if joint exists
   * @param joint_name Name of the joint
   * @return true if joint exists
   */
  bool has_joint(const std::string &joint_name) const;

  /**
   * @brief Get joint index by name
   * @param joint_name Name of the joint
   * @return Joint index (0-based, excluding universe joint)
   * @throws std::runtime_error if joint not found
   */
  JointIndex get_joint_id(const std::string &joint_name) const;

  /**
   * @brief Get configuration-space index for a joint
   * @param joint_name Name of the joint
   * @return Starting index in configuration vector q (idx_q)
   * @throws std::runtime_error if joint not found
   */
  int get_joint_config_index(const std::string &joint_name) const;

  /**
   * @brief Get number of configuration variables for a joint
   * @param joint_name Name of the joint
   * @return Number of configuration variables (nq, typically 1 for revolute, 2 for continuous)
   * @throws std::runtime_error if joint not found
   */
  int get_joint_config_size(const std::string &joint_name) const;

  /**
   * @brief Get velocity-space index for a joint
   * @param joint_name Name of the joint
   * @return Starting index in velocity vector v (idx_v)
   * @throws std::runtime_error if joint not found
   */
  int get_joint_velocity_index(const std::string &joint_name) const;

  /**
   * @brief Get number of velocity variables for a joint
   * @param joint_name Name of the joint
   * @return Number of velocity variables (nv, typically 1)
   * @throws std::runtime_error if joint not found
   */
  int get_joint_velocity_size(const std::string &joint_name) const;

  /**
   * @brief Integrate a velocity vector into a configuration using Lie group
   * operations.
   *
   * For standard revolute/prismatic joints this is equivalent to q + v*dt,
   * but for floating-base (SE3), spherical (quaternion), and other
   * non-Euclidean joint types it performs the correct manifold integration
   * (e.g. quaternion exponential map).
   *
   * @param q Current configuration vector (size nq)
   * @param v Velocity / tangent vector (size nv)
   * @param dt Time step (default 1.0, i.e. v is already scaled)
   * @return Integrated configuration vector (size nq)
   */
  Eigen::VectorXd integrate(const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                            double dt = 1.0) const;

  /**
   * @brief Compute the tangent-vector difference between two configurations.
   *
   * Returns the velocity v such that q1 = integrate(q0, v).
   * For Euclidean joints this is simply q1 - q0, but for quaternion /
   * floating-base joints the result lives in the tangent space (size nv).
   *
   * @param q0 Start configuration (size nq)
   * @param q1 End configuration (size nq)
   * @return Tangent vector v (size nv)
   */
  Eigen::VectorXd difference(const Eigen::VectorXd &q0,
                             const Eigen::VectorXd &q1) const;

  /**
   * @brief Return the neutral (zero / home) configuration for this model.
   *
   * For floating-base robots this includes a valid unit quaternion for the
   * base orientation rather than all-zeros.
   *
   * @return Neutral configuration vector (size nq)
   */
  Eigen::VectorXd neutral_configuration() const;

  /**
   * @brief Generate a random valid configuration within joint limits.
   *
   * Uses Pinocchio's randomConfiguration which respects the joint topology
   * (e.g. generates valid quaternions for floating-base).
   *
   * @return Random configuration vector (size nq)
   */
  Eigen::VectorXd random_configuration() const;

  /**
   * @brief Normalize a configuration vector in-place.
   *
   * For joints that live on a manifold (quaternion components of
   * floating-base or spherical joints) this re-normalizes the quaternion
   * part. For standard revolute/prismatic joints this is a no-op.
   *
   * @param q Configuration vector (modified in-place, size nq)
   * @return Normalized configuration vector (same as input, modified in-place)
   */
  Eigen::VectorXd normalize(const Eigen::VectorXd &q) const;

  // =========================================================================
  // Inverse dynamics / gravity
  // =========================================================================

  /**
   * @brief Compute the generalized gravity torque vector.
   *
   * Returns the joint torques required to compensate gravity at the given
   * configuration (equivalent to RNEA with zero velocity and acceleration).
   *
   * @param q Joint configuration vector (size nq)
   * @return Gravity torque vector g(q) (size nv)
   */
  Eigen::VectorXd compute_generalized_gravity(const Eigen::VectorXd &q) const;

  /**
   * @brief Compute inverse dynamics using the Recursive Newton-Euler Algorithm.
   *
   * Returns the joint torques required to produce the given accelerations
   * at the specified configuration and velocity:
   *   tau = M(q)*a + C(q,v)*v + g(q)
   *
   * Common usage patterns:
   *  - Gravity only:   rnea(q, zeros, zeros)  == compute_generalized_gravity(q)
   *  - Coriolis+grav:  rnea(q, v, zeros)
   *  - Full dynamics:  rnea(q, v, a)
   *
   * @param q Joint configuration vector (size nq)
   * @param v Joint velocity vector (size nv)
   * @param a Joint acceleration vector (size nv)
   * @return Joint torque vector tau (size nv)
   */
  Eigen::VectorXd rnea(const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                       const Eigen::VectorXd &a) const;

  /**
   * @brief Compute the joint-space mass/inertia matrix M(q).
   *
   * Uses the Composite Rigid Body Algorithm (CRBA). The returned matrix is
   * symmetric positive-definite and has size nv x nv.
   *
   * @param q Joint configuration vector (size nq)
   * @return Mass matrix M(q) (size nv x nv)
   */
  Eigen::MatrixXd compute_mass_matrix(const Eigen::VectorXd &q) const;

  /**
   * @brief Compute the Coriolis + centrifugal torque vector C(q,v)*v.
   *
   * Computed as: rnea(q, v, 0) - g(q), where g(q) is the gravity vector.
   *
   * @param q Joint configuration vector (size nq)
   * @param v Joint velocity vector (size nv)
   * @return Coriolis torque vector (size nv)
   */
  Eigen::VectorXd compute_coriolis(const Eigen::VectorXd &q,
                                   const Eigen::VectorXd &v) const;

  /**
   * @brief Set the gravity vector for the model.
   *
   * Default is [0, 0, -9.81]. This affects gravity torque computations.
   *
   * @param gravity 3D gravity vector (e.g. [0, 0, -9.81])
   */
  void set_gravity(const Eigen::Vector3d &gravity);

  /**
   * @brief Get the current gravity vector.
   * @return 3D gravity vector
   */
  Eigen::Vector3d get_gravity() const;

  // =========================================================================
  // Boolean collision checking
  // =========================================================================

  /**
   * @brief Check whether any collision pair is in contact at the current
   * configuration.
   *
   * Returns true if the minimum distance across all active collision pairs
   * is <= 0 (i.e. geometries are overlapping or touching).
   *
   * @return true if any collision detected, false otherwise
   */
  bool check_collision() const;

  /**
   * @brief Check whether any collision pair has distance below a threshold.
   *
   * @param min_distance Distance threshold (default 0.0 = contact check)
   * @return true if any pair has distance <= min_distance
   */
  bool check_collision(double min_distance) const;

  /**
   * @brief Get current joint configuration
   * @return Current q vector
   */
  const Eigen::VectorXd &get_current_configuration() const {
    return current_q_;
  }

  /**
   * @brief Get current joint velocities
   * @return Current v vector
   */
  const Eigen::VectorXd &get_current_velocity() const { return current_v_; }

  // Getters for direct access
  const Model &model() const { return model_; }
  Data &data() { return data_; }
  const Data &data() const { return data_; }
  int nq() const { return model_.nq; }
  int nv() const { return model_.nv; }
  bool is_floating_base() const { return floating_base_; }

  // Access to visual and collision models (if loaded)
  pinocchio::GeometryModel *visual_model() { return visual_model_.get(); }
  const pinocchio::GeometryModel *visual_model() const {
    return visual_model_.get();
  }
  pinocchio::GeometryModel *collision_model() { return collision_model_.get(); }
  const pinocchio::GeometryModel *collision_model() const {
    return collision_model_.get();
  }
  pinocchio::GeometryData *collision_data() { return collision_data_.get(); }
  const pinocchio::GeometryData *collision_data() const {
    return collision_data_.get();
  }
  pinocchio::GeometryData *visual_data() { return visual_data_.get(); }
  const pinocchio::GeometryData *visual_data() const {
    return visual_data_.get();
  }

  /**
   * @brief Get the names of all collision geometry objects (empty if none
   * loaded).
   */
  std::vector<std::string> get_collision_geometry_names() const;

  /**
   * @brief Get the names of all collision pairs (as "object_a|object_b").
   */
  std::vector<std::pair<std::string, std::string>>
  get_collision_pair_names() const;

  /**
   * @brief Check whether collision geometry is available.
   */
  bool has_collision_geometry() const {
    return collision_model_ != nullptr && collision_data_ != nullptr;
  }

  /**
   * @brief Compute minimum collision distance at current configuration.
   *
   * Updates geometry placements and computes distances for all collision pairs,
   * returning the minimum distance found.
   *
   * @return Minimum collision distance, or infinity if no collision geometry
   */
  double compute_min_collision_distance() const;

  /**
   * @brief Compute collision distances for all pairs at current configuration.
   *
   * Updates geometry placements and computes distances for all collision pairs.
   *
   * @return Vector of distances for each collision pair, in order of
   * collision_model.collisionPairs
   */
  std::vector<double> compute_collision_distances() const;

  // Access to URDF path for visualization
  const std::string &urdf_path() const { return urdf_path_; }

  // Access to controlled joints (for visualization)
  std::vector<std::string> controlled_joint_names;
  std::unordered_map<std::string, int> controlled_joint_indices;

  /**
   * @brief Apply collision exclusions described as SRDF disable pairs.
   * @param collision_pairs Pairs of collision object names to disable.
   */
  void apply_collision_exclusions(
      const std::vector<std::pair<std::string, std::string>> &collision_pairs);

private:
  /**
   * @brief Get frame index from name
   * @param frame_name Name of the frame
   * @return Frame index
   * @throws std::runtime_error if frame not found
   */
  FrameIndex get_frame_id(const std::string &frame_name) const;

  /**
   * @brief Build frame name to index mapping
   */
  void build_frame_map();

  // Pinocchio model and data
  Model model_;
  mutable Data data_;

  // Current state
  Eigen::VectorXd current_q_;
  Eigen::VectorXd current_v_;

  // Frame name to index mapping for fast lookup
  std::unordered_map<std::string, FrameIndex> frame_map_;

  // Robot properties
  bool floating_base_;
  std::string urdf_path_;

  // Custom limits (optional)
  std::optional<Eigen::VectorXd> custom_acceleration_limits_;

  // Flags for lazy evaluation
  mutable bool kinematics_updated_ = false;
  mutable bool jacobians_updated_ = false;
  mutable bool com_updated_ = false;

  // Optional geometry models
  std::unique_ptr<pinocchio::GeometryModel> visual_model_;
  std::unique_ptr<pinocchio::GeometryModel> collision_model_;
  mutable std::unique_ptr<pinocchio::GeometryData> visual_data_;
  mutable std::unique_ptr<pinocchio::GeometryData> collision_data_;
};

} // namespace embodik
