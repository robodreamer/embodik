/*
 * MIT License
 *
 * Copyright (c) 2025 Andy Park <andypark.purdue@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
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
