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

#include <Eigen/Geometry>
#include <algorithm>
#include <cstdlib>
#include <embodik/robot_model.hpp>
#include <fstream>
#include <iostream>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/collision/distance.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace embodik {

RobotModel::RobotModel(const std::string &urdf_path, bool floating_base)
    : floating_base_(floating_base), urdf_path_(urdf_path) {

  // Check if file exists
  std::ifstream file(urdf_path);
  if (!file.good()) {
    throw std::runtime_error("URDF file not found: " + urdf_path);
  }
  file.close();

  // Build model from URDF
  if (floating_base) {
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(),
                                model_);
  } else {
    pinocchio::urdf::buildModel(urdf_path, model_);
  }

  // Create data structure
  data_ = pinocchio::Data(model_);

  // Try to load geometry models independently so broken visuals do not
  // suppress collision support.
  std::string package_dir =
      urdf_path.substr(0, urdf_path.find_last_of("/\\"));

  try {
    visual_model_ = std::make_unique<pinocchio::GeometryModel>();
    pinocchio::urdf::buildGeom(model_, urdf_path, pinocchio::VISUAL,
                               *visual_model_, package_dir);
    visual_data_ = std::make_unique<pinocchio::GeometryData>(*visual_model_);
  } catch (const std::exception &) {
    visual_model_.reset();
    visual_data_.reset();
  }

  try {
    collision_model_ = std::make_unique<pinocchio::GeometryModel>();
    pinocchio::urdf::buildGeom(model_, urdf_path, pinocchio::COLLISION,
                               *collision_model_, package_dir);
    if (collision_model_->collisionPairs.empty()) {
      collision_model_->addAllCollisionPairs();
    }
    collision_data_ =
        std::make_unique<pinocchio::GeometryData>(*collision_model_);
  } catch (const std::exception &) {
    collision_model_.reset();
    collision_data_.reset();
  }

  // Initialize configuration vectors
  current_q_ = pinocchio::neutral(model_);
  current_v_ = Eigen::VectorXd::Zero(model_.nv);

  // Build frame mapping
  build_frame_map();

  // Initial forward kinematics
  update_configuration(current_q_);
}

namespace {
template <typename T>
bool is_in_vector(const std::vector<T> &vec, const T &elt) {
  return std::find(vec.begin(), vec.end(), elt) != vec.end();
}
}  // namespace

RobotModel::RobotModel(const std::string &urdf_path,
                       const std::vector<std::string> &actuated_joint_names,
                       bool floating_base)
    : floating_base_(floating_base), urdf_path_(urdf_path) {
  // Check if file exists
  std::ifstream file(urdf_path);
  if (!file.good()) {
    throw std::runtime_error("URDF file not found: " + urdf_path);
  }
  file.close();

  // Build full model from URDF
  Model full_model;
  if (floating_base) {
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(),
                               full_model);
  } else {
    pinocchio::urdf::buildModel(urdf_path, full_model);
  }

  // Validate actuated joint names and build set of actuated joint IDs
  std::vector<JointIndex> actuated_joint_ids;
  actuated_joint_ids.reserve(actuated_joint_names.size());
  for (const std::string &name : actuated_joint_names) {
    if (!full_model.existJointName(name)) {
      std::ostringstream oss;
      oss << "Actuated joint '" << name << "' not found in model.";
      throw std::runtime_error(oss.str());
    }
    actuated_joint_ids.push_back(full_model.getJointId(name));
  }

  // Build list of joints to lock: all joints not in actuated set.
  // Skip universe (j=0) always. For floating base, also skip the root
  // freeflyer (j=1) — it must remain unlocked.
  JointIndex start_j = floating_base ? 2 : 1;
  std::vector<JointIndex> joints_to_lock;
  for (JointIndex j = start_j;
       j < static_cast<JointIndex>(full_model.njoints); ++j) {
    if (!is_in_vector(actuated_joint_ids, j)) {
      joints_to_lock.push_back(j);
    }
  }

  Eigen::VectorXd reference_config = pinocchio::neutral(full_model);

  std::string package_dir =
      urdf_path.substr(0, urdf_path.find_last_of("/\\"));
  pinocchio::buildReducedModel(full_model, joints_to_lock, reference_config,
                               model_);
  data_ = pinocchio::Data(model_);

  try {
    visual_model_ = std::make_unique<pinocchio::GeometryModel>();
    pinocchio::urdf::buildGeom(model_, urdf_path, pinocchio::VISUAL,
                               *visual_model_, package_dir);
    visual_data_ = std::make_unique<pinocchio::GeometryData>(*visual_model_);
  } catch (const std::exception &) {
    visual_model_.reset();
    visual_data_.reset();
  }

  try {
    collision_model_ = std::make_unique<pinocchio::GeometryModel>();
    pinocchio::urdf::buildGeom(model_, urdf_path, pinocchio::COLLISION,
                               *collision_model_, package_dir);
    if (collision_model_->collisionPairs.empty()) {
      collision_model_->addAllCollisionPairs();
    }
    collision_data_ =
        std::make_unique<pinocchio::GeometryData>(*collision_model_);
  } catch (const std::exception &) {
    collision_model_.reset();
    collision_data_.reset();
  }

  // Initialize configuration vectors
  current_q_ = pinocchio::neutral(model_);
  current_v_ = Eigen::VectorXd::Zero(model_.nv);

  // Build frame mapping
  build_frame_map();

  // Initial forward kinematics
  update_configuration(current_q_);
}

std::unique_ptr<RobotModel>
RobotModel::from_xacro(const std::string &xacro_path, bool floating_base) {
  // Check if xacro file exists
  std::ifstream file(xacro_path);
  if (!file.good()) {
    throw std::runtime_error("XACRO file not found: " + xacro_path);
  }
  file.close();

  // Process xacro to URDF using xacro command
  std::string temp_urdf =
      "/tmp/embodik_temp_" + std::to_string(std::rand()) + ".urdf";
  std::string command =
      "xacro " + xacro_path + " > " + temp_urdf + " 2>/dev/null";

  int result = std::system(command.c_str());
  if (result != 0) {
    throw std::runtime_error("Failed to process XACRO file: " + xacro_path);
  }

  // Create model from generated URDF
  auto model = std::make_unique<RobotModel>(temp_urdf, floating_base);

  // Clean up temporary file
  std::remove(temp_urdf.c_str());

  return model;
}

void RobotModel::update_configuration(const Eigen::VectorXd &q) {
  if (q.size() != model_.nq) {
    throw std::runtime_error("Configuration vector size mismatch. Expected " +
                             std::to_string(model_.nq) + ", got " +
                             std::to_string(q.size()));
  }

  current_q_ = q;
  current_v_.setZero();

  // Compute forward kinematics
  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);

  kinematics_updated_ = true;
  jacobians_updated_ = false;
  com_updated_ = false;
  jacobian_time_variation_updated_ = false;
  com_acceleration_updated_ = false;
}

void RobotModel::update_kinematics(const Eigen::VectorXd &q,
                                   const Eigen::VectorXd &v) {
  if (q.size() != model_.nq) {
    throw std::runtime_error("Configuration vector size mismatch. Expected " +
                             std::to_string(model_.nq) + ", got " +
                             std::to_string(q.size()));
  }

  bool has_velocity = (v.size() > 0);
  if (has_velocity && v.size() != model_.nv) {
    throw std::runtime_error("Velocity vector size mismatch. Expected " +
                             std::to_string(model_.nv) + ", got " +
                             std::to_string(v.size()));
  }

  current_q_ = q;
  current_v_ = has_velocity ? v : Eigen::VectorXd::Zero(model_.nv);

  // Compute forward kinematics
  if (has_velocity) {
    pinocchio::forwardKinematics(model_, data_, q, v);
  } else {
    pinocchio::forwardKinematics(model_, data_, q);
  }
  pinocchio::updateFramePlacements(model_, data_);

  kinematics_updated_ = true;
  jacobians_updated_ = false;
  com_updated_ = false;
  jacobian_time_variation_updated_ = false;
  com_acceleration_updated_ = false;
}

RobotModel::SE3
RobotModel::get_frame_pose(const std::string &frame_name) const {
  if (!kinematics_updated_) {
    throw std::runtime_error(
        "Kinematics not updated. Call update_configuration() first.");
  }

  FrameIndex frame_id = get_frame_id(frame_name);
  return data_.oMf[frame_id];
}

Eigen::Matrix<double, 6, Eigen::Dynamic>
RobotModel::get_frame_jacobian(const std::string &frame_name,
                               pinocchio::ReferenceFrame ref) const {

  if (!kinematics_updated_) {
    throw std::runtime_error(
        "Kinematics not updated. Call update_configuration() first.");
  }

  FrameIndex frame_id = get_frame_id(frame_name);

  if (!jacobians_updated_) {
    pinocchio::computeJointJacobians(model_, data_, current_q_);
    jacobians_updated_ = true;
  }

  Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, model_.nv);
  J.setZero();
  pinocchio::getFrameJacobian(model_, data_, frame_id, ref, J);

  return J;
}

Eigen::Matrix<double, 6, 1> RobotModel::get_frame_jacobian_bias(
    const std::string &frame_name, pinocchio::ReferenceFrame ref) const {
  if (!kinematics_updated_) {
    throw std::runtime_error(
        "Kinematics not updated. Call update_kinematics() first.");
  }

  if (current_v_.isZero(0.0)) {
    return Eigen::Matrix<double, 6, 1>::Zero();
  }

  const FrameIndex frame_id = get_frame_id(frame_name);
  if (!jacobian_time_variation_updated_) {
    pinocchio::computeJointJacobiansTimeVariation(model_, data_, current_q_,
                                                   current_v_);
    jacobian_time_variation_updated_ = true;
    jacobians_updated_ = true;
  }

  Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian_time_variation(6,
                                                                    model_.nv);
  jacobian_time_variation.setZero();
  pinocchio::getFrameJacobianTimeVariation(
      model_, data_, frame_id, ref, jacobian_time_variation);
  return jacobian_time_variation * current_v_;
}

Eigen::Matrix<double, 3, Eigen::Dynamic>
RobotModel::get_point_jacobian(const std::string &frame_name,
                               const Eigen::Vector3d &local_point) const {
  if (!kinematics_updated_) {
    throw std::runtime_error(
        "Kinematics not updated. Call update_configuration() first.");
  }

  FrameIndex frame_id = get_frame_id(frame_name);
  Eigen::Matrix<double, 6, Eigen::Dynamic> frame_jacobian =
      get_frame_jacobian(frame_name, pinocchio::LOCAL_WORLD_ALIGNED);

  Eigen::Matrix<double, 3, Eigen::Dynamic> linear = frame_jacobian.topRows<3>();
  const Eigen::Matrix<double, 3, Eigen::Dynamic> angular =
      frame_jacobian.bottomRows<3>();

  const Eigen::Vector3d point_world =
      data_.oMf[frame_id].rotation() * local_point;

  // Account for rotational contribution to point velocity: v = v_o + omega x r
  for (Eigen::Index col = 0; col < linear.cols(); ++col) {
    linear.col(col) += angular.col(col).cross(point_world);
  }

  return linear;
}

std::vector<std::string> RobotModel::get_collision_geometry_names() const {
  std::vector<std::string> names;
  if (!collision_model_) {
    return names;
  }
  names.reserve(collision_model_->geometryObjects.size());
  for (const auto &obj : collision_model_->geometryObjects) {
    names.push_back(obj.name);
  }
  return names;
}

std::vector<std::pair<std::string, std::string>>
RobotModel::get_collision_pair_names() const {
  std::vector<std::pair<std::string, std::string>> pairs;
  if (!collision_model_) {
    return pairs;
  }
  for (const auto &pair : collision_model_->collisionPairs) {
    const auto &obj_a = collision_model_->geometryObjects[pair.first];
    const auto &obj_b = collision_model_->geometryObjects[pair.second];
    pairs.emplace_back(obj_a.name, obj_b.name);
  }
  return pairs;
}

void RobotModel::apply_collision_exclusions(
    const std::vector<std::pair<std::string, std::string>> &collision_pairs) {
  if (!collision_model_ || collision_pairs.empty()) {
    return;
  }

  auto canonical_key = [](const std::string &a, const std::string &b) {
    return (a <= b) ? (a + "|" + b) : (b + "|" + a);
  };

  std::unordered_set<std::string> excluded;
  excluded.reserve(collision_pairs.size());
  for (const auto &pair : collision_pairs) {
    excluded.insert(canonical_key(pair.first, pair.second));
  }

  auto &pairs = collision_model_->collisionPairs;
  pairs.erase(
      std::remove_if(pairs.begin(), pairs.end(),
                     [&](const pinocchio::CollisionPair &pair) {
                       const auto &name_a =
                           collision_model_->geometryObjects[pair.first].name;
                       const auto &name_b =
                           collision_model_->geometryObjects[pair.second].name;
                       return excluded.find(canonical_key(name_a, name_b)) !=
                              excluded.end();
                     }),
      pairs.end());

  collision_data_ =
      std::make_unique<pinocchio::GeometryData>(*collision_model_);
}

double RobotModel::compute_min_collision_distance() const {
  if (!has_collision_geometry()) {
    return std::numeric_limits<double>::infinity();
  }

  if (!kinematics_updated_) {
    throw std::runtime_error(
        "Kinematics not updated. Call update_configuration() first.");
  }

  // Update geometry placements
  pinocchio::updateGeometryPlacements(model_, data_, *collision_model_,
                                      *collision_data_);

  double min_distance = std::numeric_limits<double>::infinity();
  const auto &pairs = collision_model_->collisionPairs;

  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    // Honor active pair mask
    if (!collision_data_->activeCollisionPairs.empty() &&
        !collision_data_->activeCollisionPairs[idx]) {
      continue;
    }

    pinocchio::computeDistance(*collision_model_, *collision_data_, idx);
    double distance = collision_data_->distanceResults[idx].min_distance;

    if (std::isfinite(distance) && distance < min_distance) {
      min_distance = distance;
    }
  }

  return min_distance;
}

std::vector<double> RobotModel::compute_collision_distances() const {
  std::vector<double> distances;

  if (!has_collision_geometry()) {
    return distances;
  }

  if (!kinematics_updated_) {
    throw std::runtime_error(
        "Kinematics not updated. Call update_configuration() first.");
  }

  // Update geometry placements
  pinocchio::updateGeometryPlacements(model_, data_, *collision_model_,
                                      *collision_data_);

  const auto &pairs = collision_model_->collisionPairs;
  distances.reserve(pairs.size());

  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    // Honor active pair mask - return infinity for inactive pairs
    if (!collision_data_->activeCollisionPairs.empty() &&
        !collision_data_->activeCollisionPairs[idx]) {
      distances.push_back(std::numeric_limits<double>::infinity());
      continue;
    }

    pinocchio::computeDistance(*collision_model_, *collision_data_, idx);
    double distance = collision_data_->distanceResults[idx].min_distance;
    distances.push_back(distance);
  }

  return distances;
}

Eigen::Vector3d RobotModel::get_com_position() const {
  if (!kinematics_updated_) {
    throw std::runtime_error(
        "Kinematics not updated. Call update_configuration() first.");
  }

  if (!com_updated_) {
    pinocchio::centerOfMass(model_, data_, current_q_, false);
    com_updated_ = true;
  }

  return data_.com[0];
}

Eigen::Vector3d RobotModel::get_com_velocity() const {
  if (!kinematics_updated_) {
    throw std::runtime_error(
        "Kinematics not updated. Call update_configuration() first.");
  }

  if (current_v_.isZero()) {
    return Eigen::Vector3d::Zero();
  }

  if (!com_updated_) {
    pinocchio::centerOfMass(model_, data_, current_q_, current_v_, false);
    com_updated_ = true;
  }

  return data_.vcom[0];
}

Eigen::Matrix<double, 3, Eigen::Dynamic> RobotModel::get_com_jacobian() const {
  if (!kinematics_updated_) {
    throw std::runtime_error(
        "Kinematics not updated. Call update_configuration() first.");
  }

  Eigen::Matrix<double, 3, Eigen::Dynamic> Jcom(3, model_.nv);
  pinocchio::jacobianCenterOfMass(model_, data_, current_q_, false);
  Jcom = data_.Jcom;

  return Jcom;
}

Eigen::Vector3d RobotModel::get_com_jacobian_bias() const {
  if (!kinematics_updated_) {
    throw std::runtime_error(
        "Kinematics not updated. Call update_kinematics() first.");
  }

  if (current_v_.isZero(0.0)) {
    return Eigen::Vector3d::Zero();
  }

  if (!com_acceleration_updated_) {
    pinocchio::centerOfMass(model_, data_, current_q_, current_v_,
                            Eigen::VectorXd::Zero(model_.nv), false);
    com_updated_ = true;
    com_acceleration_updated_ = true;
  }
  return data_.acom[0];
}

std::vector<std::string> RobotModel::get_frame_names() const {
  std::vector<std::string> names;
  names.reserve(model_.frames.size());

  for (const auto &frame : model_.frames) {
    names.push_back(frame.name);
  }

  return names;
}

std::vector<std::string> RobotModel::get_joint_names() const {
  std::vector<std::string> names;
  names.reserve(model_.joints.size());

  for (size_t i = 1; i < model_.joints.size(); ++i) { // Skip universe joint
    names.push_back(model_.names[i]);
  }

  return names;
}

bool RobotModel::has_joint(const std::string &joint_name) const {
  return model_.existJointName(joint_name);
}

RobotModel::JointIndex
RobotModel::get_joint_id(const std::string &joint_name) const {
  if (!model_.existJointName(joint_name)) {
    throw std::runtime_error("Joint not found: " + joint_name);
  }
  return model_.getJointId(joint_name);
}

int RobotModel::get_joint_config_index(const std::string &joint_name) const {
  JointIndex jid = get_joint_id(joint_name);
  return model_.idx_qs[jid];
}

int RobotModel::get_joint_config_size(const std::string &joint_name) const {
  JointIndex jid = get_joint_id(joint_name);
  return model_.nqs[jid];
}

int RobotModel::get_joint_velocity_index(const std::string &joint_name) const {
  JointIndex jid = get_joint_id(joint_name);
  return model_.idx_vs[jid];
}

int RobotModel::get_joint_velocity_size(const std::string &joint_name) const {
  JointIndex jid = get_joint_id(joint_name);
  return model_.nvs[jid];
}

Eigen::VectorXd RobotModel::integrate(const Eigen::VectorXd &q,
                                      const Eigen::VectorXd &v,
                                      double dt) const {
  if (q.size() != model_.nq) {
    throw std::runtime_error("Configuration vector size mismatch. Expected " +
                             std::to_string(model_.nq) + ", got " +
                             std::to_string(q.size()));
  }
  if (v.size() != model_.nv) {
    throw std::runtime_error("Velocity vector size mismatch. Expected " +
                             std::to_string(model_.nv) + ", got " +
                             std::to_string(v.size()));
  }
  return pinocchio::integrate(model_, q, v * dt);
}

Eigen::VectorXd RobotModel::difference(const Eigen::VectorXd &q0,
                                       const Eigen::VectorXd &q1) const {
  if (q0.size() != model_.nq) {
    throw std::runtime_error(
        "q0 configuration vector size mismatch. Expected " +
        std::to_string(model_.nq) + ", got " + std::to_string(q0.size()));
  }
  if (q1.size() != model_.nq) {
    throw std::runtime_error(
        "q1 configuration vector size mismatch. Expected " +
        std::to_string(model_.nq) + ", got " + std::to_string(q1.size()));
  }
  return pinocchio::difference(model_, q0, q1);
}

Eigen::VectorXd RobotModel::neutral_configuration() const {
  return pinocchio::neutral(model_);
}

Eigen::VectorXd RobotModel::random_configuration() const {
  return pinocchio::randomConfiguration(model_);
}

Eigen::VectorXd RobotModel::normalize(const Eigen::VectorXd &q) const {
  if (q.size() != model_.nq) {
    throw std::runtime_error("Configuration vector size mismatch. Expected " +
                             std::to_string(model_.nq) + ", got " +
                             std::to_string(q.size()));
  }
  Eigen::VectorXd q_out = q;
  pinocchio::normalize(model_, q_out);
  return q_out;
}

// =========================================================================
// Inverse dynamics / gravity
// =========================================================================

Eigen::VectorXd
RobotModel::compute_generalized_gravity(const Eigen::VectorXd &q) const {
  if (q.size() != model_.nq) {
    throw std::runtime_error("Configuration vector size mismatch. Expected " +
                             std::to_string(model_.nq) + ", got " +
                             std::to_string(q.size()));
  }
  return pinocchio::computeGeneralizedGravity(model_, data_, q);
}

Eigen::VectorXd RobotModel::rnea(const Eigen::VectorXd &q,
                                 const Eigen::VectorXd &v,
                                 const Eigen::VectorXd &a) const {
  if (q.size() != model_.nq) {
    throw std::runtime_error("Configuration vector size mismatch. Expected " +
                             std::to_string(model_.nq) + ", got " +
                             std::to_string(q.size()));
  }
  if (v.size() != model_.nv) {
    throw std::runtime_error("Velocity vector size mismatch. Expected " +
                             std::to_string(model_.nv) + ", got " +
                             std::to_string(v.size()));
  }
  if (a.size() != model_.nv) {
    throw std::runtime_error("Acceleration vector size mismatch. Expected " +
                             std::to_string(model_.nv) + ", got " +
                             std::to_string(a.size()));
  }
  return pinocchio::rnea(model_, data_, q, v, a);
}

Eigen::MatrixXd
RobotModel::compute_mass_matrix(const Eigen::VectorXd &q) const {
  if (q.size() != model_.nq) {
    throw std::runtime_error("Configuration vector size mismatch. Expected " +
                             std::to_string(model_.nq) + ", got " +
                             std::to_string(q.size()));
  }
  pinocchio::crba(model_, data_, q);
  // CRBA only fills upper triangle; symmetrise
  data_.M.triangularView<Eigen::StrictlyLower>() =
      data_.M.transpose().triangularView<Eigen::StrictlyLower>();
  return data_.M;
}

Eigen::VectorXd RobotModel::compute_coriolis(const Eigen::VectorXd &q,
                                             const Eigen::VectorXd &v) const {
  if (q.size() != model_.nq) {
    throw std::runtime_error("Configuration vector size mismatch. Expected " +
                             std::to_string(model_.nq) + ", got " +
                             std::to_string(q.size()));
  }
  if (v.size() != model_.nv) {
    throw std::runtime_error("Velocity vector size mismatch. Expected " +
                             std::to_string(model_.nv) + ", got " +
                             std::to_string(v.size()));
  }
  Eigen::VectorXd a_zero = Eigen::VectorXd::Zero(model_.nv);
  Eigen::VectorXd tau_cv_g = pinocchio::rnea(model_, data_, q, v, a_zero);
  Eigen::VectorXd g = pinocchio::computeGeneralizedGravity(model_, data_, q);
  return tau_cv_g - g;
}

void RobotModel::set_gravity(const Eigen::Vector3d &gravity) {
  model_.gravity.linear() = gravity;
}

Eigen::Vector3d RobotModel::get_gravity() const {
  return model_.gravity.linear();
}

// =========================================================================
// Boolean collision checking
// =========================================================================

bool RobotModel::check_collision() const { return check_collision(0.0); }

bool RobotModel::check_collision(double min_distance) const {
  if (!has_collision_geometry()) {
    return false;
  }

  if (!kinematics_updated_) {
    throw std::runtime_error(
        "Kinematics not updated. Call update_configuration() first.");
  }

  pinocchio::updateGeometryPlacements(model_, data_, *collision_model_,
                                      *collision_data_);

  const auto &pairs = collision_model_->collisionPairs;
  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    if (!collision_data_->activeCollisionPairs.empty() &&
        !collision_data_->activeCollisionPairs[idx]) {
      continue;
    }
    pinocchio::computeDistance(*collision_model_, *collision_data_, idx);
    double distance = collision_data_->distanceResults[idx].min_distance;
    if (std::isfinite(distance) && distance <= min_distance) {
      return true;
    }
  }
  return false;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
RobotModel::get_joint_limits() const {
  return std::make_pair(model_.lowerPositionLimit, model_.upperPositionLimit);
}

void RobotModel::set_joint_limits(const Eigen::VectorXd &lower,
                                  const Eigen::VectorXd &upper) {
  if (lower.size() != model_.nq || upper.size() != model_.nq) {
    throw std::invalid_argument("Joint limit vectors must have size nq (" +
                                std::to_string(model_.nq) + ")");
  }

  for (Eigen::Index i = 0; i < lower.size(); ++i) {
    if (lower[i] > upper[i]) {
      throw std::invalid_argument(
          "Joint limit lower bound greater than upper bound at index " +
          std::to_string(i));
    }
  }

  model_.lowerPositionLimit = lower;
  model_.upperPositionLimit = upper;

  // Clamp current configuration to new limits
  current_q_ = current_q_.cwiseMin(upper).cwiseMax(lower);
}

Eigen::VectorXd RobotModel::get_velocity_limits() const {
  return model_.velocityLimit;
}

Eigen::VectorXd RobotModel::get_acceleration_limits() const {
  if (custom_acceleration_limits_.has_value()) {
    return custom_acceleration_limits_.value();
  }
  // Most URDF files don't specify acceleration limits
  // Return a default high value (100 rad/s^2 or m/s^2)
  // Users can override this if they have specific acceleration limits
  return Eigen::VectorXd::Constant(model_.nv, 100.0);
}

bool RobotModel::has_custom_acceleration_limits() const {
  return custom_acceleration_limits_.has_value();
}

void RobotModel::set_acceleration_limits(const Eigen::VectorXd &accel_limits) {
  if (accel_limits.size() != model_.nv) {
    throw std::invalid_argument(
        "Acceleration limits size must match number of velocity DoFs (nv)");
  }
  custom_acceleration_limits_ = accel_limits;
}

Eigen::VectorXd RobotModel::get_effort_limits() const {
  return model_.effortLimit;
}

bool RobotModel::has_frame(const std::string &frame_name) const {
  return frame_map_.find(frame_name) != frame_map_.end();
}

RobotModel::FrameIndex
RobotModel::get_frame_id(const std::string &frame_name) const {
  auto it = frame_map_.find(frame_name);
  if (it == frame_map_.end()) {
    throw std::runtime_error("Frame not found: " + frame_name);
  }
  return it->second;
}

void RobotModel::build_frame_map() {
  frame_map_.clear();

  for (size_t i = 0; i < model_.frames.size(); ++i) {
    frame_map_[model_.frames[i].name] = static_cast<FrameIndex>(i);
  }
}

} // namespace embodik
