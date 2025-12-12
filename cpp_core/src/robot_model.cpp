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
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <embodik/robot_model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <Eigen/Geometry>

namespace embodik {

RobotModel::RobotModel(const std::string& urdf_path, bool floating_base)
    : floating_base_(floating_base), urdf_path_(urdf_path) {

    // Check if file exists
    std::ifstream file(urdf_path);
    if (!file.good()) {
        throw std::runtime_error("URDF file not found: " + urdf_path);
    }
    file.close();

    // Build model from URDF
    if (floating_base) {
        pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model_);
    } else {
        pinocchio::urdf::buildModel(urdf_path, model_);
    }

    // Create data structure
    data_ = pinocchio::Data(model_);

    // Try to load geometry models (optional - may fail if meshes not found)
    try {
        // Extract directory path from URDF path
        std::string package_dir = urdf_path.substr(0, urdf_path.find_last_of("/\\"));

        // Load visual geometry
        visual_model_ = std::make_unique<pinocchio::GeometryModel>();
        pinocchio::urdf::buildGeom(model_, urdf_path, pinocchio::VISUAL,
                                  *visual_model_, package_dir);
        visual_data_ = std::make_unique<pinocchio::GeometryData>(*visual_model_);

        // Load collision geometry
        collision_model_ = std::make_unique<pinocchio::GeometryModel>();
        pinocchio::urdf::buildGeom(model_, urdf_path, pinocchio::COLLISION,
                                  *collision_model_, package_dir);
        if (collision_model_->collisionPairs.empty()) {
            collision_model_->addAllCollisionPairs();
        }
        collision_data_ = std::make_unique<pinocchio::GeometryData>(*collision_model_);
    } catch (const std::exception& e) {
        // Geometry loading is optional - continue without it
        visual_model_.reset();
        visual_data_.reset();
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

std::unique_ptr<RobotModel> RobotModel::from_xacro(const std::string& xacro_path,
                                                    bool floating_base) {
    // Check if xacro file exists
    std::ifstream file(xacro_path);
    if (!file.good()) {
        throw std::runtime_error("XACRO file not found: " + xacro_path);
    }
    file.close();

    // Process xacro to URDF using xacro command
    std::string temp_urdf = "/tmp/embodik_temp_" + std::to_string(std::rand()) + ".urdf";
    std::string command = "xacro " + xacro_path + " > " + temp_urdf + " 2>/dev/null";

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

void RobotModel::update_configuration(const Eigen::VectorXd& q) {
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
}

void RobotModel::update_kinematics(const Eigen::VectorXd& q,
                                  const Eigen::VectorXd& v) {
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
}

RobotModel::SE3 RobotModel::get_frame_pose(const std::string& frame_name) const {
    if (!kinematics_updated_) {
        throw std::runtime_error("Kinematics not updated. Call update_configuration() first.");
    }

    FrameIndex frame_id = get_frame_id(frame_name);
    return data_.oMf[frame_id];
}

Eigen::Matrix<double, 6, Eigen::Dynamic> RobotModel::get_frame_jacobian(
    const std::string& frame_name,
    pinocchio::ReferenceFrame ref) const {

    if (!kinematics_updated_) {
        throw std::runtime_error("Kinematics not updated. Call update_configuration() first.");
    }

    FrameIndex frame_id = get_frame_id(frame_name);

    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, model_.nv);
    J.setZero();

    // Compute frame Jacobian
    pinocchio::computeFrameJacobian(model_, data_, current_q_, frame_id, ref, J);

    return J;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> RobotModel::get_point_jacobian(
    const std::string& frame_name,
    const Eigen::Vector3d& local_point) const {
    if (!kinematics_updated_) {
        throw std::runtime_error("Kinematics not updated. Call update_configuration() first.");
    }

    FrameIndex frame_id = get_frame_id(frame_name);
    Eigen::Matrix<double, 6, Eigen::Dynamic> frame_jacobian =
        get_frame_jacobian(frame_name, pinocchio::LOCAL_WORLD_ALIGNED);

    Eigen::Matrix<double, 3, Eigen::Dynamic> linear = frame_jacobian.topRows<3>();
    const Eigen::Matrix<double, 3, Eigen::Dynamic> angular = frame_jacobian.bottomRows<3>();

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
    for (const auto& obj : collision_model_->geometryObjects) {
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
    for (const auto& pair : collision_model_->collisionPairs) {
        const auto& obj_a = collision_model_->geometryObjects[pair.first];
        const auto& obj_b = collision_model_->geometryObjects[pair.second];
        pairs.emplace_back(obj_a.name, obj_b.name);
    }
    return pairs;
}

void RobotModel::apply_collision_exclusions(
    const std::vector<std::pair<std::string, std::string>>& collision_pairs) {
    if (!collision_model_ || collision_pairs.empty()) {
        return;
    }

    auto canonical_key = [](const std::string& a, const std::string& b) {
        return (a <= b) ? (a + "|" + b) : (b + "|" + a);
    };

    std::unordered_set<std::string> excluded;
    excluded.reserve(collision_pairs.size());
    for (const auto& pair : collision_pairs) {
        excluded.insert(canonical_key(pair.first, pair.second));
    }

    auto& pairs = collision_model_->collisionPairs;
    pairs.erase(
        std::remove_if(
            pairs.begin(), pairs.end(),
            [&](const pinocchio::CollisionPair& pair) {
                const auto& name_a = collision_model_->geometryObjects[pair.first].name;
                const auto& name_b = collision_model_->geometryObjects[pair.second].name;
                return excluded.find(canonical_key(name_a, name_b)) != excluded.end();
            }),
        pairs.end());

    collision_data_ = std::make_unique<pinocchio::GeometryData>(*collision_model_);
}

Eigen::Vector3d RobotModel::get_com_position() const {
    if (!kinematics_updated_) {
        throw std::runtime_error("Kinematics not updated. Call update_configuration() first.");
    }

    if (!com_updated_) {
        pinocchio::centerOfMass(model_, data_, current_q_, false);
        com_updated_ = true;
    }

    return data_.com[0];
}

Eigen::Vector3d RobotModel::get_com_velocity() const {
    if (!kinematics_updated_) {
        throw std::runtime_error("Kinematics not updated. Call update_configuration() first.");
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
        throw std::runtime_error("Kinematics not updated. Call update_configuration() first.");
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> Jcom(3, model_.nv);
    pinocchio::jacobianCenterOfMass(model_, data_, current_q_, false);
    Jcom = data_.Jcom;

    return Jcom;
}

std::vector<std::string> RobotModel::get_frame_names() const {
    std::vector<std::string> names;
    names.reserve(model_.frames.size());

    for (const auto& frame : model_.frames) {
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

std::pair<Eigen::VectorXd, Eigen::VectorXd> RobotModel::get_joint_limits() const {
    return std::make_pair(model_.lowerPositionLimit, model_.upperPositionLimit);
}

void RobotModel::set_joint_limits(const Eigen::VectorXd& lower,
                                  const Eigen::VectorXd& upper) {
    if (lower.size() != model_.nq || upper.size() != model_.nq) {
        throw std::invalid_argument(
            "Joint limit vectors must have size nq (" + std::to_string(model_.nq) + ")");
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

void RobotModel::set_acceleration_limits(const Eigen::VectorXd& accel_limits) {
    if (accel_limits.size() != model_.nv) {
        throw std::invalid_argument("Acceleration limits size must match number of velocity DoFs (nv)");
    }
    custom_acceleration_limits_ = accel_limits;
}

Eigen::VectorXd RobotModel::get_effort_limits() const {
    return model_.effortLimit;
}

bool RobotModel::has_frame(const std::string& frame_name) const {
    return frame_map_.find(frame_name) != frame_map_.end();
}

RobotModel::FrameIndex RobotModel::get_frame_id(const std::string& frame_name) const {
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
