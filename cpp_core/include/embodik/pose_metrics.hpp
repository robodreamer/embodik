#pragma once
#include <Eigen/Dense>
#include <utility>

namespace embodik {

std::pair<Eigen::VectorXd, double> joint_limit_distance(
    const Eigen::VectorXd &q,
    const Eigen::VectorXd &q_lower,
    const Eigen::VectorXd &q_upper,
    double epsilon = 0.04);

Eigen::VectorXd joint_limit_distance_gradient(
    const Eigen::VectorXd &q,
    const Eigen::VectorXd &q_lower,
    const Eigen::VectorXd &q_upper,
    double epsilon = 0.04);

double velocity_manipulability(const Eigen::MatrixXd &jacobian);

double singularity_joint_limit_metric(
    const Eigen::VectorXd &q,
    const Eigen::MatrixXd &jacobian,
    const Eigen::VectorXd &q_lower,
    const Eigen::VectorXd &q_upper,
    double epsilon = 0.04);

} // namespace embodik
