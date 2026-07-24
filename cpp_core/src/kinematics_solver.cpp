/**
 * @file kinematics_solver.cpp
 * @brief Implementation of high-level kinematics solver
 */

#include <stdexcept>
#include <utility>

#include <embodik/kinematics_solver.hpp>

namespace embodik {

KinematicsSolver::KinematicsSolver(std::shared_ptr<RobotModel> robot)
    : robot_(robot) {
  if (!robot_) {
    throw std::invalid_argument("Robot model cannot be null");
  }
}

} // namespace embodik
