#include <embodik/kinematics_solver.hpp>

int main() {
  embodik::AccelerationSolver *solver = nullptr;
  embodik::PositionStepOptions position_step_options;
  position_step_options.current_joint_velocity = Eigen::VectorXd::Zero(1);
  return solver == nullptr &&
                 position_step_options.current_joint_velocity.size() == 1
             ? 0
             : 1;
}
