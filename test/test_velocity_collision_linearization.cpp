#include "velocity_constraint_lift.hpp"
#include "velocity_constraint_test_observer.hpp"

#include <embodik/kinematics_solver.hpp>
#include <embodik/robot_model.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>

namespace embodik::test {
namespace {

constexpr double kTolerance = 1e-10;

void create_collision_urdf(const std::string &filename) {
  std::ofstream file(filename);
  file << R"(<?xml version="1.0"?>
<robot name="collision_linearization_test_robot">
  <link name="world"/>

  <link name="obstacle">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="obstacle_fixed" type="fixed">
    <parent link="world"/>
    <child link="obstacle"/>
  </joint>

  <link name="moving">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/>
    <child link="moving"/>
    <origin xyz="0.12 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.1" upper="0.1" effort="100" velocity="1"/>
  </joint>
</robot>)";
}

void create_multi_collision_urdf(const std::string &filename) {
  std::ofstream file(filename);
  file << R"(<?xml version="1.0"?>
<robot name="multi_collision_linearization_test_robot">
  <link name="base">
    <collision name="left_sphere">
      <origin xyz="-0.14 0 0"/>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
    <collision name="right_sphere">
      <origin xyz="0.17 0 0"/>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>

  <link name="moving">
    <collision name="moving_sphere">
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="base"/>
    <child link="moving"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.1" upper="0.1" effort="100" velocity="1"/>
  </joint>
</robot>)";
}

class VelocityCollisionLinearizationTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_path_ = "/tmp/embodik_velocity_collision_linearization.urdf";
    multi_urdf_path_ =
        "/tmp/embodik_multi_velocity_collision_linearization.urdf";
    create_collision_urdf(urdf_path_);
    create_multi_collision_urdf(multi_urdf_path_);
  }

  void TearDown() override {
    std::remove(urdf_path_.c_str());
    std::remove(multi_urdf_path_.c_str());
  }

  std::shared_ptr<RobotModel> make_robot(const std::string &path) const {
    auto robot = std::make_shared<RobotModel>(path, false);
    Eigen::VectorXd q(1);
    q << 0.0;
    robot->update_configuration(q);
    return robot;
  }

  KinematicsSolver make_solver(std::shared_ptr<RobotModel> robot,
                               double solver_dt,
                               double min_distance = 0.04) const {
    KinematicsSolver solver(robot);
    solver.set_dt(solver_dt);
    solver.configure_collision_constraint(min_distance, {}, {}, true, 1);
    solver.set_proximity_gated_collision_activation_enabled(false);
    solver.set_collision_tuning_mode(CollisionTuningMode::kPrecise);
    return solver;
  }

  std::string urdf_path_;
  std::string multi_urdf_path_;
};

TEST(VelocityConstraintLiftTest,
     LiftedBoundsAreEquivalentToNextVelocityRows) {
  Eigen::MatrixXd A(2, 3);
  A << 1.0, -2.0, 0.5,
       -0.25, 0.75, 1.5;
  Eigen::VectorXd dq(3);
  dq << 0.3, -0.1, 0.2;
  Eigen::VectorXd ddq(3);
  ddq << -0.5, 0.25, 0.1;
  const Eigen::VectorXd row_velocity = A * dq;
  Eigen::VectorXd lower(2);
  lower << row_velocity[0] - 0.2, row_velocity[1] - 0.3;
  Eigen::VectorXd upper(2);
  upper << row_velocity[0] + 0.4, row_velocity[1] + 0.5;
  const double dt = 0.02;

  const auto lifted = detail::lift_velocity_constraint_to_acceleration(
      A, lower, upper, dq, dt);

  EXPECT_TRUE(lifted.coefficient_matrix.isApprox(A, kTolerance));
  EXPECT_TRUE(((A * ddq).array() >=
               (lifted.lower_bounds.array() - kTolerance))
                  .all());
  EXPECT_TRUE(((A * ddq).array() <=
               (lifted.upper_bounds.array() + kTolerance))
                  .all());
  EXPECT_TRUE(((A * (dq + dt * ddq)).array() >=
               (lower.array() - kTolerance))
                  .all());
  EXPECT_TRUE(((A * (dq + dt * ddq)).array() <=
               (upper.array() + kTolerance))
                  .all());
  EXPECT_TRUE(lifted.lower_bounds.isApprox(
      (lower - A * dq) / dt, kTolerance));
  EXPECT_TRUE(lifted.upper_bounds.isApprox(
      (upper - A * dq) / dt, kTolerance));
}

TEST(VelocityConstraintLiftTest, RejectsInvalidInputs) {
  const Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2, 2);
  const Eigen::VectorXd bounds = Eigen::VectorXd::Zero(2);
  const Eigen::VectorXd dq = Eigen::VectorXd::Zero(2);

  EXPECT_THROW(detail::lift_velocity_constraint_to_acceleration(
                   A, bounds, bounds, dq, 0.0),
               std::invalid_argument);
  EXPECT_THROW(detail::lift_velocity_constraint_to_acceleration(
                   A, Eigen::VectorXd::Zero(1), bounds, dq, 0.01),
               std::invalid_argument);
  EXPECT_THROW(detail::lift_velocity_constraint_to_acceleration(
                   Eigen::MatrixXd(0, 2), Eigen::VectorXd{},
                   Eigen::VectorXd{}, dq, 0.01),
               std::invalid_argument);
  EXPECT_THROW(detail::lift_velocity_constraint_to_acceleration(
                   A, Eigen::Vector2d(1.0, -1.0),
                   Eigen::Vector2d(0.0, 1.0), dq, 0.01),
               std::invalid_argument);
}

TEST_F(VelocityCollisionLinearizationTest,
       LinearizationMatchesCanonicalCollisionRows) {
  auto direct_robot = make_robot(urdf_path_);
  auto linearized_robot = make_robot(urdf_path_);
  ASSERT_TRUE(direct_robot->has_collision_geometry());
  ASSERT_TRUE(linearized_robot->has_collision_geometry());

  auto direct_solver = make_solver(direct_robot, 0.01);
  auto linearized_solver = make_solver(linearized_robot, 0.01);

  const auto direct = detail::VelocityConstraintTestObserver::
      compute_collision_constraint(direct_solver, 0.01);
  const auto linearized = detail::VelocityConstraintTestObserver::
      linearize_collision_velocity_constraint(linearized_solver, 0.01);

  ASSERT_TRUE(direct.has_value());
  ASSERT_TRUE(linearized.has_value());
  EXPECT_TRUE(linearized->coefficient_matrix.isApprox(direct->jacobian,
                                                      kTolerance));
  EXPECT_TRUE(linearized->lower_bounds.isApprox(direct->lower_bounds,
                                                kTolerance));
  EXPECT_TRUE(linearized->upper_bounds.isApprox(direct->upper_bounds,
                                                kTolerance));
  EXPECT_NEAR(linearized->distance, direct->distance, kTolerance);
  EXPECT_EQ(linearized->object_a, direct->object_a);
  EXPECT_EQ(linearized->object_b, direct->object_b);
  EXPECT_EQ(linearized->coefficient_matrix.rows(), 1);
  EXPECT_EQ(linearized->coefficient_matrix.cols(), direct_robot->nv());
}

TEST_F(VelocityCollisionLinearizationTest,
       ExplicitRowDtScalesVelocityBoundsWithoutChangingRows) {
  auto fast_robot = make_robot(urdf_path_);
  auto slow_robot = make_robot(urdf_path_);
  ASSERT_TRUE(fast_robot->has_collision_geometry());
  ASSERT_TRUE(slow_robot->has_collision_geometry());

  auto fast_solver = make_solver(fast_robot, 0.01, 0.005);
  auto slow_solver = make_solver(slow_robot, 0.01, 0.005);

  const auto fast = detail::VelocityConstraintTestObserver::
      linearize_collision_velocity_constraint(fast_solver, 0.01);
  const auto slow = detail::VelocityConstraintTestObserver::
      linearize_collision_velocity_constraint(slow_solver, 0.02);

  ASSERT_TRUE(fast.has_value());
  ASSERT_TRUE(slow.has_value());
  EXPECT_TRUE(fast->coefficient_matrix.isApprox(slow->coefficient_matrix,
                                                kTolerance));
  EXPECT_TRUE((2.0 * slow->lower_bounds).isApprox(fast->lower_bounds,
                                                  1e-9));
  EXPECT_TRUE((2.0 * slow->upper_bounds).isApprox(fast->upper_bounds,
                                                  1e-9));
}

TEST_F(VelocityCollisionLinearizationTest,
       SameSolverExplicitRowDtDoesNotReuseStaleBounds) {
  auto robot = make_robot(urdf_path_);
  ASSERT_TRUE(robot->has_collision_geometry());
  auto solver = make_solver(robot, 0.01, 0.005);

  const auto first = detail::VelocityConstraintTestObserver::
      linearize_collision_velocity_constraint(solver, 0.01);
  const auto second = detail::VelocityConstraintTestObserver::
      linearize_collision_velocity_constraint(solver, 0.02);
  const auto third = detail::VelocityConstraintTestObserver::
      linearize_collision_velocity_constraint(solver, 0.01);
  const auto near_different = detail::VelocityConstraintTestObserver::
      linearize_collision_velocity_constraint(solver, 0.0100005);

  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  ASSERT_TRUE(third.has_value());
  ASSERT_TRUE(near_different.has_value());
  EXPECT_TRUE(first->coefficient_matrix.isApprox(second->coefficient_matrix,
                                                 kTolerance));
  EXPECT_TRUE((2.0 * second->lower_bounds).isApprox(first->lower_bounds,
                                                    1e-9));
  EXPECT_TRUE((2.0 * second->upper_bounds).isApprox(first->upper_bounds,
                                                    1e-9));
  EXPECT_TRUE(third->lower_bounds.isApprox(first->lower_bounds, 1e-9));
  EXPECT_TRUE(third->upper_bounds.isApprox(first->upper_bounds, 1e-9));
  EXPECT_FALSE(near_different->lower_bounds.isApprox(first->lower_bounds,
                                                     1e-12));
  EXPECT_FALSE(near_different->upper_bounds.isApprox(first->upper_bounds,
                                                     1e-12));
}

TEST_F(VelocityCollisionLinearizationTest, RejectsInvalidRowDt) {
  auto robot = make_robot(urdf_path_);
  ASSERT_TRUE(robot->has_collision_geometry());
  auto solver = make_solver(robot, 0.01);

  EXPECT_THROW(detail::VelocityConstraintTestObserver::
                   compute_collision_constraint(solver, 0.0),
               std::invalid_argument);
  EXPECT_THROW(detail::VelocityConstraintTestObserver::
                   linearize_collision_velocity_constraint(
                       solver, std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
}

TEST_F(VelocityCollisionLinearizationTest,
       NoArgCollisionConstraintPreservesLegacyDtClamp) {
  auto robot = make_robot(urdf_path_);
  ASSERT_TRUE(robot->has_collision_geometry());
  auto solver = make_solver(robot, 0.01);
  solver.set_dt(0.0);

  EXPECT_NO_THROW({
    const auto rows =
        detail::VelocityConstraintTestObserver::compute_collision_constraint(
            solver);
    EXPECT_TRUE(rows.has_value());
  });
}

TEST_F(VelocityCollisionLinearizationTest,
       IncludeAndExcludeUseCanonicalCollisionPairFilter) {
  auto robot = make_robot(multi_urdf_path_);
  ASSERT_TRUE(robot->has_collision_geometry());
  KinematicsSolver all_solver(robot);
  all_solver.configure_collision_constraint(0.01, {}, {}, true, 3);
  const auto all_pairs = all_solver.get_active_collision_pairs();
  ASSERT_GE(all_pairs.size(), 2U);

  KinematicsSolver include_solver(robot);
  include_solver.configure_collision_constraint(
      0.01, {all_pairs.front()}, {}, true, 3);
  const auto included_pairs = include_solver.get_active_collision_pairs();
  ASSERT_EQ(included_pairs.size(), 1U);
  EXPECT_EQ(included_pairs.front(), all_pairs.front());

  KinematicsSolver exclude_solver(robot);
  exclude_solver.configure_collision_constraint(
      0.01, {}, {all_pairs.front()}, true, 3);
  const auto excluded_pairs = exclude_solver.get_active_collision_pairs();
  EXPECT_EQ(excluded_pairs.size(), all_pairs.size() - 1);
  EXPECT_TRUE(std::find(excluded_pairs.begin(), excluded_pairs.end(),
                        all_pairs.front()) == excluded_pairs.end());
}

TEST_F(VelocityCollisionLinearizationTest,
       MultiRowBudgetKeepsDeterministicCollisionRowOrder) {
  auto direct_robot = make_robot(multi_urdf_path_);
  auto linearized_robot = make_robot(multi_urdf_path_);
  ASSERT_TRUE(direct_robot->has_collision_geometry());
  ASSERT_TRUE(linearized_robot->has_collision_geometry());

  auto direct_solver = make_solver(direct_robot, 0.01, 0.005);
  auto linearized_solver = make_solver(linearized_robot, 0.01, 0.005);
  direct_solver.configure_collision_constraint(0.005, {}, {}, true, 2);
  linearized_solver.configure_collision_constraint(0.005, {}, {}, true, 2);
  direct_solver.set_proximity_gated_collision_activation_enabled(false);
  linearized_solver.set_proximity_gated_collision_activation_enabled(false);

  const auto direct = detail::VelocityConstraintTestObserver::
      compute_collision_constraint(direct_solver, 0.01);
  const auto direct_debug = direct_solver.get_last_collision_debug_list();
  const auto linearized = detail::VelocityConstraintTestObserver::
      linearize_collision_velocity_constraint(linearized_solver, 0.01);

  ASSERT_TRUE(direct.has_value());
  ASSERT_TRUE(linearized.has_value());
  ASSERT_EQ(direct->jacobian.rows(), 2);
  ASSERT_EQ(direct_debug.size(), 2U);
  EXPECT_LE(direct_debug[0].distance, direct_debug[1].distance);
  EXPECT_TRUE(linearized->coefficient_matrix.isApprox(direct->jacobian,
                                                      kTolerance));
  EXPECT_TRUE(linearized->lower_bounds.isApprox(direct->lower_bounds,
                                                kTolerance));
  EXPECT_TRUE(linearized->upper_bounds.isApprox(direct->upper_bounds,
                                                kTolerance));
}

} // namespace
} // namespace embodik::test
