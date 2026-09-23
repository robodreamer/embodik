#include <embodik/acceleration_solver.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

namespace embodik::test {
namespace {

constexpr double kTolerance = 1e-8;

Eigen::VectorXd vector(std::initializer_list<double> values) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(values.size()));
  Eigen::Index index = 0;
  for (double value : values) {
    out(index++) = value;
  }
  return out;
}

Eigen::MatrixXd matrix(Eigen::Index rows, Eigen::Index cols,
                       std::initializer_list<double> values) {
  Eigen::MatrixXd out(rows, cols);
  Eigen::Index index = 0;
  for (double value : values) {
    out(index / cols, index % cols) = value;
    ++index;
  }
  return out;
}

void expect_no_motion_outputs(const AccelerationSolverResult &result) {
  EXPECT_TRUE(result.solution.empty());
  EXPECT_EQ(result.joint_accelerations.size(), 0);
  EXPECT_EQ(result.joint_velocities_next.size(), 0);
  EXPECT_EQ(result.q_solution.size(), 0);
  EXPECT_EQ(result.predicted_torques.size(), 0);
}

class ScopedUrdf {
public:
  ScopedUrdf(std::string filename, const std::string &contents)
      : path_("/tmp/" + std::move(filename)) {
    std::ofstream file(path_);
    file << contents;
  }
  ~ScopedUrdf() { std::remove(path_.c_str()); }

  const std::string &path() const { return path_; }

private:
  std::string path_;
};

std::string two_link_contact_urdf(const std::string &effort1 = "100.0",
                                  const std::string &effort2 = "100.0") {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_contact_test_robot">
  <link name="base_link"/>
  <link name="link1">
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="50.0" effort=")" +
         effort1 + R"("/>
  </joint>
  <link name="tip">
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="50.0" effort=")" +
         effort2 + R"("/>
  </joint>
  <link name="contact"/>
  <joint name="tool_fixed" type="fixed">
    <parent link="tip"/>
    <child link="contact"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
</robot>)";
}

AccelerationSolveOptions unconstrained_options() {
  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({100.0, 100.0});
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  return options;
}

AccelerationTaskReference reference(const Eigen::VectorXd &acceleration) {
  AccelerationTaskReference out;
  out.desired_acceleration = acceleration;
  return out;
}

Eigen::VectorXd frame_acceleration(RobotModel &robot, const Eigen::VectorXd &q,
                                   const Eigen::VectorXd &dq,
                                   const Eigen::VectorXd &ddq,
                                   const std::string &frame_name,
                                   Eigen::Index rows) {
  robot.update_kinematics(q, dq);
  const auto jacobian = robot.get_frame_jacobian(frame_name);
  const auto bias = robot.get_frame_jacobian_bias(frame_name);
  return jacobian.topRows(rows) * ddq + bias.head(rows);
}

class AccelerationSolverContactTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_contact_test.urdf", two_link_contact_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
    robot_->set_gravity(Eigen::Vector3d::Zero());
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(AccelerationSolverContactTest, CapabilitiesExposeContactScope) {
  const auto capabilities = AccelerationSolver::capabilities();
  EXPECT_TRUE(capabilities.supports_fixed_base_contact_kinematics);
  EXPECT_FALSE(capabilities.supports_dynamic_contact);
  EXPECT_FALSE(capabilities.supports_floating_base);
}

TEST_F(AccelerationSolverContactTest, PointContactCancelsJdotBias) {
  AccelerationSolver solver(robot_);
  auto options = unconstrained_options();
  ContactAccelerationConstraint contact;
  contact.source_id = "tip_point_contact";
  contact.frame_name = "contact";
  contact.type = ContactType::kPointContact;
  options.contact_acceleration_constraints.push_back(contact);

  const Eigen::VectorXd q = vector({0.4, -0.2});
  const Eigen::VectorXd dq = vector({0.7, -0.3});
  robot_->update_kinematics(q, dq);
  ASSERT_GT(robot_->get_frame_jacobian_bias("contact").head<3>().norm(),
            1e-3);

  const auto result = solver.solve(q, dq, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_LT(frame_acceleration(*robot_, q, dq, result.joint_accelerations,
                               "contact", 3)
                .norm(),
            kTolerance);
}

TEST_F(AccelerationSolverContactTest,
       RigidContactEnforcesSixDimensionalEquality) {
  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");
  solver.set_task_reference("posture", reference(vector({4.0, -2.0})));

  auto options = unconstrained_options();
  ContactAccelerationConstraint contact;
  contact.source_id = "tip_rigid_contact";
  contact.frame_name = "contact";
  contact.type = ContactType::kRigidContact;
  options.contact_acceleration_constraints.push_back(contact);

  const Eigen::VectorXd q = vector({0.2, -0.3});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  const auto result = solver.solve(q, dq, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_LT(frame_acceleration(*robot_, q, dq, result.joint_accelerations,
                               "contact", 6)
                .norm(),
            kTolerance);
}

TEST_F(AccelerationSolverContactTest, InvalidRecordsClearMotionOutputs) {
  AccelerationSolver solver(robot_);
  const Eigen::VectorXd q = Eigen::Vector2d::Zero();
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();

  auto missing_id = unconstrained_options();
  missing_id.contact_acceleration_constraints.push_back(
      ContactAccelerationConstraint{"", "contact",
                                    ContactType::kPointContact});
  const auto missing_id_result = solver.solve(q, dq, 0.01, missing_id);
  EXPECT_EQ(missing_id_result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(missing_id_result);

  auto missing_frame = unconstrained_options();
  missing_frame.contact_acceleration_constraints.push_back(
      ContactAccelerationConstraint{"missing_frame", "not_a_frame",
                                    ContactType::kPointContact});
  const auto missing_frame_result = solver.solve(q, dq, 0.01, missing_frame);
  EXPECT_EQ(missing_frame_result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(missing_frame_result);

  auto invalid_type = unconstrained_options();
  invalid_type.contact_acceleration_constraints.push_back(
      ContactAccelerationConstraint{
          "invalid_type", "tip", static_cast<ContactType>(999)});
  const auto invalid_type_result = solver.solve(q, dq, 0.01, invalid_type);
  EXPECT_EQ(invalid_type_result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(invalid_type_result);
}

TEST_F(AccelerationSolverContactTest, NonFiniteContactRowsFailClosed) {
  AccelerationSolver solver(robot_);
  auto options = unconstrained_options();
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  options.contact_acceleration_constraints.push_back(
      ContactAccelerationConstraint{"huge_velocity_contact", "contact",
                                    ContactType::kPointContact});

  const Eigen::VectorXd q = Eigen::Vector2d::Zero();
  const Eigen::VectorXd dq = vector({1.0e200, -1.0e200});
  const auto result = solver.solve(q, dq, 0.01, options);

  EXPECT_NE(result.status, SolverStatus::kSuccess);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverContactTest, DuplicateSourceIdsCrossFamilies) {
  AccelerationSolver solver(robot_);
  auto options = unconstrained_options();
  AffineAccelerationConstraint affine;
  affine.source_id = "same_physics";
  affine.coefficient_matrix = matrix(1, 2, {1.0, 0.0});
  affine.affine_bias = vector({0.0});
  affine.lower_bounds = vector({-1.0});
  affine.upper_bounds = vector({1.0});
  options.affine_constraints.push_back(affine);
  options.contact_acceleration_constraints.push_back(
      ContactAccelerationConstraint{"same_physics", "contact",
                                    ContactType::kPointContact});

  const auto result = solver.solve(Eigen::Vector2d::Zero(),
                                   Eigen::Vector2d::Zero(), 0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverContactTest,
       ContactRowsRemainPhysicalUnderAccelerationAllocation) {
  AccelerationSolver solver(robot_);
  auto options = unconstrained_options();
  GeneralizedAccelerationAllocation allocation;
  allocation.metric_diagonal = vector({4.0, 0.25});
  allocation.reference_acceleration = vector({3.0, -4.0});
  options.generalized_acceleration_allocation = allocation;
  options.contact_acceleration_constraints.push_back(
      ContactAccelerationConstraint{"allocated_contact", "contact",
                                    ContactType::kPointContact});

  const Eigen::VectorXd q = vector({0.5, -0.4});
  const Eigen::VectorXd dq = vector({0.4, -0.2});
  const auto result = solver.solve(q, dq, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.allocation_diagnostics.applied);
  EXPECT_LT(frame_acceleration(*robot_, q, dq, result.joint_accelerations,
                               "contact", 3)
                .norm(),
            kTolerance);
}

TEST_F(AccelerationSolverContactTest,
       ContactRowsComposeWithEffortEnvelopeAndLocks) {
  ScopedUrdf bounded_urdf("embodik_acceleration_contact_effort_test.urdf",
                          two_link_contact_urdf("20.0", "20.0"));
  auto bounded_robot =
      std::make_shared<RobotModel>(bounded_urdf.path(), false);
  bounded_robot->set_gravity(Eigen::Vector3d::Zero());

  AccelerationSolver solver(bounded_robot);
  auto options = unconstrained_options();
  options.zero_acceleration_joint_indices.push_back(1);
  options.effort_constraints = EffortConstraintOptions{};
  options.contact_acceleration_constraints.push_back(
      ContactAccelerationConstraint{"locked_effort_contact", "contact",
                                    ContactType::kPointContact});

  const Eigen::VectorXd q = vector({0.3, -0.1});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  const auto result = solver.solve(q, dq, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.joint_accelerations(1), 0.0, kTolerance);
  EXPECT_TRUE(result.effort_limits_applied);
  ASSERT_EQ(result.predicted_torques.size(), bounded_robot->nv());
  EXPECT_LT(frame_acceleration(*bounded_robot, q, dq,
                               result.joint_accelerations, "contact", 3)
                .norm(),
            kTolerance);
  EXPECT_LE(result.predicted_torques.cwiseAbs().maxCoeff(), 20.0 + kTolerance);
}

} // namespace
} // namespace embodik::test
