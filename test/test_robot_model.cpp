#include <Eigen/Dense>
#include <embodik/robot_model.hpp>
#include <fstream>
#include <gtest/gtest.h>

using namespace embodik;

// Helper function to create a simple test URDF
void create_test_urdf(const std::string &filename) {
  std::ofstream file(filename);
  file << R"(<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="1.0" effort="10.0"/>
  </joint>

  <link name="end_effector">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="end_effector"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" velocity="1.0" effort="10.0"/>
  </joint>
</robot>)";
  file.close();
}

class RobotModelTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create test URDF file
    test_urdf_path_ = "/tmp/test_robot.urdf";
    create_test_urdf(test_urdf_path_);
  }

  void TearDown() override {
    // Clean up test file
    std::remove(test_urdf_path_.c_str());
  }

  std::string test_urdf_path_;
};

TEST_F(RobotModelTest, LoadFromURDF) {
  EXPECT_NO_THROW({
    RobotModel model(test_urdf_path_, false);
    EXPECT_EQ(model.nq(), 2); // 2 revolute joints
    EXPECT_EQ(model.nv(), 2);
    EXPECT_FALSE(model.is_floating_base());
  });
}

TEST_F(RobotModelTest, LoadWithFloatingBase) {
  RobotModel model(test_urdf_path_, true);
  EXPECT_EQ(model.nq(), 9); // 7 for floating base + 2 joints
  EXPECT_EQ(model.nv(), 8); // 6 for floating base + 2 joints
  EXPECT_TRUE(model.is_floating_base());
}

TEST_F(RobotModelTest, UpdateConfiguration) {
  RobotModel model(test_urdf_path_, false);

  Eigen::VectorXd q(2);
  q << 0.5, -0.5;

  EXPECT_NO_THROW({ model.update_configuration(q); });

  // Check that configuration is stored
  EXPECT_TRUE(model.get_current_configuration().isApprox(q));
}

TEST_F(RobotModelTest, GetFramePose) {
  RobotModel model(test_urdf_path_, false);

  Eigen::VectorXd q(2);
  q.setZero();
  model.update_configuration(q);

  // Check end effector pose at zero configuration
  auto pose = model.get_frame_pose("end_effector");

  // At zero configuration, end effector should be at (0, 0, 2)
  EXPECT_NEAR(pose.translation()(2), 2.0, 1e-6);
}

TEST_F(RobotModelTest, GetFrameJacobian) {
  RobotModel model(test_urdf_path_, false);

  Eigen::VectorXd q(2);
  q.setZero();
  model.update_configuration(q);

  auto J = model.get_frame_jacobian("end_effector");

  EXPECT_EQ(J.rows(), 6);
  EXPECT_EQ(J.cols(), 2);
}

TEST_F(RobotModelTest, GetCenterOfMass) {
  RobotModel model(test_urdf_path_, false);

  Eigen::VectorXd q(2);
  q.setZero();
  model.update_configuration(q);

  auto com = model.get_com_position();
  EXPECT_EQ(com.size(), 3);

  // COM should be somewhere positive in Z
  EXPECT_GT(com(2), 0.0);
}

TEST_F(RobotModelTest, GetJointLimits) {
  RobotModel model(test_urdf_path_, false);

  auto [lower, upper] = model.get_joint_limits();

  EXPECT_EQ(lower.size(), 2);
  EXPECT_EQ(upper.size(), 2);

  // Check limits match URDF
  EXPECT_NEAR(lower(0), -3.14, 1e-6);
  EXPECT_NEAR(upper(0), 3.14, 1e-6);
}

TEST_F(RobotModelTest, InvalidURDFPath) {
  EXPECT_THROW({ RobotModel model("/nonexistent/path.urdf", false); },
               std::runtime_error);
}

TEST_F(RobotModelTest, InvalidConfigurationSize) {
  RobotModel model(test_urdf_path_, false);

  Eigen::VectorXd q_wrong(3); // Wrong size
  q_wrong.setZero();

  EXPECT_THROW({ model.update_configuration(q_wrong); }, std::runtime_error);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
