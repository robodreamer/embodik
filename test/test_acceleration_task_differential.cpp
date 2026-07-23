#include "acceleration_task_differential.hpp"

#include <embodik/robot_model.hpp>
#include <embodik/tasks.hpp>

#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>

namespace embodik::test {
namespace {

constexpr double kTolerance = 1e-10;

void create_test_urdf(const std::string &filename) {
  std::ofstream file(filename);
  file << R"(<?xml version="1.0"?>
<robot name="acceleration_task_test_robot">
  <link name="base_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.3 0 0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="4.0" effort="20.0"/>
  </joint>

  <link name="end_effector">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0.2 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="end_effector"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="4.0" effort="20.0"/>
  </joint>
</robot>)";
}

class AccelerationTaskDifferentialTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_path_ = "/tmp/embodik_acceleration_task_test.urdf";
    create_test_urdf(urdf_path_);
    model_ = std::make_shared<RobotModel>(urdf_path_, false);
    q_.resize(2);
    q_ << 0.4, -0.25;
    v_.resize(2);
    v_ << 0.7, -0.35;
    model_->update_kinematics(q_, v_);
  }

  void TearDown() override { std::remove(urdf_path_.c_str()); }

  detail::AccelerationTaskDifferential evaluate(Task &task) {
    task.update(*model_);
    return detail::evaluate_acceleration_task_differential(task, *model_);
  }

  std::string urdf_path_;
  std::shared_ptr<RobotModel> model_;
  Eigen::VectorXd q_;
  Eigen::VectorXd v_;
};

void expect_success(const detail::AccelerationTaskDifferential &result) {
  ASSERT_EQ(result.status,
            detail::AccelerationTaskDifferentialStatus::kSuccess)
      << result.message;
  EXPECT_TRUE(result.physical_jacobian.allFinite());
  EXPECT_TRUE(result.control_jacobian.allFinite());
  EXPECT_TRUE(result.position_error.allFinite());
  EXPECT_TRUE(result.jacobian_bias.allFinite());
  EXPECT_TRUE(result.reference_row_scale.allFinite());
  EXPECT_EQ(result.physical_jacobian.rows(), result.position_error.size());
  EXPECT_EQ(result.physical_jacobian.rows(), result.jacobian_bias.size());
  EXPECT_EQ(result.physical_jacobian.rows(), result.reference_row_scale.size());
  EXPECT_EQ(result.physical_jacobian.rows(), result.control_jacobian.rows());
  EXPECT_EQ(result.physical_jacobian.cols(), result.control_jacobian.cols());
}

TEST_F(AccelerationTaskDifferentialTest,
       FrameTaskExposesPhysicalAndControlJacobians) {
  FrameTask task("frame", model_, "end_effector", TaskType::FRAME_POSE);
  const auto pose = model_->get_frame_pose("end_effector");
  task.setTargetPose(pose.translation() + Eigen::Vector3d(0.1, -0.05, 0.02),
                     pose.rotation());
  task.setPositionMask(Eigen::Vector3d(1.0, 0.0, 1.0));
  task.set_excluded_joint_indices({0});

  const auto result = evaluate(task);

  expect_success(result);
  EXPECT_EQ(result.physical_jacobian.rows(), 6);
  EXPECT_EQ(result.physical_jacobian.cols(), model_->nv());
  EXPECT_TRUE(result.control_jacobian.isApprox(task.getJacobian(),
                                                kTolerance));
  EXPECT_GT(result.physical_jacobian.col(0).norm(), kTolerance);
  EXPECT_NEAR(result.control_jacobian.col(0).norm(), 0.0, kTolerance);
  EXPECT_NEAR(result.physical_jacobian.row(1).norm(), 0.0, kTolerance);
  EXPECT_NEAR(result.jacobian_bias(1), 0.0, kTolerance);
}

TEST_F(AccelerationTaskDifferentialTest, FrameAndComBiasAreZeroAtZeroVelocity) {
  model_->update_kinematics(q_, Eigen::VectorXd::Zero(model_->nv()));

  FrameTask frame("frame", model_, "end_effector", TaskType::FRAME_POSE);
  frame.setTargetPose(model_->get_frame_pose("end_effector").translation(),
                      Eigen::Matrix3d::Identity());
  const auto frame_result = evaluate(frame);
  expect_success(frame_result);
  EXPECT_NEAR(frame_result.jacobian_bias.norm(), 0.0, kTolerance);

  COMTask com("com", model_);
  com.setTargetPosition(model_->get_com_position());
  const auto com_result = evaluate(com);
  expect_success(com_result);
  EXPECT_NEAR(com_result.jacobian_bias.norm(), 0.0, kTolerance);
}

TEST_F(AccelerationTaskDifferentialTest,
       HeldFrameAndComTargetsExposeRobotModelBiasAtNonzeroVelocity) {
  FrameTask frame("frame", model_, "end_effector", TaskType::FRAME_POSE);
  const auto pose = model_->get_frame_pose("end_effector");
  frame.setTargetPose(pose.translation(), pose.rotation());
  const auto frame_result = evaluate(frame);
  expect_success(frame_result);
  EXPECT_TRUE(frame_result.position_error.isZero(kTolerance));
  EXPECT_TRUE(frame_result.jacobian_bias.isApprox(
      model_->get_frame_jacobian_bias("end_effector"), kTolerance));

  COMTask com("com", model_);
  com.setTargetPosition(model_->get_com_position());
  const auto com_result = evaluate(com);
  expect_success(com_result);
  EXPECT_TRUE(com_result.position_error.isZero(kTolerance));
  EXPECT_TRUE(com_result.jacobian_bias.isApprox(model_->get_com_jacobian_bias(),
                                                kTolerance));
}

TEST_F(AccelerationTaskDifferentialTest, ComTaskAppliesMaskAndExclusion) {
  COMTask task("com", model_);
  task.setTargetPosition(model_->get_com_position() +
                         Eigen::Vector3d(0.03, -0.02, 0.01));
  task.setPositionMask(Eigen::Vector3d(1.0, 0.0, 1.0));
  task.set_excluded_joint_indices({0});

  const auto result = evaluate(task);

  expect_success(result);
  EXPECT_TRUE(result.control_jacobian.isApprox(task.getJacobian(),
                                                kTolerance));
  EXPECT_GT(result.physical_jacobian.col(0).norm(), kTolerance);
  EXPECT_NEAR(result.control_jacobian.col(0).norm(), 0.0, kTolerance);
  EXPECT_NEAR(result.physical_jacobian.row(1).norm(), 0.0, kTolerance);
  EXPECT_NEAR(result.jacobian_bias(1), 0.0, kTolerance);
}

TEST_F(AccelerationTaskDifferentialTest,
       JointSpaceTasksUseZeroBiasAndAdapterExclusions) {
  PostureTask posture("posture", model_);
  Eigen::VectorXd q_target = q_;
  q_target(0) += 0.1;
  posture.setTargetConfiguration(q_target);
  posture.set_excluded_joint_indices({0});
  auto posture_result = evaluate(posture);
  expect_success(posture_result);
  EXPECT_NEAR(posture_result.jacobian_bias.norm(), 0.0, kTolerance);
  EXPECT_TRUE(posture_result.control_jacobian.isApprox(posture.getJacobian(),
                                                       kTolerance));

  JointTask joint("joint", model_, 0, q_(0) + 0.2);
  joint.set_excluded_joint_indices({0});
  auto joint_result = evaluate(joint);
  expect_success(joint_result);
  EXPECT_NEAR(joint_result.jacobian_bias.norm(), 0.0, kTolerance);
  EXPECT_GT(joint_result.physical_jacobian.col(0).norm(), kTolerance);
  EXPECT_NEAR(joint_result.control_jacobian.col(0).norm(), 0.0, kTolerance);

  MultiJointTask multi("multi", model_, std::vector<int>{0, 1},
                       (Eigen::Vector2d() << q_(0) + 0.1, q_(1) - 0.1)
                           .finished());
  multi.set_excluded_joint_indices({1});
  auto multi_result = evaluate(multi);
  expect_success(multi_result);
  EXPECT_NEAR(multi_result.jacobian_bias.norm(), 0.0, kTolerance);
  EXPECT_GT(multi_result.physical_jacobian.col(1).norm(), kTolerance);
  EXPECT_NEAR(multi_result.control_jacobian.col(1).norm(), 0.0, kTolerance);
}

TEST_F(AccelerationTaskDifferentialTest,
       FloatingBasePostureAllJointRowsMatchVelocityJacobian) {
  auto floating_model = std::make_shared<RobotModel>(urdf_path_, true);
  Eigen::VectorXd q = floating_model->neutral_configuration();
  Eigen::VectorXd v = Eigen::VectorXd::Zero(floating_model->nv());
  floating_model->update_kinematics(q, v);

  PostureTask posture("floating_posture", floating_model);
  Eigen::VectorXd weights(floating_model->nq());
  for (Eigen::Index index = 0; index < weights.size(); ++index) {
    weights(index) = 1.0 + 0.17 * static_cast<double>(index);
  }
  posture.setJointWeights(weights);
  posture.update(*floating_model);

  const auto result =
      detail::evaluate_acceleration_task_differential(posture,
                                                      *floating_model);

  expect_success(result);
  EXPECT_TRUE(result.control_jacobian.isApprox(posture.getJacobian(),
                                               kTolerance));
  EXPECT_TRUE(result.physical_jacobian.isApprox(posture.getJacobian(),
                                                kTolerance));
}

TEST_F(AccelerationTaskDifferentialTest,
       EctsTasksProduceFinitePoseDifferentials) {
  RelativeFrameTask relative("relative", model_, "link1", "end_effector");
  relative.update(*model_);
  relative.captureCurrentAsTarget();
  auto relative_result = evaluate(relative);
  expect_success(relative_result);
  EXPECT_EQ(relative_result.physical_jacobian.rows(), 6);
  EXPECT_EQ(relative_result.physical_jacobian.cols(), model_->nv());

  AbsoluteFrameTask absolute("absolute", model_, "link1", "end_effector", 0.4);
  absolute.update(*model_);
  absolute.setTargetPose(absolute.getCurrentPosition(),
                         absolute.getCurrentOrientation());
  auto absolute_result = evaluate(absolute);
  expect_success(absolute_result);
  EXPECT_EQ(absolute_result.physical_jacobian.rows(), 6);
  EXPECT_EQ(absolute_result.physical_jacobian.cols(), model_->nv());
}

TEST_F(AccelerationTaskDifferentialTest,
       RelativeAndAbsoluteJacobiansMatchVelocityTaskUnderMasksAndExclusions) {
  RelativeFrameTask relative("relative", model_, "link1", "end_effector");
  relative.update(*model_);
  relative.captureCurrentAsTarget();
  relative.setPositionMask(Eigen::Vector3d(1.0, 0.0, 1.0));
  relative.setOrientationMask(Eigen::Vector3d(0.0, 1.0, 1.0));
  relative.set_excluded_joint_indices({0});
  const auto relative_result = evaluate(relative);
  expect_success(relative_result);
  EXPECT_TRUE(relative_result.control_jacobian.isApprox(relative.getJacobian(),
                                                        kTolerance));

  AbsoluteFrameTask absolute("absolute", model_, "link1", "end_effector",
                             0.35);
  Eigen::Matrix4d offset_a = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d offset_b = Eigen::Matrix4d::Identity();
  offset_a(0, 3) = 0.11;
  offset_a(1, 3) = -0.03;
  offset_b(0, 3) = -0.07;
  offset_b(2, 3) = 0.05;
  absolute.setTcpOffsets(offset_a, offset_b);
  absolute.update(*model_);
  absolute.setTargetPose(absolute.getCurrentPosition(),
                         absolute.getCurrentOrientation());
  absolute.setPositionMask(Eigen::Vector3d(1.0, 0.0, 1.0));
  absolute.setOrientationMask(Eigen::Vector3d(0.0, 1.0, 1.0));
  absolute.set_excluded_joint_indices({0});

  const auto absolute_result = evaluate(absolute);

  expect_success(absolute_result);
  EXPECT_TRUE(absolute_result.control_jacobian.isApprox(absolute.getJacobian(),
                                                        kTolerance));
  EXPECT_GT(absolute_result.physical_jacobian.col(0).norm(), kTolerance);
  EXPECT_NEAR(absolute_result.control_jacobian.col(0).norm(), 0.0,
              kTolerance);
}

TEST_F(AccelerationTaskDifferentialTest,
       AbsoluteBiasMatchesVelocityJacobianFiniteDifference) {
  AbsoluteFrameTask absolute("absolute", model_, "link1", "end_effector",
                             0.35);
  Eigen::Matrix4d offset_a = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d offset_b = Eigen::Matrix4d::Identity();
  offset_a(0, 3) = 0.11;
  offset_b(1, 3) = -0.09;
  absolute.setTcpOffsets(offset_a, offset_b);
  absolute.update(*model_);
  absolute.setTargetPose(absolute.getCurrentPosition(),
                         absolute.getCurrentOrientation());

  const auto result = evaluate(absolute);
  expect_success(result);

  auto absolute_jacobian_at = [&](const Eigen::VectorXd &q) {
    model_->update_kinematics(q, v_);
    absolute.update(*model_);
    return absolute.getJacobian();
  };

  constexpr double dt = 1e-7;
  const Eigen::MatrixXd jacobian_now = absolute_jacobian_at(q_);
  const Eigen::VectorXd q_next = model_->integrate(q_, v_, dt);
  const Eigen::MatrixXd jacobian_next = absolute_jacobian_at(q_next);
  const Eigen::VectorXd numerical_bias =
      ((jacobian_next - jacobian_now) * v_) / dt;
  model_->update_kinematics(q_, v_);
  absolute.update(*model_);

  EXPECT_TRUE(result.physical_jacobian.isApprox(jacobian_now, kTolerance));
  EXPECT_TRUE(result.jacobian_bias.isApprox(numerical_bias, 1e-5))
      << "analytic: " << result.jacobian_bias.transpose()
      << "\nnumeric: " << numerical_bias.transpose();
}

TEST_F(AccelerationTaskDifferentialTest,
       ContinuityMutationsAreVisibleToDifferentialRows) {
  FrameTask task("frame", model_, "end_effector", TaskType::FRAME_POSE);
  task.setTargetPose(model_->get_frame_pose("end_effector").translation(),
                     Eigen::Matrix3d::Identity());
  task.update(*model_);
  const auto initial_target_revision = task.getContinuityTargetRevision();
  const auto initial_state_revision = task.getContinuityRevision();

  task.setTargetPose(model_->get_frame_pose("end_effector").translation() +
                         Eigen::Vector3d(0.02, 0.0, 0.0),
                     Eigen::Matrix3d::Identity());
  EXPECT_GT(task.getContinuityTargetRevision(), initial_target_revision);

  auto before_mask = evaluate(task);
  expect_success(before_mask);
  task.setPositionMask(Eigen::Vector3d(0.0, 1.0, 1.0));
  EXPECT_GT(task.getContinuityRevision(), initial_state_revision);
  auto after_mask = evaluate(task);
  expect_success(after_mask);
  EXPECT_GT(before_mask.physical_jacobian.row(0).norm(), kTolerance);
  EXPECT_NEAR(after_mask.physical_jacobian.row(0).norm(), 0.0, kTolerance);
}

class CustomTask final : public Task {
public:
  CustomTask() : Task("custom") {}
  void update(const RobotModel &) override {}
  Eigen::VectorXd getError() const override { return Eigen::VectorXd::Ones(1); }
  Eigen::MatrixXd getJacobian() const override {
    return Eigen::MatrixXd::Ones(1, 2);
  }
  int getDimension() const override { return 1; }
  TaskType getType() const override { return TaskType::POSTURE; }
};

TEST_F(AccelerationTaskDifferentialTest,
       CustomTasksKeepVelocityCompatibilityButRejectAccelerationRows) {
  CustomTask task;
  EXPECT_EQ(task.getVelocity().size(), 1);
  EXPECT_EQ(task.getJacobian().rows(), 1);

  const auto result =
      detail::evaluate_acceleration_task_differential(task, *model_);

  EXPECT_EQ(result.status,
            detail::AccelerationTaskDifferentialStatus::kUnsupportedTask);
  EXPECT_NE(result.message.find("custom task"), std::string::npos);
}

} // namespace
} // namespace embodik::test
