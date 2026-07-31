#include <Eigen/Dense>
#include <embodik/robot_model.hpp>
#include <gtest/gtest.h>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

namespace embodik::test {
namespace {

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

std::string asymmetric_centroidal_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="centroidal_asymmetric_test_robot">
  <link name="base_link">
    <inertial>
      <origin xyz="0.07 -0.03 0.11" rpy="0.05 -0.02 0.04"/>
      <mass value="3.25"/>
      <inertia ixx="0.31" ixy="0.012" ixz="-0.017" iyy="0.43" iyz="0.023" izz="0.52"/>
    </inertial>
  </link>
  <link name="shoulder">
    <inertial>
      <origin xyz="0.34 0.06 -0.02" rpy="-0.03 0.04 0.07"/>
      <mass value="1.75"/>
      <inertia ixx="0.08" ixy="-0.006" ixz="0.004" iyy="0.11" iyz="-0.003" izz="0.13"/>
    </inertial>
  </link>
  <joint name="yaw" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder"/>
    <origin xyz="0.13 -0.08 0.22" rpy="0.01 0.03 -0.02"/>
    <axis xyz="0.2 0.1 0.97"/>
    <limit lower="-2.4" upper="2.3" velocity="7.0" effort="40.0"/>
  </joint>
  <link name="forearm">
    <inertial>
      <origin xyz="-0.08 0.27 0.05" rpy="0.09 -0.06 0.02"/>
      <mass value="0.95"/>
      <inertia ixx="0.044" ixy="0.005" ixz="-0.002" iyy="0.061" iyz="0.007" izz="0.073"/>
    </inertial>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="shoulder"/>
    <child link="forearm"/>
    <origin xyz="0.41 0.18 -0.09" rpy="-0.04 0.08 0.03"/>
    <axis xyz="-0.1 0.95 0.2"/>
    <limit lower="-0.35" upper="0.45" velocity="3.0" effort="35.0"/>
  </joint>
  <link name="tool">
    <inertial>
      <origin xyz="0.12 -0.16 0.19" rpy="-0.02 0.11 -0.05"/>
      <mass value="0.55"/>
      <inertia ixx="0.019" ixy="-0.002" ixz="0.001" iyy="0.026" iyz="-0.003" izz="0.031"/>
    </inertial>
  </link>
  <joint name="pitch" type="revolute">
    <parent link="forearm"/>
    <child link="tool"/>
    <origin xyz="-0.23 0.36 0.14" rpy="0.06 -0.03 0.05"/>
    <axis xyz="0.25 -0.4 0.88"/>
    <limit lower="-1.7" upper="1.8" velocity="5.0" effort="20.0"/>
  </joint>
</robot>)";
}

Eigen::VectorXd fixed_q() {
  Eigen::VectorXd q(3);
  q << 0.37, -0.11, -0.42;
  return q;
}

Eigen::VectorXd fixed_v() {
  Eigen::VectorXd v(3);
  v << -0.23, 0.31, 0.17;
  return v;
}

Eigen::VectorXd fixed_a() {
  Eigen::VectorXd a(3);
  a << 0.41, -0.19, 0.29;
  return a;
}

Eigen::VectorXd floating_q(const RobotModel &robot) {
  Eigen::VectorXd q = robot.neutral_configuration();
  q.segment<3>(0) << 0.18, -0.27, 0.44;
  Eigen::Quaterniond quat(
      Eigen::AngleAxisd(0.29, Eigen::Vector3d(0.3, -0.4, 0.86).normalized()));
  q.segment<4>(3) << quat.x(), quat.y(), quat.z(), quat.w();
  q.tail<3>() = fixed_q();
  return q;
}

Eigen::VectorXd floating_v() {
  Eigen::VectorXd v(9);
  v << 0.21, -0.16, 0.09, -0.13, 0.24, -0.07, -0.23, 0.31, 0.17;
  return v;
}

Eigen::VectorXd floating_a() {
  Eigen::VectorXd a(9);
  a << -0.11, 0.08, 0.19, 0.14, -0.05, 0.22, 0.41, -0.19, 0.29;
  return a;
}

Eigen::VectorXd momentum_vector(const pinocchio::Force &momentum) {
  return momentum.toVector();
}

Eigen::MatrixXd expected_ag(const RobotModel &robot, const Eigen::VectorXd &q,
                            const Eigen::VectorXd &v) {
  pinocchio::Data data(robot.model());
  return pinocchio::ccrba(robot.model(), data, q, v);
}

Eigen::MatrixXd finite_difference_dag(const RobotModel &robot,
                                      const Eigen::VectorXd &q,
                                      const Eigen::VectorXd &v) {
  constexpr double eps = 1e-7;
  const Eigen::VectorXd q_plus = pinocchio::integrate(robot.model(), q, eps * v);
  const Eigen::VectorXd q_minus =
      pinocchio::integrate(robot.model(), q, -eps * v);
  pinocchio::Data data_plus(robot.model());
  pinocchio::Data data_minus(robot.model());
  const Eigen::MatrixXd ag_plus =
      pinocchio::computeCentroidalMap(robot.model(), data_plus, q_plus);
  const Eigen::MatrixXd ag_minus =
      pinocchio::computeCentroidalMap(robot.model(), data_minus, q_minus);
  return (ag_plus - ag_minus) / (2.0 * eps);
}

void expect_centroidal_outputs(RobotModel &robot, const Eigen::VectorXd &q,
                               const Eigen::VectorXd &v,
                               const Eigen::VectorXd &a) {
  robot.update_kinematics(q, v);

  pinocchio::Data data(robot.model());
  const Eigen::MatrixXd ag = pinocchio::ccrba(robot.model(), data, q, v);
  const Eigen::VectorXd h = ag * v;
  const Eigen::MatrixXd dag = pinocchio::dccrba(robot.model(), data, q, v);
  const Eigen::VectorXd bias = dag * v;
  pinocchio::computeCentroidalMomentumTimeVariation(robot.model(), data, q, v,
                                                    a);
  const Eigen::VectorXd hdot = momentum_vector(data.dhg);

  EXPECT_NEAR(robot.get_total_mass(), 6.5, 1e-12);
  EXPECT_TRUE(robot.get_centroidal_momentum_matrix().isApprox(ag, 1e-10));
  EXPECT_TRUE(robot.compute_centroidal_momentum_matrix(q).isApprox(
      expected_ag(robot, q, Eigen::VectorXd::Zero(robot.nv())), 1e-10));
  EXPECT_TRUE(robot.compute_centroidal_momentum_matrix(q, v).isApprox(ag,
                                                                      1e-10));
  EXPECT_TRUE(robot.get_centroidal_momentum().isApprox(h, 1e-10));
  EXPECT_TRUE(robot.compute_centroidal_momentum(q, v).isApprox(h, 1e-10));
  EXPECT_TRUE(robot.get_centroidal_momentum_matrix_time_variation().isApprox(
      dag, 1e-10));
  EXPECT_TRUE(robot.compute_centroidal_momentum_matrix_time_variation(q, v)
                  .isApprox(dag, 1e-10));
  EXPECT_TRUE(robot.get_centroidal_momentum_matrix_bias().isApprox(bias,
                                                                   1e-10));
  EXPECT_TRUE(
      robot.compute_centroidal_momentum_matrix_bias(q, v).isApprox(bias,
                                                                    1e-10));
  EXPECT_TRUE((ag * a + bias).isApprox(hdot, 1e-10));
  EXPECT_TRUE(dag.isApprox(finite_difference_dag(robot, q, v), 1e-5));
}

TEST(RobotModelCentroidalTest, FixedBaseCentroidalQuantitiesMatchPinocchio) {
  ScopedUrdf urdf("embodik_centroidal_fixed.urdf",
                  asymmetric_centroidal_urdf());
  RobotModel robot(urdf.path(), false);
  expect_centroidal_outputs(robot, fixed_q(), fixed_v(), fixed_a());
}

TEST(RobotModelCentroidalTest, FloatingBaseCentroidalQuantitiesMatchPinocchio) {
  ScopedUrdf urdf("embodik_centroidal_floating.urdf",
                  asymmetric_centroidal_urdf());
  RobotModel robot(urdf.path(), true);
  expect_centroidal_outputs(robot, floating_q(robot), floating_v(),
                            floating_a());
}

TEST(RobotModelCentroidalTest, RejectsWrongSizeAndNonFiniteInputs) {
  ScopedUrdf urdf("embodik_centroidal_validation.urdf",
                  asymmetric_centroidal_urdf());
  RobotModel robot(urdf.path(), false);

  const Eigen::VectorXd q = fixed_q();
  const Eigen::VectorXd v = fixed_v();
  Eigen::VectorXd wrong_q(2);
  wrong_q.setZero();
  Eigen::VectorXd wrong_v(2);
  wrong_v.setZero();
  Eigen::VectorXd nonfinite_q = q;
  nonfinite_q(1) = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd nonfinite_v = v;
  nonfinite_v(2) = std::numeric_limits<double>::infinity();

  EXPECT_THROW(robot.compute_centroidal_momentum_matrix(wrong_q),
               std::invalid_argument);
  EXPECT_THROW(robot.compute_centroidal_momentum(q, wrong_v),
               std::invalid_argument);
  EXPECT_THROW(robot.compute_centroidal_momentum_matrix_time_variation(q,
                                                                       wrong_v),
               std::invalid_argument);
  EXPECT_THROW(robot.compute_centroidal_momentum_matrix_bias(q, wrong_v),
               std::invalid_argument);
  EXPECT_THROW(robot.compute_centroidal_momentum_matrix(nonfinite_q),
               std::invalid_argument);
  EXPECT_THROW(robot.compute_centroidal_momentum(nonfinite_q, v),
               std::invalid_argument);
  EXPECT_THROW(robot.compute_centroidal_momentum_matrix_time_variation(
                   q, nonfinite_v),
               std::invalid_argument);
  EXPECT_THROW(robot.compute_centroidal_momentum_matrix_bias(q, nonfinite_v),
               std::invalid_argument);
}

} // namespace
} // namespace embodik::test
