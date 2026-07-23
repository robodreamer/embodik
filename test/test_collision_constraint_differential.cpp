#include "collision_constraint_differential.hpp"

#include <embodik/robot_model.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace embodik::test {
namespace {

Eigen::VectorXd vector(std::initializer_list<double> values) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(values.size()));
  Eigen::Index index = 0;
  for (double value : values) {
    out(index++) = value;
  }
  return out;
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

std::string single_sphere_pair_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="collision_constraint_differential_single">
  <link name="world"/>
  <link name="obstacle">
    <collision name="obstacle_sphere">
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="obstacle_fixed" type="fixed">
    <parent link="world"/>
    <child link="obstacle"/>
  </joint>
  <link name="moving">
    <collision name="moving_sphere">
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/>
    <child link="moving"/>
    <origin xyz="0.20 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="1000" velocity="100"/>
  </joint>
</robot>)";
}

std::string multi_sphere_pair_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="collision_constraint_differential_multi">
  <link name="base">
    <collision name="left_sphere">
      <origin xyz="-0.25 0 0"/>
      <geometry><sphere radius="0.04"/></geometry>
    </collision>
    <collision name="right_sphere">
      <origin xyz="0.45 0 0"/>
      <geometry><sphere radius="0.04"/></geometry>
    </collision>
  </link>
  <link name="moving">
    <collision name="moving_sphere">
      <geometry><sphere radius="0.04"/></geometry>
    </collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="base"/>
    <child link="moving"/>
    <origin xyz="0.10 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="1000" velocity="100"/>
  </joint>
</robot>)";
}

std::string unsupported_box_pair_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="collision_constraint_differential_box">
  <link name="world"/>
  <link name="box_link">
    <collision name="box_collision">
      <geometry><box size="0.10 0.10 0.10"/></geometry>
    </collision>
  </link>
  <joint name="box_fixed" type="fixed">
    <parent link="world"/>
    <child link="box_link"/>
  </joint>
  <link name="moving">
    <collision name="moving_sphere">
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/>
    <child link="moving"/>
    <origin xyz="0.20 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="1000" velocity="100"/>
  </joint>
</robot>)";
}

TEST(CollisionConstraintDifferentialTest,
     SphereSphereDifferentialMatchesAnalyticPrismaticMotion) {
  ScopedUrdf urdf("embodik_collision_differential_single.urdf",
                  single_sphere_pair_urdf());
  RobotModel robot(urdf.path(), false);
  detail::CollisionDifferentialScratch scratch(robot);

  const auto result = detail::evaluate_bounded_collision_pair_differential_at_state(
      robot, scratch, 0, vector({0.03}), vector({0.40}));

  ASSERT_TRUE(result.satisfied()) << result.message;
  const auto &differential = *result.differential;
  EXPECT_EQ(differential.nominal.pair_index, 0U);
  EXPECT_EQ(differential.nominal.object_a.empty(), false);
  EXPECT_EQ(differential.nominal.object_b.empty(), false);
  EXPECT_NE(differential.nominal.pair_key.find("|"), std::string::npos);
  EXPECT_NEAR(differential.nominal.signed_distance, 0.13, 1e-12);
  EXPECT_TRUE(differential.nominal.center_a_world.allFinite());
  EXPECT_TRUE(differential.nominal.center_b_world.allFinite());
  EXPECT_NEAR((differential.nominal.center_b_world -
               differential.nominal.center_a_world)
                  .norm(),
              differential.nominal.signed_distance + 0.10, 1e-12);
  ASSERT_EQ(differential.nominal.jacobian.rows(), 1);
  ASSERT_EQ(differential.nominal.jacobian.cols(), 1);
  EXPECT_NEAR(differential.nominal.jacobian(0, 0), 1.0, 1e-12);
  EXPECT_NEAR(differential.nominal.rate, 0.40, 1e-12);
  EXPECT_NEAR(differential.nominal.affine_bias, 0.0, 1e-12);
  ASSERT_EQ(differential.nominal.relative_center_jacobian.rows(), 3);
  ASSERT_EQ(differential.nominal.relative_center_jacobian.cols(), 1);
  EXPECT_NEAR(
      (differential.nominal.normal_a_to_b_world.transpose() *
       differential.nominal.relative_center_jacobian)(0, 0),
      differential.nominal.jacobian(0, 0), 1e-12);
  EXPECT_NEAR(differential.nominal.normal_a_to_b_world.dot(
                  differential.nominal.relative_center_velocity_world),
              differential.nominal.rate, 1e-12);
  EXPECT_NEAR(
      differential.nominal.relative_center_affine_acceleration_world.norm(),
      0.0, 1e-12);
  EXPECT_NEAR((differential.nominal.point_b_world -
               differential.nominal.point_a_world)
                  .norm(),
              differential.nominal.signed_distance, 1e-12);
  EXPECT_EQ(differential.proof_kind,
            detail::CollisionDifferentialProofKind::kAnalyticSphereSphere);
  EXPECT_DOUBLE_EQ(differential.affine_bias_error_bound, 0.0);
  EXPECT_LE(differential.signed_distance_lower_bound,
            differential.nominal.signed_distance);
  EXPECT_GE(differential.signed_distance_upper_bound,
            differential.nominal.signed_distance);
  EXPECT_LE(differential.rate_lower_bound, differential.nominal.rate);
  EXPECT_GE(differential.rate_upper_bound, differential.nominal.rate);
}

TEST(CollisionConstraintDifferentialTest,
     AllActivePairsPreserveCatalogOrderAndAccounting) {
  ScopedUrdf urdf("embodik_collision_differential_multi.urdf",
                  multi_sphere_pair_urdf());
  RobotModel robot(urdf.path(), false);
  auto *collision_data = robot.collision_data();
  ASSERT_NE(collision_data, nullptr);
  ASSERT_EQ(collision_data->activeCollisionPairs.size(), 2U);
  collision_data->activeCollisionPairs[1] = false;
  detail::CollisionDifferentialScratch scratch(robot);

  const auto result =
      detail::evaluate_all_active_bounded_collision_pair_differentials_at_state(
          robot, scratch, vector({0.0}), vector({0.10}));

  ASSERT_TRUE(result.satisfied()) << result.message;
  ASSERT_EQ(result.differentials.size(), 1U);
  EXPECT_EQ(result.pair_evaluations, 1U);
  EXPECT_EQ(result.active_pairs_considered, 1U);
  EXPECT_EQ(result.differentials[0].nominal.pair_index, 0U);
}

TEST(CollisionConstraintDifferentialTest,
     UnsupportedNonSphereActivePairFailsClosedWithExactAccounting) {
  ScopedUrdf urdf("embodik_collision_differential_box.urdf",
                  unsupported_box_pair_urdf());
  RobotModel robot(urdf.path(), false);
  detail::CollisionDifferentialScratch scratch(robot);

  const auto result =
      detail::evaluate_all_active_bounded_collision_pair_differentials_at_state(
          robot, scratch, vector({0.0}), vector({0.0}));

  EXPECT_FALSE(result.satisfied());
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  EXPECT_NE(result.message.find("sphere-sphere"), std::string::npos)
      << result.message;
  EXPECT_EQ(result.pair_evaluations, 1U);
  EXPECT_GE(result.active_pairs_considered, 1U);
  EXPECT_TRUE(result.differentials.empty());
}

TEST(CollisionConstraintDifferentialTest,
     ExplicitActiveEvaluatorHonorsMaskWhileSelectedEvaluatorOwnsFiltering) {
  ScopedUrdf urdf("embodik_collision_differential_mask.urdf",
                  multi_sphere_pair_urdf());
  RobotModel robot(urdf.path(), false);
  auto *collision_data = robot.collision_data();
  ASSERT_NE(collision_data, nullptr);
  ASSERT_EQ(collision_data->activeCollisionPairs.size(), 2U);
  collision_data->activeCollisionPairs[0] = false;
  detail::CollisionDifferentialScratch scratch(robot);

  const auto active =
      detail::evaluate_bounded_collision_pair_differentials_at_state(
          robot, scratch, {0}, vector({0.0}), vector({0.0}));
  EXPECT_FALSE(active.satisfied());
  EXPECT_NE(active.message.find("inactive"), std::string::npos);

  const auto selected =
      detail::evaluate_selected_bounded_collision_pair_differentials_at_state(
          robot, scratch, {0}, vector({0.0}), vector({0.0}));
  ASSERT_TRUE(selected.satisfied()) << selected.message;
  ASSERT_EQ(selected.differentials.size(), 1U);
  EXPECT_EQ(selected.differentials[0].nominal.pair_index, 0U);
}

TEST(CollisionConstraintDifferentialTest,
     InvalidStateAndOutOfRangePairFailWithoutDifferential) {
  ScopedUrdf urdf("embodik_collision_differential_invalid.urdf",
                  single_sphere_pair_urdf());
  RobotModel robot(urdf.path(), false);
  detail::CollisionDifferentialScratch scratch(robot);

  auto result = detail::evaluate_bounded_collision_pair_differential_at_state(
      robot, scratch, 1, vector({0.0}), vector({0.0}));
  EXPECT_FALSE(result.satisfied());
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);

  result = detail::evaluate_bounded_collision_pair_differential_at_state(
      robot, scratch, 0, vector({std::numeric_limits<double>::quiet_NaN()}),
      vector({0.0}));
  EXPECT_FALSE(result.satisfied());
  EXPECT_EQ(result.status, SolverStatus::kNonFiniteInput);
}

} // namespace
} // namespace embodik::test
