#include "kinematics_solver_internal.hpp"

#include <gtest/gtest.h>

namespace embodik::test {
namespace {

Eigen::MatrixXd recovery_rows() {
  Eigen::MatrixXd rows = Eigen::MatrixXd::Zero(3, 5);
  rows(0, 0) = 1.0;
  rows(1, 1) = 1.0;
  rows.row(2) << -1.0, -1.0, 1.0, 1.0, 0.0;
  return rows;
}

TEST(VelocityTaskProjectionTest,
     CoupledPositiveRecoveryRowsLeaveAScalableTangentTask) {
  for (const bool keep_tangent_task : {false, true}) {
    SCOPED_TRACE(keep_tangent_task);
    Eigen::MatrixXd task = Eigen::MatrixXd::Zero(keep_tangent_task ? 2 : 1, 5);
    task.row(0) << 0.0, 0.0, 1.0, 1.0, 0.0;
    Eigen::VectorXd target(task.rows());
    target(0) = 1.0;
    if (keep_tangent_task) {
      task(1, 4) = 1.0;
      target(1) = 0.2;
    }
    std::vector<Eigen::MatrixXd> tasks{task};
    std::vector<Eigen::VectorXd> targets{target};
    const Eigen::MatrixXd recovery = recovery_rows();
    const Eigen::Vector3d lower = Eigen::Vector3d::Constant(0.5);

    // The requested velocity satisfies the third row. Retaining that normal in
    // the task would require scale >= 1.5 once the first two rows recover.
    ASSERT_TRUE(project_objectives_into_constraint_tangent_space(
        targets, tasks, {true}, recovery, lower, 1e-10, {}, 0.0, false,
        nullptr, nullptr));
    EXPECT_EQ(tasks.front().rows(), 1);
    EXPECT_NEAR((tasks.front() * recovery.transpose()).norm(), 0.0, 1e-10);
    if (!keep_tangent_task) {
      EXPECT_TRUE(tasks.front().isZero(0.0));
      EXPECT_TRUE(targets.front().isZero(0.0));
    }

    Eigen::MatrixXd constraints(8, 5);
    constraints.topRows(5).setIdentity();
    constraints.bottomRows(3) = recovery;
    Eigen::VectorXd bounds_lower = Eigen::VectorXd::Constant(8, -1.0);
    Eigen::VectorXd bounds_upper = Eigen::VectorXd::Constant(8, 1.0);
    bounds_lower.tail(3) = lower;
    bounds_upper.tail(3).setConstant(10.0);
    VelocitySolverConfig config;
    config.epsilon = 1e-10;
    config.precision_threshold = 1e-12;
    config.regularization_config.regularization_factor = 0.0;
    ObjectiveSolveConfig objective;
    objective.allow_min_error_fallback = false;
    const auto result = computeMultiObjectiveVelocitySolutionEigen(
        targets, tasks, constraints, bounds_lower, bounds_upper, config,
        {objective}, nullptr);
    ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
    const Eigen::Map<const Eigen::VectorXd> velocity(result.solution.data(), 5);
    const Eigen::VectorXd values = constraints * velocity;
    EXPECT_TRUE((values.array() >= bounds_lower.array() - 1e-10).all());
    EXPECT_TRUE((values.array() <= bounds_upper.array() + 1e-10).all());
    EXPECT_NEAR(result.task_scales.front(), 1.0, 1e-10);
    EXPECT_NEAR(velocity(4), keep_tangent_task ? 0.2 : 0.0, 1e-10);
    EXPECT_FALSE(result.task_used_fallback.front());

    // The same projection cannot make an inconsistent hard set feasible.
    bounds_lower(7) = 2.5;
    const auto impossible_legacy = computeMultiObjectiveVelocitySolutionEigen(
        targets, tasks, constraints, bounds_lower, bounds_upper, config,
        {objective}, nullptr);
    EXPECT_NE(impossible_legacy.status, SolverStatus::kSuccess);
    const auto impossible = solveHierarchicalLinearSystemEigen(
        targets, {Eigen::VectorXd::Zero(targets.front().size())}, tasks,
        constraints, bounds_lower, bounds_upper, config, {objective}, nullptr);
    EXPECT_NE(impossible.status, SolverStatus::kSuccess)
        << impossible.status_message;
  }
}

TEST(VelocityTaskProjectionTest,
     SingleRecoveryNormalKeepsCollapsedTaskRay) {
  for (const double lower : {-0.5, 0.0, 0.5}) {
    for (const int repeated_rows : {1, 2}) {
      SCOPED_TRACE(lower);
      SCOPED_TRACE(repeated_rows);
      const Eigen::MatrixXd task = Eigen::MatrixXd::Ones(1, 1);
      const Eigen::VectorXd target = Eigen::VectorXd::Constant(1, -1.0);
      std::vector<Eigen::MatrixXd> tasks{task};
      std::vector<Eigen::VectorXd> targets{target};
      EXPECT_TRUE(project_objectives_into_constraint_tangent_space(
          targets, tasks, {true}, Eigen::MatrixXd::Ones(repeated_rows, 1),
          Eigen::VectorXd::Constant(repeated_rows, lower), 1e-10, {}, 0.0,
          false, nullptr, nullptr));
      EXPECT_TRUE(tasks.front().isApprox(task, 0.0));
      EXPECT_TRUE(targets.front().isApprox(target, 0.0));
    }
  }
}

TEST(VelocityTaskProjectionTest,
     TaskSatisfyingEveryRecoveryRowKeepsItsOriginalObjective) {
  const Eigen::MatrixXd task = Eigen::MatrixXd::Identity(5, 5);
  Eigen::VectorXd target(5);
  target << 0.5, 0.5, 1.0, 1.0, 0.2;
  std::vector<Eigen::MatrixXd> tasks{task};
  std::vector<Eigen::VectorXd> targets{target};
  EXPECT_FALSE(project_objectives_into_constraint_tangent_space(
      targets, tasks, {true}, recovery_rows(),
      Eigen::Vector3d::Constant(0.5), 1e-10, {}, 0.0, false,
      nullptr, nullptr));
  EXPECT_TRUE(tasks.front().isApprox(task, 0.0));
  EXPECT_TRUE(targets.front().isApprox(target, 0.0));
}

} // namespace
} // namespace embodik::test
