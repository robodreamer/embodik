#include <embodik/ik_baseline.hpp>
#include <gtest/gtest.h>

#include <cstring>

namespace {
void same_result(const embodik::SolverResult &a, const embodik::SolverResult &b) {
  EXPECT_EQ(a.solution, b.solution);
  EXPECT_EQ(a.status, b.status);
  EXPECT_EQ(a.iterations, b.iterations);
  EXPECT_EQ(0, std::memcmp(&a.final_error, &b.final_error, sizeof(double)));
  EXPECT_EQ(0, std::memcmp(&a.condition_number, &b.condition_number, sizeof(double)));
  EXPECT_EQ(a.task_scales, b.task_scales);
  EXPECT_EQ(a.task_errors, b.task_errors);
  EXPECT_EQ(a.task_modes_effective, b.task_modes_effective);
  EXPECT_EQ(a.task_used_fallback, b.task_used_fallback);
  EXPECT_EQ(a.status_message, b.status_message);
}

TEST(NumericalBandTrace, LegacyStagesDoNotChangeAnyNonclockResult) {
  using namespace embodik;
  Eigen::MatrixXd first(1, 2), second(1, 2);
  first << 1., 0.; second << 0., 1.;
  std::vector<Eigen::MatrixXd> matrices{first, second, Eigen::Matrix2d::Identity()};
  std::vector<Eigen::VectorXd> targets{Eigen::VectorXd::Constant(1, .5),
      Eigen::VectorXd::Constant(1, .2), Eigen::Vector2d::Ones()};
  const Eigen::MatrixXd rows = Eigen::Matrix2d::Identity();
  const Eigen::VectorXd lower = -Eigen::Vector2d::Ones(), upper = Eigen::Vector2d::Ones();
  std::vector<ObjectiveSolveConfig> tasks(3);
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i].priority = i;
    tasks[i].allow_min_error_fallback = false;
    tasks[i].solve_mode = i < 2 ? TaskSolveMode::kMinError : TaskSolveMode::kScale;
  }
  VelocitySolverConfig config;
  config.regularization_config.regularization_factor = .1;
  config.refine_partial_scale = true;
  detail::HierarchicalLinearSolverExecutionOptions options;
  options.policy = detail::HierarchicalLinearSolverPolicy::kLegacyVelocity;
  options.validate_legacy_primary_range = false;
  const auto disabled = solveHierarchicalLinearSystemEigen(targets, {}, matrices,
      rows, lower, upper, config, tasks, nullptr, options);
  detail::NumericalBandTrace trace;
  options.numerical_trace = &trace;
  const auto enabled = solveHierarchicalLinearSystemEigen(targets, {}, matrices,
      rows, lower, upper, config, tasks, nullptr, options);
  same_result(disabled, enabled);
  ASSERT_EQ(trace.stages.size(), 3);
  EXPECT_TRUE(trace.least_squares.empty());
  for (size_t i = 0; i < trace.stages.size(); ++i) {
    const auto &stage = trace.stages[i];
    EXPECT_EQ(stage.objective_index, i);
    EXPECT_TRUE(stage.legacy);
    EXPECT_TRUE(stage.completed_normally);
    EXPECT_TRUE(stage.original_rows.isApprox(rows, 0.));
    EXPECT_TRUE(stage.original_lower.isApprox(lower, 0.));
    EXPECT_TRUE(stage.original_upper.isApprox(upper, 0.));
    EXPECT_TRUE(stage.prefix_images.isApprox(stage.prefix_rows*stage.entry_velocity, 0.));
    EXPECT_DOUBLE_EQ(stage.exit_raw_violation, detail::ComputeMaxLinearConstraintViolation(
        rows, lower, upper, stage.exit_velocity));
  }
  const Eigen::MatrixXd stored = trace.stages[0].matrix;
  matrices[0].setZero();
  EXPECT_TRUE(trace.stages[0].matrix.isApprox(stored, 0.));
}

TEST(NumericalBandTrace, RecordsOnlyActualLeastSquaresCallsWithoutChangingResults) {
  using namespace embodik;
  const Eigen::MatrixXd rows = Eigen::Matrix2d::Identity();
  const Eigen::VectorXd lower = -Eigen::Vector2d::Ones(), upper = Eigen::Vector2d::Ones();
  std::vector<Eigen::MatrixXd> matrices{rows};
  std::vector<Eigen::VectorXd> targets{Eigen::Vector2d(2., -3.)};
  std::vector<Eigen::VectorXd> biases{Eigen::Vector2d::Zero()};
  ObjectiveSolveConfig task;
  task.solve_mode = TaskSolveMode::kMinError;
  task.allow_min_error_fallback = false;
  VelocitySolverConfig config;
  config.regularization_config.regularization_factor = 0.;
  detail::HierarchicalLinearSolverExecutionOptions options;
  options.prefer_direct_min_error_qp = true;
  const auto disabled = solveHierarchicalLinearSystemEigen(targets, biases, matrices,
      rows, lower, upper, config, {task}, nullptr, options);
  detail::NumericalBandTrace trace;
  options.numerical_trace = &trace;
  const auto enabled = solveHierarchicalLinearSystemEigen(targets, biases, matrices,
      rows, lower, upper, config, {task}, nullptr, options);
  same_result(disabled, enabled);
  ASSERT_EQ(trace.least_squares.size(), 1);
  const auto &band = trace.least_squares.front();
  EXPECT_TRUE(band.success);
  EXPECT_TRUE(band.caller_seed_present);
  EXPECT_TRUE(band.rows.isApprox(rows, 0.));
  EXPECT_TRUE(band.target.isApprox(targets.front(), 0.));
  EXPECT_DOUBLE_EQ(band.ridge, 1e-12);
  EXPECT_TRUE(band.candidate.isApprox(Eigen::Map<const Eigen::VectorXd>(
      enabled.solution.data(), enabled.solution.size()), 0.));
}

TEST(NumericalBandTrace, EarlyReturnIsAnIncompleteOuterCheckpoint) {
  using namespace embodik;
  Eigen::MatrixXd task(1, 2);
  task << 1., 0.;
  const std::vector<Eigen::MatrixXd> matrices{task, Eigen::Matrix2d::Identity()};
  const std::vector<Eigen::VectorXd> targets{
      Eigen::VectorXd::Constant(1, 2.), Eigen::Vector2d::Ones()};
  const Eigen::MatrixXd rows = Eigen::Matrix2d::Identity();
  const Eigen::VectorXd lower = -Eigen::Vector2d::Ones(), upper = Eigen::Vector2d::Ones();
  std::vector<ObjectiveSolveConfig> tasks(2);
  tasks[0].solve_mode = TaskSolveMode::kMinError;
  tasks[1].solve_mode = TaskSolveMode::kScale;
  for (auto &policy : tasks) policy.allow_min_error_fallback = false;
  VelocitySolverConfig config;
  config.iteration_limit = 0;
  detail::HierarchicalLinearSolverExecutionOptions options;
  options.policy = detail::HierarchicalLinearSolverPolicy::kLegacyVelocity;
  const auto disabled = solveHierarchicalLinearSystemEigen(targets, {}, matrices,
      rows, lower, upper, config, tasks, nullptr, options);
  detail::NumericalBandTrace trace;
  options.numerical_trace = &trace;
  const auto enabled = solveHierarchicalLinearSystemEigen(targets, {}, matrices,
      rows, lower, upper, config, tasks, nullptr, options);
  same_result(disabled, enabled);
  ASSERT_EQ(trace.stages.size(), 1);
  EXPECT_FALSE(trace.stages[0].completed_normally);
  EXPECT_TRUE(trace.least_squares.empty());
  EXPECT_EQ(trace.stages[0].exit_velocity.size(), 2);
}
struct StageProblem {
  std::vector<Eigen::MatrixXd> matrices;
  std::vector<Eigen::VectorXd> targets;
  Eigen::MatrixXd rows;
  Eigen::VectorXd lower, upper;
  std::vector<embodik::ObjectiveSolveConfig> tasks;
  embodik::VelocitySolverConfig config;
  embodik::detail::HierarchicalLinearSolverExecutionOptions options;
  StageProblem() : rows(Eigen::MatrixXd::Zero(10,5)),
      lower(Eigen::VectorXd::Constant(10,-1.)),
      upper(Eigen::VectorXd::Constant(10,1.)), tasks(3) {
    Eigen::MatrixXd first=Eigen::MatrixXd::Zero(1,5);first(0,0)=1.;
    Eigen::MatrixXd second(3,5);
    second << 0.,-.94,.78,-1.21,0., 0.,.68,-.21,-.13,0.,
        0.,-.16,-1.66,-1.66,0.;
    Eigen::MatrixXd third=Eigen::MatrixXd::Zero(1,5);third(0,4)=1.;
    matrices={first,second,third};
    targets={Eigen::VectorXd::Constant(1,.2),Eigen::Vector3d(-2.4,2.6,6.),
        Eigen::VectorXd::Constant(1,.5)};
    rows.topRows(5).setIdentity();
    rows.bottomRows(5) << 0.,.76,1.46,.25,0., 0.,.30,.90,.09,0.,
        0.,-1.91,-1.48,-.09,0., 0.,1.22,-.30,0.,0.,
        0.,-.11,.27,-.03,0.;
    for(size_t i=0;i<tasks.size();++i) {
      tasks[i].priority=i; tasks[i].allow_min_error_fallback=false;
      tasks[i].solve_mode=i<2?embodik::TaskSolveMode::kMinError:embodik::TaskSolveMode::kScale;
    }
    config.refine_partial_scale=true;
    config.regularization_config.regularization_factor=.1;
    options.policy=embodik::detail::HierarchicalLinearSolverPolicy::kLegacyVelocity;
    options.validate_legacy_primary_range=false;
  }
  embodik::SolverResult solve(embodik::detail::NumericalBandTrace *trace=nullptr,
      const Eigen::VectorXd *softening=nullptr,
      const std::vector<Eigen::VectorXd> &bias={}) {
    options.numerical_trace=trace;
    return embodik::solveHierarchicalLinearSystemEigen(targets,bias,matrices,
        rows,lower,upper,config,tasks,softening,options);
  }
};

TEST(MinErrorStageValidity, ConstrainedRepairKeepsPrefixAndPermitsLowerScale) {
  using namespace embodik;
  StageProblem p; detail::NumericalBandTrace trace;
  const auto disabled=p.solve();
  const auto result=p.solve(&trace);
  same_result(disabled,result);
  ASSERT_EQ(result.status,SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(trace.stages.size(),3);
  ASSERT_EQ(trace.least_squares.size(),1);
  EXPECT_TRUE(trace.least_squares[0].success);
  EXPECT_EQ(trace.least_squares[0].objective_index,1);
  Eigen::Map<const Eigen::VectorXd> solution(result.solution.data(),result.solution.size());
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(p.rows,p.lower,p.upper,solution),p.config.epsilon);
  EXPECT_NEAR(solution[0],trace.stages[1].entry_velocity[0],1e-10);
  EXPECT_GT(solution[4],.49);
  EXPECT_EQ(result.task_scales[0],-1.);
  EXPECT_EQ(result.task_scales[1],-1.);
  EXPECT_EQ(result.task_modes_effective[2],TaskSolveMode::kScale);
  for(bool fallback:result.task_used_fallback) EXPECT_FALSE(fallback);
}

TEST(MinErrorStageValidity, CertifiedLegacyScalePrefixRetainsItsMode) {
  using namespace embodik;
  StageProblem p;
  p.tasks[0].solve_mode=TaskSolveMode::kScale;
  p.tasks[2].allow_min_error_fallback=true;
  detail::NumericalBandTrace trace;
  const auto result=p.solve(&trace);
  EXPECT_EQ(result.status,SolverStatus::kSuccess);
  EXPECT_EQ(trace.stages.size(),3);
  ASSERT_EQ(trace.least_squares.size(),1);
  EXPECT_EQ(result.iterations,3);
  EXPECT_EQ(result.task_modes_effective[0],TaskSolveMode::kScale);
  EXPECT_NEAR(result.solution[0],result.task_scales[0]*.2,1e-10);
  EXPECT_EQ(result.task_modes_effective[2],TaskSolveMode::kScale);
  EXPECT_FALSE(result.task_used_fallback[2]);
}

TEST(MinErrorStageValidity, RelaxableRowsDoNotEnterOriginalBoundRepair) {
  StageProblem p;
  const Eigen::VectorXd softening=Eigen::VectorXd::Constant(p.rows.rows(),2.);
  embodik::detail::NumericalBandTrace trace;
  p.solve(&trace,&softening);
  EXPECT_TRUE(trace.least_squares.empty());
}

TEST(MinErrorStageValidity, AffineScalePrefixUsesItsOriginalAchievedImage) {
  using namespace embodik;
  StageProblem p;
  p.options.policy=detail::HierarchicalLinearSolverPolicy::kGeneralizedEsns;
  p.tasks[0].solve_mode=TaskSolveMode::kScale;
  std::vector<Eigen::VectorXd> biases{Eigen::VectorXd::Constant(1,.05),
      Eigen::Vector3d::Zero(),Eigen::VectorXd::Zero(1)};
  detail::NumericalBandTrace trace;
  const auto result=p.solve(&trace,nullptr,biases);
  EXPECT_EQ(result.status,SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(trace.stages.size(),3);
  ASSERT_EQ(trace.least_squares.size(),1);
  const Eigen::Map<const Eigen::VectorXd> solution(result.solution.data(),result.solution.size());
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(p.rows,p.lower,p.upper,solution),p.config.epsilon);
  EXPECT_NEAR(solution[0]+.05,result.task_scales[0]*.2,1e-10);
  EXPECT_EQ(result.task_modes_effective[0],TaskSolveMode::kScale);
}
TEST(MinErrorStageValidity, AllMinErrorRetainsWholeProblemRecoveryOwnership) {
  using namespace embodik;
  StageProblem p;
  for(auto &task:p.tasks) task.solve_mode=TaskSolveMode::kMinError;
  detail::NumericalBandTrace trace;
  const auto result=p.solve(&trace);
  ASSERT_EQ(result.status,SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(result.solution[0],.2,1e-10);
  const Eigen::Map<const Eigen::VectorXd> solution(result.solution.data(),result.solution.size());
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(p.rows,p.lower,p.upper,solution),p.config.epsilon);
}
TEST(MinErrorStageValidity, AuthorizedPrefixFallbackMetadataSurvivesRepair) {
  using namespace embodik;
  StageProblem p;
  p.upper[0]=0.;
  p.tasks[0].solve_mode=TaskSolveMode::kScale;
  p.tasks[0].allow_min_error_fallback=true;
  detail::NumericalBandTrace trace;
  const auto result=p.solve(&trace);
  ASSERT_EQ(result.status,SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(trace.stages.size(),6);
  ASSERT_EQ(trace.least_squares.size(),2);
  EXPECT_EQ(result.task_modes_effective[0],TaskSolveMode::kMinError);
  EXPECT_TRUE(result.task_used_fallback[0]);
  EXPECT_EQ(result.task_scales[0],-1.);
  EXPECT_NEAR(result.solution[0],trace.stages[4].entry_velocity[0],1e-10);
  EXPECT_EQ(result.task_modes_effective[2],TaskSolveMode::kScale);
  EXPECT_FALSE(result.task_used_fallback[2]);
}
TEST(MinErrorStageValidity, TrailingMinErrorKeepsExistingSuffixRecoveryOwnership) {
  using namespace embodik;
  StageProblem p;
  p.tasks[0].solve_mode=TaskSolveMode::kScale;
  p.tasks[2].solve_mode=TaskSolveMode::kMinError;
  detail::NumericalBandTrace trace;
  const auto result=p.solve(&trace);
  ASSERT_EQ(result.status,SolverStatus::kSuccess) << result.status_message;
  EXPECT_EQ(result.status_message,"hard-constrained trailing minimum-error numerical recovery");
  EXPECT_NEAR(result.solution[0],result.task_scales[0]*.2,1e-10);
  EXPECT_EQ(result.task_modes_effective[0],TaskSolveMode::kScale);
  EXPECT_EQ(result.task_modes_effective[1],TaskSolveMode::kMinError);
  EXPECT_EQ(result.task_modes_effective[2],TaskSolveMode::kMinError);
  EXPECT_FALSE(result.task_used_fallback[0]);
  const Eigen::Map<const Eigen::VectorXd> solution(result.solution.data(),result.solution.size());
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(p.rows,p.lower,p.upper,solution),p.config.epsilon);
}
TEST(MinErrorStageValidity, InvalidPrefixStopsBeforeUnenteredFallback) {
  using namespace embodik;
  StageProblem p;
  p.matrices[0]=Eigen::MatrixXd::Zero(2,5);
  p.matrices[0](0,0)=1.;p.matrices[0](1,0)=1.;
  p.targets[0]=Eigen::Vector2d(.2,.3);
  p.tasks[0].solve_mode=TaskSolveMode::kScale;
  p.tasks[2].allow_min_error_fallback=true;
  detail::NumericalBandTrace trace;
  const auto result=p.solve(&trace);
  EXPECT_EQ(result.status,SolverStatus::kNumericalError);
  EXPECT_EQ(trace.stages.size(),2);
  EXPECT_TRUE(trace.least_squares.empty());
  EXPECT_EQ(result.iterations,2);
  EXPECT_EQ(result.task_modes_effective[2],TaskSolveMode::kScale);
  EXPECT_FALSE(result.task_used_fallback[2]);
}
} // namespace
