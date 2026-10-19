#include <embodik/acceleration_solver.hpp>
#include <embodik/robot_model.hpp>
#include <embodik/tasks.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace embodik {
namespace {
constexpr double kElasticBoundaryTolerance = 1e-6;
constexpr double kElasticLowScale = 0.5;
constexpr double kElasticStalledScale = 1e-6;
constexpr double kElasticTaskError = 1e-4;
constexpr double kElasticDeltaTolerance = 1e-8;
} // namespace

void AccelerationSolver::reset_elastic_band() {
  elastic_was_active_ = false;
  elastic_band_delta_.resize(0);
  elastic_nominal_lower_.resize(0);
  elastic_nominal_upper_.resize(0);
  elastic_capacity_lower_.resize(0);
  elastic_capacity_upper_.resize(0);
}

Eigen::VectorXd AccelerationSolver::get_elastic_band_deltas() const {
  return elastic_band_delta_.size() == robot_->nv()
             ? elastic_band_delta_
             : Eigen::VectorXd::Zero(robot_->nv());
}

SolverStatus AccelerationSolver::prepare_elastic_bounds(
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &acceleration_limits,
    const AccelerationSolveOptions &options,
    AccelerationSolveOptions *effective_options, bool *active,
    std::string *message) {
  *active =
      std::any_of(tasks_.begin(), tasks_.end(),
                  [](const std::shared_ptr<Task> &task) {
                    return task && task->isActive() &&
                           task->getSolveMode() == TaskSolveMode::kScaleElastic;
                  }) ||
      std::any_of(options.centroidal_momentum_rate_objectives.begin(),
                  options.centroidal_momentum_rate_objectives.end(),
                  [](const CentroidalMomentumRateObjective &objective) {
                    return objective.solve_mode == TaskSolveMode::kScaleElastic;
                  });
  const bool has_lower = options.nominal_position_lower_bounds.has_value();
  const bool has_upper = options.nominal_position_upper_bounds.has_value();
  if (has_lower != has_upper || (has_lower && !options.apply_position_limits)) {
    *message = "nominal position bounds require both vectors and "
               "apply_position_limits";
    return SolverStatus::kInvalidInput;
  }
  if (!std::isfinite(options.elastic_delta_max) ||
      options.elastic_delta_max < 0.0 ||
      !std::isfinite(options.elastic_expand_rate) ||
      options.elastic_expand_rate < 0.0 ||
      !std::isfinite(options.elastic_decay_rate) ||
      options.elastic_decay_rate < 0.0 || options.elastic_decay_rate > 1.0) {
    *message = "elastic band parameters must be finite, nonnegative, and decay "
               "at most one";
    return SolverStatus::kInvalidInput;
  }
  if (!options.apply_position_limits) {
    reset_elastic_band();
    elastic_was_active_ = *active;
    elastic_band_delta_ = Eigen::VectorXd::Zero(q.size());
    elastic_capacity_lower_ = Eigen::VectorXd::Zero(q.size());
    elastic_capacity_upper_ = Eigen::VectorXd::Zero(q.size());
    return SolverStatus::kSuccess;
  }
  const std::pair<Eigen::VectorXd, Eigen::VectorXd> hard_limits =
      robot_->get_joint_limits();
  const Eigen::VectorXd nominal_lower =
      options.nominal_position_lower_bounds.value_or(hard_limits.first);
  const Eigen::VectorXd nominal_upper =
      options.nominal_position_upper_bounds.value_or(hard_limits.second);
  if (nominal_lower.size() != q.size() || nominal_upper.size() != q.size() ||
      !nominal_lower.allFinite() || !nominal_upper.allFinite() ||
      (nominal_lower.array() > nominal_upper.array()).any() ||
      (nominal_lower.array() < hard_limits.first.array()).any() ||
      (nominal_upper.array() > hard_limits.second.array()).any()) {
    *message = "nominal position bounds must be finite, ordered full-nq "
               "vectors inside model hard limits";
    return SolverStatus::kInvalidInput;
  }
  const bool initialize =
      (*active && !elastic_was_active_) ||
      elastic_band_delta_.size() != q.size() ||
      elastic_nominal_lower_.size() != q.size() ||
      !elastic_nominal_lower_.isApprox(nominal_lower, 0.0) ||
      !elastic_nominal_upper_.isApprox(nominal_upper, 0.0);
  elastic_was_active_ = *active;
  if (initialize) {
    elastic_band_delta_ = Eigen::VectorXd::Zero(q.size());
    elastic_nominal_lower_ = nominal_lower;
    elastic_nominal_upper_ = nominal_upper;
  }
  elastic_capacity_lower_ =
      (nominal_lower - hard_limits.first).cwiseMin(options.elastic_delta_max);
  elastic_capacity_upper_ =
      (hard_limits.second - nominal_upper).cwiseMin(options.elastic_delta_max);
  if (!*active || !options.apply_position_limits) {
    elastic_band_delta_.setZero();
  } else {
    for (Eigen::Index index = 0; index < q.size(); ++index) {
      const double capacity = std::max(elastic_capacity_lower_(index),
                                       elastic_capacity_upper_(index));
      if (initialize &&
          (q(index) - nominal_lower(index) <= kElasticBoundaryTolerance ||
           nominal_upper(index) - q(index) <= kElasticBoundaryTolerance)) {
        elastic_band_delta_(index) =
            std::min(options.elastic_expand_rate, capacity);
      }
      // A decaying nominal band must retain already occupied space and its
      // outward braking reserve. Immutable physical limits still cap this.
      const double braking =
          dq(index) * dq(index) / (2.0 * acceleration_limits(index));
      const double lower_reserve =
          nominal_lower(index) - q(index) + (dq(index) < 0.0 ? braking : 0.0);
      const double upper_reserve =
          q(index) - nominal_upper(index) + (dq(index) > 0.0 ? braking : 0.0);
      elastic_band_delta_(index) =
          std::clamp(std::max(elastic_band_delta_(index),
                              std::max(lower_reserve, upper_reserve)),
                     0.0, capacity);
    }
  }
  if (options.apply_position_limits) {
    effective_options->nominal_position_lower_bounds =
        nominal_lower - elastic_band_delta_.cwiseMin(elastic_capacity_lower_);
    effective_options->nominal_position_upper_bounds =
        nominal_upper + elastic_band_delta_.cwiseMin(elastic_capacity_upper_);
  }
  return SolverStatus::kSuccess;
}

void AccelerationSolver::update_elastic_band(
    const AccelerationSolverResult &result,
    const AccelerationSolveOptions &options) {
  if (result.status != SolverStatus::kSuccess) {
    return;
  }
  double scale = 1.0;
  const auto include_uniform_scale = [&scale](double task_scale) {
    // MIN_ERROR reports -1 rather than a uniform scale. Such diagnostics must
    // not manufacture a stall or bypass saturated-only band adaptation.
    if (std::isfinite(task_scale) && task_scale >= 0.0 && task_scale <= 1.0) {
      scale = std::min(scale, task_scale);
    }
  };
  bool task_error =
      std::any_of(result.task_errors.begin(), result.task_errors.end(),
                  [](double error) { return error > kElasticTaskError; });
  for (double task_scale : result.task_scales) {
    include_uniform_scale(task_scale);
  }
  for (const CentroidalMomentumRateDiagnostics &diagnostic :
       result.centroidal_momentum_rate_diagnostics) {
    include_uniform_scale(diagnostic.scale);
    task_error = task_error || diagnostic.residual.norm() > kElasticTaskError;
  }
  if (result.task_scales.empty() &&
      result.centroidal_momentum_rate_diagnostics.empty()) {
    return;
  }
  task_error = task_error || scale < kElasticStalledScale;
  const bool expand = scale < kElasticLowScale && task_error;
  for (Eigen::Index index = 0; index < elastic_band_delta_.size(); ++index) {
    const bool saturated =
        std::find(result.saturated_position_indices.begin(),
                  result.saturated_position_indices.end(),
                  index) != result.saturated_position_indices.end();
    if (expand && (!options.elastic_expand_only_saturated || saturated ||
                   scale < kElasticStalledScale)) {
      const double expansion =
          options.elastic_expand_rate * std::max(0.0, 1.0 - 2.0 * scale);
      const double boost = scale < kElasticStalledScale
                               ? options.elastic_delta_max * kElasticLowScale
                               : 0.0;
      elastic_band_delta_(index) =
          std::min(std::max(elastic_band_delta_(index) + expansion, boost),
                   std::max(elastic_capacity_lower_(index),
                            elastic_capacity_upper_(index)));
    } else if (!saturated || !task_error) {
      elastic_band_delta_(index) *= 1.0 - options.elastic_decay_rate;
      if (elastic_band_delta_(index) < kElasticDeltaTolerance) {
        elastic_band_delta_(index) = 0.0;
      }
    }
  }
}
} // namespace embodik
