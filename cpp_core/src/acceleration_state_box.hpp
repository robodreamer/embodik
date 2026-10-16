#pragma once

#include <embodik/acceleration_solver.hpp>

#include <Eigen/Core>

#include <cstdint>
#include <string>
#include <vector>

namespace embodik::detail {

struct ScalarBounds {
  double lower = 0.0;
  double upper = 0.0;
  bool lower_active = false;
  bool upper_active = false;
};

struct ReachableRecoveryRateResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  double rate = 0.0;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

/** Compute a positive-toward-recovery one-tick rate target. */
ReachableRecoveryRateResult compute_reachable_recovery_rate(
    double violation, double current_recovery_rate, double dt,
    double recovery_scale, double minimum_recovery_rate,
    double maximum_recovery_rate, double acceleration_authority,
    double boundary_epsilon, double authority_fraction = 0.9);

/** Snap only machine-roundoff excursions onto an acceleration interval face.
 * Larger violations and invalid inputs remain unchanged for caller validation.
 */
double canonicalize_state_box_acceleration_roundoff(double acceleration,
                                                   double lower,
                                                   double upper);

/** Polish a successful numerical solution into its original physical box.
 * Already in-box vectors are unchanged. A proposed projection is committed
 * only after original physical/normalized hard rows and all achieved objective
 * images pass their caller-provided, unchanged numerical tolerances. Failure
 * leaves the vector unchanged. This does not validate nonlinear geometry.
 */
bool polish_state_box_acceleration(
    const Eigen::VectorXd &lower, const Eigen::VectorXd &upper,
    const Eigen::MatrixXd &hard_matrix, const Eigen::VectorXd &hard_lower,
    const Eigen::VectorXd &hard_upper,
    const std::vector<Eigen::MatrixXd> &objective_matrices,
    double box_tolerance, double normalized_hard_tolerance,
    double objective_tolerance, Eigen::VectorXd *acceleration,
    bool *changed = nullptr);

struct StateBoxRowInput {
  double value = 0.0;
  double rate = 0.0;
  ScalarBounds state_bounds;
  ScalarBounds rate_bounds;
  ScalarBounds acceleration_bounds;
  double lower_braking_acceleration = 0.0;
  double upper_braking_acceleration = 0.0;
};

enum class StateBoxBoundCause : std::uint32_t {
  kAcceleration = 1U << 0U,
  kRate = 1U << 1U,
  kStateRate = 1U << 2U,
  kStateEndpoint = 1U << 3U,
  kContinuousPath = 1U << 4U,
  kNextStateViability = 1U << 5U,
};

using StateBoxCauseSet = std::uint32_t;

constexpr StateBoxCauseSet state_box_cause(StateBoxBoundCause cause) {
  return static_cast<StateBoxCauseSet>(cause);
}

constexpr bool has_state_box_cause(StateBoxCauseSet causes,
                                   StateBoxBoundCause cause) {
  return (causes & state_box_cause(cause)) != 0U;
}

enum class StateBoxFailure {
  kNone,
  kInvalidSpecification,
  kCurrentStateOutside,
  kOutwardRateAtBoundary,
  kOutsideBrakingEnvelope,
  kNonFiniteDerivedValue,
  kEmptyInterval,
};

struct StateBoxRowResult {
  SolverStatus status = SolverStatus::kSuccess;
  StateBoxFailure failure = StateBoxFailure::kNone;
  std::string message;
  double lower_acceleration = 0.0;
  double upper_acceleration = 0.0;
  bool lower_active = false;
  bool upper_active = false;
  StateBoxCauseSet lower_causes = 0U;
  StateBoxCauseSet upper_causes = 0U;
};

StateBoxRowResult shape_state_box_row(const StateBoxRowInput &input,
                                      double dt);

struct LinearizedStateBoxInput {
  std::string source_id;
  Eigen::MatrixXd coefficient_matrix;
  Eigen::VectorXd affine_bias;
  std::vector<StateBoxRowInput> rows;
};

struct LinearizedStateBoxResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  AffineAccelerationConstraint physical_constraint;
  std::vector<StateBoxRowInput> state_rows;
  std::vector<StateBoxRowResult> shaped_rows;
};

LinearizedStateBoxResult prepare_linearized_state_box(
    const LinearizedStateBoxInput &input, double dt,
    Eigen::Index variable_count);

} // namespace embodik::detail
