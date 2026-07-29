/**
 * @file kinematics_solver_constraints.cpp
 * @brief Constraint assembly and support policies for KinematicsSolver
 */

#include "kinematics_solver_internal.hpp"
#include "support_polygon_geometry.hpp"

namespace embodik {

namespace {
ComSupportPolygonConstraintDefinition
make_support_definition(const Eigen::MatrixXd &support_polygon, double margin) {
  ComSupportPolygonConstraintDefinition definition;
  definition.support_polygon = support_polygon;
  definition.margin = margin;
  definition.proximity_fraction = 0.0;
  return definition;
}

double configured_capture_omega(double explicit_omega, double gravity_z,
                                double height) {
  if (std::isfinite(explicit_omega) && explicit_omega > 0.0) {
    return explicit_omega;
  }
  if (!(std::isfinite(height) && height > 0.0) ||
      !std::isfinite(gravity_z)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::sqrt(std::abs(gravity_z) / height);
}
} // namespace

void KinematicsSolver::configure_relative_pose_constraint(
    const std::string &frame_a, const std::string &frame_b,
    const Eigen::VectorXd &lower_bounds, const Eigen::VectorXd &upper_bounds,
    const Eigen::VectorXd &axis_mask) {
  if (lower_bounds.size() != 6 || upper_bounds.size() != 6)
    throw std::invalid_argument(
        "lower_bounds and upper_bounds must be 6D (pos xyz + ori xyz)");

  RelativePoseConstraintConfig cfg;
  cfg.enabled = true;
  cfg.frame_a = frame_a;
  cfg.frame_b = frame_b;
  cfg.lower_bounds = lower_bounds;
  cfg.upper_bounds = upper_bounds;

  if (axis_mask.size() == 0) {
    cfg.axis_mask = Eigen::VectorXd::Ones(6);
  } else if (axis_mask.size() == 6) {
    cfg.axis_mask = axis_mask;
  } else {
    throw std::invalid_argument("axis_mask must be empty or 6D");
  }

  relative_pose_constraint_ = std::move(cfg);
}

void KinematicsSolver::clear_relative_pose_constraint() {
  relative_pose_constraint_.reset();
}

void KinematicsSolver::add_contact_frame(const std::string &frame_name,
                                         ContactType type) {
  if (frame_name.empty()) {
    throw std::invalid_argument("contact frame name must not be empty");
  }
  if (!robot_->has_frame(frame_name)) {
    throw std::invalid_argument("unknown frame for contact projection: " +
                                frame_name);
  }
  auto it = std::find_if(
      contact_frames_.begin(), contact_frames_.end(),
      [&](const ContactFrameConfig &cfg) { return cfg.frame_name == frame_name; });
  if (it != contact_frames_.end()) {
    it->type = type;
    return;
  }
  contact_frames_.push_back(ContactFrameConfig{frame_name, type});
}

void KinematicsSolver::configure_contact_frames(
    const std::vector<std::string> &frame_names, ContactType type) {
  clear_contact_frames();
  for (const auto &frame_name : frame_names) {
    add_contact_frame(frame_name, type);
  }
}

void KinematicsSolver::clear_contact_frames() { contact_frames_.clear(); }

void KinematicsSolver::set_linear_velocity_constraints(
    const Eigen::MatrixXd &C, const Eigen::VectorXd &lower_bounds,
    const Eigen::VectorXd &upper_bounds) {
  std::string err;
  if (!validate_linear_constraint_triplet(C, lower_bounds, upper_bounds,
                                          robot_->nv(), &err)) {
    throw std::invalid_argument(err);
  }
  LinearVelocityConstraintConfig cfg;
  cfg.enabled = true;
  cfg.C = C;
  cfg.lower_bounds = lower_bounds;
  cfg.upper_bounds = upper_bounds;
  linear_velocity_constraints_ = std::move(cfg);
}

void KinematicsSolver::append_linear_velocity_constraints(
    const Eigen::MatrixXd &C, const Eigen::VectorXd &lower_bounds,
    const Eigen::VectorXd &upper_bounds) {
  std::string err;
  if (!validate_linear_constraint_triplet(C, lower_bounds, upper_bounds,
                                          robot_->nv(), &err)) {
    throw std::invalid_argument(err);
  }
  if (!linear_velocity_constraints_.has_value() ||
      !linear_velocity_constraints_->enabled ||
      linear_velocity_constraints_->C.rows() == 0) {
    set_linear_velocity_constraints(C, lower_bounds, upper_bounds);
    return;
  }

  auto &cfg = *linear_velocity_constraints_;
  const int old_rows = static_cast<int>(cfg.C.rows());
  const int add_rows = static_cast<int>(C.rows());
  Eigen::MatrixXd C_all(old_rows + add_rows, robot_->nv());
  Eigen::VectorXd lower_all(old_rows + add_rows);
  Eigen::VectorXd upper_all(old_rows + add_rows);
  C_all.topRows(old_rows) = cfg.C;
  C_all.bottomRows(add_rows) = C;
  lower_all.head(old_rows) = cfg.lower_bounds;
  lower_all.tail(add_rows) = lower_bounds;
  upper_all.head(old_rows) = cfg.upper_bounds;
  upper_all.tail(add_rows) = upper_bounds;
  cfg.C = std::move(C_all);
  cfg.lower_bounds = std::move(lower_all);
  cfg.upper_bounds = std::move(upper_all);
}

void KinematicsSolver::clear_linear_velocity_constraints() {
  linear_velocity_constraints_.reset();
}

void KinematicsSolver::configure_centroidal_momentum_bounds(
    const Eigen::VectorXd &lower_h, const Eigen::VectorXd &upper_h,
    const Eigen::VectorXd &axis_mask) {
  Eigen::VectorXd mask;
  if (axis_mask.size() == 0) {
    mask = Eigen::VectorXd::Ones(6);
  } else if (axis_mask.size() == 6 && axis_mask.allFinite()) {
    mask = axis_mask;
  } else {
    throw std::invalid_argument(
        "centroidal momentum axis_mask must be empty or finite 6D");
  }

  int selected = 0;
  for (Eigen::Index i = 0; i < mask.size(); ++i) {
    if (mask(i) != 0.0) {
      ++selected;
    }
  }
  if (selected == 0) {
    throw std::invalid_argument(
        "centroidal momentum bounds must select at least one axis");
  }
  if (lower_h.size() != selected || upper_h.size() != selected) {
    throw std::invalid_argument(
        "centroidal momentum bounds size must match selected axis count");
  }
  if (!lower_h.allFinite() || !upper_h.allFinite()) {
    throw std::invalid_argument("centroidal momentum bounds must be finite");
  }
  for (Eigen::Index i = 0; i < lower_h.size(); ++i) {
    if (lower_h(i) > upper_h(i)) {
      throw std::invalid_argument(
          "centroidal momentum lower bound exceeds upper bound");
    }
  }

  CentroidalMomentumBoundsConfig cfg;
  cfg.enabled = true;
  cfg.lower_h = lower_h;
  cfg.upper_h = upper_h;
  cfg.axis_mask = std::move(mask);
  centroidal_momentum_bounds_ = std::move(cfg);
}

void KinematicsSolver::clear_centroidal_momentum_bounds() {
  centroidal_momentum_bounds_.reset();
}

Eigen::VectorXd KinematicsSolver::get_centroidal_momentum_bounds_lower() const {
  if (!centroidal_momentum_bounds_.has_value() ||
      !centroidal_momentum_bounds_->enabled) {
    return Eigen::VectorXd();
  }
  return centroidal_momentum_bounds_->lower_h;
}

Eigen::VectorXd KinematicsSolver::get_centroidal_momentum_bounds_upper() const {
  if (!centroidal_momentum_bounds_.has_value() ||
      !centroidal_momentum_bounds_->enabled) {
    return Eigen::VectorXd();
  }
  return centroidal_momentum_bounds_->upper_h;
}

Eigen::VectorXd
KinematicsSolver::get_centroidal_momentum_bounds_axis_mask() const {
  if (!centroidal_momentum_bounds_.has_value() ||
      !centroidal_momentum_bounds_->enabled) {
    return Eigen::VectorXd();
  }
  return centroidal_momentum_bounds_->axis_mask;
}

void KinematicsSolver::configure_capture_point_constraint(
    const Eigen::MatrixXd &support_polygon, double margin,
    const std::string &frame_name, double height, double omega,
    double gravity_z) {
  if (frame_name != "world" && !robot_->has_frame(frame_name)) {
    throw std::invalid_argument("unknown capture-point support frame: " +
                                frame_name);
  }
  const auto prepared = detail::prepare_support_polygon_geometry(
      make_support_definition(support_polygon, margin), "capture-point", "ik");
  if (!prepared.satisfied()) {
    throw std::invalid_argument(prepared.message);
  }
  const double effective_omega =
      configured_capture_omega(omega, gravity_z, height);
  if (!(std::isfinite(effective_omega) && effective_omega > 0.0)) {
    throw std::invalid_argument(
        "capture-point omega must be explicit positive or derivable from "
        "positive height and finite gravity_z");
  }
  SupportHalfspaceConfig cfg;
  cfg.enabled = true;
  cfg.support_polygon = support_polygon;
  cfg.margin = margin;
  cfg.frame_name = frame_name;
  cfg.height = height;
  cfg.omega = effective_omega;
  cfg.gravity_z = gravity_z;
  cfg.A = prepared.geometry.halfspace_normals;
  cfg.b = prepared.geometry.halfspace_offsets;
  capture_point_constraint_ = std::move(cfg);
}

void KinematicsSolver::clear_capture_point_constraint() {
  capture_point_constraint_.reset();
}

void KinematicsSolver::configure_velocity_zmp_constraint(
    const Eigen::MatrixXd &support_polygon, double margin,
    const std::string &frame_name, double fz_min, double gravity_z) {
  if (frame_name != "world" && !robot_->has_frame(frame_name)) {
    throw std::invalid_argument("unknown velocity-ZMP support frame: " +
                                frame_name);
  }
  if (!(std::isfinite(fz_min) && fz_min > 0.0) ||
      !std::isfinite(gravity_z)) {
    throw std::invalid_argument(
        "velocity-ZMP fz_min must be positive and gravity_z finite");
  }
  const auto prepared = detail::prepare_support_polygon_geometry(
      make_support_definition(support_polygon, margin), "velocity-ZMP", "ik");
  if (!prepared.satisfied()) {
    throw std::invalid_argument(prepared.message);
  }
  SupportHalfspaceConfig cfg;
  cfg.enabled = true;
  cfg.support_polygon = support_polygon;
  cfg.margin = margin;
  cfg.frame_name = frame_name;
  cfg.fz_min = fz_min;
  cfg.gravity_z = gravity_z;
  cfg.A = prepared.geometry.halfspace_normals;
  cfg.b = prepared.geometry.halfspace_offsets;
  velocity_zmp_constraint_ = std::move(cfg);
}

void KinematicsSolver::clear_velocity_zmp_constraint() {
  velocity_zmp_constraint_.reset();
}

int KinematicsSolver::get_linear_velocity_constraint_rows() const {
  if (!linear_velocity_constraints_.has_value() ||
      !linear_velocity_constraints_->enabled) {
    return 0;
  }
  return static_cast<int>(linear_velocity_constraints_->C.rows());
}

void KinematicsSolver::add_tight_frame_pose_constraint(
    const std::string &frame_name, const Eigen::Matrix4d &target_pose,
    double position_epsilon, double orientation_epsilon,
    const Eigen::VectorXd &axis_mask) {
  if (!robot_->has_frame(frame_name)) {
    throw std::invalid_argument("unknown frame for tight pose constraint: " +
                                frame_name);
  }
  if (!std::isfinite(position_epsilon) || position_epsilon <= 0.0 ||
      !std::isfinite(orientation_epsilon) || orientation_epsilon <= 0.0) {
    throw std::invalid_argument(
        "tight pose constraint epsilons must be finite and > 0");
  }
  TightFramePoseConstraintConfig cfg;
  cfg.frame_name = frame_name;
  cfg.target_pose = pinocchio::SE3(target_pose.block<3, 3>(0, 0),
                                   target_pose.block<3, 1>(0, 3));
  cfg.position_epsilon = position_epsilon;
  cfg.orientation_epsilon = orientation_epsilon;
  if (axis_mask.size() == 0) {
    cfg.axis_mask = Eigen::VectorXd::Ones(6);
  } else if (axis_mask.size() == 6 && axis_mask.allFinite()) {
    cfg.axis_mask = axis_mask;
  } else {
    throw std::invalid_argument(
        "tight pose constraint axis_mask must be empty or finite 6D");
  }
  tight_frame_pose_constraints_.push_back(std::move(cfg));
}

void KinematicsSolver::clear_tight_frame_pose_constraints() {
  tight_frame_pose_constraints_.clear();
}

void KinematicsSolver::add_tight_point_constraint(
    const std::string &frame_name, const Eigen::Vector3d &target_point,
    double position_epsilon, const Eigen::Vector3d &axis_mask) {
  if (!robot_->has_frame(frame_name)) {
    throw std::invalid_argument("unknown frame for tight point constraint: " +
                                frame_name);
  }
  if (!target_point.allFinite()) {
    throw std::invalid_argument("tight point constraint target must be finite");
  }
  if (!std::isfinite(position_epsilon) || position_epsilon <= 0.0) {
    throw std::invalid_argument(
        "tight point constraint position_epsilon must be finite and > 0");
  }
  if (!axis_mask.allFinite()) {
    throw std::invalid_argument("tight point constraint axis_mask must be finite");
  }
  TightPointConstraintConfig cfg;
  cfg.frame_name = frame_name;
  cfg.target_point = target_point;
  cfg.position_epsilon = position_epsilon;
  cfg.axis_mask = axis_mask;
  tight_point_constraints_.push_back(std::move(cfg));
}

void KinematicsSolver::clear_tight_point_constraints() {
  tight_point_constraints_.clear();
}

Eigen::MatrixXd KinematicsSolver::compute_contact_projector() const {
  const int nv = robot_->nv();
  if (contact_frames_.empty()) {
    return Eigen::MatrixXd::Identity(nv, nv);
  }

  int total_rows = 0;
  for (const auto &cfg : contact_frames_) {
    total_rows += (cfg.type == ContactType::kPointContact) ? 3 : 6;
  }
  if (total_rows <= 0) {
    return Eigen::MatrixXd::Identity(nv, nv);
  }

  Eigen::MatrixXd J_c(total_rows, nv);
  J_c.setZero();
  int row = 0;
  for (const auto &cfg : contact_frames_) {
    const Matrix6Xd J_full = robot_->get_frame_jacobian(cfg.frame_name);
    if (cfg.type == ContactType::kPointContact) {
      J_c.block(row, 0, 3, nv) = J_full.topRows(3);
      row += 3;
    } else {
      J_c.block(row, 0, 6, nv) = J_full;
      row += 6;
    }
  }

  Eigen::MatrixXd J_c_pinv;
  detail::ComputeGeneralizedInverse(J_c, constraint_tolerance_, &J_c_pinv);
  return Eigen::MatrixXd::Identity(nv, nv) - J_c_pinv * J_c;
}

std::optional<KinematicsSolver::LinearVelocityConstraintResult>
KinematicsSolver::compute_linear_velocity_constraints() {
  if (!linear_velocity_constraints_.has_value() ||
      !linear_velocity_constraints_->enabled ||
      linear_velocity_constraints_->C.rows() == 0) {
    return std::nullopt;
  }
  LinearVelocityConstraintResult out;
  out.jacobian = linear_velocity_constraints_->C;
  out.lower_bounds = linear_velocity_constraints_->lower_bounds;
  out.upper_bounds = linear_velocity_constraints_->upper_bounds;
  out.violated_rows = Eigen::ArrayXi::Zero(out.jacobian.rows());
  return out;
}

std::optional<KinematicsSolver::LinearVelocityConstraintResult>
KinematicsSolver::compute_tight_frame_pose_constraints() {
  if (tight_frame_pose_constraints_.empty()) {
    return std::nullopt;
  }

  int total_rows = 0;
  for (const auto &cfg : tight_frame_pose_constraints_) {
    for (int i = 0; i < 6; ++i) {
      if (cfg.axis_mask(i) > 0.5) {
        ++total_rows;
      }
    }
  }
  if (total_rows <= 0) {
    return std::nullopt;
  }

  LinearVelocityConstraintResult out;
  out.jacobian = Eigen::MatrixXd::Zero(total_rows, robot_->nv());
  out.lower_bounds = Eigen::VectorXd::Constant(total_rows, -1e10);
  out.upper_bounds = Eigen::VectorXd::Constant(total_rows, 1e10);
  out.violated_rows = Eigen::ArrayXi::Zero(total_rows);

  int row = 0;
  for (const auto &cfg : tight_frame_pose_constraints_) {
    const pinocchio::SE3 pose = robot_->get_frame_pose(cfg.frame_name);
    const Matrix6Xd J = robot_->get_frame_jacobian(cfg.frame_name);

    Eigen::VectorXd err = Eigen::VectorXd::Zero(6);
    err.head<3>() = pose.translation() - cfg.target_pose.translation();
    err.tail<3>() =
        pinocchio::log3(cfg.target_pose.rotation().transpose() * pose.rotation());

    for (int i = 0; i < 6; ++i) {
      if (cfg.axis_mask(i) <= 0.5) {
        continue;
      }
      const double eps = (i < 3) ? cfg.position_epsilon : cfg.orientation_epsilon;
      const double slack_lower = err(i) + eps;
      const double slack_upper = eps - err(i);
      const double vel_limit = (i < 3) ? 0.5 : 1.0;
      const double acc_limit = (i < 3) ? 1.0 : 2.0;
      const double min_headroom = 0.05 * vel_limit;
      const double activation_margin = 0.5 * eps;
      auto [lower, upper] = calculate_velocity_box_constraint(
          slack_lower, slack_upper, vel_limit, acc_limit, dt_, min_headroom,
          activation_margin);
      out.jacobian.row(row) = J.row(i);
      out.lower_bounds(row) = lower;
      out.upper_bounds(row) = upper;
      if (slack_lower < -constraint_tolerance_ ||
          slack_upper < -constraint_tolerance_) {
        out.violated_rows(row) = 1;
      }
      ++row;
    }
  }

  return out;
}

std::optional<KinematicsSolver::LinearVelocityConstraintResult>
KinematicsSolver::compute_tight_point_constraints() {
  if (tight_point_constraints_.empty()) {
    return std::nullopt;
  }

  int total_rows = 0;
  for (const auto &cfg : tight_point_constraints_) {
    for (int i = 0; i < 3; ++i) {
      if (cfg.axis_mask(i) > 0.5) {
        ++total_rows;
      }
    }
  }
  if (total_rows <= 0) {
    return std::nullopt;
  }

  LinearVelocityConstraintResult out;
  out.jacobian = Eigen::MatrixXd::Zero(total_rows, robot_->nv());
  out.lower_bounds = Eigen::VectorXd::Constant(total_rows, -1e10);
  out.upper_bounds = Eigen::VectorXd::Constant(total_rows, 1e10);
  out.violated_rows = Eigen::ArrayXi::Zero(total_rows);

  int row = 0;
  for (const auto &cfg : tight_point_constraints_) {
    const pinocchio::SE3 pose = robot_->get_frame_pose(cfg.frame_name);
    const Matrix6Xd J = robot_->get_frame_jacobian(cfg.frame_name);
    const Eigen::Vector3d err = pose.translation() - cfg.target_point;
    for (int i = 0; i < 3; ++i) {
      if (cfg.axis_mask(i) <= 0.5) {
        continue;
      }
      const double eps = cfg.position_epsilon;
      const double slack_lower = err(i) + eps;
      const double slack_upper = eps - err(i);
      constexpr double kTightPointVelLimit = 0.5;
      constexpr double kTightPointAccLimit = 1.0;
      const double min_headroom = 0.05 * kTightPointVelLimit;
      const double activation_margin = 0.5 * eps;
      auto [lower, upper] = calculate_velocity_box_constraint(
          slack_lower, slack_upper, kTightPointVelLimit, kTightPointAccLimit,
          dt_, min_headroom, activation_margin);
      out.jacobian.row(row) = J.row(i);
      out.lower_bounds(row) = lower;
      out.upper_bounds(row) = upper;
      if (slack_lower < -constraint_tolerance_ ||
          slack_upper < -constraint_tolerance_) {
        out.violated_rows(row) = 1;
      }
      ++row;
    }
  }

  return out;
}

std::optional<KinematicsSolver::RelativePoseConstraintResult>
KinematicsSolver::compute_relative_pose_constraint() {
  if (!relative_pose_constraint_.has_value() ||
      !relative_pose_constraint_->enabled)
    return std::nullopt;

  const auto &cfg = *relative_pose_constraint_;

  pinocchio::SE3 T_a = robot_->get_frame_pose(cfg.frame_a);
  pinocchio::SE3 T_b = robot_->get_frame_pose(cfg.frame_b);

  pinocchio::SE3 T_rel = compute_relative_frame(T_a, T_b);

  Matrix6Xd J_a = robot_->get_frame_jacobian(cfg.frame_a);
  Matrix6Xd J_b = robot_->get_frame_jacobian(cfg.frame_b);

  Eigen::MatrixXd J_rel_full =
      compute_relative_jacobian(J_a, J_b, T_a.rotation(), T_rel.translation());

  // Current relative pose as 6D vector: [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z]
  Eigen::Vector3d rel_pos = T_rel.translation();
  Eigen::Vector3d rel_ori_log = Eigen::Vector3d::Zero();
  {
    Eigen::Matrix3d R = T_rel.rotation();
    double trace = R.trace();
    double cos_theta = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
    double theta = std::acos(cos_theta);
    if (std::abs(theta) > 1e-6) {
      rel_ori_log = (theta / (2.0 * std::sin(theta))) *
                    Eigen::Vector3d(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0),
                                    R(1, 0) - R(0, 1));
    } else {
      rel_ori_log = 0.5 * Eigen::Vector3d(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0),
                                           R(1, 0) - R(0, 1));
    }
  }

  Eigen::VectorXd rel_state(6);
  rel_state.head<3>() = rel_pos;
  rel_state.tail<3>() = rel_ori_log;

  // Count active axes
  int num_active = 0;
  for (int i = 0; i < 6; ++i)
    if (cfg.axis_mask(i) > 0.5)
      num_active++;

  if (num_active == 0)
    return std::nullopt;

  RelativePoseConstraintResult result;
  result.jacobian.resize(num_active, robot_->nv());
  result.lower_bounds.resize(num_active);
  result.upper_bounds.resize(num_active);
  result.violated_rows = Eigen::ArrayXi::Zero(num_active);

  int row = 0;
  for (int i = 0; i < 6; ++i) {
    if (cfg.axis_mask(i) > 0.5) {
      result.jacobian.row(row) = J_rel_full.row(i);
      // Velocity bounds to keep within positional bounds
      double slack_lower = rel_state(i) - cfg.lower_bounds(i);
      double slack_upper = cfg.upper_bounds(i) - rel_state(i);
      const double dt_safe = std::max(dt_, 1e-6);
      const double lower_clamped = std::max(0.0, slack_lower);
      const double upper_clamped = std::max(0.0, slack_upper);
      result.lower_bounds(row) = -lower_clamped / dt_safe;
      result.upper_bounds(row) = upper_clamped / dt_safe;
      if (slack_lower < -1e-4 || slack_upper < -1e-4) {
        result.violated_rows(row) = 1;
      }
      row++;
    }
  }

  return result;
}


void KinematicsSolver::set_base_position_bounds(const Eigen::Vector3d &lower,
                                                const Eigen::Vector3d &upper) {
  base_position_lower_ = lower;
  base_position_upper_ = upper;
}

void KinematicsSolver::set_base_orientation_bounds(
    const Eigen::Vector3d &lower, const Eigen::Vector3d &upper) {
  base_orientation_lower_ = lower;
  base_orientation_upper_ = upper;
}

void KinematicsSolver::clear_base_bounds() {
  base_position_lower_.reset();
  base_position_upper_.reset();
  base_orientation_lower_.reset();
  base_orientation_upper_.reset();
}

// ============================================================
// CoM support-polygon constraint helpers
// ============================================================
namespace {

// 2D cross product of vectors OA and OB.
static double cross2d(const Eigen::Vector2d &O, const Eigen::Vector2d &A,
                      const Eigen::Vector2d &B) {
  return (A.x() - O.x()) * (B.y() - O.y()) -
         (A.y() - O.y()) * (B.x() - O.x());
}

// Graham scan: returns convex hull vertices in CCW order.
static std::vector<Eigen::Vector2d>
convex_hull_2d(std::vector<Eigen::Vector2d> pts) {
  const int n = static_cast<int>(pts.size());
  if (n < 3)
    return pts;

  std::sort(pts.begin(), pts.end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
              return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
            });
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());

  if (static_cast<int>(pts.size()) < 3)
    return pts;

  std::vector<Eigen::Vector2d> hull;
  hull.reserve(2 * pts.size());

  // Lower hull
  for (const auto &p : pts) {
    while (hull.size() >= 2 &&
           cross2d(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0.0)
      hull.pop_back();
    hull.push_back(p);
  }

  // Upper hull
  const int lower_size = static_cast<int>(hull.size()) + 1;
  for (int i = static_cast<int>(pts.size()) - 2; i >= 0; --i) {
    while (static_cast<int>(hull.size()) >= lower_size &&
           cross2d(hull[hull.size() - 2], hull[hull.size() - 1], pts[i]) <=
               0.0)
      hull.pop_back();
    hull.push_back(pts[i]);
  }
  hull.pop_back();
  return hull;
}

// Build half-plane representation A * x <= b from a CCW convex polygon.
// For each edge (v_i -> v_{i+1}), the outward normal points to the right.
static void polygon_to_halfplanes(const std::vector<Eigen::Vector2d> &hull,
                                  Eigen::MatrixXd &A, Eigen::VectorXd &b) {
  const int n = static_cast<int>(hull.size());
  A.resize(n, 2);
  b.resize(n);
  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto &v : hull)
    centroid += v;
  centroid /= static_cast<double>(n);

  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d &v0 = hull[i];
    const Eigen::Vector2d &v1 = hull[(i + 1) % n];
    Eigen::Vector2d edge = v1 - v0;
    // Candidate outward normal (CCW hull: right side is outside).
    Eigen::Vector2d normal(edge.y(), -edge.x());
    const double len = normal.norm();
    if (len < 1e-12)
      normal = Eigen::Vector2d(1.0, 0.0);
    else
      normal /= len;

    double bi = normal.dot(v0);
    // Robust orientation guard: enforce that polygon centroid lies in the
    // feasible half-space A*x <= b (inside polygon), regardless of winding.
    if (normal.dot(centroid) > bi) {
      normal = -normal;
      bi = -bi;
    }
    A.row(i) = normal.transpose();
    b(i) = bi;
  }
}

// Shrink polygon vertices toward centroid by fractional margin in [0, 1].
// Uses mean distance from centroid to vertices (char_size) to match the
// Uses char_size (mean centroid→vertex distance) for consistent shrink behavior.
static std::vector<Eigen::Vector2d>
shrink_polygon(const std::vector<Eigen::Vector2d> &hull, double margin) {
  if (margin <= 0.0 || hull.empty())
    return hull;

  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto &v : hull)
    centroid += v;
  centroid /= static_cast<double>(hull.size());

  double sum_dist = 0.0;
  for (const auto &v : hull)
    sum_dist += (v - centroid).norm();
  const double char_size =
      (hull.size() > 0) ? (sum_dist / static_cast<double>(hull.size())) : 0.0;

  const double shrink_dist =
      std::clamp(margin, 0.0, 1.0) * std::max(0.0, char_size - 1e-9);

  std::vector<Eigen::Vector2d> shrunk;
  shrunk.reserve(hull.size());
  for (const auto &v : hull) {
    Eigen::Vector2d dir = centroid - v;
    const double d = dir.norm();
    if (d < 1e-12)
      shrunk.push_back(v);
    else
      shrunk.push_back(v + (shrink_dist / d) * dir);
  }
  return shrunk;
}

// Minimum perpendicular distance from the polygon centroid to any edge.
// This is the radius of the largest inscribed circle (inradius) and serves
// as the natural distance scale for the polygon.
static double polygon_inradius_2d(const std::vector<Eigen::Vector2d> &hull) {
  if (hull.size() < 3)
    return 0.0;

  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto &v : hull)
    centroid += v;
  centroid /= static_cast<double>(hull.size());

  double min_dist = std::numeric_limits<double>::infinity();
  const int n = static_cast<int>(hull.size());
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d &a = hull[i];
    const Eigen::Vector2d &b = hull[(i + 1) % n];
    const Eigen::Vector2d edge = b - a;
    const double edge_len = edge.norm();
    if (edge_len < 1e-12)
      continue;
    // Outward unit normal (same convention as polygon_to_halfplanes: CCW hull)
    const Eigen::Vector2d normal(edge.y(), -edge.x());
    const double dist = std::fabs((centroid - a).dot(normal) / edge_len);
    min_dist = std::min(min_dist, dist);
  }
  return std::isfinite(min_dist) ? min_dist : 0.0;
}

} // namespace

void KinematicsSolver::configure_com_constraint(
    const Eigen::MatrixXd &vertices_xy, double margin,
    const std::string &frame_name, double com_vel_max, double com_acc_max,
    bool use_acceleration_limits, double proximity_fraction) {
  if (vertices_xy.rows() < 3) {
    throw std::invalid_argument(
        "configure_com_constraint: support polygon must have at least 3 "
        "vertices.");
  }
  if (vertices_xy.cols() < 2) {
    throw std::invalid_argument(
        "configure_com_constraint: vertices must have at least 2 columns (xy).");
  }
  if (frame_name != "world" && !robot_->has_frame(frame_name)) {
    throw std::runtime_error("configure_com_constraint: frame '" + frame_name +
                             "' not found in robot model.");
  }

  // Collect 2D points
  std::vector<Eigen::Vector2d> pts;
  pts.reserve(vertices_xy.rows());
  for (int i = 0; i < vertices_xy.rows(); ++i)
    pts.emplace_back(vertices_xy(i, 0), vertices_xy(i, 1));

  // Compute convex hull (CCW)
  auto hull = convex_hull_2d(pts);
  if (hull.size() < 3) {
    throw std::runtime_error(
        "configure_com_constraint: convex hull has fewer than 3 vertices "
        "(points may be collinear).");
  }

  // Apply inward margin
  if (margin > 0.0)
    hull = shrink_polygon(hull, margin);

  // Build half-plane representation in frame_name
  ComConstraintConfig cfg;
  cfg.enabled = true;
  cfg.vertices_xy = vertices_xy;
  cfg.margin = margin;
  cfg.frame_name = frame_name;
  cfg.com_vel_max = com_vel_max;
  cfg.com_acc_max = com_acc_max;
  cfg.use_acceleration_limits = use_acceleration_limits;
  // Auto-compute proximity threshold from the convex hull inradius so the
  // caller never needs to reason about polygon geometry themselves.
  if (proximity_fraction > 0.0)
    cfg.proximity_threshold = proximity_fraction * polygon_inradius_2d(hull);
  else
    cfg.proximity_threshold = std::numeric_limits<double>::infinity();
  polygon_to_halfplanes(hull, cfg.A, cfg.b);
  com_constraint_ = std::move(cfg);
}

void KinematicsSolver::clear_com_constraint() { com_constraint_.reset(); }

double KinematicsSolver::get_com_proximity_threshold() const {
  if (!com_constraint_.has_value())
    return 0.0;
  return com_constraint_->proximity_threshold;
}

std::optional<KinematicsSolver::ComConstraintResult>
KinematicsSolver::compute_com_constraint() {
  if (!com_constraint_.has_value() || !com_constraint_->enabled)
    return std::nullopt;

  const auto &cfg = *com_constraint_;

  // CoM position (world) and Jacobian (3 x nv)
  const Eigen::Vector3d com_world = robot_->get_com_position();
  const Eigen::MatrixXd J_com = robot_->get_com_jacobian(); // 3 x nv

  // Half-planes in frame_name; transform to world if necessary
  Eigen::MatrixXd A_world = cfg.A; // #hp x 2
  Eigen::VectorXd b_world = cfg.b; // #hp

  if (cfg.frame_name != "world") {
    const auto frame_pose = robot_->get_frame_pose(cfg.frame_name);
    const Eigen::Matrix3d R = frame_pose.rotation();
    const Eigen::Vector3d t = frame_pose.translation();
    const Eigen::Matrix2d R_xy = R.topLeftCorner<2, 2>();

    // x_F = R_xy^T * (x_world_xy - t_xy)
    // A_F * x_F <= b_F
    // => A_F * R_xy^T * x_world_xy <= b_F + A_F * R_xy^T * t_xy
    A_world = cfg.A * R_xy.transpose(); // #hp x 2
    b_world = cfg.b;
    for (int i = 0; i < static_cast<int>(b_world.size()); ++i)
      b_world(i) += A_world.row(i).dot(t.head<2>());
  }

  // Slack per half-plane: positive when CoM is inside polygon.
  const Eigen::VectorXd slack = b_world - A_world * com_world.head<2>();

  // Constraint Jacobian: all half-planes (#hp x nv)
  const Eigen::MatrixXd J_all = A_world * J_com.topRows(2);

  const double vel_max = cfg.com_vel_max;
  const double acc_max = cfg.com_acc_max;
  const double prox = cfg.proximity_threshold;

  // Anti-chattering: small epsilon dead-zone at the boundary.  When the
  // slack is within [-eps, 0] the CoM is treated as "at the boundary"
  // rather than outside, preventing sign-flip oscillations between
  // frames.  Matches the kMarginEpsilon pattern in
  // calculate_velocity_box_constraint().
  constexpr double kSlackEps = 1e-4; // 0.1 mm
  constexpr double kComRecoveryScale = 0.2;
  constexpr double kComMinRecoverySpeed = 0.01;

  // All half-plane rows always participate so that vel_max and acceleration
  // limits are enforced everywhere (matching the Spot Flex IK pattern).
  //
  // Three bound layers, from coarsest to tightest:
  //   1. vel_max              — always active, caps speed in every direction.
  //   2. sqrt(2*acc*slack_c)  — always active when use_acceleration_limits is
  //      set; starts tapering velocity well before the boundary, ensuring
  //      the CoM can decelerate smoothly (bounded tipping energy).
  //   3. slack_c/dt           — only active when slack < proximity_threshold;
  //      the hard position-based limit that prevents overshooting the
  //      boundary in a single time step.
  //
  // slack_c = max(0, slack): clamped to non-negative so that position and
  // acceleration terms never flip sign (same as the joint-limit pattern in
  // calculate_velocity_box_constraint).  When the CoM is slightly outside
  // (slack < 0 but > -eps), slack_c = 0 produces upper = 0: the solver
  // stops outward motion without commanding a recovery kick that would
  // cause chattering.
  const auto com_bounds =
      compute_halfspace_velocity_bounds(slack, dt_, vel_max, acc_max,
                                        cfg.use_acceleration_limits, prox,
                                        kSlackEps, kComRecoveryScale,
                                        kComMinRecoverySpeed);

  ComConstraintResult result;
  result.jacobian = J_all;
  result.lower_bounds = com_bounds.lower;
  result.upper_bounds = com_bounds.upper;
  result.violated_rows = com_bounds.violated_rows;
  return result;
}

std::optional<KinematicsSolver::ComConstraintResult>
KinematicsSolver::compute_centroidal_momentum_bounds_constraint() {
  if (!centroidal_momentum_bounds_.has_value() ||
      !centroidal_momentum_bounds_->enabled) {
    return std::nullopt;
  }
  const auto &cfg = *centroidal_momentum_bounds_;
  const Eigen::MatrixXd Ag = robot_->get_centroidal_momentum_matrix();
  const int selected = static_cast<int>(cfg.lower_h.size());
  ComConstraintResult result;
  result.jacobian.resize(selected, robot_->nv());
  result.lower_bounds = cfg.lower_h;
  result.upper_bounds = cfg.upper_h;
  result.violated_rows = Eigen::ArrayXi::Zero(selected);
  int out_row = 0;
  for (int row = 0; row < 6; ++row) {
    if (cfg.axis_mask(row) == 0.0) {
      continue;
    }
    result.jacobian.row(out_row) = Ag.row(row);
    ++out_row;
  }
  return result;
}

std::optional<KinematicsSolver::ComConstraintResult>
KinematicsSolver::compute_capture_point_constraint() {
  if (!capture_point_constraint_.has_value() ||
      !capture_point_constraint_->enabled) {
    return std::nullopt;
  }
  const auto &cfg = *capture_point_constraint_;
  if (cfg.frame_name != "world") {
    const Matrix6Xd J_frame = robot_->get_frame_jacobian(cfg.frame_name);
    if (J_frame.norm() > 1e-10) {
      ComConstraintResult fail;
      fail.jacobian = Eigen::MatrixXd::Zero(1, robot_->nv());
      fail.lower_bounds = Eigen::VectorXd::Ones(1);
      fail.upper_bounds = Eigen::VectorXd::Zero(1);
      fail.violated_rows = Eigen::ArrayXi::Ones(1);
      return fail;
    }
  }
  const Eigen::Vector3d com = robot_->get_com_position();
  const Eigen::MatrixXd Jcom = robot_->get_com_jacobian();
  ComConstraintResult result;
  result.jacobian = (cfg.A * Jcom.topRows(2)) / cfg.omega;
  result.lower_bounds = Eigen::VectorXd::Constant(cfg.A.rows(), -1e100);
  result.upper_bounds = cfg.b - cfg.A * com.head<2>();
  result.violated_rows = Eigen::ArrayXi::Zero(cfg.A.rows());
  return result;
}

std::optional<KinematicsSolver::ComConstraintResult>
KinematicsSolver::compute_velocity_zmp_constraint(
    const Eigen::VectorXd &current_dq) {
  if (!velocity_zmp_constraint_.has_value() ||
      !velocity_zmp_constraint_->enabled) {
    return std::nullopt;
  }
  const auto &cfg = *velocity_zmp_constraint_;
  if (cfg.frame_name != "world") {
    const Matrix6Xd J_frame = robot_->get_frame_jacobian(cfg.frame_name);
    if (J_frame.norm() > 1e-10) {
      ComConstraintResult fail;
      fail.jacobian = Eigen::MatrixXd::Zero(1, robot_->nv());
      fail.lower_bounds = Eigen::VectorXd::Ones(1);
      fail.upper_bounds = Eigen::VectorXd::Zero(1);
      fail.violated_rows = Eigen::ArrayXi::Ones(1);
      return fail;
    }
  }
  const Eigen::MatrixXd Ag = robot_->get_centroidal_momentum_matrix();
  const Eigen::VectorXd bias = robot_->get_centroidal_momentum_matrix_bias();
  const Eigen::Vector3d c = robot_->get_com_position();
  const double mass_g = robot_->get_total_mass() * std::abs(cfg.gravity_z);
  const Eigen::MatrixXd G = Ag / dt_;
  const Eigen::VectorXd k = bias - (Ag * current_dq) / dt_;

  const int hp = static_cast<int>(cfg.A.rows());
  ComConstraintResult result;
  result.jacobian.resize(hp + 1, robot_->nv());
  result.lower_bounds = Eigen::VectorXd::Constant(hp + 1, -1e100);
  result.upper_bounds = Eigen::VectorXd::Constant(hp + 1, 1e100);
  result.violated_rows = Eigen::ArrayXi::Zero(hp + 1);
  for (int i = 0; i < hp; ++i) {
    const double ax = cfg.A(i, 0);
    const double ay = cfg.A(i, 1);
    result.jacobian.row(i) =
        ax * (c.x() * G.row(2) - G.row(4) - c.z() * G.row(0)) +
        ay * (c.y() * G.row(2) + G.row(3) - c.z() * G.row(1)) -
        cfg.b(i) * G.row(2);
    const double constant =
        ax * (c.x() * (mass_g + k(2)) - k(4) - c.z() * k(0)) +
        ay * (c.y() * (mass_g + k(2)) + k(3) - c.z() * k(1)) -
        cfg.b(i) * (mass_g + k(2));
    result.upper_bounds(i) = -constant;
  }
  result.jacobian.row(hp) = G.row(2);
  result.lower_bounds(hp) = cfg.fz_min - mass_g - k(2);
  return result;
}

KinematicsSolver::CentroidalSupportDebug
KinematicsSolver::evaluate_capture_point_constraint(
    const Eigen::VectorXd &current_q, const Eigen::VectorXd &dq_command) {
  CentroidalSupportDebug debug;
  if (current_q.size() != robot_->nq() || dq_command.size() != robot_->nv()) {
    debug.status = SolverStatus::kInvalidInput;
    debug.message = "q/dq size mismatch";
    return debug;
  }
  if (!current_q.allFinite() || !dq_command.allFinite()) {
    debug.status = SolverStatus::kNonFiniteInput;
    debug.message = "q/dq contains non-finite values";
    return debug;
  }
  if (!capture_point_constraint_.has_value() ||
      !capture_point_constraint_->enabled) {
    debug.status = SolverStatus::kInvalidInput;
    debug.message = "capture-point constraint is not configured";
    return debug;
  }
  robot_->update_kinematics(current_q);
  const auto &cfg = *capture_point_constraint_;
  const Eigen::Vector3d com = robot_->get_com_position();
  const Eigen::MatrixXd Jcom = robot_->get_com_jacobian();
  debug.point = com.head<2>() + (Jcom.topRows(2) * dq_command) / cfg.omega;
  debug.slacks = cfg.b - cfg.A * debug.point;
  return debug;
}

KinematicsSolver::CentroidalSupportDebug
KinematicsSolver::evaluate_velocity_zmp_constraint(
    const Eigen::VectorXd &current_q, const Eigen::VectorXd &current_dq,
    const Eigen::VectorXd &dq_command) {
  CentroidalSupportDebug debug;
  if (current_q.size() != robot_->nq() || current_dq.size() != robot_->nv() ||
      dq_command.size() != robot_->nv()) {
    debug.status = SolverStatus::kInvalidInput;
    debug.message = "q/current_dq/dq_command size mismatch";
    return debug;
  }
  if (!current_q.allFinite() || !current_dq.allFinite() ||
      !dq_command.allFinite()) {
    debug.status = SolverStatus::kNonFiniteInput;
    debug.message = "q/current_dq/dq_command contains non-finite values";
    return debug;
  }
  if (!velocity_zmp_constraint_.has_value() ||
      !velocity_zmp_constraint_->enabled) {
    debug.status = SolverStatus::kInvalidInput;
    debug.message = "velocity-ZMP constraint is not configured";
    return debug;
  }
  robot_->update_kinematics(current_q);
  const auto &cfg = *velocity_zmp_constraint_;
  const Eigen::VectorXd hdot =
      robot_->get_centroidal_momentum_matrix() *
          ((dq_command - current_dq) / dt_) +
      robot_->get_centroidal_momentum_matrix_bias();
  const Eigen::Vector3d c = robot_->get_com_position();
  const double Fz = robot_->get_total_mass() * std::abs(cfg.gravity_z) + hdot(2);
  debug.force_z = Fz;
  if (Fz <= 0.0 || !std::isfinite(Fz)) {
    debug.status = SolverStatus::kInfeasible;
    debug.message = "velocity-ZMP force denominator is not positive finite";
    return debug;
  }
  debug.point.x() = c.x() - (hdot(4) + c.z() * hdot(0)) / Fz;
  debug.point.y() = c.y() + (hdot(3) - c.z() * hdot(1)) / Fz;
  debug.slacks = cfg.b - cfg.A * debug.point;
  if (Fz < cfg.fz_min || debug.slacks.minCoeff() < -constraint_tolerance_) {
    debug.status = SolverStatus::kInfeasible;
  }
  return debug;
}

} // namespace embodik
