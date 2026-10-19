#include <embodik/cartesian_corridor_recovery.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace embodik;

void bind_cartesian_corridor_recovery(nb::module_ &m) {
  nb::enum_<CartesianRecoveryPhase>(m, "CartesianRecoveryPhase")
      .value("IDLE", CartesianRecoveryPhase::kIdle)
      .value("ACTIVE", CartesianRecoveryPhase::kActive)
      .value("SETTLED", CartesianRecoveryPhase::kSettled)
      .value("INTERRUPTED", CartesianRecoveryPhase::kInterrupted)
      .value("BLOCKED", CartesianRecoveryPhase::kBlocked);
  nb::class_<CartesianCorridorOptions>(m, "CartesianCorridorOptions")
      .def(nb::init<>())
      .def_rw("dt", &CartesianCorridorOptions::dt)
      .def_rw("guide_gain", &CartesianCorridorOptions::guide_gain)
      .def_rw("guide_mode", &CartesianCorridorOptions::guide_mode)
      .def_rw("budget_velocity_constraints", &CartesianCorridorOptions::budget_velocity_constraints)
      .def_rw("endpoint_safety_steering", &CartesianCorridorOptions::endpoint_safety_steering)
      .def_rw("allow_measured_state_updates", &CartesianCorridorOptions::allow_measured_state_updates)
      .def_rw("allow_pose_zone_reentry", &CartesianCorridorOptions::allow_pose_zone_reentry)
      .def_rw("warm_start_guide_backoff", &CartesianCorridorOptions::warm_start_guide_backoff)
      .def_rw("pose_caps", &CartesianCorridorOptions::pose_caps)
      .def_rw("terminal_tolerances", &CartesianCorridorOptions::terminal_tolerances)
      .def_rw("support_halfplanes", &CartesianCorridorOptions::support_halfplanes)
      .def_rw("support_upper", &CartesianCorridorOptions::support_upper)
      .def_rw("support_frame", &CartesianCorridorOptions::support_frame)
      .def_rw("collision_floor_pairs", &CartesianCorridorOptions::collision_floor_pairs)
      .def_rw("collision_floors", &CartesianCorridorOptions::collision_floors);
  nb::class_<CartesianCorridorResult>(m, "CartesianCorridorResult")
      .def_ro("accepted", &CartesianCorridorResult::accepted)
      .def_ro("phase", &CartesianCorridorResult::phase)
      .def_ro("reason", &CartesianCorridorResult::reason)
      .def_ro("q_next", &CartesianCorridorResult::q_next)
      .def_ro("velocity", &CartesianCorridorResult::velocity)
      .def_ro("pose_errors", &CartesianCorridorResult::pose_errors)
      .def_ro("mode_requested", &CartesianCorridorResult::mode_requested)
      .def_ro("solver_result", &CartesianCorridorResult::solver_result)
      .def_ro("computation_time_ms", &CartesianCorridorResult::computation_time_ms)
      .def_ro("attempts", &CartesianCorridorResult::attempts)
      .def_ro("accepted_backoff", &CartesianCorridorResult::accepted_backoff);
  nb::class_<CartesianGuideResult>(m, "CartesianGuideResult")
      .def_ro("found", &CartesianGuideResult::found)
      .def_ro("q_guide", &CartesianGuideResult::q_guide)
      .def_ro("pose_errors", &CartesianGuideResult::pose_errors)
      .def_ro("seed_index", &CartesianGuideResult::seed_index)
      .def_ro("iterations", &CartesianGuideResult::iterations)
      .def_ro("total_iterations", &CartesianGuideResult::total_iterations)
      .def_ro("corridor_verified", &CartesianGuideResult::corridor_verified)
      .def_ro("rollout_steps", &CartesianGuideResult::rollout_steps)
      .def_ro("cancelled", &CartesianGuideResult::cancelled)
      .def_ro("computation_time_ms", &CartesianGuideResult::computation_time_ms);
  nb::class_<CartesianCorridorRecovery>(m, "CartesianCorridorRecovery")
      .def(nb::init<std::shared_ptr<KinematicsSolver>, std::vector<std::string>, CartesianCorridorOptions>(),
           nb::arg("solver"), nb::arg("frames"), nb::arg("options"))
      .def("start", &CartesianCorridorRecovery::start, nb::arg("q"), nb::arg("guide"),
           nb::arg("targets"), nb::arg("both_hands_free"))
      .def("find_guide", &CartesianCorridorRecovery::find_guide, nb::arg("seeds"),
           nb::arg("targets"), nb::arg("max_iterations") = 300,
           nb::call_guard<nb::gil_scoped_release>())
      .def("find_corridor_guide", &CartesianCorridorRecovery::find_corridor_guide,
           nb::arg("q"), nb::arg("seeds"), nb::arg("targets"),
           nb::arg("max_iterations") = 300, nb::arg("max_steps") = 600,
           nb::call_guard<nb::gil_scoped_release>())
      .def("step", &CartesianCorridorRecovery::step, nb::arg("q"), nb::arg("targets"),
           nb::arg("both_hands_free"))
      .def("interrupt", &CartesianCorridorRecovery::interrupt, nb::arg("reason"))
      .def("cancel_search", &CartesianCorridorRecovery::cancel_search)
      .def_prop_ro("search_running", &CartesianCorridorRecovery::search_running)
      .def_prop_ro("phase", &CartesianCorridorRecovery::phase)
      .def_prop_ro("reason", &CartesianCorridorRecovery::reason)
      .def_prop_ro("accepted_steps", &CartesianCorridorRecovery::accepted_steps);
}
