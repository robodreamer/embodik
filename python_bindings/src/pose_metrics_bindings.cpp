#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>

#include <embodik/pose_metrics.hpp>

namespace nb = nanobind;
using namespace embodik;

void bind_pose_metrics(nb::module_ &m) {
  m.def("joint_limit_distance", &joint_limit_distance,
        nb::arg("q"), nb::arg("q_lower"), nb::arg("q_upper"),
        nb::arg("epsilon") = 0.04,
        "Per-joint and aggregate joint-limit distance metric.");

  m.def("joint_limit_distance_gradient", &joint_limit_distance_gradient,
        nb::arg("q"), nb::arg("q_lower"), nb::arg("q_upper"),
        nb::arg("epsilon") = 0.04,
        "Analytical gradient of joint-limit distance (descent direction).");

  m.def("velocity_manipulability", &velocity_manipulability,
        nb::arg("jacobian"),
        "Velocity manipulability: sqrt(det(J * J^T)).");

  m.def("singularity_joint_limit_metric", &singularity_joint_limit_metric,
        nb::arg("q"), nb::arg("jacobian"), nb::arg("q_lower"), nb::arg("q_upper"),
        nb::arg("epsilon") = 0.04,
        "Combined singularity + joint-limit metric.");
}
