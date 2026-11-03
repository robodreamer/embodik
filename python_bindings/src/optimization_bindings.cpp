#include <embodik/optimization.hpp>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <limits>

namespace nb = nanobind;
using namespace embodik;

namespace {
SolveResult backend_failure(const std::string &backend) {
  SolveResult result;
  result.backend = "none";
  result.status = backend == "gpu" ? SolveStatus::kUnsupported : SolveStatus::kInvalidInput;
  result.message = backend == "gpu" ? "standalone GPU math solver not yet qualified" : "backend must be auto, cpu or gpu";
  return result;
}

struct Constraints {
  Eigen::MatrixXd A;
  Eigen::VectorXd l, u;
};
Constraints constraints(Eigen::Index n, const std::optional<Eigen::MatrixXd> &A,
                        const std::optional<Eigen::VectorXd> &l,
                        const std::optional<Eigen::VectorXd> &u) {
  Constraints result;
  result.A = A.value_or(Eigen::MatrixXd(0, n));
  result.l = l.value_or(Eigen::VectorXd::Constant(result.A.rows(), -std::numeric_limits<double>::infinity()));
  result.u = u.value_or(Eigen::VectorXd::Constant(result.A.rows(), std::numeric_limits<double>::infinity()));
  return result;
}
} // namespace

void bind_optimization(nb::module_ &m) {
  nb::enum_<SolveStatus>(m, "SolveStatus")
      .value("SOLVED", SolveStatus::kSolved)
      .value("INFEASIBLE", SolveStatus::kInfeasible)
      .value("NO_PROGRESS", SolveStatus::kNoProgress)
      .value("INVALID_INPUT", SolveStatus::kInvalidInput)
      .value("UNSUPPORTED", SolveStatus::kUnsupported);
  nb::class_<SolveResult>(m, "SolveResult")
      .def_prop_ro("success", &SolveResult::success)
      .def_prop_ro("x", [](const SolveResult &r) { return r.x; })
      .def_ro("status", &SolveResult::status)
      .def_ro("message", &SolveResult::message)
      .def_ro("backend", &SolveResult::backend)
      .def_ro("objective", &SolveResult::objective)
      .def_ro("max_violation", &SolveResult::max_violation)
      .def_ro("normalized_violation", &SolveResult::normalized_violation)
      .def_ro("elapsed_ms", &SolveResult::elapsed_ms)
      .def_ro("iterations", &SolveResult::iterations);
  m.def("solve_ls", [](const Eigen::MatrixXd &J, const Eigen::VectorXd &target,
          const std::optional<Eigen::MatrixXd> &A,
          const std::optional<Eigen::VectorXd> &l,
          const std::optional<Eigen::VectorXd> &u,
          double ridge, const std::optional<Eigen::VectorXd> &x0,
          double tol, unsigned int max_iter, const std::string &backend,
          const std::optional<Eigen::MatrixXd> &S,
          const std::optional<Eigen::VectorXd> &sl,
          const std::optional<Eigen::VectorXd> &su,
          const std::optional<Eigen::VectorXd> &weights) {
    if (backend != "auto" && backend != "cpu") return backend_failure(backend);
    const auto c = constraints(J.cols(), A, l, u);
    if (S) {
      if (weights && weights->size() != S->rows()) {
        SolveResult invalid;
        invalid.message = "weights must have one value per soft row";
        return invalid;
      }
      const auto soft = constraints(J.cols(), S, sl, su);
      return solve_ls(J, target, c.A, c.l, c.u,
          SoftBounds{soft.A, soft.l, soft.u, weights.value_or(Eigen::VectorXd())},
          ridge, SolveOptions{tol, max_iter, x0});
    }
    if (sl || su || weights) {
      SolveResult invalid;
      invalid.message = "soft bounds and weights require S";
      return invalid;
    }
    return solve_ls(J, target, c.A, c.l, c.u, ridge, SolveOptions{tol, max_iter, x0});
  }, nb::arg("J"), nb::arg("target"), nb::arg("A") = nb::none(),
     nb::arg("l") = nb::none(), nb::arg("u") = nb::none(), nb::kw_only(),
     nb::arg("ridge") = 0.0, nb::arg("x0") = nb::none(), nb::arg("tol") = 1e-8,
     nb::arg("max_iter") = 200, nb::arg("backend") = "auto",
     nb::arg("S") = nb::none(), nb::arg("sl") = nb::none(),
     nb::arg("su") = nb::none(), nb::arg("weights") = nb::none(),
     "Minimize .5*||J*x-target||^2 + .5*ridge*||x||^2 subject to l<=A*x<=u. "
     "Optional S, sl, su, weights add .5*sum(weights*dist(S*x,[sl,su])^2); "
     "weights default to one and soft bounds default to infinite. "
     "CPU dense arrays only; tol applies to normalized hard rows.");
  m.def("solve_qp", [](const Eigen::MatrixXd &P, const Eigen::VectorXd &q,
          const std::optional<Eigen::MatrixXd> &A,
          const std::optional<Eigen::VectorXd> &l,
          const std::optional<Eigen::VectorXd> &u,
          const std::optional<Eigen::VectorXd> &x0,
          double tol, unsigned int max_iter, const std::string &backend) {
    if (backend != "auto" && backend != "cpu") return backend_failure(backend);
    const auto c = constraints(q.size(), A, l, u);
    return solve_qp(P, q, c.A, c.l, c.u, SolveOptions{tol, max_iter, x0});
  }, nb::arg("P"), nb::arg("q"), nb::arg("A") = nb::none(),
     nb::arg("l") = nb::none(), nb::arg("u") = nb::none(), nb::kw_only(),
     nb::arg("x0") = nb::none(), nb::arg("tol") = 1e-8,
     nb::arg("max_iter") = 200, nb::arg("backend") = "auto",
     "Minimize .5*x.T*P*x+q.T*x subject to l<=A*x<=u. Initial CPU backend "
     "requires positive-definite P; other families report UNSUPPORTED.");
}
