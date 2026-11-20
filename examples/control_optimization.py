"""Run small control problems with the standalone CPU LS and PD-QP interfaces.

    pixi run python examples/control_optimization.py
    pixi run python examples/control_optimization.py --case fit

No robot assets, viewer or optional dependencies are needed. GPU dispatch,
general semidefinite QPs and linear programs are not supplied by these APIs yet.
"""

import argparse
import json

import numpy as np

import embodik


def problems() -> dict[str, dict]:
    """Assemble ordinary matrices directly from three control formulations."""
    # Allocate bounded actuator efforts u to desired total/differential effort.
    # min .5 ||B u - desired||² + .5 ridge ||u||², 0 <= u <= .8.
    B = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0]])
    allocation = dict(
        method="ls",
        J=B,
        target=np.array([2.6, 1.2]),
        ridge=0.02,
        A=np.eye(4),
        l=np.zeros(4),
        u=np.full(4, 0.8),
    )

    # Project a command onto a simplex: min .5 ||x - command||²,
    # x.sum() = 1, x >= 0. The QP omits the constant .5 ||command||².
    command = np.array([2.0, -1.0])
    projection = dict(
        method="qp",
        P=np.eye(2),
        q=-command,
        A=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        l=np.array([1.0, 0.0, 0.0]),
        u=np.array([1.0, np.inf, np.inf]),
    )

    # Fit affine measurements M x + bias to desired values with diagonal weights:
    # min .5 ||sqrt(W)(M x + bias - desired)||² + .5 ridge ||x||².
    # Fold the bias into the target; apply sqrt(weights) to BOTH residual sides.
    M = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    bias, desired = np.array([0.2, -0.1, 0.3]), np.array([1.0, -0.4, 0.7])
    root_weight = np.sqrt(np.array([4.0, 1.0, 0.25]))
    fit = dict(
        method="ls",
        J=root_weight[:, None] * M,
        target=root_weight * (desired - bias),
        ridge=0.1,
        A=np.eye(2),
        l=np.full(2, -0.5),
        u=np.full(2, 0.5),
    )
    return dict(allocation=allocation, projection=projection, fit=fit)


def solve(name: str, *, backend: str = "cpu") -> dict:
    """Return owned solution data, leaving failed/unsupported results explicit."""
    problem = problems()[name]
    method = problem.pop("method")
    result = (
        embodik.solve_ls(**problem, backend=backend)
        if method == "ls"
        else embodik.solve_qp(**problem, backend=backend)
    )
    report = dict(
        status=result.status.name,
        backend=result.backend,
        message=result.message,
        x=result.x.tolist(),
        objective=float(result.objective) if result.success else None,
    )
    if result.success:
        report["max_hard_violation"] = float(result.max_violation)
        if method == "ls":
            report["achieved"] = (problem["J"] @ result.x).tolist()
            report["target"] = problem["target"].tolist()
            report["ridge"] = problem["ridge"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "allocation", "projection", "fit"), default="all")
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    names = problems() if args.case == "all" else [args.case]
    reports = {name: solve(name, backend=args.backend) for name in names}
    print(json.dumps(reports, indent=2, allow_nan=False))
    return 0 if all(r["status"] == "SOLVED" for r in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
