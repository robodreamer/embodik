"""Lazy numerical-backend discovery, separate from robot execution admission.

The native handle exposes the existing native functions unchanged. Configure a
handle outside the hot loop and normally omit its functions' legacy ``backend``
keyword; those callables retain their original signature and dispatch semantics.
Optional numerical providers and control/reference handles are not registered yet.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .._embodik_impl import SolveResult


@dataclass(frozen=True)
class Qualification:
    """Evidence scoped to an exact version, device, dtype and problem/profile."""

    version: str
    device: str
    dtype: str
    problem: str
    evidence_id: str
    robot_profile: str | None = None


@dataclass(frozen=True)
class BackendInfo:
    """Immutable runtime capabilities; availability is not a quality certificate."""

    id: str
    role: str
    version: str | None
    devices: tuple[str, ...]
    dtypes: tuple[str, ...]
    capabilities: tuple[str, ...]
    available: bool
    reason: str | None = None
    qualifications: tuple[Qualification, ...] = ()


@dataclass(frozen=True)
class Backend:
    """Numerical-only handle; successful results do not authorize robot motion."""

    info: BackendInfo
    solve_ls: Callable[..., SolveResult]
    solve_qp: Callable[..., SolveResult]


_ALIASES = {"native": "native", "cpu": "native", "auto": "native"}
_ROLES = ("numerical", "control", "reference")
_NATIVE_CAPABILITIES = (
    "least_squares",
    "rank_deficient_least_squares",
    "positive_definite_qp",
    "linear_hard_bounds",
    "squared_soft_bounds",
    "primal_initial_guess",
)


def _native_module() -> ModuleType | None:
    # Importing embodik already attempted to load its extension. Discovery must
    # describe that state, not retry loading a possibly incompatible library.
    return sys.modules.get("embodik._embodik_impl")


def _canonical(name: str) -> str:
    if not isinstance(name, str) or name not in _ALIASES:
        raise ValueError(f"Unknown numerical backend: {name!r}")
    return _ALIASES[name]


def info(name: str = "native") -> BackendInfo:
    """Describe a known provider even when its native runtime is unavailable.

    ``native``, ``cpu`` and ``auto`` identify the same native numerical provider.
    Version identifies the loaded extension, not installed package metadata.
    Dtypes describe arithmetic; existing array conversion semantics are unchanged.
    Unknown provider IDs raise ValueError. No optional engine is imported.
    """
    canonical = _canonical(name)
    module = _native_module()
    usable = module is not None and all(
        callable(getattr(module, entry, None)) for entry in ("solve_ls", "solve_qp")
    )
    version = getattr(module, "__version__", None)
    return BackendInfo(
        id=canonical,
        role="numerical",
        version=version if isinstance(version, str) else None,
        devices=("cpu",),
        dtypes=("float64",),
        capabilities=_NATIVE_CAPABILITIES,
        available=usable,
        reason=None if usable else "Native standalone math entry points are not loaded",
    )


def available(*, role: str = "numerical") -> tuple[str, ...]:
    """Return canonical usable IDs for a role, without loading optional engines.

    Only numerical handles are registered in this initial catalog. Valid control
    and reference roles currently return an empty tuple; unknown roles raise
    ValueError. Runtime availability does not imply robot-profile qualification.
    """
    if not isinstance(role, str) or role not in _ROLES:
        raise ValueError(f"Unknown backend role: {role!r}; expected one of {_ROLES}")
    if role != "numerical":
        return ()
    return ("native",) if info("native").available else ()


def get(name: str = "native") -> Backend:
    """Get a numerical handle without changing default solver dispatch.

    Unknown IDs raise ValueError; known but unusable native math raises
    RuntimeError. There is no fallback. Resolve the handle once, before a hot
    loop: solve attributes are the exact native function objects, not wrappers.
    Native results retain their existing ``backend == 'cpu'`` device label.
    """
    descriptor = info(name)
    if not descriptor.available:
        raise RuntimeError(f"Backend {descriptor.id!r} is unavailable: {descriptor.reason}")
    module = _native_module()
    return Backend(descriptor, module.solve_ls, module.solve_qp)


__all__ = ["Backend", "BackendInfo", "Qualification", "available", "get", "info"]
