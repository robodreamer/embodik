"""Native backend discovery is lazy and never changes mathematical dispatch."""

import importlib
import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import ModuleType

import numpy as np
import pytest

import embodik as eik


def catalog():
    return importlib.import_module("embodik.backends")


@pytest.mark.parametrize("alias", ["native", "cpu", "auto"])
def test_native_handle_preserves_exact_callables_and_legacy_identity(alias):
    module = catalog()
    native = importlib.import_module("embodik._embodik_impl")
    handle = module.get(alias)
    assert handle.info.id == "native" and handle.info.role == "numerical"
    assert handle.solve_ls is native.solve_ls is eik.solve_ls
    assert handle.solve_qp is native.solve_qp is eik.solve_qp
    assert handle.solve_ls(np.eye(1), np.ones(1)).backend == "cpu"


def test_native_metadata_is_deeply_immutable_and_uses_extension_version(monkeypatch):
    module = catalog()
    native = importlib.import_module("embodik._embodik_impl")
    monkeypatch.setattr(eik, "__version__", "999.installed-metadata-is-not-the-binary")
    info = module.info("native")
    assert info.version == native.__version__
    assert info.available
    assert info.devices == ("cpu",) and info.dtypes == ("float64",)
    assert isinstance(info.capabilities, tuple) and info.capabilities
    assert info.qualifications == ()
    assert not hasattr(info, "qualified")
    with pytest.raises(FrozenInstanceError):
        info.id = "other"
    with pytest.raises(TypeError):
        info.devices[0] = "cuda"
    handle = module.get("native")
    with pytest.raises(FrozenInstanceError):
        handle.solve_ls = lambda: None


def test_catalog_only_advertises_implemented_numerical_handles():
    module = catalog()
    assert module.available() == ("native",)
    assert module.available(role="control") == ()
    assert module.available(role="reference") == ()


@pytest.mark.parametrize("role", ["physics", "", None])
def test_unknown_roles_are_explicit_errors(role):
    with pytest.raises(ValueError, match="role"):
        catalog().available(role=role)


@pytest.mark.parametrize("name", ["missing", "gpu", None, []])
def test_unknown_provider_has_no_implicit_fallback(name):
    for action in (catalog().info, catalog().get):
        with pytest.raises(ValueError, match="backend"):
            action(name)


def test_unloaded_native_is_describable_but_not_usable(monkeypatch):
    module = catalog()
    monkeypatch.delitem(sys.modules, "embodik._embodik_impl")
    info = module.info("native")
    assert not info.available and info.version is None and info.reason
    assert module.available() == ()
    with pytest.raises(RuntimeError, match="native"):
        module.get("native")


def test_native_without_both_math_entry_points_is_not_advertised(monkeypatch):
    module = catalog()
    incomplete = ModuleType("embodik._embodik_impl")
    incomplete.__version__ = "old-build"
    incomplete.solve_ls = lambda: None
    monkeypatch.setitem(sys.modules, "embodik._embodik_impl", incomplete)
    descriptor = module.info("native")
    assert not descriptor.available and descriptor.version == "old-build"
    assert module.available() == ()
    with pytest.raises(RuntimeError, match="entry points"):
        module.get("native")


@pytest.mark.parametrize(
    "case", ["ls", "rank_loss", "soft", "infeasible", "qp", "psd", "gpu", "unknown"]
)
def test_handle_results_match_direct_native_results(case):
    handle = catalog().get("native")
    kwargs = {}
    if case in ("qp", "psd"):
        fn, wrapped = eik.solve_qp, handle.solve_qp
        args = (np.eye(2) if case == "qp" else np.zeros((2, 2)), -np.ones(2))
    else:
        fn, wrapped = eik.solve_ls, handle.solve_ls
        args = (np.eye(2), np.ones(2))
        if case == "rank_loss":
            args = (np.ones((1, 2)), np.ones(1))
        elif case == "soft":
            kwargs = dict(S=np.eye(2), sl=2 * np.ones(2), weights=np.ones(2))
        elif case == "infeasible":
            kwargs = dict(A=np.zeros((1, 2)), l=np.ones(1), u=2 * np.ones(1))
        elif case in ("gpu", "unknown"):
            kwargs = dict(backend="gpu" if case == "gpu" else "missing")
    direct, actual = fn(*args, **kwargs), wrapped(*args, **kwargs)
    for field in ("status", "success", "message", "backend", "iterations"):
        assert getattr(actual, field) == getattr(direct, field)
    np.testing.assert_array_equal(actual.x, direct.x)
    for field in ("objective", "max_violation", "normalized_violation"):
        np.testing.assert_equal(getattr(actual, field), getattr(direct, field))


def test_root_catalog_access_is_lazy_and_adds_no_optional_engine_imports():
    code = """
import sys
import embodik as eik
assert 'embodik.backends' not in sys.modules
before=set(sys.modules)
assert eik.backends.info('native').available
added=set(sys.modules)-before
for prefix in ('jax','torch','elastiqp','frax','frdv','clarabel','osqp'):
    assert not any(name==prefix or name.startswith(prefix+'.') for name in added), (prefix,added)
assert eik.backends is sys.modules['embodik.backends']
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
