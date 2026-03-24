from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckPlan:
    run_build: bool = False
    run_docs: bool = False
    run_tests: bool = False
    run_lint: bool = False
    test_targets: list[str] | None = None


def _run_capture(command: list[str]) -> str:
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "command failed"
        raise RuntimeError(f"{' '.join(command)}: {msg}")
    return proc.stdout


def _collect_changed_files() -> list[str]:
    # Includes staged, unstaged, and untracked paths.
    raw = _run_capture(["git", "status", "--porcelain"])
    files: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        path_part = line[3:]
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1]
        files.append(path_part)
    return sorted(set(files))


def _is_python_file(path: str) -> bool:
    return path.endswith(".py")


def _under(path: str, prefix: str) -> bool:
    return path == prefix.rstrip("/") or path.startswith(prefix)


def _plan_checks(changed_files: list[str]) -> CheckPlan:
    plan = CheckPlan(test_targets=[])
    for path in changed_files:
        if _under(path, "cpp_core/") or _under(path, "python_bindings/") or path in {
            "CMakeLists.txt",
            "pyproject.toml",
            "pixi.toml",
        }:
            plan.run_build = True
            plan.run_tests = True

        if _under(path, "docs/") or path in {"mkdocs.yml"}:
            plan.run_docs = True

        if (
            _under(path, "python/")
            or _under(path, "examples/")
            or _under(path, "scripts/")
            or _under(path, "test/")
        ):
            plan.run_tests = True

        if _is_python_file(path) and (
            _under(path, "python/")
            or _under(path, "test/")
            or _under(path, "examples/")
            or _under(path, "scripts/")
        ):
            plan.run_lint = True

        if _under(path, "test/") and path.endswith(".py"):
            plan.test_targets.append(path)

    if not changed_files:
        plan.test_targets = []
    elif not plan.test_targets:
        plan.test_targets = []
    else:
        plan.test_targets = sorted(set(plan.test_targets))

    return plan


def _print_plan(changed_files: list[str], plan: CheckPlan) -> None:
    print("Changed files:")
    if not changed_files:
        print("  (none)")
    else:
        for path in changed_files:
            print(f"  - {path}")

    print("\nPlanned checks:")
    if not any([plan.run_build, plan.run_docs, plan.run_tests, plan.run_lint]):
        print("  - No checks required for current changes.")
        return

    if plan.run_lint:
        print("  - pixi run lint")
    if plan.run_build:
        print("  - pixi run build")
    if plan.run_tests:
        if plan.test_targets:
            joined = " ".join(plan.test_targets)
            print(f"  - pytest {joined}")
            print("    (via pixi env: pixi run pytest ...)")
        else:
            print("  - pixi run test")
    if plan.run_docs:
        print("  - pixi run docs-build")


def _run_command(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    proc = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}")


def _execute(plan: CheckPlan) -> None:
    if not any([plan.run_build, plan.run_docs, plan.run_tests, plan.run_lint]):
        print("\nNo checks to run.")
        return

    if plan.run_lint:
        _run_command(["pixi", "run", "lint"])
    if plan.run_build:
        _run_command(["pixi", "run", "build"])
    if plan.run_tests:
        if plan.test_targets:
            _run_command(["pixi", "run", "pytest", *plan.test_targets])
        else:
            _run_command(["pixi", "run", "test"])
    if plan.run_docs:
        _run_command(["pixi", "run", "docs-build"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Path-aware verification helper for changed files."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute planned checks (default is dry-run).",
    )
    args = parser.parse_args()

    try:
        changed_files = _collect_changed_files()
        plan = _plan_checks(changed_files)
        _print_plan(changed_files, plan)
        if args.run:
            _execute(plan)
    except Exception as exc:  # noqa: BLE001
        print(f"verify_changed error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
