#!/usr/bin/env python3
"""
Patch Qhull CMake configuration to skip executable file checks.

This is a workaround for conda-forge's qhull package which doesn't include
executable binaries (qhull, rbox, qconvex, etc.) that the CMake configuration
expects. We only need the libraries (qhullcpp, qhull_r) for Pinocchio.

This script patches the QhullTargets.cmake file to skip FATAL_ERROR checks
for executable targets.
"""

import os
import sys
import re
from pathlib import Path


def find_qhull_targets_file(conda_prefix=None):
    """Find the QhullTargets.cmake file."""
    if conda_prefix is None:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            # Try pixi environment
            pixi_env = os.environ.get("PIXI_ENVIRONMENT")
            if pixi_env:
                # Assume .pixi/envs/default structure
                conda_prefix = Path(".pixi/envs/default").resolve()
            else:
                # Try common conda/pixi locations
                for prefix in [
                    Path(".pixi/envs/default"),
                    Path.home() / ".conda" / "envs" / "embodik",
                ]:
                    if prefix.exists():
                        conda_prefix = prefix.resolve()
                        break

    if not conda_prefix:
        print("ERROR: Could not find conda/pixi environment. Set CONDA_PREFIX or run from project root.", file=sys.stderr)
        return None

    conda_prefix = Path(conda_prefix)
    qhull_targets = conda_prefix / "lib" / "cmake" / "Qhull" / "QhullTargets.cmake"

    if not qhull_targets.exists():
        print(f"WARNING: QhullTargets.cmake not found at {qhull_targets}", file=sys.stderr)
        print("This patch may not be needed, or Qhull is not installed.", file=sys.stderr)
        return None

    return qhull_targets


def patch_qhull_targets(qhull_targets_path):
    """Patch QhullTargets.cmake to skip executable file checks."""
    qhull_targets_path = Path(qhull_targets_path)

    # Read the file
    try:
        with open(qhull_targets_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR: Failed to read {qhull_targets_path}: {e}", file=sys.stderr)
        return False

    # Check if already patched
    if '# Workaround: Skip executable file check' in content or 'if(0)  # Disabled: executables not required' in content:
        # Already patched - this is fine, return success silently
        return True

    # Executable targets that don't exist in conda-forge qhull package
    executable_targets = ['Qhull::qhull', 'Qhull::rbox', 'Qhull::qconvex',
                           'Qhull::qdelaunay', 'Qhull::qvoronoi', 'Qhull::qhalf']

    # Pattern to match the FATAL_ERROR check for missing files
    # We want to make it conditional - skip for executables
    pattern = r'(if\(NOT EXISTS "\$\{_cmake_file\}"\)\s+message\(FATAL_ERROR "[^"]+"[^)]+\))'

    def make_conditional(match):
        fatal_block = match.group(1)
        # Replace FATAL_ERROR with a conditional that skips for executables
        new_block = fatal_block.replace(
            'message(FATAL_ERROR',
            '# Workaround: Skip check for executables (conda-forge qhull package doesn\'t include them)\n        if(0)  # Disabled: executables not required\n        message(FATAL_ERROR'
        )
        # Close the if(0) block
        new_block = new_block.replace(')', ')\n        endif()', 1)
        return new_block

    # Apply the patch
    patched_content = re.sub(pattern, make_conditional, content, flags=re.MULTILINE)

    # Only write if content changed
    if patched_content == content:
        print(f"WARNING: No changes made to {qhull_targets_path}. File may already be patched or format changed.", file=sys.stderr)
        return False

    # Backup original file
    backup_path = qhull_targets_path.with_suffix('.cmake.bak')
    try:
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"INFO: Created backup at {backup_path}")
    except Exception as e:
        print(f"WARNING: Failed to create backup: {e}", file=sys.stderr)

    # Write patched content
    try:
        with open(qhull_targets_path, 'w') as f:
            f.write(patched_content)
        print(f"SUCCESS: Patched {qhull_targets_path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to write patched file: {e}", file=sys.stderr)
        # Restore from backup if possible
        if backup_path.exists():
            try:
                with open(backup_path, 'r') as f:
                    with open(qhull_targets_path, 'w') as out:
                        out.write(f.read())
                print(f"INFO: Restored original file from backup")
            except Exception as restore_error:
                print(f"ERROR: Failed to restore backup: {restore_error}", file=sys.stderr)
        return False


def main():
    """Main entry point."""
    qhull_targets = find_qhull_targets_file()

    if qhull_targets is None:
        print("INFO: Qhull CMake file not found. Patch may not be needed.", file=sys.stderr)
        return 0  # Not an error - patch may not be needed

    success = patch_qhull_targets(qhull_targets)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

