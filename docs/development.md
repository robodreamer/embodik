# Development Guide

Guide for contributing to EmbodiK and developing with the source code.

## Development Setup

### Clone Repository

```bash
git clone https://github.com/robodreamer/embodik.git
cd embodik
```

### Install With Pixi

```bash
pixi run install
```

Pixi is the canonical development environment. It installs EmbodiK in editable
mode and manages Pinocchio, Eigen, Nanobind, CMake, test tools, and docs tools.

### Rebuild After Native Changes

```bash
pixi run install-rebuild
```

## Project Structure

```
embodik/
├── cpp_core/           # C++ core library
│   ├── include/        # Header files
│   └── src/            # Source files
├── python/             # Python package
│   └── embodik/       # Package source
├── python_bindings/    # Nanobind bindings
│   └── src/            # Binding code
├── examples/           # Example scripts
├── test/               # Test suite
├── docs/               # Documentation
└── CMakeLists.txt      # CMake configuration
```

## Building

### Using CMake (Direct Debugging)

Use direct CMake only when debugging the native build. Otherwise use Pixi.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Using pip (Source-Build Fallback)

```bash
python -m pip install --no-build-isolation -e .
```

If CMake cannot find Pinocchio, Eigen, or URDFDOM, use the source-build
fallback in the [Installation Guide](installation.md).

## Testing

Run the test suite:

```bash
# Run all tests
pixi run test

# Run specific test file
pixi run python -m pytest test/test_robot_model.py

# Hardware-style seed recovery (joint limits + self-collision)
pixi run python -m pytest test/test_hardware_seed_recovery.py
```

## Code Style

EmbodiK follows PEP 8 for Python code:

```bash
# Format code
pixi run format

# Check style
pixi run lint
```

## Documentation

### Building Documentation

```bash
# Build static site
pixi run docs-build
```

### Writing Documentation

- API documentation is auto-generated from docstrings
- Add docstrings to all public functions and classes
- Use NumPy-style docstrings for consistency

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure relevant tests pass: `pixi run test`
6. Format code: `pixi run format`
7. Submit a pull request

## Release Process

### Versioning Contract

EmbodiK follows Semantic Versioning while it remains in initial `0.y.z`
development. The public API includes exported C++ headers, Python bindings, and
documented solver, task, constraint, result, and supported-model behavior.

- **Patch (`0.Y.Z`)**: backward-compatible bug fixes, documentation, CI/build
  changes, performance work, and implementation refinements that do not add a
  public capability.
- **Minor (`0.Y.0`)**: new public solvers, tasks, constraints, behavior modes,
  supported model families, or deprecations. An unavoidable compatibility
  change during pre-1.0 development also requires a minor release and prominent
  migration notes.
- **`1.0.0`**: declares that the supported public API is stable and that future
  incompatible changes require a major release.

A release may include lower-level changes together with its highest applicable
category. This policy applies prospectively from `0.21.0`; earlier `0.20.x`
release numbers are not reclassified.

1. Bump the version with `pixi run version --bump <patch|minor|major>`. The
   command updates `pyproject.toml`; manually synchronize `pixi.toml` and
   `CMakeLists.txt`, then verify all three files contain the same version.
2. Move the release notes into a dated version section in `CHANGELOG.md`.
3. Open and merge the release pull request into `main`.
4. The `Prepare Release` workflow validates the synchronized version and dated
   changelog entry, creates the missing `v<version>` tag, and dispatches
   `wheels.yml` on that tag.
5. The tag workflow builds Linux/macOS wheels and the source distribution,
   publishes them to PyPI through trusted publishing, and creates the GitHub
   release after publication succeeds.

The merge-triggered workflow is idempotent: an existing version tag makes a
normal `main` push a no-op. For release recovery, manually dispatch `release.yml`
on `main`; it reuses the existing tag and reruns the package workflow.

The release workflow explicitly dispatches `wheels.yml` after creating the tag.
GitHub suppresses ordinary workflow runs caused by tag pushes made with
`GITHUB_TOKEN`, while `workflow_dispatch` is allowed to start a new run.

The PyPI trusted publisher for `embodik` must match the tag workflow exactly:

- Owner: `robodreamer`
- Repository: `embodik`
- Workflow: `wheels.yml`
- Environment: `pypi`

The `Publish to PyPI` job declares `environment: pypi` so PyPI receives a
stable environment claim in GitHub's OIDC token. If PyPI reports
`invalid-publisher`, compare the rendered claims from the Actions log against
the PyPI trusted publisher configuration. A valid token with
`environment: pypi` can still fail when the PyPI project has no matching
publisher for the owner, repository, workflow filename, and environment.

## Debugging

### C++ Extension Issues

If the C++ extension fails to load:

```python
import sys
print(sys.path)  # Check Python path
import embodik._embodik_impl  # Try direct import
```

### CMake Debugging

Enable verbose CMake output:

```bash
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
```

## Questions?

- Open an issue on GitHub
- Check existing documentation
- Review example code in `examples/`
