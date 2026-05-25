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

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions will build and publish to PyPI

The PyPI trusted publisher for `embodik` must match the tag workflow exactly:

- Owner: `robodreamer`
- Repository: `embodik`
- Workflow: `wheels.yml`
- Environment: `pypi`

The `Publish to PyPI` job declares `environment: pypi` so PyPI receives a
stable environment claim in GitHub's OIDC token. If PyPI reports
`invalid-publisher` with `environment: MISSING`, the workflow environment and
the PyPI trusted publisher configuration are out of sync.

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
