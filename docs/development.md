# Development Guide

Guide for contributing to EmbodiK and developing with the source code.

## Development Setup

### Clone Repository

```bash
git clone https://github.com/embodik/embodik.git
cd embodik
```

### Install in Development Mode

```bash
pip install -e ".[dev]"
```

This installs:
- EmbodiK in editable mode
- Development dependencies (pytest, black, isort, etc.)

### Build from Source

```bash
# Install system dependencies (see Installation guide)
bash build.sh
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

### Using CMake (Direct)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Using pip (Recommended)

```bash
pip install -e .
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_robot_model.py

# Run with coverage
pytest --cov=embodik --cov-report=html
```

## Code Style

EmbodiK follows PEP 8 for Python code:

```bash
# Format code
black python/

# Sort imports
isort python/

# Check style
flake8 python/
```

## Documentation

### Building Documentation

```bash
# Install docs dependencies
pip install mkdocs-material mkdocstrings[python]

# Serve locally
mkdocs serve

# Build static site
mkdocs build
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
5. Ensure all tests pass: `pytest`
6. Format code: `black . && isort .`
7. Submit a pull request

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions will build and publish to PyPI

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
