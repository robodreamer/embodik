# Contributing to embodiK

Thank you for your interest in contributing to embodiK! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Check if the issue already exists
2. Use a clear, descriptive title
3. Provide a minimal example that reproduces the issue
4. Include your environment details (OS, Python version, etc.)

### Submitting Pull Requests

1. **Fork the repository** and create a branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Run tests** locally before submitting
6. **Submit a pull request** with a clear description

### Development Setup

#### Option 1: Using Pixi (Recommended)

Pixi provides a reproducible development environment with all dependencies managed automatically.

1. Install Pixi:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/embodik.git
   cd embodik
   ```

3. Install the package in development mode:
   ```bash
   pixi run install
   # Or with auto-rebuild on import:
   pixi run install-rebuild
   ```

4. Activate the pixi environment:
   ```bash
   pixi shell
   ```

All system dependencies (CMake, Eigen, Pinocchio, etc.) are automatically managed by pixi.

#### Option 2: Manual Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/embodik.git
   cd embodik
   ```

2. Install system dependencies:
   - CMake 3.16+
   - C++17 compiler (GCC 7+, Clang 5+)
   - Eigen3 development headers (`libeigen3-dev` on Ubuntu)
   - Pinocchio library

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

### Coding Standards

- **Python**: Follow PEP 8 style guide
- **C++**: Follow the existing code style (C++17 standard)
- **Documentation**: Use docstrings for all public functions/classes
- **Tests**: Write tests for new features and bug fixes

### Running Tests

**With Pixi:**
```bash
pixi run test
pixi run test-verbose
pixi run test-cov  # With coverage (requires dev feature)
```

**Without Pixi:**
```bash
# Run Python tests
pytest

# Run with coverage
pytest --cov=embodik --cov-report=html

# Run C++ tests
cd build && ctest
```

### Code Formatting

We use `black` and `isort` for Python code formatting:

**With Pixi:**
```bash
pixi run format      # Format code
pixi run lint        # Check formatting
```

**Without Pixi:**
```bash
# Format code
black python/ test/ examples/
isort python/ test/ examples/

# Check formatting
black --check python/ test/ examples/
isort --check python/ test/ examples/
```

### Documentation

- Update docstrings for any new public APIs
- Update relevant documentation files in `docs/`
- Build docs locally to verify:
  ```bash
  # With Pixi
  pixi run docs-serve

  # Without Pixi
  mkdocs serve
  ```

### Commit Messages

Use clear, descriptive commit messages:
- Start with a verb in imperative mood (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters
- Add more details in the body if needed

Example:
```
Add support for joint limit constraints

- Implement joint limit checking in kinematics solver
- Add tests for limit enforcement
- Update documentation with examples
```

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

Thank you for contributing to embodiK!
