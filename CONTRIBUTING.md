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

Pixi is the canonical development environment. It manages CMake, Eigen,
Pinocchio, Nanobind, test tools, and docs tools.

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
   ```

4. Optional: activate the Pixi shell:
   ```bash
   pixi shell
   ```

For rebuilds after native C++ changes, run `pixi run install-rebuild`.
For manual source-build debugging outside Pixi, use the maintained setup in the
[Installation Guide](docs/installation.md#if-pip-builds-from-source).

### Coding Standards

- **Python**: Follow PEP 8 style guide
- **C++**: Follow the existing code style (C++17 standard)
- **Documentation**: Use docstrings for all public functions/classes
- **Tests**: Write tests for new features and bug fixes

### Running Tests

```bash
pixi run test
```

For a targeted test:

```bash
pixi run python -m pytest test/test_robot_model.py
```

### Code Formatting

We use `black` and `isort` for Python code formatting:

```bash
pixi run format
pixi run lint
```

### Documentation

- Update docstrings for any new public APIs
- Update relevant documentation files in `docs/`
- Build docs locally to verify:
  ```bash
  pixi run docs-build
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
