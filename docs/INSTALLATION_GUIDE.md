# Installation Guide: Pixi vs pip

This guide explains when to use **Pixi** vs **pip install** for embodiK.

## Quick Decision Guide

| Scenario | Use | Why |
|----------|-----|-----|
| **End user installing from PyPI** | `pip install embodik` | Simple, standard Python package installation |
| **Developer working on embodiK** | Pixi | Automatic dependency management, reproducible environment |
| **CI/CD pipelines** | Pixi (recommended) or pip | Pixi ensures consistency; pip works if system deps pre-installed |
| **System without conda/pixi** | pip | Fallback option, requires manual system dependency installation |
| **Quick test/experiment** | pip | Faster if you already have system dependencies |

## Detailed Comparison

### Pixi Installation (Recommended for Development)

**What it does:**
- ✅ Manages **all dependencies** automatically (system + Python)
- ✅ Creates isolated, reproducible environment
- ✅ No manual system dependency installation needed
- ✅ Works across different Linux distributions
- ✅ Ensures consistent versions across team

**When to use:**
- You're developing embodiK or contributing to it
- You want reproducible builds
- You're working in a team environment
- You don't want to manually install CMake, Eigen, Pinocchio, etc.

**Installation:**
```bash
# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Clone repository
git clone https://github.com/embodik/embodik.git
cd embodik

# Install in development mode
pixi run install

# Activate environment
pixi shell
```

**How it works:**
1. Pixi reads `pixi.toml` configuration
2. Downloads and installs system dependencies from conda-forge:
   - CMake, ninja (build tools)
   - Eigen, Pinocchio (C++ libraries)
   - nanobind, scikit-build-core (Python build tools)
3. Uses `pip install -e . --no-build-isolation` to install the package
4. The `--no-build-isolation` flag tells pip to use pixi's environment instead of creating an isolated build environment

**Key advantage:** Everything is managed automatically. No need to install system packages manually.

---

### pip Installation (Standard Python Package Installation)

**What it does:**
- ✅ Standard Python package installation
- ✅ Works with any Python environment (venv, conda, system)
- ✅ Requires system dependencies to be pre-installed
- ✅ Simpler for end users who just want to use the package

**When to use:**
- You're an end user installing from PyPI
- You already have system dependencies installed
- You prefer standard Python tooling
- You're in an environment where pixi isn't available

**Installation:**
```bash
# Install system dependencies first (Ubuntu example)
sudo apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    pkg-config

# Install Pinocchio (see installation.md for options)
sudo apt-get install robotpkg-pinocchio

# Install from PyPI
pip install embodik

# Or from source
git clone https://github.com/embodik/embodik.git
cd embodik
pip install -e .
```

**How it works:**
1. You manually install system dependencies (CMake, Eigen, Pinocchio)
2. pip installs Python dependencies (numpy, etc.)
3. scikit-build-core builds the C++ extensions using your system's CMake
4. Package is installed in your Python environment

**Key requirement:** System dependencies must be installed separately before pip install.

---

## How They Coexist

### For End Users (PyPI Installation)

**Use pip:**
```bash
pip install embodik
```

This is the standard way. Users don't need pixi - they just install the pre-built package from PyPI. The package maintainers handle building wheels with all dependencies.

### For Developers

**Option 1: Pixi (Recommended)**
```bash
pixi run install
```

**Option 2: pip (If you prefer)**
```bash
# Install system deps manually first
pip install -e .
```

Both work! Pixi just automates the system dependency management.

---

## Technical Details: How They Work Together

### The `--no-build-isolation` Flag

When using pixi, the key is this flag:
```bash
pip install -e . --no-build-isolation
```

**What it does:**
- **Without `--no-build-isolation`**: pip creates an isolated build environment, downloads its own cmake/ninja/nanobind
- **With `--no-build-isolation`**: pip uses the current environment (pixi's conda environment) which already has cmake/ninja/nanobind

This is why pixi works seamlessly - it provides the dependencies, and pip uses them instead of downloading its own.

### pyproject.toml vs pixi.toml

- **`pyproject.toml`**: Defines Python package metadata, build system, Python dependencies
- **`pixi.toml`**: Defines system dependencies and development workflow

They work together:
- pixi.toml provides the environment (system deps)
- pyproject.toml defines the package (Python deps, build config)
- pip reads pyproject.toml but uses pixi's environment

---

## Migration Path

### If you're currently using pip:

**You can switch to pixi anytime:**
```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# In your project directory
pixi run install
```

**Or continue with pip:**
- Just ensure system dependencies are installed
- Works exactly the same as before

### If you're using pixi:

**You can still use pip commands:**
```bash
pixi shell  # Activate pixi environment
pip install some-package  # Works normally
```

---

## CI/CD Considerations

### Using Pixi in CI (Recommended)
```yaml
- uses: prefix-dev/setup-pixi@v1
- run: pixi run test
```

**Benefits:**
- Consistent environment across all CI runs
- No need to install system dependencies manually
- Faster setup

### Using pip in CI
```yaml
- run: sudo apt-get install cmake libeigen3-dev ...
- run: pip install embodik
```

**Works but:**
- Requires manual system dependency installation
- More steps in CI configuration
- Potential version inconsistencies

---

## Summary

| Aspect | Pixi | pip |
|--------|------|-----|
| **System deps** | Automatic | Manual |
| **Python deps** | Automatic | Automatic |
| **Reproducibility** | High | Medium |
| **Ease of use** | High (dev) | High (end user) |
| **Setup time** | Fast (first time) | Fast (if deps exist) |
| **Best for** | Development | End users |

**Bottom line:**
- **Developers**: Use Pixi for automatic dependency management
- **End users**: Use pip for standard Python package installation
- **Both work**: Choose based on your preference and environment
