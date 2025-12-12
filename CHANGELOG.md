# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-12

### Added
- Initial release of embodiK
- High-performance inverse kinematics solver with hierarchical task resolution
- Python bindings using nanobind
- Support for multiple task types (FrameTask, PostureTask, COMTask, JointTask, MultiJointTask)
- Position and velocity IK solving
- Optional visualization tools using Viser
- Comprehensive test suite
- Example scripts demonstrating various use cases

### Technical Details
- C++17 core library built on Pinocchio and Eigen3
- Python 3.8+ support
- Linux x86_64 support
- CMake-based build system
- scikit-build-core for Python packaging
