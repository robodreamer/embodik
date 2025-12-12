#!/usr/bin/env python3
"""
Setup script for SwiftIK Python package.
This provides a fallback installation method if pyproject.toml doesn't work.
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Get the directory containing this script
ROOT_DIR = Path(__file__).parent.absolute()

def get_version():
    """Get version from pyproject.toml or default."""
    try:
        import tomli
        with open(ROOT_DIR / "pyproject.toml", "rb") as f:
            data = tomli.load(f)
        return data["project"]["version"]
    except:
        return "0.1.0"

def build_cmake_extension():
    """Build the C++ extension using CMake."""
    build_dir = ROOT_DIR / "build"
    build_dir.mkdir(exist_ok=True)

    # Configure with CMake
    subprocess.run([
        "cmake",
        str(ROOT_DIR),
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={build_dir / 'install'}"
    ], cwd=build_dir, check=True)

    # Build
    subprocess.run([
        "cmake", "--build", ".", "--parallel"
    ], cwd=build_dir, check=True)

    return build_dir

class CMakeBuild(build_ext):
    """Custom build extension that uses CMake."""

    def build_extension(self, ext):
        build_cmake_extension()

if __name__ == "__main__":
    setup(
        name="swift_ik",
        version=get_version(),
        author="SwiftIK Team",
        author_email="contact@swiftik.org",
        description="High-performance inverse kinematics solver",
        long_description=(ROOT_DIR / "README.md").read_text(),
        long_description_content_type="text/markdown",
        url="https://github.com/swiftik/swift_ik",
        packages=find_packages(where="python"),
        package_dir={"": "python"},
        python_requires=">=3.8",
        install_requires=[
            "numpy>=1.20.0",
            "pinocchio>=2.6.0",
        ],
        extras_require={
            "visualization": [
                "viser>=0.1.0",
                "yourdfpy>=0.0.52",
                "spatialmath-python>=1.1.0",
            ],
            "examples": [
                "robot_descriptions>=1.0.0",
                "scipy>=1.7.0",
            ],
            "dev": [
                "pytest>=6.0",
                "pytest-cov>=2.0",
                "black>=22.0",
                "isort>=5.0",
            ]
        },
        ext_modules=[],
        cmdclass={"build_ext": CMakeBuild},
        zip_safe=False,
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: C++",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
