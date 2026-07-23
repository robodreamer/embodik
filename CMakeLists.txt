cmake_minimum_required(VERSION 3.16)
project(embodik VERSION 0.20.14)

# Python is used to auto-discover Pinocchio (PyPI `pin`) CMake config during sdist builds.
find_package(Python REQUIRED COMPONENTS Interpreter)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CheckIPOSupported)

# macOS + Xcode CLT: Clang searches an incomplete libc++ under CommandLineTools before the SDK,
# which breaks <cmath> / std headers. Prefer the active macOS SDK's libc++ (Pixi, local pip, CI).
if(APPLE)
    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-path
        OUTPUT_VARIABLE _EMBODIK_MACOS_SDK
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_EMBODIK_MACOS_SDK AND EXISTS "${_EMBODIK_MACOS_SDK}/usr/include/c++/v1/cmath")
        include_directories(SYSTEM "${_EMBODIK_MACOS_SDK}/usr/include/c++/v1")
    endif()
endif()

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(EMBODIK_ENABLE_IPO "Enable interprocedural optimization for release builds" ON)

# If the user doesn't have a system Pinocchio install, try to locate the CMake config
# shipped with the PyPI `pin` wheel (import name: `pinocchio`).
#
# This helps `pip install` from sdist (when a wheel isn't available) in a plain venv.
if(NOT DEFINED pinocchio_DIR AND NOT DEFINED ENV{pinocchio_DIR})
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            [=[
import pathlib
import sys

try:
    import importlib.metadata as im
    prefix = pathlib.Path(im.distribution("pin").locate_file("cmeel.prefix")).resolve()
    if (prefix / "lib" / "cmake" / "pinocchio" / "pinocchioConfig.cmake").is_file():
        print(prefix)
        sys.exit(0)
except Exception:
    pass

try:
    import pinocchio
except Exception:
    sys.exit(1)

p = pathlib.Path(pinocchio.__file__).resolve()
# .../cmeel.prefix/lib/pythonX/site-packages/pinocchio/__init__.py
print(str(p.parents[4]))
]=]
        RESULT_VARIABLE _pin_prefix_rc
        OUTPUT_VARIABLE _pin_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_pin_prefix_rc EQUAL 0 AND _pin_prefix)
        message(STATUS "Detected PyPI `pin` prefix: ${_pin_prefix}")
        # Prefer the `pin` wheel's CMake packages over any local Pinocchio installs.
        list(PREPEND CMAKE_PREFIX_PATH "${_pin_prefix}")
        # Force pinocchio_DIR to the wheel-provided config to avoid accidentally
        # picking up a locally-built Pinocchio (which can bake in incompatible Boost SONAMEs).
        set(pinocchio_DIR "${_pin_prefix}/lib/cmake/pinocchio" CACHE PATH "Pinocchio CMake package directory" FORCE)
        message(STATUS "Forcing pinocchio_DIR=${pinocchio_DIR}")
        if(NOT DEFINED Eigen3_DIR AND NOT DEFINED ENV{Eigen3_DIR})
            set(_pin_eigen_dir "${_pin_prefix}/share/eigen3/cmake")
            if(EXISTS "${_pin_eigen_dir}/Eigen3Config.cmake")
                set(Eigen3_DIR "${_pin_eigen_dir}" CACHE PATH "Eigen3 CMake package directory" FORCE)
                message(STATUS "Forcing Eigen3_DIR=${Eigen3_DIR}")
            endif()
        endif()
    else()
        message(STATUS
            "EmbodiK: PyPI `pin` was not importable, so CMake will search "
            "for Pinocchio using the existing CMAKE_PREFIX_PATH/pinocchio_DIR. "
            "For pip source builds, install `pin>=3.8.0` or set pinocchio_DIR "
            "to a compatible Pinocchio CMake package.")
    endif()
else()
    message(STATUS
        "EmbodiK: using caller-provided pinocchio_DIR or environment "
        "pinocchio_DIR; PyPI `pin` auto-detection is skipped.")
endif()

# Find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "EmbodiK: using Eigen3_DIR='${Eigen3_DIR}'")

find_package(pinocchio REQUIRED)
message(STATUS "EmbodiK: using pinocchio_DIR='${pinocchio_DIR}'")

# Add compile options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Create main library
add_library(embodik_core SHARED
    cpp_core/src/robot_model.cpp
    cpp_core/src/tasks.cpp
    cpp_core/src/dual_arm_ects.cpp
    cpp_core/src/frame_kinematic_differential.cpp
    cpp_core/src/dual_arm_ects_differential.cpp
    cpp_core/src/acceleration_task_differential.cpp
    cpp_core/src/acceleration_state_box.cpp
    cpp_core/src/acceleration_solver.cpp
    cpp_core/src/kinematics_solver.cpp
    cpp_core/src/pose_metrics.cpp
    cpp_core/src/sphere_broadphase.cpp
)

target_include_directories(embodik_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp_core/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(embodik_core PUBLIC
    Eigen3::Eigen
    pinocchio::pinocchio
)

if(EMBODIK_ENABLE_IPO AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    check_ipo_supported(RESULT _EMBODIK_IPO_SUPPORTED OUTPUT _EMBODIK_IPO_ERROR)
    if(_EMBODIK_IPO_SUPPORTED)
        set_property(TARGET embodik_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO/LTO not enabled for embodik_core: ${_EMBODIK_IPO_ERROR}")
    endif()
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python_bindings)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
        add_subdirectory(test)
    endif()
endif()

# Examples
if(BUILD_EXAMPLES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
        add_subdirectory(examples)
    endif()
endif()

# Installation
install(TARGETS embodik_core
    EXPORT embodik_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY cpp_core/include/
    DESTINATION include
)

install(EXPORT embodik_targets
    FILE embodik_targets.cmake
    NAMESPACE embodik::
    DESTINATION lib/cmake/embodik
)

# Python package installation
# Note: scikit-build-core handles Python package installation via wheel.packages
# The extension module is installed by python_bindings/CMakeLists.txt
# Python source files are included via wheel.packages in pyproject.toml

# Install examples directory for pip-installed users
install(DIRECTORY examples/
    DESTINATION share/embodik/examples
    FILES_MATCHING
        PATTERN "harnesses" EXCLUDE
        PATTERN "*.py"
        PATTERN "*.yaml"
        PATTERN "*.md"
        PATTERN "LICENSE"
        PATTERN "*.urdf"
        PATTERN "*.xml"
        PATTERN "*.obj"
        PATTERN "*.mtl"
        PATTERN "*.png"
        PATTERN "*.stl"
        PATTERN "*.json"
        PATTERN "*.onnx"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
    PATTERN "_scratch*" EXCLUDE
)
