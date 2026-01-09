cmake_minimum_required(VERSION 3.16)
project(embodik VERSION 0.1.1)

# Python is used to auto-discover Pinocchio (PyPI `pin`) CMake config during sdist builds.
find_package(Python REQUIRED COMPONENTS Interpreter)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# If the user doesn't have a system Pinocchio install, try to locate the CMake config
# shipped with the PyPI `pin` wheel (import name: `pinocchio`).
#
# This helps `pip install` from sdist (when a wheel isn't available) in a plain venv.
if(NOT DEFINED pinocchio_DIR AND NOT DEFINED ENV{pinocchio_DIR})
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import pathlib, sys\n"
            "try:\n"
            "  import pinocchio\n"
            "except Exception:\n"
            "  sys.exit(1)\n"
            "p = pathlib.Path(pinocchio.__file__).resolve()\n"
            "# .../cmeel.prefix/lib/pythonX/site-packages/pinocchio/__init__.py\n"
            "prefix = p.parents[4]\n"
            "print(str(prefix))\n"
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
    endif()
endif()

find_package(pinocchio REQUIRED)

# Add compile options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Create main library
add_library(embodik_core SHARED
    cpp_core/src/robot_model.cpp
    cpp_core/src/tasks.cpp
    cpp_core/src/kinematics_solver.cpp
)

target_include_directories(embodik_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp_core/include>
    $<INSTALL_INTERFACE:include>
)

# For inplace editable installs, copy core library to python/embodik directory
# Always copy for editable installs (editable.mode = "inplace" in pyproject.toml)
add_custom_command(TARGET embodik_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:embodik_core>
        "${CMAKE_CURRENT_SOURCE_DIR}/python/embodik/"
    COMMENT "Copying core library to python/embodik for editable install"
)

target_link_libraries(embodik_core PUBLIC
    Eigen3::Eigen
    pinocchio::pinocchio
)

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
        PATTERN "*.py"
        PATTERN "*.yaml"
        PATTERN "*.md"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
)
