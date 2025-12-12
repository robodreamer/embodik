cmake_minimum_required(VERSION 3.16)
project(embodik VERSION 0.1.0)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
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
