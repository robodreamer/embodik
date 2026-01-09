# Python - always find Python regardless of build system
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Nanobind discovery
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    RESULT_VARIABLE nanobind_result
    OUTPUT_VARIABLE nanobind_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(nanobind_result AND NOT nanobind_result EQUAL 0)
    message(FATAL_ERROR "Failed to get Nanobind CMake directory! Make sure nanobind is installed: pip install nanobind")
endif()

find_package(nanobind REQUIRED CONFIG PATHS "${nanobind_cmake_dir}")

# Build the nanobind module
nanobind_add_module(_embodik_impl
    NB_STATIC
    LTO
    NOMINSIZE
    src/bindings.cpp
    src/robot_model_bindings.cpp
    src/tasks_bindings.cpp
    src/kinematics_solver_bindings.cpp
)

# Link with core library
target_link_libraries(_embodik_impl PRIVATE
    embodik_core
)

# For inplace editable installs, copy extension to python/embodik directory
# Always copy for editable installs (editable.mode = "inplace" in pyproject.toml)
add_custom_command(TARGET _embodik_impl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:_embodik_impl>
        "${CMAKE_SOURCE_DIR}/python/embodik/"
    COMMENT "Copying extension module to python/embodik for editable install"
)

# Set RPATH so Python can find libembodik_core.so
# For editable installs (inplace mode), library is in same directory as extension
# For regular installs, library is in ./lib relative to extension (installed under the package dir)
set_target_properties(_embodik_impl PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
    BUILD_RPATH "$ORIGIN:$<TARGET_FILE_DIR:embodik_core>"
)

# Generate Python type stubs (non-fatal - stub generation may fail if dependencies aren't available)
# The module will work fine without .pyi files - they're just for IDE autocomplete
# Note: We need to set LD_LIBRARY_PATH to find shared libraries (boost, pinocchio, embodik_core, etc.)
# Collect library directories at configure time
set(STUBGEN_LIB_PATHS "")

# Add build directories (will be resolved at build time via generator expressions)
# Add common library paths
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND STUBGEN_LIB_PATHS "$ENV{CONDA_PREFIX}/lib")
endif()
if(DEFINED ENV{LD_LIBRARY_PATH})
    list(APPEND STUBGEN_LIB_PATHS "$ENV{LD_LIBRARY_PATH}")
endif()

# Try to get pinocchio library directory
if(TARGET pinocchio::pinocchio)
    get_target_property(PINOCCHIO_LOC pinocchio::pinocchio IMPORTED_LOCATION)
    if(PINOCCHIO_LOC)
        get_filename_component(PINOCCHIO_DIR "${PINOCCHIO_LOC}" DIRECTORY)
        list(APPEND STUBGEN_LIB_PATHS "${PINOCCHIO_DIR}")
    endif()
endif()

# Try to get boost library directory
if(TARGET Boost::filesystem)
    get_target_property(BOOST_LOC Boost::filesystem IMPORTED_LOCATION)
    if(BOOST_LOC)
        get_filename_component(BOOST_DIR "${BOOST_LOC}" DIRECTORY)
        list(APPEND STUBGEN_LIB_PATHS "${BOOST_DIR}")
    endif()
endif()

# Convert to colon-separated string
string(REPLACE ";" ":" STUBGEN_LIB_PATH_STR "${STUBGEN_LIB_PATHS}")

# Create stub generation command with proper library paths
# Use generator expressions for build-time directories
add_custom_command(
    TARGET _embodik_impl POST_BUILD
    COMMAND sh -c "LD_LIBRARY_PATH=\"$<TARGET_FILE_DIR:embodik_core>:$<TARGET_FILE_DIR:_embodik_impl>:${STUBGEN_LIB_PATH_STR}:$$LD_LIBRARY_PATH\" PYTHONPATH=\"${CMAKE_SOURCE_DIR}/python:$$PYTHONPATH\" ${Python_EXECUTABLE} -m nanobind.stubgen -q -i \"$<TARGET_FILE_DIR:_embodik_impl>\" -M py.typed -m _embodik_impl -o \"$<TARGET_FILE_DIR:_embodik_impl>/__init__.pyi\" && touch \"$<TARGET_FILE_DIR:_embodik_impl>/py.typed\" || echo 'Stub generation skipped (dependencies may be missing)'"
    COMMENT "Generating Python type stubs (non-fatal)"
    VERBATIM
)

# Install the nanobind module
# scikit-build-core will handle the installation path based on wheel.install-dir
install(TARGETS _embodik_impl
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

# Copy stubs to source package directory so they're included in the wheel
# This happens after stub generation (which is non-fatal)
# Use shell command to make copy optional (won't fail if files don't exist)
add_custom_command(
    TARGET _embodik_impl POST_BUILD
    COMMAND sh -c "${CMAKE_COMMAND} -E copy_if_different \"$<TARGET_FILE_DIR:_embodik_impl>/__init__.pyi\" \"${CMAKE_SOURCE_DIR}/python/embodik/__init__.pyi\" 2>/dev/null || true"
    COMMAND sh -c "${CMAKE_COMMAND} -E copy_if_different \"$<TARGET_FILE_DIR:_embodik_impl>/py.typed\" \"${CMAKE_SOURCE_DIR}/python/embodik/py.typed\" 2>/dev/null || true"
    COMMENT "Copying type stubs to source package directory (if generated)"
    VERBATIM
)

