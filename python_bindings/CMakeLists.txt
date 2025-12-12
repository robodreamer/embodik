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
# For regular installs, library is in ../lib relative to extension
set_target_properties(_embodik_impl PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
    BUILD_RPATH "$ORIGIN:$<TARGET_FILE_DIR:embodik_core>"
)

# Generate Python type stubs
nanobind_add_stub(_embodik_stub
    MODULE _embodik_impl
    OUTPUT __init__.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_embodik_impl>
    DEPENDS _embodik_impl
    MARKER_FILE py.typed)

# Install the nanobind module
# scikit-build-core will handle the installation path based on wheel.install-dir
install(TARGETS _embodik_impl
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

# Install Python type stubs if generated
if(TARGET _embodik_stub)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/__init__.pyi
        ${CMAKE_CURRENT_BINARY_DIR}/py.typed
        DESTINATION .
        OPTIONAL
    )
endif()

# Copy stubs into the source package dir for IDEs/type checkers (commented out to avoid build issues)
# add_custom_command(TARGET _embodik_impl POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#         ${CMAKE_BINARY_DIR}/__init__.pyi
#         ${CMAKE_SOURCE_DIR}/embodik_py/__init__.pyi
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#         ${CMAKE_BINARY_DIR}/py.typed
#         ${CMAKE_SOURCE_DIR}/embodik_py/py.typed)


