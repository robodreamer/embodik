# Unit tests for embodik
enable_testing()

# Find GTest
find_package(GTest REQUIRED)

# Test executable
add_executable(test_robot_model
    test_robot_model.cpp
)

target_link_libraries(test_robot_model
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

# Add test
add_test(NAME test_robot_model COMMAND test_robot_model)

add_executable(test_hierarchical_linear_solver
    test_hierarchical_linear_solver.cpp
)

target_link_libraries(test_hierarchical_linear_solver
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

add_test(NAME test_hierarchical_linear_solver
    COMMAND test_hierarchical_linear_solver
)

add_executable(test_acceleration_task_differential
    test_acceleration_task_differential.cpp
)

target_link_libraries(test_acceleration_task_differential
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_task_differential
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_task_differential
    COMMAND test_acceleration_task_differential
)

add_executable(test_velocity_collision_linearization
    test_velocity_collision_linearization.cpp
)

target_link_libraries(test_velocity_collision_linearization
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_velocity_collision_linearization
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_velocity_collision_linearization
    COMMAND test_velocity_collision_linearization
)

add_executable(test_generalized_constraint_set
    test_generalized_constraint_set.cpp
)

target_link_libraries(test_generalized_constraint_set
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_generalized_constraint_set
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_generalized_constraint_set
    COMMAND test_generalized_constraint_set
)

add_executable(test_acceleration_state_box
    test_acceleration_state_box.cpp
)

target_link_libraries(test_acceleration_state_box
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_state_box
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_state_box
    COMMAND test_acceleration_state_box
)

add_executable(test_acceleration_solver
    test_acceleration_solver.cpp
)

target_link_libraries(test_acceleration_solver
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver
    COMMAND test_acceleration_solver
)

add_executable(test_acceleration_solver_affine_constraints
    test_acceleration_solver_affine_constraints.cpp
)

target_link_libraries(test_acceleration_solver_affine_constraints
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver_affine_constraints
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver_affine_constraints
    COMMAND test_acceleration_solver_affine_constraints
)

# Python tests
find_package(Python COMPONENTS Interpreter REQUIRED)

add_test(NAME test_python_robot_model
    COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_robot_model.py -v
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Set Python path for tests
set_tests_properties(test_python_robot_model PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python_bindings:$ENV{PYTHONPATH}"
)
