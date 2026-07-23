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

add_executable(test_acceleration_solver_task_bounds
    test_acceleration_solver_task_bounds.cpp
)

target_link_libraries(test_acceleration_solver_task_bounds
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver_task_bounds
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver_task_bounds
    COMMAND test_acceleration_solver_task_bounds
)

add_executable(test_acceleration_solver_frozen_locks
    test_acceleration_solver_frozen_locks.cpp
)

target_link_libraries(test_acceleration_solver_frozen_locks
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver_frozen_locks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver_frozen_locks
    COMMAND test_acceleration_solver_frozen_locks
)

add_executable(test_acceleration_allocation_transform
    test_acceleration_allocation_transform.cpp
)

target_link_libraries(test_acceleration_allocation_transform
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_allocation_transform
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_allocation_transform
    COMMAND test_acceleration_allocation_transform
)

add_executable(test_acceleration_solver_allocation
    test_acceleration_solver_allocation.cpp
)

target_link_libraries(test_acceleration_solver_allocation
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver_allocation
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver_allocation
    COMMAND test_acceleration_solver_allocation
)

add_executable(test_acceleration_solver_effort
    test_acceleration_solver_effort.cpp
)

target_link_libraries(test_acceleration_solver_effort
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver_effort
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver_effort
    COMMAND test_acceleration_solver_effort
)

add_executable(test_acceleration_solver_contact
    test_acceleration_solver_contact.cpp
)

target_link_libraries(test_acceleration_solver_contact
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver_contact
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver_contact
    COMMAND test_acceleration_solver_contact
)

add_executable(test_acceleration_solver_velocity_collision_lift
    test_acceleration_solver_velocity_collision_lift.cpp
)

target_link_libraries(test_acceleration_solver_velocity_collision_lift
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver_velocity_collision_lift
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver_velocity_collision_lift
    COMMAND test_acceleration_solver_velocity_collision_lift
)

add_executable(test_acceleration_solver_tight_point
    test_acceleration_solver_tight_point.cpp
)

target_link_libraries(test_acceleration_solver_tight_point
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver_tight_point
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver_tight_point
    COMMAND test_acceleration_solver_tight_point
)

add_executable(test_acceleration_solver_tight_frame_pose
    test_acceleration_solver_tight_frame_pose.cpp
)

target_link_libraries(test_acceleration_solver_tight_frame_pose
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver_tight_frame_pose
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver_tight_frame_pose
    COMMAND test_acceleration_solver_tight_frame_pose
)

add_executable(test_acceleration_solver_relative_pose
    test_acceleration_solver_relative_pose.cpp
)

target_link_libraries(test_acceleration_solver_relative_pose
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver_relative_pose
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver_relative_pose
    COMMAND test_acceleration_solver_relative_pose
)

add_executable(test_acceleration_solver_torso_pose_bounds
    test_acceleration_solver_torso_pose_bounds.cpp
)

target_link_libraries(test_acceleration_solver_torso_pose_bounds
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver_torso_pose_bounds
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver_torso_pose_bounds
    COMMAND test_acceleration_solver_torso_pose_bounds
)

add_executable(test_acceleration_solver_com_support_polygon
    test_acceleration_solver_com_support_polygon.cpp
)

target_link_libraries(test_acceleration_solver_com_support_polygon
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_acceleration_solver_com_support_polygon
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_acceleration_solver_com_support_polygon
    COMMAND test_acceleration_solver_com_support_polygon
)

add_executable(test_geometric_constraint_differential
    test_geometric_constraint_differential.cpp
)

target_link_libraries(test_geometric_constraint_differential
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_geometric_constraint_differential
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_geometric_constraint_differential
    COMMAND test_geometric_constraint_differential
)

add_executable(test_collision_constraint_differential
    test_collision_constraint_differential.cpp
)

target_link_libraries(test_collision_constraint_differential
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(test_collision_constraint_differential
    PRIVATE
        ${CMAKE_SOURCE_DIR}/cpp_core/src
)

add_test(NAME test_collision_constraint_differential
    COMMAND test_collision_constraint_differential
)

add_executable(test_shared_constraint_definitions
    test_shared_constraint_definitions.cpp
)

target_link_libraries(test_shared_constraint_definitions
    embodik_core
    GTest::gtest_main
    GTest::gtest
)

add_test(NAME test_shared_constraint_definitions
    COMMAND test_shared_constraint_definitions
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
