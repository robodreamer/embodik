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
