# CMake generated Testfile for 
# Source directory: /path/to/local/Projects/git-worktrees/embodik-elastic-band/test
# Build directory: /path/to/local/Projects/git-worktrees/embodik-elastic-band/build-cp311/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_robot_model "/path/to/local/Projects/git-worktrees/embodik-elastic-band/build-cp311/test/test_robot_model")
set_tests_properties(test_robot_model PROPERTIES  _BACKTRACE_TRIPLES "/path/to/local/Projects/git-worktrees/embodik-elastic-band/test/CMakeLists.txt;19;add_test;/path/to/local/Projects/git-worktrees/embodik-elastic-band/test/CMakeLists.txt;0;")
add_test(test_python_robot_model "/path/to/local/Projects/validation-repos/validation/validation_robot/.pixi/envs/default/bin/python3.11" "-m" "pytest" "/path/to/local/Projects/git-worktrees/embodik-elastic-band/test/test_robot_model.py" "-v")
set_tests_properties(test_python_robot_model PROPERTIES  ENVIRONMENT "PYTHONPATH=/path/to/local/Projects/git-worktrees/embodik-elastic-band/build-cp311/python_bindings:" WORKING_DIRECTORY "/path/to/local/Projects/git-worktrees/embodik-elastic-band/build-cp311" _BACKTRACE_TRIPLES "/path/to/local/Projects/git-worktrees/embodik-elastic-band/test/CMakeLists.txt;24;add_test;/path/to/local/Projects/git-worktrees/embodik-elastic-band/test/CMakeLists.txt;0;")
