# CMake generated Testfile for 
# Source directory: /home/andypark/Projects/git-worktrees/embodik-elastic-band/test
# Build directory: /home/andypark/Projects/git-worktrees/embodik-elastic-band/build-cp311/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_robot_model "/home/andypark/Projects/git-worktrees/embodik-elastic-band/build-cp311/test/test_robot_model")
set_tests_properties(test_robot_model PROPERTIES  _BACKTRACE_TRIPLES "/home/andypark/Projects/git-worktrees/embodik-elastic-band/test/CMakeLists.txt;19;add_test;/home/andypark/Projects/git-worktrees/embodik-elastic-band/test/CMakeLists.txt;0;")
add_test(test_python_robot_model "/home/andypark/Projects/hmnd-repos/hmnd/hmnd_robot/.pixi/envs/default/bin/python3.11" "-m" "pytest" "/home/andypark/Projects/git-worktrees/embodik-elastic-band/test/test_robot_model.py" "-v")
set_tests_properties(test_python_robot_model PROPERTIES  ENVIRONMENT "PYTHONPATH=/home/andypark/Projects/git-worktrees/embodik-elastic-band/build-cp311/python_bindings:" WORKING_DIRECTORY "/home/andypark/Projects/git-worktrees/embodik-elastic-band/build-cp311" _BACKTRACE_TRIPLES "/home/andypark/Projects/git-worktrees/embodik-elastic-band/test/CMakeLists.txt;24;add_test;/home/andypark/Projects/git-worktrees/embodik-elastic-band/test/CMakeLists.txt;0;")
