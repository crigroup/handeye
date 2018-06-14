cmake_minimum_required(VERSION 2.8.3)
project(handeye)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  std_msgs
)

catkin_python_setup()

add_service_files(
  FILES
  CalibrateHandEye.srv
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_runtime
    std_msgs
)

# Tests
if (CATKIN_ENABLE_TESTING)
  # nosetests
  catkin_add_nosetests(tests)
  # rostests
  find_package(rostest REQUIRED)
  add_rostest(tests/test_handeye_server.test
    DEPENDENCIES ${${PROJECT_NAME}_EXPORTED_TARGETS})
endif()

# Install python scripts
catkin_install_python(
  PROGRAMS
    scripts/handeye_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
