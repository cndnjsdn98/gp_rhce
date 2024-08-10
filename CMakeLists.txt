cmake_minimum_required(VERSION 3.0.2)
project(gp_rhce)

# Set build type to Release for optimization
set(CMAKE_BUILD_TYPE Release)

# Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

# Additional optimization flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O3)  # Enable high-level optimization
endif()

# Find catkin macros and libraries
set(${PROJECT_NAME}_CATKIN_DEPS
    roscpp
    roslib
    rospy
    std_msgs
    mav_msgs
    sensor_msgs
    geometry_msgs
    quadrotor_msgs
    nav_msgs
)

find_package(catkin REQUIRED COMPONENTS
    ${${PROJECT_NAME}_CATKIN_DEPS}
)

# Declare ROS messages, services, and actions
add_message_files(
  FILES
  ReferenceTrajectory.msg
)

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES 
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES gp_mhe_node
  CATKIN_DEPENDS ${${PROJECT_NAME}_CATKIN_DEPS}
)

# Retrieve the ACADOS source directory from the environment variable
if(DEFINED ENV{ACADOS_SOURCE_DIR})
  set(ACADOS_SOURCE_DIR $ENV{ACADOS_SOURCE_DIR})
else()
  message(FATAL_ERROR "Environment variable ACADOS_SOURCE_DIR is not set.")
endif()

# Build
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  acados_ocp/
  ${ACADOS_SOURCE_DIR}/include 
  ${ACADOS_SOURCE_DIR}/include/blasfeo/include
  ${ACADOS_SOURCE_DIR}/include/hpipm/include
)

# Specify the directory containing precompiled Acados Models
set(ACADOS_OCP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/acados_ocp")

add_executable(gp_mhe_node 
               src/mhe/gp_mhe_node.cpp 
               src/mhe/gp_mhe.cpp)
add_executable(gp_mpc_node 
               src/mpc/gp_mpc_node.cpp 
               src/mpc/gp_mpc.cpp)

add_dependencies(gp_mhe_node 
                 ${${PROJECT_NAME}_EXPORTED_TARGETS} 
                 ${catkin_EXPORTED_TARGETS})
add_dependencies(gp_mpc_node 
                 ${${PROJECT_NAME}_EXPORTED_TARGETS} 
                 ${catkin_EXPORTED_TARGETS})

# List the precompiled ACADOS libraries explicitly
set(ACADOS_LIBS
  ${ACADOS_SOURCE_DIR}/lib/libacados.so
  ${ACADOS_SOURCE_DIR}/lib/libblasfeo.so
  ${ACADOS_SOURCE_DIR}/lib/libhpipm.so
)

target_link_libraries(gp_mhe_node
  ${catkin_LIBRARIES}
  ${ACADOS_LIBS}
  ${ACADOS_OCP_DIR}/mhe/libacados_ocp_solver_mhe.so
)

target_link_libraries(gp_mpc_node
  ${catkin_LIBRARIES}
  ${ACADOS_LIBS}
  ${ACADOS_OCP_DIR}/mpc/libacados_ocp_solver_mpc.so
)
