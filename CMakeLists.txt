cmake_minimum_required(VERSION 3.10)
project(jetson_lio  VERSION 0.1.0
                    LANGUAGES CXX CUDA)

#set(CMAKE_C_COMPILER "gcc-10")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-10")

option(BUILD_TESTS "Build tests" On)
option(LTO_OPT "Enable LTO" Off)

if (LTO_OPT)
  MESSAGE(STATUS "using Link Time Optimizations, this will take time...")
  add_compile_options(-flto)
endif()

find_package(CUDA REQUIRED)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP QUIET)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
endif()

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  message_generation
  eigen_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED QUIET) # TODO fixed in 1.13, QUIET only for disabling PCL not finding pcap warnings when building
find_package(TBB REQUIRED)

add_definitions(-DUSE_CUDA)

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  include
)

add_definitions( ${PCL_DEFINITIONS} )

add_message_files(
  FILES
  Pose6D.msg
)

add_service_files(
  FILES
  SetRunning.srv
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
 sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

add_library(gpu_preprocess src/gpu_preprocess.cu)
target_link_libraries(gpu_preprocess ${CUDA_LIBRARIES})

add_library(idk_gpu_search include/ikd-Tree/ikd_Tree_gpu_search.cu)
target_link_libraries(idk_gpu_search ${CUDA_LIBRARIES} ${CUSOLVER_LIB})
target_include_directories(idk_gpu_search PRIVATE ${EIGEN3_INCLUDE_DIR})

add_executable(${PROJECT_NAME} src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} pthread TBB::tbb gpu_preprocess idk_gpu_search)
target_include_directories(${PROJECT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_generate_messages_cpp)

add_compile_options(-std=c++17)
#add_compile_options(-Wall)


target_compile_options(gpu_preprocess PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
-O3 --use_fast_math -arch=sm_72
>)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
  -O3 -march=native
  >)
else()
  target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
  -O0 -g -fasynchronous-unwind-tables
  >)
endif()


if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()