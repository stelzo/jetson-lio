cmake_minimum_required(VERSION 3.10)
set(TEST_TARGET pca)
project(jetson_lio_${TEST_TARGET}_test CXX CUDA)

set(TEST_SRC
  pca_test.cu
  ${jetson_lio_SOURCE_DIR}/include/ikd-Tree/ikd_Tree.cpp
)

catkin_add_gtest(${TEST_TARGET} ${TEST_SRC})
target_include_directories(${TEST_TARGET} PUBLIC ${Eigen3_INCLUDE_DIRS})
target_link_libraries(${TEST_TARGET} GTest::gtest_main Eigen3::Eigen ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} TBB::tbb glog gpu_preprocess idk_gpu_search)
gtest_discover_tests(${TEST_TARGET})