cmake_minimum_required(VERSION 3.10)
set(TEST_TARGET rm_mtk_interop)
project(jetson_lio_${TEST_TARGET}_test CXX CUDA)

set(TEST_SRC
  rm_mtk_interop_test.cu
)

catkin_add_gtest(${TEST_TARGET} ${TEST_SRC})
target_include_directories(${TEST_TARGET} PUBLIC ${Eigen3_INCLUDE_DIRS} ../../include)
target_link_libraries(${TEST_TARGET} GTest::gtest_main Eigen3::Eigen ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} TBB::tbb glog gpu_preprocess idk_gpu_search)
gtest_discover_tests(${TEST_TARGET})