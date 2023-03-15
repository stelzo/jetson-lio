cmake_minimum_required(VERSION 3.10)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

enable_testing()
include(GoogleTest)

add_subdirectory(nearest_search)
add_subdirectory(pca)
add_subdirectory(rm_mtk_interop)