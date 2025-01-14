cmake_minimum_required(VERSION 3.5)
project(clustering_cpp)

## Use C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

set(GTEST_ROOT "/usr/src/gtest" CACHE PATH "Path to googletest")
find_package(GTest REQUIRED)

find_package(PCL 1.12 REQUIRED)
find_package(Eigen3 REQUIRED)

## Setup PCL Library
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

## Specify additional locations of header files
include_directories(
    include
    ${GTEST_INCLUDE_DIRS}
    ${EIGEN3_INLCUDE_DIRS}
    ${PCL_INLCUDE_DIRS}
)

add_subdirectory(src)
set(PROJECT_LIBRARIES pcl_k_means euc_dist_seg)
add_subdirectory(test)