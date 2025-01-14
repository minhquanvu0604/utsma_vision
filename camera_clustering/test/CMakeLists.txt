enable_testing()

add_executable(algo_test algo_test.cpp)

target_link_libraries(algo_test PUBLIC
    ${GTEST_LIBRARIES}
    ${PROJECT_LIBRARIES}
    ${PCL_LIBRARIES}
)