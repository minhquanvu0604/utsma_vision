## Build Naive K Means algorithm
add_library(pcl_k_means pcl_k_means.cpp)

target_link_libraries(pcl_k_means
    ${PCL_LIBRARIES}
    )

## Build Naive K Means algorithm
add_library(euc_dist_seg euc_dist_seg.cpp)
add_library(detect_cone detect_cone.cpp)

target_link_libraries(euc_dist_seg
    ${PCL_LIBRARIES}
    )

target_link_libraries(detect_cone
    ${PCL_LIBRARIES}
    )