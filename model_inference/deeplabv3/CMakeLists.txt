# Download and move the LibTorch library to the /opt directory
# CUDA 12.4 Configuration
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export CMAKE_PREFIX_PATH=/opt/libtorch:$CMAKE_PREFIX_PATH

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(model_inference)

# ==============================
# Arguments
# ==============================
option(INSTALL_IN_WORKSPACE "Install the package to the ROS workspace" ON)
set(CMAKE_INSTALL_PREFIX /home/quanvu/ros/apple_ws/devel)
set(PROJECT_VERSION 1.0.0)

find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to LibTorch
# .bashrc
if(NOT CMAKE_PREFIX_PATH) 
    set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
endif()
# message(STATUS "====== CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH} ======")

# Include directories
include_directories(
  include
  include/model_inference
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# ==============================
# Set installation prefix
# ==============================
if(INSTALL_IN_WORKSPACE)
  # Install to workspace
  set(INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
else()
  # Install within its own build directory
  set(INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)
endif()

# ==============================
# Add the library
# ==============================
add_library(${PROJECT_NAME} SHARED src/predict.cpp)

# Link against LibTorch
target_link_libraries(${PROJECT_NAME} 
                            ${TORCH_LIBRARIES} 
                            ${OpenCV_LIBS}
                            opencv_imgcodecs opencv_highgui)

# Ensure RPATH is set correctly (optional but recommended)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

# To ensure that the library is built with the same flags as LibTorch
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Necessary to link LibTorch properly
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Ensure that the necessary libraries are included for the GPU support
if (TORCH_CUDA_ARCH_LIST)
  target_link_libraries(${PROJECT_NAME} ${TORCH_CUDA_LIBRARIES})
endif()

# Optional: Install targets
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)


# ==============================
# Add an executable for testing
# ==============================
add_executable(predict_demo src/predict_demo.cpp)

# Link the executable with the inference library and other dependencies
target_link_libraries(predict_demo ${PROJECT_NAME} ${OpenCV_LIBS} opencv_imgcodecs opencv_highgui)

# Set properties for the executable
set_property(TARGET predict_demo PROPERTY CXX_STANDARD 17)

# Optional: Install the executable
install(TARGETS predict_demo RUNTIME DESTINATION bin)

# ==============================
# Exporting
# ==============================
# Set the include directory variable
set(PACKAGE_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include/model_inference")
set(model_inference_INCLUDE_DIRS ${PACKAGE_INCLUDE_INSTALL_DIR})

set(PACKAGE_LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib")


include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/model_inferenceConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Configure the config file
configure_file(cmake/model_inferenceConfig.cmake.in
"${CMAKE_CURRENT_BINARY_DIR}/model_inferenceConfig.cmake"
@ONLY
)

configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/cmake/model_inferenceConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/model_inferenceConfig.cmake"
  INSTALL_DESTINATION lib/cmake/model_inference
)

  
# Install the config and version files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/model_inferenceConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/model_inferenceConfigVersion.cmake"
  DESTINATION lib/cmake/model_inference
)

# Set the appropriate installation destinations based on the INSTALL_PREFIX
install(TARGETS ${PROJECT_NAME}
  EXPORT model_inferenceTargets
  LIBRARY DESTINATION ${INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${INSTALL_PREFIX}/lib
  RUNTIME DESTINATION ${INSTALL_PREFIX}/bin
  INCLUDES DESTINATION ${INSTALL_PREFIX}/include
)

install(TARGETS predict_demo
  EXPORT model_inferenceTargets
  RUNTIME DESTINATION ${INSTALL_PREFIX}/bin
)

# Headers
install(DIRECTORY include/ DESTINATION ${INSTALL_PREFIX}/include)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/model_inferenceConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/model_inferenceConfigVersion.cmake"
  DESTINATION ${INSTALL_PREFIX}/lib/cmake/model_inference
)

install(EXPORT model_inferenceTargets
  FILE model_inferenceTargets.cmake
  NAMESPACE model_inference::
  DESTINATION ${INSTALL_PREFIX}/lib/cmake/model_inference
)

# message(STATUS "====== CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR} ======")
message(STATUS "====== TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS} ======")
message(STATUS "====== TORCH_LIBRARIES: ${TORCH_LIBRARIES} ======")
