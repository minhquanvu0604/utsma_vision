# Problem Log
- Latency in image transmission - high even when streaming raw images without ROS, possible solutions are
    - GST image transmission pipeline
    - To improve performance with ROS, use Zero-Copy 
- Examine why the [vanilla zed docker](DOCKER_ZED_ROS2_WRAPPER.md) doesn't work 


# Check Out
- Resource for docker images
    - https://github.com/dusty-nv/jetson-containers
    - https://hub.docker.com/r/dustynv/ros/tags?name=humble-ros


# Working Setup
Long Train's [Dockerfile](zed_2i_LONG/zed-2i-LONG.Dockerfile) 

## Fixme
- Resolve @TODO's