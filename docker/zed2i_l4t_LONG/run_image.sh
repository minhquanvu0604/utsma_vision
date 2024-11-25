IMAGE_NAME=utsma/zed_image
CONTAINER_NAME=zed_container

docker run -it --runtime=nvidia \
   --name $CONTAINER_NAME \
   --privileged \
   --network host \
   --volume /dev:/dev \
   --volume /tmp/argus_socket:/tmp/argus_socket \
   --env DISPLAY=$DISPLAY \
   --env NVIDIA_DRIVER_CAPABILITIES=all \
   --env NVIDIA_VISIBLE_DEVICES=all \
   --env QT_X11_NO_MITSHM=1 \
   --volume /tmp/.X11-unix:/tmp/.X11-unix \
   --volume /home/utsma/zedsdk_ros2_docker/sdk_config:/usr/local/zed/resources \
   $IMAGE_NAME