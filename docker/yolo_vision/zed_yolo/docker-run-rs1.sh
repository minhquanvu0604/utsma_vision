DOCKERFILE=Dockerfile.yolo
IMAGE_NAME=utsma/yolo
CONTAINER_NAME=yolo_container
USER=long

xhost +local:root
# remove old containers
docker ps -a | grep $IMAGE_NAME | awk '{print $1}' | xargs -r docker rm

# remove old images
docker rmi -f $IMAGE_NAME
docker rm -f $CONTAINER_NAME

# build
docker build -t $IMAGE_NAME -f $DOCKERFILE .

# run
docker run -it $1 \
   --name $CONTAINER_NAME \
   --network host --privileged \
   --device /dev/dri:/dev/dri \
   --ipc=host \
   --pid=host \
   -h $(hostname) \
   -e DISPLAY=$DISPLAY \
   -e XAUTHORITY=/tmp/xauth \
   -v ~/.Xauthority:/tmp/xauth \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v /dev:/dev \
   -v /home/$USER/utsma_ws/zed_yolo/src:/home/ros2_ws/src/computer_vision \
   $IMAGE_NAME
