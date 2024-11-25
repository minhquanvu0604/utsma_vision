DOCKERFILE=zed2i-l4t-LONG.Dockerfile
IMAGE_NAME=utsma/zed_image
CONTAINER_NAME=zed_container

# Allow X11 access for the container
xhost +local:root

# Remove old containers
docker ps -a | grep $IMAGE_NAME | awk '{print $1}' | xargs -r docker rm

# Remove old images
docker rmi -f $IMAGE_NAME
docker rm -f $CONTAINER_NAME

# Build the Docker image
docker build -t $IMAGE_NAME -f $DOCKERFILE .