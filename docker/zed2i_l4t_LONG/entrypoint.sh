#!/bin/bash

echo "Starting entrypoint.sh"

# Source ROS 2 setup
echo "Sourcing ROS 2 setup..."
source /opt/ros/humble/setup.bash

# Check if workspace is built
if [ ! -d "~/ros2_ws/install" ]; then
    echo "Workspace not found. Building the workspace..."
    cd ~/ros2_ws
    rosdep install --from-paths src --ignore-src -r -y
    colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc)
else
    echo "Workspace already built. Skipping build."
fi

# Source workspace setup
echo "Sourcing workspace setup..."
source ~/ros2_ws/install/setup.bash

# Final debug message
echo "Finished entrypoint.sh, starting bash..."
exec bash
