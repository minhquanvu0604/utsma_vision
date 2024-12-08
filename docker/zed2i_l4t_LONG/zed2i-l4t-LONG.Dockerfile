FROM stereolabs/iot:0.81.0-devel-jetson-l4t36.3-zed4.2.1

# Install prerequisites
RUN apt-get update && apt-get install -y \
    wget \
    git \
    lsb-release \
    gnupg2 \
    zstd \
    curl \
    python3-pip \
    build-essential \
    # python3-rosdep \
    # python3-colcon-common-extensions \
    nano \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install ROS 2 Humble
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - \
    && sh -c 'echo "deb [arch=arm64] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list' \
    && apt-get update && apt-get install -y \
    ros-humble-desktop \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for ZED
RUN pip3 install PyOpenGL PyOpenGL_accelerate requests && \
    pip3 install -U colcon-bash

# @TODO: What is this for?
RUN git clone https://github.com/stereolabs/zed-sdk.git

# Add environment variables to .bashrc
# @TODO: This doesn't work
RUN echo 'export ZED_SDK_ROOT=/usr/local/zed' >> ~/.bashrc && \
    echo 'export PYTHONPATH=$ZED_SDK_ROOT/lib/python3.10/site-packages:$PYTHONPATH' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=$ZED_SDK_ROOT/lib:$LD_LIBRARY_PATH' >> ~/.bashrc \
    echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc

# Add environment variables globally
ENV ZED_SDK_ROOT=/usr/local/zed
ENV PYTHONPATH=$ZED_SDK_ROOT/lib/python3.10/site-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=$ZED_SDK_ROOT/lib:$LD_LIBRARY_PATH

# Execute get_python_api.py script
RUN cd $ZED_SDK_ROOT && python3 get_python_api.py

RUN apt-get update && apt-get install -y \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

# Set up ROS2 workspace and clone the ZED ROS2 wrapper repository
RUN mkdir -p ~/ros2_ws/src/ && \
    cd ~/ros2_ws/src/ && \
    git clone --recursive https://github.com/stereolabs/zed-ros2-wrapper.git && \
    git clone -b ros2 https://gitlab.com/eufs/eufs_msgs.git

# Install dependencies and Python packages
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-pip \
    && pip3 install -U colcon-bash


# @TODO: This doesn't work, exported variables only exist in the current layer
RUN export ROS_PYTHON_VERSION=3
RUN export ROS_DISTRO=humble

# Update package list, source ROS setup, initialize ROS dependencies, install dependencies, and build the workspace
RUN . /opt/ros/humble/setup.sh && \
    cd ~/ros2_ws && \
    export ROS_PYTHON_VERSION=3 && \ 
    export ROS_DISTRO=humble && \
    export PYTHONPATH=/opt/ros/humble/lib/python3.8/site-packages:$PYTHONPATH && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc) 
    # echo "source ~/ros2_ws/install/local_setup.bash" >> ~/.bashrc

RUN pip3 install torch pandas    

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
SHELL ["/bin/bash", "-c"]
CMD ["bash"]

