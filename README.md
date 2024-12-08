# Running - prepare for push test 4-12-2024
```bash
# colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc)
colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc) --packages-select eufs_msgs
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

# Set up
Check if the camera is connected using any camera app (there are `vlc` or `cheese`)

# Problem Log
- Latency in image transmission - high even when streaming raw images without ROS, possible solutions are
    - GST image transmission pipeline
    - To improve performance with ROS, use Zero-Copy 
- Examine why the zed docker provided by [Stereolabs](wiki/DOCKER_ZED_ROS2_WRAPPER.md) doesn't work 

# Check Out
- Resource for docker images
    - https://github.com/dusty-nv/jetson-containers
    - https://hub.docker.com/r/dustynv/ros/tags?name=humble-ros

# Working Setup
Long Train's [Dockerfile](zed_2i_LONG/zed-2i-LONG.Dockerfile) 

## Fixme
- Resolve @TODO's
- It has to setup everytime the container is restarted
```bash
[component_container_isolated-2] [2024-12-04 22:26:08 UTC][ZED][INFO] [Init]  No calibration file found for SN 28503076. Downloading... 
[component_container_isolated-2] [2024-12-04 22:26:08 UTC][ZED][INFO] [Init]  Calibration file downloaded.
```
- A lot of setup when loading model:
```bash
Collecting ultralytics
  Downloading ultralytics-8.3.41-py3-none-any.whl (899 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 899.1/899.1 KB 8.5 MB/s eta 0:00:00
Requirement already satisfied: torch>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.5.1)
Requirement already satisfied: matplotlib>=3.3.0 in /usr/lib/python3/dist-packages (from ultralytics) (3.5.1)
Collecting opencv-python>=4.6.0
  Downloading opencv_python-4.10.0.84-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl (41.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.7/41.7 MB 6.7 MB/s eta 0:00:00
Requirement already satisfied: pandas>=1.1.4 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.2.3)
Requirement already satisfied: pyyaml>=5.3.1 in /usr/lib/python3/dist-packages (from ultralytics) (5.4.1)
Collecting tqdm>=4.64.0
  Downloading tqdm-4.67.1-py3-none-any.whl (78 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.5/78.5 KB 4.4 MB/s eta 0:00:00
Requirement already satisfied: psutil in /usr/lib/python3/dist-packages (from ultralytics) (5.9.0)
Collecting ultralytics-thop>=2.0.0
  Downloading ultralytics_thop-2.0.12-py3-none-any.whl (26 kB)
Collecting seaborn>=0.11.0
  Downloading seaborn-0.13.2-py3-none-any.whl (294 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.9/294.9 KB 7.7 MB/s eta 0:00:00
Requirement already satisfied: pillow>=7.1.2 in /usr/lib/python3/dist-packages (from ultralytics) (9.0.1)
Collecting torchvision>=0.9.0
  Downloading torchvision-0.20.1-cp310-cp310-manylinux2014_aarch64.whl (14.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.3/14.3 MB 6.4 MB/s eta 0:00:00
Collecting py-cpuinfo
  Downloading py_cpuinfo-9.0.0-py3-none-any.whl (22 kB)
Requirement already satisfied: numpy>=1.23.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.26.4)
Requirement already satisfied: scipy>=1.4.1 in /usr/lib/python3/dist-packages (from ultralytics) (1.8.0)
Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.32.3)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.1.4->ultralytics) (2.9.0.post0)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.1.4->ultralytics) (2024.2)
Requirement already satisfied: pytz>=2020.1 in /usr/lib/python3/dist-packages (from pandas>=1.1.4->ultralytics) (2022.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (3.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (2024.8.30)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (2.2.3)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (3.10)
Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (4.12.2)
Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (2024.10.0)
Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (1.13.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.1.4)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.16.1)
Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.4.2)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy==1.13.1->torch>=1.8.0->ultralytics) (1.3.0)
Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.8.2->pandas>=1.1.4->ultralytics) (1.16.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.8.0->ultralytics) (3.0.2)
Installing collected packages: py-cpuinfo, tqdm, opencv-python, ultralytics-thop, torchvision, seaborn, ultralytics
Successfully installed opencv-python-4.10.0.84 py-cpuinfo-9.0.0 seaborn-0.13.2 torchvision-0.20.1 tqdm-4.67.1 ultralytics-8.3.41 ultralytics-thop-2.0.12
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
Creating new Ultralytics Settings v0.0.6 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
requirements: Ultralytics requirements ['gitpython>=3.1.30', 'Pillow>=10.0.1', 'setuptools>=65.5.1'] not found, attempting AutoUpdate...
```

- Container dies
```bash
[ERROR] [component_container_isolated-2]: process has died [pid 33633, exit code -11, cmd '/opt/ros/humble/lib/rclcpp_components/component_container_isolated --ros-args -r __node:=zed_container -r __ns:=/zed'].
```

- Running the zed alone takes 85-98% CPU
```bash
top
```
  - Reso to 720
  - Jetson Orin Nano runs Issac ROS's image segmentation graph at 2.22 fps Hz - [link](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation)


