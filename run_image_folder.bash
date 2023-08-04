#! /bin/bash

colcon build
source install/setup.bash
echo "build & source"
ros2 launch image_folder_publisher publisher_launch.py

