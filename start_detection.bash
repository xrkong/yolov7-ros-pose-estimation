#!/bin/bash 

source /opt/ros/foxy/install/setup.bash
colcon build
source install/setup.bash

ros2 launch yolov7_ros detector_launch.py
