#!/bin/bash
set -e

export DISPLAY=:0
export ROS_DOMAIN_ID=5
export PYTHONPATH=$PYTHONPATH:/ws/yolov7-ros-pose-estimation/src/yolov7_ros/yolov7_ros/:/ws/yolov7-ros-pose-estimation/src/mo_tracker/:/ws/yolov7-ros-pose-estimation/install/yolov7_ros/lib/python3.8/site-packages/yolov7_ros/

exec "$@"
