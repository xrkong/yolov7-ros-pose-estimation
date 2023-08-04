#! /bin/bash
export PYTHONPATH=$PYTHONPATH:/home/kong/my_ws/yolov7-ros-pose-estimation/src/yolov7_ros/yolov7_ros
export PYTHONPATH=$PYTHONPATH:/home/kong/my_ws/yolov7-ros-pose-estimation/src/mo_tracker/mo_tracker

colcon build
source install/setup.bash
echo "build & source"
ros2 launch yolov7_ros image_mot_launch.py
