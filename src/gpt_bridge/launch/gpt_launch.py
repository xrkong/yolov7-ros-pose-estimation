from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gpt_bridge',
            executable='gpt_bridge',
            name='gpt_bridge',
            output='screen',
            emulate_tty=True,
            parameters=[{    
                "pose_topic": "/yolov7/kpt",
                "bbox_topic": "/yolov7/bbox",
                "out_topic": "gpt_answer",
                "queue_size": 10,
                "classes_path": "/home/kong/my_ws/yolov7-ros-pose-estimation/src/yolov7_ros/class_labels/coco.txt"}]
        ),
    ])
