from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mo_tracker',
            executable='mo_tracker',
            name='mo_tracker',
            output='screen',
            emulate_tty=True,
            parameters=[{    
                "img_topic": "image_raw",
                "pose_topic": "/yolov7/kpt",
                "bbox_topic": "/yolov7/bbox",
                "out_topic": "action_est",
                "queue_size": 10,
                "visualize": True,
                "classes_path": "/home/kong/my_ws/yolov7-ros-pose-estimation/src/yolov7_ros/class_labels/coco.txt"}]
        ),
    ])
