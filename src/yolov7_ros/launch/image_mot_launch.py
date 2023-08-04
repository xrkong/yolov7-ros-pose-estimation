import launch
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_folder_publisher',
            executable='image_folder_publisher',
            name='image_folder_publisher',
            output='screen',
            emulate_tty=True,
            parameters=[os.path.join(get_package_share_directory('image_folder_publisher'), "config/conf.yaml")]
        ),
        Node(
            package='yolov7_ros',
            executable='detect_car',
            name='detect_car',
            output='screen',
            emulate_tty=True,
            parameters=[{    
                'weights_path': '/home/kong/my_ws/nn_models/yolov7/yolov7.pt',
                'classes_path': '/home/kong/my_ws/yolov7-ros-pose-estimation/src/yolov7_ros/class_labels/coco.txt',
                'img_topic': '/image_raw',
                'conf_thresh': 0.75,
                'iou_thresh': 0.60,
                'queue_size': 10,
                'img_width': 1280,
                'img_height': 320,
                'visualize': False,
                'device': 'cuda'} ]
        ),
        Node(
            package='yolov7_ros',
            executable='detect_ped',
            name='detect_ped',
            output='screen',
            emulate_tty=True,
            parameters=[{    
                'weights_path': '/home/kong/my_ws/nn_models/yolov7/yolov7-w6-pose.pt',
                'classes_path': '/home/kong/my_ws/yolov7-ros-pose-estimation/src/yolov7_ros/class_labels/coco.txt',
                'img_topic': '/image_raw',
                'conf_thresh': 0.60,
                'iou_thresh': 0.60,
                'queue_size': 10,
                'img_width': 1280,
                'img_height': 320,
                'visualize': False,
                'device': 'cuda'} ]
        ),
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
                "visualize": False,
                "classes_path": "/home/kong/my_ws/yolov7-ros-pose-estimation/src/yolov7_ros/class_labels/coco.txt"}]
        ),
    ])
