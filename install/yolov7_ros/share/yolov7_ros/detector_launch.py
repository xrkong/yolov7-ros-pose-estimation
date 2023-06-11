from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
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
                'conf_thresh': 0.35,
                'iou_thresh': 0.45,
                'queue_size': 10,
                'img_width': 1280,
                'img_height': 320,
                'visualize': True,
                'device': 'cuda'} ]
        )
    ])
