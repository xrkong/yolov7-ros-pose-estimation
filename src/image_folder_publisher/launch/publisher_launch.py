from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_folder_publisher',
            executable='image_folder_publisher',
            name='image_folder_publisher',
            output='screen',
            emulate_tty=True,
            parameters=[{    
                'topic_name': '/image_raw' ,
                'publish_rate': 30,
                'sort_files': True ,
                'frame_id': 'camera', 
                'sleep': 5, 
                'loop': -1, 
                'image_folder': '/home/kong/dataset/kitti/data_tracking_image_2/training/image_02/0007/' }] # 0019 pedestrian, 0005 highway, 007 right
        ),
    ])
