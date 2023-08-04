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
    ])
