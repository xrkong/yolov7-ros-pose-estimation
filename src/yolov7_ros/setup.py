import os
from glob import glob
from setuptools import setup
from setuptools import find_packages

package_name = 'yolov7_ros'
submodules = ["yolov7_ros.utils", "yolov7_ros.models"]

setup(
    name=package_name,
    version='0.0.0',
    #packages=[package_name],
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kong',
    maintainer_email='xiangrui.kong@research.uwa.edu.au',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_car = yolov7_ros.detect_car:main',
            'detect_ped = yolov7_ros.detect_ped:main'
        ],
    },
)
