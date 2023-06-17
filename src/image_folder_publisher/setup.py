import os
from glob import glob
from setuptools import setup
from setuptools import find_packages

package_name = 'image_folder_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    #packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        #(os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
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
            'image_folder_publisher = image_folder_publisher.image_folder_publisher:main'
        ],
    },
)
