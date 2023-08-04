#!/usr/bin/env python3
from __future__ import print_function
import sys
import os
from os import listdir
from os.path import isfile, join
import time

import rclpy
import rclpy.node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_folder_publisher(rclpy.node.Node):
    def __init__(self):
        super().__init__('image_folder_publisher')

        self._cv_bridge = CvBridge()

        self.declare_parameter('topic_name', '/image_raw', ParameterDescriptor(description='image topic name'))
        self.declare_parameter('publish_rate', 5, ParameterDescriptor(description='rate to publish images'))
        self.declare_parameter('sort_files', True, ParameterDescriptor(description='sort files in the folder'))
        self.declare_parameter('frame_id', 'camera', ParameterDescriptor(description='frame id of the image'))
        self.declare_parameter('loop', 1, ParameterDescriptor(description='loop over the images in the folder'))
        self.declare_parameter('image_folder', '/home/kong', ParameterDescriptor(description='folder containing the images'))
        self.declare_parameter('sleep', 0, ParameterDescriptor(description='sleep after each image'))
        self.declare_parameter('splits', [], ParameterDescriptor(description='select folders to publish'))

        self._topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.get_logger().info(f"(topic_name) Publishing Images to topic {self._topic_name}")

        self._image_publisher = self.create_publisher(Image, self._topic_name, 10)
        self._rate = self.get_parameter('publish_rate').get_parameter_value().integer_value
        self.get_logger().info(f"(publish_rate) Publish rate set to {self._rate} hz")

        self._sort_files = self.get_parameter('sort_files').get_parameter_value().bool_value
        self.get_logger().info(f" (sort_files) Sort Files: {self._sort_files}")

        self._sort_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.get_logger().info(f" (frame_id) Frame ID set to  {self._sort_id}")

        self._loop = self.get_parameter('loop').get_parameter_value().integer_value
        self.get_logger().info(f" (loop) Loop  {self._loop} time(s) (set it -1 for infinite)")

        self._image_folder = self.get_parameter('image_folder').get_parameter_value().string_value
        self.get_logger().info(f" (image_folder) Image folder set to {self._image_folder}")

        self._sleep = self.get_parameter('sleep').get_parameter_value().double_value
        self.get_logger().info(f" (sleep) Sleep {self._sleep} seconds")

        self._splits = self.get_parameter('splits').get_parameter_value().string_array_value
        self.get_logger().info(f" (splits) Splits {self._splits}")
        
        if self._image_folder == '' or not os.path.exists(self._image_folder) or not os.path.isdir(self._image_folder):
            self.get_logger().fatal(f" (image_folder) Invalid Image folder {self._image_folder}")
            sys.exit(0)
        self.get_logger().info(f" Reading images from {self._image_folder}")

    def run(self):
        time.sleep(self._sleep)
        while self._loop != 0:
            if self._splits != []:
                for split in self._splits:
                    self._sub_image_folder = join(self._image_folder, split)
                    self._publish_images()
            else:
                self._sub_image_folder = self._image_folder
                self._publish_images()
            
            self._loop = self._loop - 1

    def _publish_images(self):
        files_in_dir = [f for f in listdir(self._sub_image_folder) if isfile(join(self._sub_image_folder, f))]
        if self._sort_files:
            files_in_dir.sort()

        try:
            for f in files_in_dir:
                #self.get_logger().info(f" Reading {f}")
                if rclpy.ok():
                    if isfile(join(self._sub_image_folder, f)):
                        cv_image = cv2.imread(join(self._sub_image_folder, f))
                        if cv_image is not None:
                            file_name = os.path.splitext(f)[0]
                            ros_msg = Image()
                            ros_msg = self._cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")
                            #print(ros_msg.width, ros_msg.height, ros_msg.encoding, ros_msg.step)
                            ros_msg.header.frame_id = self._sub_image_folder.split('/')[-1]+'/'+file_name # for kitti dataset, image name is the frame id
                            print(ros_msg.header.frame_id)
                            ros_msg.header.stamp = self.get_clock().now().to_msg()
                            self._image_publisher.publish(ros_msg)
                            self.get_logger().info(f"Published {join(self._sub_image_folder, f)}")
                            # cv2.imshow('image', cv_image)
                            # cv2.waitKey(1)
                        else:
                            self.get_logger().info(f"Invalid image file {f}")
                        time.sleep(1/self._rate)
                else:
                    return
            #self.get_logger().info(f"Looping {self._loop} time(s) left")
        except CvBridgeError as e:
            self.get_logger().error(e)


def main():
    # rclpy.init_node('image_folder_publisher', anonymous=True)
    # image_publisher = image_folder_publisher()
    # image_publisher.run()
    rclpy.init()
    image_publisher = image_folder_publisher()
    image_publisher.run()

if __name__ == '__main__':
    main()