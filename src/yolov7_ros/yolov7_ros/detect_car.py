#!/usr/bin/python3
from models.experimental import attempt_load
from utils.general import non_max_suppression
from .visualizer import draw_detections

from typing import Tuple, Union

import os
import torch
import cv2
from torchvision.transforms import ToTensor
import numpy as np
import rclpy
import rclpy.node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import json
import csv 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String

def parse_classes_file(path):
    classes = []
    with open(path, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            classes.append(line)
    return classes

def save_string_to_csv(file_path, data):
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header if the file doesn't exist
        if not file_exists:
            writer.writerow(['objects'])
        
        # Write the string to the CSV file
        writer.writerow([data])

def rescale(ori_shape: Tuple[int, int], boxes: Union[torch.Tensor, np.ndarray],
            target_shape: Tuple[int, int]):
    """Rescale the output to the original image shape
    :param ori_shape: original width and height [width, height].
    :param boxes: original bounding boxes as a torch.Tensor or np.array or shape
        [num_boxes, >=4], where the first 4 entries of each element have to be
        [x1, y1, x2, y2].
    :param target_shape: target width and height [width, height].
    """
    xscale = target_shape[1] / ori_shape[1]
    yscale = target_shape[0] / ori_shape[0]

    boxes[:, [0, 2]] *= xscale
    boxes[:, [1, 3]] *= yscale

    return boxes

def save_images(img, save_path):
    """
    Saves an image to a specified directory with names incrementally.
    
    Parameters:
    img: The image to be saved.
    save_path: Directory in which images will be saved.
    """

    # Check if the directory exists, if not, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if the image counter exists
    if not hasattr(save_images, "counter"):
        save_images.counter = 1  # it doesn't exist yet, so initialize it

    # Construct the full path for the image
    img_name = f"{save_images.counter}.jpg"
    path = os.path.join(save_path, img_name)

    # Save the image
    cv2.imwrite(path, img)

    # Increment the counter
    save_images.counter += 1

class YoloV7:
    def __init__(self, weights, conf_thresh: float = 0.5, iou_thresh: float = 0.45,
                 device: str = "cuda"):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.model = attempt_load(weights, map_location=device)
        self.model.eval()

    @torch.no_grad()
    def inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        img = img.unsqueeze(0)
        pred_results = self.model(img)[0]
        detections = non_max_suppression(
            pred_results, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh
        )

        if detections:
            detections = detections[0]
         
        return detections

class Yolov7Publisher(rclpy.node.Node):
    def __init__(self):
        super().__init__('detect_car')
        weights_path_des = ParameterDescriptor(description='absolute path to the weights file')
        classs_path_des = ParameterDescriptor(description='absolute path to the classes file for yolo')
        img_topic_des = ParameterDescriptor(description='name of the image topic to listen to')
        conf_thresh_des = ParameterDescriptor(description='confidence threshold')
        iou_thresh_des = ParameterDescriptor(description='intersection over union threshold')
        queue_size_des = ParameterDescriptor(description='queue size for publishers')
        img_width_des = ParameterDescriptor(description='width of the image')
        img_height_des = ParameterDescriptor(description='height of the image')
        visualize_des = ParameterDescriptor(description='flag to enable publishing the detections visualized in the image')
        device_des = ParameterDescriptor(description='device to do inference on (e.g., "cuda" or "cpu")')

        self.declare_parameter('weights_path', '/home/kong/my_ws/nn_models/yolov7/yolov7-w6-pose.pt', weights_path_des)
        self.declare_parameter('classes_path', '/home/kong/my_ws/yolov7-ros-pose-estimation/src/yolov7_ros/class_labels/berkeley.txt', classs_path_des)
        self.declare_parameter('img_topic', '/image_raw', img_topic_des)
        self.declare_parameter('conf_thresh', 0.35, conf_thresh_des)
        self.declare_parameter('iou_thresh', 0.45, iou_thresh_des)
        self.declare_parameter('queue_size', 10, queue_size_des)
        self.declare_parameter('img_width', 1280, img_width_des)
        self.declare_parameter('img_height', 320, img_height_des)
        self.declare_parameter('visualize', True, visualize_des)
        self.declare_parameter('device', 'cuda', device_des)
        
        self.weights = self.get_parameter('weights_path').get_parameter_value().string_value
        self.classes_path = self.get_parameter('classes_path').get_parameter_value().string_value
        self.img_topic = self.get_parameter('img_topic').get_parameter_value().string_value
        self.conf_thresh = self.get_parameter('conf_thresh').get_parameter_value().double_value
        self.iou_thresh = self.get_parameter('iou_thresh').get_parameter_value().double_value
        self.queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        self.img_width = self.get_parameter('img_width').get_parameter_value().integer_value
        self.img_height = self.get_parameter('img_height').get_parameter_value().integer_value
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.img_size = (self.img_width, self.img_height)
        self.class_labels = parse_classes_file(self.get_parameter('classes_path').get_parameter_value().string_value)
        
        #print("class labels: ", self.class_labels)
        self.visualization_publisher = self.create_publisher(Image, '/yolov7/visualization', 10)

        self.bridge = CvBridge()

        self.tensorize = ToTensor()
        self.model = YoloV7(
            weights=self.weights, conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh,
            device=self.device
        )
        self.camera_info_sub = self.create_subscription(
            Image, self.img_topic, self.process_img_msg, 10)

        #bbox_topic = self.create_publisher(String, '/yolov7/bbox', 10)
        self.detection_publisher = self.create_publisher(
            String, "/yolov7/bbox", 10)
        self.get_logger().info('Hello %s!' % "YOLOv7 initialization complete. Ready to start inference")

    def process_img_msg(self, img_msg: Image):
        """ callback function for publisher """
        img_id = img_msg.header.frame_id
        np_img_orig = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='bgr8'
        )
        # cv2.imshow("Object Detector Raw", np_img_orig)
        # cv2.waitKey(1)
        # handle possible different img formats
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis=2)

        h_orig, w_orig, c = np_img_orig.shape

        # automatically resize the image to the next smaller possible size
        w_scaled, h_scaled = self.img_size
        np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))

        # conversion to torch tensor (copied from original yolov7 repo)
        img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.device)

        # inference & rescaling the output to original img size
        detections = self.model.inference(img)
        detections[:, :4] = rescale(
            [h_scaled, w_scaled], detections[:, :4], [h_orig, w_orig])
        detections[:, :4] = detections[:, :4].round()

        # publishing
        #print("objects: ", detections.tolist())
        detection_msg = json.dumps(detections.tolist())
        save_string_to_csv('/home/kong/my_ws/llm_chatgpt/data/'+img_id, detections.tolist())
        msg = String()
        msg.data = detection_msg
        self.detection_publisher.publish(msg)

        # visualizing if required
        if self.visualization_publisher:
            bboxes = [[int(x1), int(y1), int(x2), int(y2)]
                      for x1, y1, x2, y2 in detections[:, :4].tolist()]
            classes = [int(c) for c in detections[:, 5].tolist()]
            vis_img = draw_detections(np_img_orig, bboxes, classes,
                                      self.class_labels)
            vis_msg = self.bridge.cv2_to_imgmsg(cv2.resize(vis_img,(w_scaled, h_scaled)))
            cv2.imshow("Object Detector", vis_img)
            cv2.waitKey(1)
            #save_images(vis_img, './obj_images')
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img)
            self.visualization_publisher.publish(vis_msg)

def main():
    rclpy.init()
    node = Yolov7Publisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()