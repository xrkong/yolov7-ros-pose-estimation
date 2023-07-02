#!/usr/bin/python3
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt
from .utils.plots import output_to_keypoint
from .utils.plots import plot_one_box_kpt
from .visualizer import draw_detections

from typing import Tuple, Union
import time

import torch
import cv2
from torchvision.transforms import ToTensor
import numpy as np
import rclpy
import rclpy.node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import json
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import warnings
warnings.filterwarnings("ignore") 

def parse_classes_file(path):
    classes = []
    with open(path, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            classes.append(line)
    return classes

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

def rescale_detection(detections, new_shape : Tuple[int, int],ori_shape: Tuple[int, int]):
    xscale = new_shape[0] / ori_shape[0]
    yscale = new_shape[1] / ori_shape[1]
    #print(detections)
    detections[:, 0] *= xscale
    detections[:, 2] *= xscale
    detections[:, 1] *= yscale
    detections[:, 3] *= yscale

    for det in detections:
        if len(det) > 7:
            det[6::3] *= xscale
            det[7::3] *= yscale

    return detections

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
        :returns: tensor of shape [num_boxes, 57], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id, K1_x, K1_y, K1_conf, ... , K17_x, K17_y, K17_conf]
            and image with keypoints drawn on it.
        """
        img = img.unsqueeze(0)
        pred_results = self.model(img)[0]
        detections_pdst = non_max_suppression_kpt(pred_results,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.5, # IoU Threshold.
                                            nc=self.model.yaml['nc'], # Number of classes.
                                            nkpt=self.model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
        
        '''
        detections[i][0:6] -> [x,y,w,h,conf,class_id]
        detections[i][6:j:57] -> [x, y, conf]
        '''
        output_pdst = output_to_keypoint(detections_pdst)

        im0 = img[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
        im0 = im0.cpu().numpy().astype(np.uint8)
        
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        return detections_pdst

class Yolov7Publisher(rclpy.node.Node):
    def __init__(self):
        super().__init__('detect_ped')
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
        self.declare_parameter('visualize', False, visualize_des)
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
        self.visualization_publisher = self.create_publisher(Image, '/yolov7/visualization', 10)

        self.bridge = CvBridge()
        self.tensorize = ToTensor()
        self.model = YoloV7(
            weights=self.weights, conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh,
            device=self.device
        )
        self.camera_info_sub = self.create_subscription(
            Image, self.img_topic, self.process_img_msg, 10)

        self.detection_publisher = self.create_publisher(
            String, "/yolov7/kpt", 10)
        self.get_logger().info('Hello %s!' % "YOLOv7 initialization complete. Ready to start inference")

    def process_img_msg(self, img_msg: Image):
        """ callback function for publisher """
        start_time = time.time()
        np_img_orig = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='bgr8'
        )

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

        if detections is None:
            return
        # publishing
        detections[0] = rescale_detection(detections[0], (w_orig, h_orig),(w_scaled, h_scaled))
        detection_msg = json.dumps(detections[0].tolist())
        msg = String()
        msg.data = detection_msg
        self.detection_publisher.publish(msg)

        # visualizing if required
        if self.visualize:
            
            end_time = time.time()  #Calculation for FPS
            fps = 1 / (end_time - start_time)
            for i, det in enumerate(detections):  # detections per image
                for c in det[:, 5].unique(): # Print results
                    n = (det[:, 5] == c).sum()  # detections per class                
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])): #loop over poses for drawing on frame
                    c = int(cls)  # integer class
                    kpts = det[det_index, 6:]
                    label = None # if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                    plot_one_box_kpt(xyxy, np_img_orig, label=label, #color=colors(c, True), 
                                line_thickness=3,kpt_label=True, kpts=kpts, steps=3, 
                                orig_shape=np_img_orig.shape[:2])
                    #plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=1)
            
            cv2.putText(np_img_orig, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
            im0 = cv2.resize(np_img_orig,(w_orig, h_orig))
        
            #cv2.imshow("Pedestrian Detector", im0)
            #cv2.waitKey(1) 
            

def main():
    rclpy.init()
    node = Yolov7Publisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
