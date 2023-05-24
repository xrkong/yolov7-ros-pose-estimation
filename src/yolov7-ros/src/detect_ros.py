#!/usr/bin/python3

from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt
from utils.ros import create_detection_msg
from visualizer import draw_detections
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt
import time

import os
from typing import Tuple, Union, List

import torch
import cv2
from torchvision.transforms import ToTensor
import numpy as np
import rospy
import json

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


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
        frame_count = 0  #count no of frames
        total_fps = 0  #count total fps
        time_list = []   #list to store time
        fps_list = []    #list to store fps
        start_time = time.time()
        img = img.unsqueeze(0)
        pred_results = self.model(img)[0]
        # detections = non_max_suppression(
        #     pred_results, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh
        # )
        # output_data, _ = self.model(img)
        detections = non_max_suppression_kpt(pred_results,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=self.model.yaml['nc'], # Number of classes.
                                            nkpt=self.model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
        
        '''
        detections[i][0:6] -> 
            x coordinate of the center of the bounding box
            y coordinate of the center of the bounding box
            w - width of the bounding box
            h - height of the bounding box
            conf - confidence in the bounding box
            class_id - class id of the object

        detections[i][6:j:57] -> 
            x coordinate of the (j-6)/3 keypoint
            y coordinate of the j keypoint
            conf - confidence in the j keypoint
        '''
        output = output_to_keypoint(detections)

        im0 = img[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
        im0 = im0.cpu().numpy().astype(np.uint8)
        
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        for i, det in enumerate(detections):  # detections per image
            if len(detections):  #check if no pose
                for c in det[:, 5].unique(): # Print results
                    n = (det[:, 5] == c).sum()  # detections per class
                    print("{} Objects in Current Frame".format(n))
                
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])): #loop over poses for drawing on frame
                    c = int(cls)  # integer class
                    kpts = det[det_index, 6:]
                    label = None # if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                    plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                line_thickness=3,kpt_label=True, kpts=kpts, steps=3, 
                                orig_shape=im0.shape[:2])

        
        end_time = time.time()  #Calculation for FPS
        fps = 1 / (end_time - start_time)
        total_fps += fps
        frame_count += 1
        
        fps_list.append(total_fps) #append FPS in list
        time_list.append(end_time - start_time) #append time in list
        
        cv2.putText(im0, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
        # Stream results
        #if view_img:
        #cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
        cv2.waitKey(1)  # 1 millisecond

        #out.write(im0)  #writing the video frame
        #print(f'{detections}')

        return detections, im0


class Yolov7Publisher:
    def __init__(self, img_topic: str, weights: str, conf_thresh: float = 0.5,
                 iou_thresh: float = 0.45, pub_topic: str = "yolov7",
                 device: str = "cuda",
                 img_size: Union[Tuple[int, int], None] = (640, 640),
                 queue_size: int = 1, visualize: bool = True,
                 class_labels: Union[List, None] = None):
        """
        :param img_topic: name of the image topic to listen to
        :param weights: path/to/yolo_weights.pt
        :param conf_thresh: confidence threshold
        :param iou_thresh: intersection over union threshold
        :param pub_topic: name of the output topic (will be published under the
            namespace '/yolov7')
        :param device: device to do inference on (e.g., 'cuda' or 'cpu')
        :param queue_size: queue size for publishers
        :visualize: flag to enable publishing the detections visualized in the image
        :param img_size: (height, width) to which the img is resized before being
            fed into the yolo network. Final output coordinates will be rescaled to
            the original img size.
        :param class_labels: List of length num_classes, containing the class
            labels. The i-th element in this list corresponds to the i-th
            class id. Only for viszalization. If it is None, then no class
            labels are visualized.
        """
        self.img_size = img_size
        self.device = device
        self.class_labels = class_labels

        vis_topic = pub_topic + "image" if pub_topic.endswith("/") else \
           pub_topic + "/image"
        self.visualization_publisher = rospy.Publisher(
            vis_topic, Image, queue_size=queue_size
        ) if visualize else None

        self.bridge = CvBridge()

        self.tensorize = ToTensor()
        self.model = YoloV7(
            weights=weights, conf_thresh=conf_thresh, iou_thresh=iou_thresh,
            device=device
        )
        self.img_subscriber = rospy.Subscriber(
            img_topic, Image, self.process_img_msg
        )

        # pubilsh detections as an array of Detection results
        pos_topic = pub_topic + "kpt" if pub_topic.endswith("/") else \
           pub_topic + "/kpt"
        self.detection_publisher = rospy.Publisher(
            pos_topic, String, queue_size=queue_size
        )
        rospy.loginfo("YOLOv7 initialization complete. Ready to start inference")

    def process_img_msg(self, img_msg: Image):
        """ callback function for publisher """
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
        # Apply NMS
        detections, im0 = self.model.inference(img)
        '''
        detections example:
        tensor([[5.12900e+00, 4.38904e+01, 6.39624e+02, 6.38738e+02, 9.39202e-01, 0.00000e+00, # C_x, C_y, W, H, conf, class
        4.47898e+02, 1.87086e+02, 9.98347e-01, # K0_x, K0_y, K0_conf
        4.74549e+02, 1.60572e+02, 9.96893e-01, # K1_x, K1_y, K1_conf
        4.09181e+02, 1.62192e+02, 9.97104e-01, # K2_x, K2_y, K2_conf
        5.11035e+02, 2.07350e+02, 9.33736e-01, # K3_x, K3_y, K3_conf
        3.57368e+02, 2.20482e+02, 8.57085e-01, # K4_x, K4_y, K4_conf
        5.80041e+02, 4.27738e+02, 8.05406e-01, 
        2.72383e+02, 4.23904e+02, 9.34467e-01, 
        6.24336e+02, 6.13686e+02, 8.06896e-02, 
        1.70459e+02, 6.15567e+02, 3.00836e-01, 
        5.65607e+02, 5.99629e+02, 7.81805e-02, 
        9.35799e+01, 6.16238e+02, 2.07950e-01, # ...
        4.89327e+02, 6.28644e+02, 2.17725e-02, 
        3.02137e+02, 6.24425e+02, 3.63526e-02, 
        4.20300e+02, 5.49449e+02, 6.38045e-03,
        2.77955e+02, 5.59165e+02, 9.96765e-03, 
        3.54366e+02, 5.52154e+02, 6.15365e-03, 
        2.96931e+02, 5.64074e+02, 7.98799e-03]], # K{16}_x, K{16}_y, K{16}_conf
        device='cuda:0')
        '''

        if detections is None:
            return

        # publish detections as a string of json
        # print(detections)
        detection_msg = json.dumps(detections[0].tolist())
        self.detection_publisher.publish(detection_msg)

        # visualizing if required
        if self.visualization_publisher:
            vis_msg = self.bridge.cv2_to_imgmsg(im0)
            self.visualization_publisher.publish(vis_msg)
        else:
            pass


if __name__ == "__main__":
    rospy.init_node("yolov7_node")

    ns = rospy.get_name() + "/"

    weights_path = rospy.get_param(ns + "weights_path")
    classes_path = rospy.get_param(ns + "classes_path")
    img_topic = rospy.get_param(ns + "img_topic")
    conf_thresh = rospy.get_param(ns + "conf_thresh")
    iou_thresh = rospy.get_param(ns + "iou_thresh")
    queue_size = rospy.get_param(ns + "queue_size")
    img_width = rospy.get_param(ns + "img_width")
    img_height = rospy.get_param(ns + "img_height")
    visualize = rospy.get_param(ns + "visualize")
    device = rospy.get_param(ns + "device")

    # some sanity checks
    if not os.path.isfile(weights_path):
        raise FileExistsError(f"Weights not found ({weights_path}).")
    
    if classes_path: 
        if not os.path.isfile(classes_path):
            raise FileExistsError(f"Classes file not found ({classes_path}).")
        classes = parse_classes_file(classes_path)
    else:
        rospy.loginfo("No class file provided. Class labels will not be visualized.")
        classes = None

    if not ("cuda" in device or "cpu" in device):
        raise ValueError("Check your device.")


    publisher = Yolov7Publisher(
        img_topic=img_topic,
        pub_topic=rospy.get_namespace(),
        weights=weights_path,
        device=device,
        visualize=visualize,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        img_size=(img_width, img_height),
        queue_size=queue_size,
        class_labels=classes
    )

    rospy.spin()
