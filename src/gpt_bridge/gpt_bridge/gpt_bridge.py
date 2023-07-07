#!/usr/bin/env python3
import rclpy
import rclpy.node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from std_msgs.msg import String
import json


kpt_name = ["nose", "left eye", "right eye", "left ear", "right ear", 
            "left shoulder", "right shoulder"]       
limbs = [[5,7], [6,8], [7,9], [8,10], [11,13], [12,14], [13,15], [14,16]]     
limb_name = ["left upper arm", "right upper arm", "left lower arm", "right lower arm",
                "left upper leg", "right upper leg", "left lower leg", "right lower leg"]   

def parse_classes_file(path):
    classes = []
    with open(path, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            classes.append(line)
    return classes

def check_overlap(box1, box2): # box format: x1y1x2y2
    if box1[0] > box2[2] or box1[2] < box2[0] or box1[1] > box2[3] or box1[3] < box2[1]:
        combox = [0,0,0,0]
        return (False,-1, )
    else:
        x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
        y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
        overlap_area = int(x_overlap * y_overlap)
        combox = [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])]
        return (True, overlap_area, combox)

class GptBridgeNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('gpt_bridge')
        self.declare_parameter('pose_topic', '/yolov7/kpt', 
                               ParameterDescriptor(description='pose topic name'))
        self.declare_parameter('bbox_topic', '/yolov7/object/bbox', 
                               ParameterDescriptor(description='bbox topic name'))
        self.declare_parameter('queue_size', 1, 
                               ParameterDescriptor(description='rate to publish images'))
        self.declare_parameter('out_topic', 'tracks', 
                               ParameterDescriptor(description='out topic name'))
        self.declare_parameter('classes_path', '/home/kong/my_ws/yolov7-ros-pose-estimation/src/yolov7_ros/class_labels/berkeley.txt', 
                               ParameterDescriptor(description='class path'))

        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.bbox_topic = self.get_parameter('bbox_topic').get_parameter_value().string_value
        self.queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        self.out_topic = self.get_parameter('out_topic').get_parameter_value().string_value
        self.class_labels = parse_classes_file(self.get_parameter('classes_path').get_parameter_value().string_value)

        self.ped_dets = []
        self.obj_dets = []

        self.pose_subscriber = self.create_subscription(
            String, self.pose_topic, self.ped_det_callback, 10)
        self.bbox_subscriber = self.create_subscription(
            String, self.bbox_topic, self.obj_det_callback, 10)
        bbox_topic = self.create_publisher(String, '/limbs_track', 10)

    def ped_det_callback(self, msg):
        '''output: {[ID, label, (x,y,conf), 
        left shoulder(x,y,conf), left elbow, left wrist, 
        right shoulder, right elbow, right wrist, 
        overlapping object label = none, overlapping object confidence = none]}'''
        self.ped_dets = json.loads(msg.data)
        if len(self.ped_dets) <= 0: return

        kpt = {"left shoulder":5,  "left elbow":7,  "left wrist":9,
            "right shoulder":6, "right elbow":8, "right wrist":10}
        for i in range(len(self.ped_dets)):
            id = int(self.ped_dets[i][5])
            conf = self.ped_dets[i][4]
            centre = [int((self.ped_dets[i][0]+self.ped_dets[i][2])/2), int((self.ped_dets[i][1]+self.ped_dets[i][3])/2)]
            xyxy_ped = [int(x) for x in self.ped_dets[i][0:4]]
            x0 = self.ped_dets[i][0]
            y1 = self.ped_dets[i][3] # transform Right Down Coodinate to Rigth Up Coordinate
            left_shoulder = [int(self.ped_dets[i][6+kpt["left shoulder"]*3]-x0), int(y1-self.ped_dets[i][6+kpt["left shoulder"]*3+1])]
            left_elbow = [int(self.ped_dets[i][6+kpt["left elbow"]*3]-x0), int(y1-self.ped_dets[i][6+kpt["left elbow"]*3+1])]
            left_wrist = [int(self.ped_dets[i][6+kpt["left wrist"]*3]-x0), int(y1-self.ped_dets[i][6+kpt["left wrist"]*3+1])]
            right_shoulder = [int(self.ped_dets[i][6+kpt["right shoulder"]*3]-x0), int(y1-self.ped_dets[i][6+kpt["right shoulder"]*3+1])]
            right_elbow = [int(self.ped_dets[i][6+kpt["right elbow"]*3]-x0), int(y1-self.ped_dets[i][6+kpt["right elbow"]*3+1])]
            right_wrist = [int(self.ped_dets[i][6+kpt["right wrist"]*3]-x0), int(y1-self.ped_dets[i][6+kpt["right wrist"]*3+1])]
            limbes = [[int(value - x0) for value in self.ped_dets[i][6::3]],
                [int(y1 - value) for value in self.ped_dets[i][7::3] ]]
            
            # calaulate overlapping objects
            op_flg, op_area, op_conf, op_label = (False, 0, 0, 'None')
            if self.obj_dets is not None:
                for j in range(len(self.obj_dets)):
                    xyxy_obj = [int(x) for x in self.obj_dets[j][0:4]]
                    lb = self.class_labels[int(self.obj_dets[j][5])]
                    if lb == 'person':
                        continue

                    if check_overlap(xyxy_ped, xyxy_obj)[0] and check_overlap(xyxy_ped, xyxy_obj)[1] > op_area:
                        op_flg = True
                        op_area = check_overlap(xyxy_ped, xyxy_obj)[1]
                        op_label = self.class_labels[int(self.obj_dets[j][5])]
                        op_conf = self.obj_dets[j][4]

            print([id, 'person', xyxy_ped, "{:.3f}".format(conf), 
                  left_shoulder, left_elbow, left_wrist,
                  right_shoulder, right_elbow, right_wrist, op_label, op_flg, op_area, "{:.3f}".format(op_conf)])
            print(limbes)
            # TODO: link images and results by bbox

    def obj_det_callback(self, msg):
        '''output: [[label, xyxy,conf,],...]'''
        self.obj_dets = json.loads(msg.data)
        if len(self.obj_dets) <= 0: return
        for i in range(len(self.obj_dets)):
            if self.class_labels:
                label = self.class_labels[int(self.obj_dets[i][5])]
                xyxy = self.obj_dets[i][0:4] # TODO: combine with lane detection to [left],[middle],[right]
                conf = self.obj_dets[i][4]
            #print(label, "{:.3f}".format(conf))
  

def main(args=None):
    rclpy.init(args=args)
    node = GptBridgeNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
