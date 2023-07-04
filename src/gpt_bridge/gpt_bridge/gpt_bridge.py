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
            #label = self.class_labels[int(self.obj_dets[i][5])]
            id = int(self.ped_dets[i][5])
            conf = self.ped_dets[i][4]
            centre = [int((self.ped_dets[i][0]+self.ped_dets[i][2])/2), int((self.ped_dets[i][1]+self.ped_dets[i][3])/2)]
            #xyxy = self.obj_dets[i][0:4]
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
            # overlapping objects
            print([id, 'person', self.ped_dets[i][0:4], "{:.3f}".format(conf), 
                  left_shoulder, left_elbow, left_wrist,
                  right_shoulder, right_elbow, right_wrist, 'None', 0])
            print(limbes)

    def obj_det_callback(self, msg):
        '''output: [[label, xyxy,conf,],...]'''
        self.obj_dets = json.loads(msg.data)
        if len(self.obj_dets) <= 0: return
        for i in range(len(self.obj_dets)):
            if self.class_labels:
                label = self.class_labels[int(self.obj_dets[i][5])]
                xyxy = self.obj_dets[i][0:4] # TODO: combine with lane detection to [left],[middle],[right]
                conf = self.obj_dets[i][4]
            print(label, "{:.3f}".format(conf))
  

def main(args=None):
    rclpy.init(args=args)
    node = GptBridgeNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
