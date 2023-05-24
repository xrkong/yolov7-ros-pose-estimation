#!/usr/bin/env python3
import rospy
import cv2
import time
import json
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from std_msgs.msg import String

prev_frame_time = 0
bridge = CvBridge()

# TODO: analysis of pose data and kpt
class ActionEstimator:
    def __init__(self, kpt: str = None):
        self.kpt = kpt # [[C_x, C_y, W, H, conf, class, kpt_x, kpt_y, kpt_conf, ..., K17_x, K17_y, K17_conf], [], ...]]
        self.action = None

    def empty(self):
        return len(self.kpt) == 0 or self.action == None

    def set_kpt(self, kpt: str):
        self.kpt = kpt
        if(len(self.kpt) > 0):
            self.action = 'Test action'

# TODO: publish action label
class EstimatorNode():
    def __init__(self, img_topic: str, pose_topic: str, queue_size: int = 1, visualize: bool = True, out_topic: str = 'action_label'):
        self.estimator = ActionEstimator()
        self.img_subscriber = rospy.Subscriber(img_topic, Image, self.process_img_msg)
        self.pose_subscriber = rospy.Subscriber(pose_topic, String, self.process_kpt_msg) 
        self.track_subscriber = rospy.Subscriber('/limbs_track', String, self.process_track_msg) 

    def process_img_msg(self, image):
        global prev_frame_time
        new_frame_time = time.time()
        #bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')

        fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        #cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        if visualize:
            if self.estimator.empty():
                cv2.putText(image, 'No action', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(image, self.estimator.action, (int(self.estimator.kpt[0][0]),int(self.estimator.kpt[0][1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Action label', image)
            cv2.waitKey(1)

    def process_kpt_msg(self, kpt):
        self.estimator.set_kpt(json.loads(kpt.data)) # yolov7 kpt format
        limbs_name = ["left upper arm", "right upper arm", "left lower arm", "right lower arm",
                      "left upper leg", "right upper leg", "left lower leg", "right lower leg"]   
        limbs = [[5,7], [6,8], [7,9], [8,10], [11,13], [12,14], [13,15], [14,16]]
        

    def process_track_msg(self, track):
        self.estimator.set_kpt(json.loads(track.data)) # 


if __name__ == '__main__':
    rospy.init_node('action_estimator')

    ns = rospy.get_name() + "/"

    img_topic = rospy.get_param(ns + "img_topic")
    pose_topic = rospy.get_param(ns + "pose_topic")
    out_topic = rospy.get_param(ns + "out_topic")
    queue_size = rospy.get_param(ns + "queue_size")
    visualize = rospy.get_param(ns + "visualize")

    publisher = EstimatorNode(
        img_topic = img_topic, 
        pose_topic = pose_topic, 
        queue_size = queue_size, 
        visualize = visualize,
        out_topic = out_topic)


    rospy.spin()
