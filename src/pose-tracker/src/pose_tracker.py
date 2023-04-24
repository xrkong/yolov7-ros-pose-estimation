#!/usr/bin/env python3
import rospy
import mediapipe as mp
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
prev_frame_time = 0
bridge = CvBridge()
pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


class poseTracker:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

def callback(img_msg):
    global prev_frame_time
    new_frame_time = time.time()
    #bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    #results = image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    '''
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15

    '''

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    

    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('MediaPipe Pose', image)
    cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('pose_tracker', anonymous=True)


    rospy.Subscriber("/image_detection", Image, callback)
    #rospy.Subscriber("/usb_cam/image_raw", Image, callback)
    # spin() simply keeps python from exiting until this node is stopped

    rospy.spin()
