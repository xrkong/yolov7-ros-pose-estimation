#!/usr/bin/env python3
from visualizer import draw_detections
import track_eval
from sort import *

import rospy
import numpy as np
import cv2
import torch

import random
from typing import Tuple, Union, List

import json
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String

kpt_dict = {"left shoulder":5, "right shoulder":6,"left elbow":7, 
            "right elbow":8, "left wrist":9, "right wrist":10}
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

#[[track.object_data, track.id, track.centroidarr]] object_label range (0 ~ i*1e4)
def action_estimate(trackers): # update
    '''
    TODO list:
    - [ ] Logic to estimate action: left hand waving, right hand waving
        left hand waving: wrist is above elbow in the track, and the angle(lower arm) belongs to [30,-30]
        - [ ] Wrist_y > elbow_y for 80% of the time
        - [ ] Calaulate the angle(lower arm) and the sign changing times (positive <--> negative)
    - [ ] print them and publish them
    '''
    pass
    # global kpt_dict
    # act_label = {}
    # # trackers is a dict which key is id and value is track 
    # for key, value in trackers.items():
    #     person_id = key
    #     kpt_track = value
    #     #print(f'{value}')
    #     mean_kpt = np.mean(kpt_track[-5:], axis=0) # mean of cols
    #     # print(f'{mean_kpt}')

    #     if mean_kpt[5 + kpt_dict["left wrist"]*3+1] > mean_kpt[5 + kpt_dict["left elbow"]*3+1]:
    #         act_label[person_id] = 'Left hand up'
        
    #     elif mean_kpt[5 + kpt_dict["right wrist"]*3+1] > mean_kpt[5 + kpt_dict["right elbow"]*3+1]:
    #         act_label[person_id] = 'Right hand up'

    #     else:
    #         act_label[person_id] = 'No action'
        
    # return act_label

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2

    return coords

def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 4, 9, 9, 7, 7, 7, 0, 6, 0, 6, 0, 12, 16, 12, 16, 12, 16, 12]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 6, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
            #cv2.putText(im, f"k{kid}", (int(x_coord), int(y_coord)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
        #cv2.putText(im, f"{sk_id}", ((pos1[0] + pos2[0])//2,(pos1[1] + pos2[1])//2) , cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
        
def plot_one_box_kpt(x, im, color=None, label=None, line_thickness=3, kpt_label=True, kpts=None, steps=3, orig_shape=None):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'

    if label:
        if len(label.split(' ')) > 1:
            label = label.split(' ')[-1]
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)
    if kpt_label:
        plot_skeleton_kpts(im, kpts, steps, orig_shape=orig_shape)
    else:
        tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(im, c1, c2, (255,0,0), thickness=tl*1//3, lineType=cv2.LINE_AA)

class TrackerNode():
    def __init__(self, img_topic: str="/usb_cam/image_raw", pose_topic: str="/yolov7/kpt",bbox_topic: str="/yolov7/object/bbox", 
                 queue_size: int = 1, visualize: bool = False, out_topic: str = 'tracks', class_labels: Union[List, None] = None):
    
        self.person_track = {} # key:id value: np.array[kpts_t1, kpt_t2, ...]
        self.img = None
        self.track = []
        self.ped_dets = []
        self.obj_dets = []
        self.sort_tracker_people = Sort(max_age=5,min_hits=2,iou_threshold=0.2) 
        self.class_labels = class_labels

        self.img_subscriber = rospy.Subscriber(img_topic, Image, self.raw_img_callback)
        self.pose_subscriber = rospy.Subscriber(pose_topic, String, self.ped_det_callback) # process_people_msg
        self.bbox_subscriber = rospy.Subscriber(bbox_topic, String, self.obj_det_callback) # process_bbox_msg
        self.track_publisher = rospy.Publisher('/limbs_track', String, queue_size=queue_size)

    def raw_img_callback(self, image):
        self.img = CvBridge().imgmsg_to_cv2(image, desired_encoding='passthrough')
        try:
            self.frame_id = int(image.header.frame_id)
        except:
            self.frame_id = -1

        img = self.img.copy()
        if len(self.ped_dets): 
            # draw key points
            for i in range(len(self.ped_dets)):
                plot_one_box_kpt(self.ped_dets[i], img, kpts=self.ped_dets[i][6:])
            self.ped_dets = []

        if len(self.obj_dets):
            # draw bbox
            car_dets = np.array(self.obj_dets).reshape((-1, 6))
            bboxes = [[int(x1), int(y1), int(x2), int(y2)]
                      for x1, y1, x2, y2 in car_dets[:, :4].tolist()]
            classes = [int(c) for c in car_dets[:, 5].tolist()]
            vis_img = draw_detections(img, bboxes, classes, self.class_labels)
            self.obj_dets = []

            # tracking 
            tracks = self.sort_tracker_people.getTrackers()
            if len(tracks)>0:
                track_color = self.sort_tracker_people.color_list
                print(f'  people tracked: {len(tracks)}')            
                for t, track in enumerate(tracks):
                    #track_color = track_color[t]
                    [cv2.line(img, (int(track.centroidarr[i][0]),
                                int(track.centroidarr[i][1])), 
                                (int(track.centroidarr[i+1][0]),
                                int(track.centroidarr[i+1][1])),
                                track_color[t], thickness=3) 
                                for i,_ in  enumerate(track.centroidarr) 
                                    if i < len(track.centroidarr)-1 ] 

        cv2.imshow("Tracker Output", img)
        cv2.waitKey(1)

    def ped_det_callback(self, objects):
        '''
        Use kpt to track and publish the result, 
        Action estimation Node subscribes to the results and estimates action 
        '''
        global limbs
        self.ped_dets = json.loads(objects.data)
        input_person = json.loads(objects.data)
            
        output = []
        # for i in range(len(input_person)):
        #     #plot_one_box_kpt(input_person[i], people_img, kpts=input_person[i][6:])
        #     dets_to_sort = np.empty((0,57))
        #     det = torch.tensor(input_person[i])
        #     #track person
        #     #person_data = np.append(np.array(det.cpu().detach().numpy()[0:5]),9999+(i+1)*1e4) 
        #     person_data = np.array(det.cpu().detach().numpy())
        #     dets_to_sort = np.vstack((dets_to_sort, person_data))

        #     tracked_dets = self.sort_tracker_people.update(dets_to_sort)
        #     tracks = self.sort_tracker_people.getTrackers()
        #     if len(tracked_dets)<=0:
        #         continue
            
        #     track_color = self.sort_tracker_people.color_list
        #     print(f'  people tracked: {len(tracks)}')            
            # for t, track in enumerate(tracks):
            #     #track_color = track_color[t]
            #     [cv2.line(people_img, (int(track.centroidarr[i][0]),
            #                 int(track.centroidarr[i][1])), 
            #                 (int(track.centroidarr[i+1][0]),
            #                 int(track.centroidarr[i+1][1])),
            #                 track_color[t], thickness=3) 
            #                 for i,_ in  enumerate(track.centroidarr) 
            #                     if i < len(track.centroidarr)-1 ] 
                
                # track_data = [self.frame_id, track.id, 'Pedestrian', -1, -1, 0, 
                #               track.object_data[0],track.object_data[1],track.object_data[2],track.object_data[3],
                #               0,0,0, 
                #               0,0, 0, 
                #               0, track.object_data[4]]
                # track_data = np.reshape(track_data, (1,18))
                # print(f'{track_data}')
                # #track_data.save('./test.txt')
                # line_to_add = " ".join(str(num) for num in track_data)
                # with open('/home/kong/my_ws/TrackEval/data/trackers/kitti/kitti_2d_box_train/myself/data/0019.txt', "a") as file:
                #     np.savetxt(file, np.array(track_data, dtype=str), delimiter=' ', fmt='%s')
                
                
            # act_label = action_estimate(self.person_track)
            # for track in tracks:
            #     person = [track.id, track.object_data, track.centroidarr]
            #     #self.track_kpt(person)      # TODO      
            #     if track.id in act_label:
            #         cv2.putText(people_img, act_label[track.id], 
            #                     [int(track.centroidarr[-1][0]),int(track.centroidarr[-1][1])], 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
            #     else:                    
            #         cv2.putText(people_img, "New One", 
            #                     [int(track.centroidarr[-1][0]),int(track.centroidarr[-1][1])], 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # track_msg = json.dumps(output)
        # self.track_publisher.publish(track_msg)
        # cv2.imshow('People track', people_img)
        # cv2.waitKey(1)

    def obj_det_callback(self, objects):
        self.obj_dets = json.loads(objects.data)
        rospy.loginfo(f'bbox: {len(self.obj_dets)} detected!')
        for i in range(len(self.obj_dets)):
            dets_to_sort = np.empty((0,6))
            det = torch.tensor(self.obj_dets[i])
            obj_data = np.array(det.cpu().detach().numpy())
            dets_to_sort = np.vstack((dets_to_sort, obj_data))

            tracked_dets = self.sort_tracker_people.update(dets_to_sort)
            if len(tracked_dets)<=0:
                continue        

if __name__ == '__main__':
    rospy.init_node('mo_tracker')

    ns = rospy.get_name() + "/"

    img_topic = rospy.get_param(ns + "img_topic")
    pose_topic = rospy.get_param(ns + "pose_topic")
    bbox_topic = rospy.get_param(ns + "bbox_topic")
    out_topic = rospy.get_param(ns + "out_topic")
    queue_size = rospy.get_param(ns + "queue_size")
    visualize = rospy.get_param(ns + "visualize")
    classes_path = rospy.get_param(ns + "classes_path")

    if classes_path: 
        if not os.path.isfile(classes_path):
            raise FileExistsError(f"Classes file not found ({classes_path}).")
        classes = parse_classes_file(classes_path)

    publisher = TrackerNode(
        img_topic = img_topic, 
        pose_topic = pose_topic, 
        bbox_topic = bbox_topic,
        queue_size = queue_size, 
        visualize = visualize,
        out_topic = out_topic,
        class_labels=classes)

    rospy.spin()
