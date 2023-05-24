#!/usr/bin/env python3
import rospy
import numpy as np
import json
import cv2
import torch
import statistics
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from sort import *

kpt_dict = {"left shoulder":5, "right shoulder":6,"left elbow":7, 
            "right elbow":8, "left wrist":9, "right wrist":10}
kpt_name = ["nose", "left eye", "right eye", "left ear", "right ear", 
            "left shoulder", "right shoulder"]       
limbs = [[5,7], [6,8], [7,9], [8,10], [11,13], [12,14], [13,15], [14,16]]     
limb_name = ["left upper arm", "right upper arm", "left lower arm", "right lower arm",
                "left upper leg", "right upper leg", "left lower leg", "right lower leg"]   

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
    global kpt_dict
    act_label = {}
    # trackers is a dict which key is id and value is track 
    for key, value in trackers.items():
        person_id = key
        kpt_track = value
        #print(f'{value}')
        mean_kpt = np.mean(kpt_track[-5:], axis=0) # mean of cols
        # print(f'{mean_kpt}')

        if mean_kpt[1 + kpt_dict["left wrist"]*3+1] > mean_kpt[1 + kpt_dict["left elbow"]*3+1]:
            act_label[person_id] = 'Left hand up'
        
        elif mean_kpt[1 + kpt_dict["right wrist"]*3+1] > mean_kpt[1 + kpt_dict["right elbow"]*3+1]:
            act_label[person_id] = 'Right hand up'

        else:
            act_label[person_id] = 'No action'
        
    return act_label

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


"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, confidences = None, names=None, colors = None):
    if len(bbox) == 0:  # If no bounding boxes detected
        cv2.putText(img, "No_Person is detected in frame", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0
            # conf = confidences[i] if confidences is not None else 0

            color = colors[cat]

    return img

# TODO: DeepSort Track each limbs
'''
Duty:
- Get img and kpt from topic
- Track obj and add tag to kpt
- Publish results and imgs
'''
class TrackerNode():
    def __init__(self, img_topic: str, pose_topic: str, queue_size: int = 1, visualize: bool = True, 
                 out_topic: str = 'action_label'):
        self.img_subscriber = rospy.Subscriber(img_topic, Image, self.process_img_msg)
        self.pose_subscriber = rospy.Subscriber(pose_topic, String, self.process_people_msg)
        self.track_publisher = rospy.Publisher('/limbs_track', String, queue_size=queue_size)

        self.person_track = {} # key:id value: np.array[kpts_t1, kpt_t2, ...]
        self.img = None

    def process_img_msg(self, image):
        '''
        No process image, only display number and trajectories
        '''
        self.img = CvBridge().imgmsg_to_cv2(image, desired_encoding='passthrough')
        pass

    def process_kpt_msg(self, objects):
        '''
        Use kpt to track and publish the result, 
        Action estimation Node subscribes to the results and estimates action 
        '''
        global limbs
        # json.loads(objects.data)
        person_id = objects[0]
        input_kpts = objects[1]
        output = []
        #for i in range(len(input_person)):

        dets_to_sort = np.empty((0,6))
        det = torch.tensor(input_kpts)
        
        # track its 4 limbs
        for index, element in enumerate(limbs):
            kpt1 = det.cpu().detach().numpy()[1+3*element[0]:1+3*element[0]+3]
            kpt2 = det.cpu().detach().numpy()[1+3*element[1]:1+3*element[1]+3]

            if kpt1[2] < 0.5 or kpt2[2] < 0.5:
                continue

            [x1,x2,y1,y2,conf] = [min(kpt1[0], kpt2[0]), min(kpt1[1], kpt2[1]), 
                             min(kpt1[1], kpt2[1]), max(kpt1[1], kpt2[1]), (kpt1[2]+kpt2[2])/2] 
            detclass = person_id * 1e4 + element[0] * 1e2 + element[1]
            dets_to_sort = np.vstack((dets_to_sort, 
                    np.array([x1,y1,x2,y2,conf,detclass])))
        
        # track its kpts 
        for index in range(17):
            kpt = det.cpu().detach().numpy()[1+3*index:6+3*index+3] # K{index}_x, K{index}_y, K{index}_conf
            [x1, x2, y1, y2, conf, detclass] = [kpt[0]-50, kpt[0]+50, kpt[1]-50, 
                                                kpt[1]+50, kpt[2], person_id*1e4 + index]
            if conf < 0.5:
                continue
            # i: person index, element: limbs index
            dets_to_sort = np.vstack((dets_to_sort, 
                    np.array([x1,y1,x2,y2,conf,detclass])))
            #tracked_dets = sort_tracker.update(dets_to_sort)
        
        tracked_dets = sort_tracker_kpt.update(dets_to_sort)
        tracks = sort_tracker_kpt.getTrackers()

        if len(tracked_dets)<=0:
            return #continue 

        #print(f'  body parts tracked: {tracked_dets[:, 4]}')
        # plot
        kpt_track_img = self.img.copy()
        for t, track in enumerate(tracks):
            track_color = [0, 0, 0]
            [cv2.line(kpt_track_img, (int(track.centroidarr[i][0]),
                        int(track.centroidarr[i][1])), 
                        (int(track.centroidarr[i+1][0]),
                        int(track.centroidarr[i+1][1])),
                        track_color, thickness=3) 
                        for i,_ in  enumerate(track.centroidarr) 
                            if i < len(track.centroidarr)-1 ] 

        # save the track of limbs and kpts
        object.append(tracks)

        cv2.imshow('Limbs track', kpt_track_img)
        cv2.waitKey(1)

    def track_kpt(self, track):
        person_id = track[0]
        kpts_pos = track[1]
        person_track = track[2]
        #kpts_pos[][]
        # update track dict of the person, if not exist, create a new one. 
        #print(f'person_id: {person_id}, kpts_pos: {self.person_track[person_id]}')
        kpt_to_sort = np.empty((0,52))
        if person_id not in self.person_track:
            self.person_track[person_id] = [kpts_pos]
        else:
            self.person_track[person_id] = np.vstack((self.person_track[person_id], kpts_pos))

        # print(f'{len(kpts_pos)}')
        print(f'{len(kpts_pos)} people tracked: {self.person_track[person_id]}')

    def process_people_msg(self, objects):
        '''
        Use kpt to track and publish the result, 
        Action estimation Node subscribes to the results and estimates action 
        '''
        global limbs
        input_person = json.loads(objects.data)
        if self.img is None:
            return
        people_img = self.img.copy()

        output = []
        for i in range(len(input_person)):
            dets_to_sort = np.empty((0,57))
            det = torch.tensor(input_person[i])
            #track person
            #person_data = np.append(np.array(det.cpu().detach().numpy()[0:5]),9999+(i+1)*1e4) 
            person_data = np.array(det.cpu().detach().numpy())
            dets_to_sort = np.vstack((dets_to_sort, person_data))
            #dets_to_sort = np.append(dets_to_sort, person_data)
            #print(f'dets_to_sort: {dets_to_sort}')
            tracked_dets = sort_tracker_people.update(dets_to_sort)
            tracks = sort_tracker_people.getTrackers()
            if len(tracked_dets)<=0:
                continue
            #if opt.show_track:
            #loop over tracks
            
            for t, track in enumerate(tracks):
                track_color = [255, 0, 0] 
                [cv2.line(people_img, (int(track.centroidarr[i][0]),
                            int(track.centroidarr[i][1])), 
                            (int(track.centroidarr[i+1][0]),
                            int(track.centroidarr[i+1][1])),
                            track_color, thickness=3) 
                            for i,_ in  enumerate(track.centroidarr) 
                                if i < len(track.centroidarr)-1 ] 
                
            act_label = action_estimate(self.person_track)
            for track in tracks:
                person = [track.id, track.object_data, track.centroidarr]
                self.track_kpt(person)           
                if track.id in act_label:
                    cv2.putText(people_img, act_label[track.id], 
                                [int(track.centroidarr[-1][0]),int(track.centroidarr[-1][1])], 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
                else:                    
                    cv2.putText(people_img, "New One", 
                                [int(track.centroidarr[-1][0]),int(track.centroidarr[-1][1])], 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # track_msg = json.dumps(output)
        # self.track_publisher.publish(track_msg)
        cv2.imshow('People track', people_img)
        cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('pose_tracker')

    ns = rospy.get_name() + "/"

    # img_topic = rospy.get_param(ns + "img_topic")
    # pose_topic = rospy.get_param(ns + "pose_topic")
    # out_topic = rospy.get_param(ns + "out_topic")
    # queue_size = rospy.get_param(ns + "queue_size")
    # visualize = rospy.get_param(ns + "visualize")
    img_topic="/yolov7/image"
    pose_topic="/yolov7/kpt"
    out_topic="action_est"
    queue_size="10"
    visualize="true"

    publisher = TrackerNode(
        img_topic = img_topic, 
        pose_topic = pose_topic, 
        queue_size = queue_size, 
        visualize = visualize,
        out_topic = out_topic)
    
    sort_tracker_kpt = Sort(max_age=5,
                    min_hits=2,
                    iou_threshold=0.2) 
    
    sort_tracker_people = Sort(max_age=5,
                min_hits=2,
                iou_threshold=0.2) 

    rospy.spin()
