#!/usr/bin/env python3
import os
import numpy as np
class kitti_obj(object):
    def __init__(self, frame_id=-1, track_id=-1, type='DontCare', truncated=-1, occluded=-1, alpha=0, bbox=[0,0,0,0], 
                 dimensions=[0,0,0], location=[0,0,0], rotation_y=0, score=0):
        self.frame_id = frame_id
        self.track_id = track_id
        self.type = type
        self.truncated = truncated
        self.occluded = occluded
        self.alpha = alpha
        self.bbox = bbox
        self.dimensions = dimensions
        self.location = location
        self.rotation_y = rotation_y
        self.score = score
    
    def save(self, path:str):
        # Assuming you have a matrix called 'matrix'
        data = [self.frame_id, self.track_id, self.type, 
                        self.truncated, self.occluded, self.alpha, 
                        self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3],
                        self.dimensions[0], self.dimensions[1], self.dimensions[2],
                        self.location[0], self.location[1], self.location[2], 
                        self.rotation_y, self.score]#np.array(, dtype=str)
        data = np.reshape(data, (1,18))

        # if not os.path.exists(path):
        #     os.makedirs(os.path.dirname(path))
        # Save the matrix to a file with spaces as the delimiter
        line_to_add = " ".join(str(num) for num in data)
        
        with open(path, "a") as file:
           np.savetxt(file, np.array(data, dtype=str), delimiter=' ', fmt='%s')
# a = kitti_obj()
# a.save('./test.txt')
#save('./test.txt', [['CAr',2,3],[4,5,6],[7,8,9]])