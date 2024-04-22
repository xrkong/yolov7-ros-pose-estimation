# yolov7-ros-pose-estimation
A ROS wrapper around detect.py for mobile robots and autonomous vehicles. 

src/video_stream_opencv from https://github.com/ros-drivers/video_stream_opencv    
src/yolov7-ros from https://github.com/lukazso/yolov7-ros    
src/pose-tracker/src/sort.py from https://github.com/abewley/sort  

I made some changes for pedestrian track. 

remember to add the path to ```$PYTHONPATH``` or python cannot find the module like    
```No module named 'xxxx'```

## todo 
### yolo node: 
- [x] images &rarr; label, bounding box
- [ ] serialize label, ground truth (not kpt), process time.

### llm node:
- [ ] write request node to those LLM/MLLM

### images publisher: None

### tracker node: None