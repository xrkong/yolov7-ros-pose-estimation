# yolov7-ros-pose-estimation
A ROS wrapper around detect.py for mobile robots and autonomous vehicles. 

src/video_stream_opencv from https://github.com/ros-drivers/video_stream_opencv    
src/yolov7-ros from https://github.com/lukazso/yolov7-ros    
src/pose-tracker/src/sort.py from https://github.com/abewley/sort  

remember to add the path to ```$PYTHONPATH``` or python cannot find the module like    
```No module named 'xxxx'```

## Prepare

### Pull image from DockerHub
``` docker pull dustynv/ros:foxy-pytorch-l4t-r34.1.1 ```
This image is for amd arch only which is for orin

### Build a image
``` DOCKER_BUILDKIT=1 docker build -t foxy-torch:test_entry . --ssh default ```

```DOCKER_BUILDKIT=1``` allows you pip install

```foxy-torch:test_entry``` is the target image tag. if you change the tag, you should also change the ```.env```

```---ssh default``` is for pull repo from GitHub repo.

### Start te container
```$ docker compose up```


### topics
```/CamerFront``` is the input camera image topic.
```/yolov7/bbox``` for object detection,
```/yolov7/kpt``` for pose estimation.

Parameters can be found in ```src/yolov7_ros/launch/detector_launch.py```

## To-do list
- [x] build a docker
- [x] re-factory 
- [ ] TBD
