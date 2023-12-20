FROM dustynv/ros:foxy-pytorch-l4t-r34.1.1 as base
SHELL ["/bin/bash", "-c"]
WORKDIR /ws
ENV DEBIAN_FRONTEND=noninteractive

# ============================== install additional dependencies  ========================================= #
FROM base as common
#RUN pip3 install tqdm matplotlib seaborn scipy
RUN pip install --trusted-host=pypi.org --trusted-host=files.pythonhosted.org --user tqdm matplotlib seaborn scipy
COPY ./ros_entrypoint.sh /ros_entrypoint.sh
RUN mkdir -p -m 0700 /root/.ssh  && ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT [ "/ros_entrypoint.sh" ]

# ============================== Download pre-train models and code  ========================================= # 
FROM common as dev
# if you have models at local, you can ignore this
RUN mkdir -p nn_models/yolov7 \
    && wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt \
    && mv yolov7.pt nn_models/yolov7/ \
    && wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt \
    && mv yolov7-w6-pose.pt nn_models/yolov7/
# Pull code from GitHub and show folder size
RUN --mount=type=ssh git clone git@github.com:xrkong/yolov7-ros-pose-estimation.git --branch nuway2  --single-branch --depth=1

# ============================== Copy project from local ========================================= #
FROM common as staging 
ENV PYTHONPATH=$PYTHONPATH:${WORKDIR}/yolov7-ros-pose-estimation/src/yolov7_ros/yolov7_ros/:${WORKDIR}/yolov7-ros-pose-estimation/src/mo_tracker/
RUN --mount=type=ssh git clone git@github.com:xrkong/yolov7-ros-pose-estimation.git --branch nuway2  --single-branch --depth=1

WORKDIR /ws/yolov7-ros-pose-estimation/
RUN source /opt/ros/foxy/install/setup.bash && colcon build && source ./install/setup.bash
