FROM osrf/ros:melodic-desktop-full

# Dev tools
RUN apt-get -y update && apt-get install -y \
    x11-apps \
    python-pip \
    build-essential \
    vim \
    nano \
    git \
    tmux

# Nvidia GPU Support (if run with nvidia-container-runtime)
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN echo "source /opt/ros/melodic/setup.bash" >> /etc/bash.bashrc
# Make the prompt a little nicer
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\u@:\w\$ '" >> /etc/bash.bashrc  

CMD /bin/bash
ENTRYPOINT /ros_entrypoint.sh

# Project specific dependencies
COPY . /catkin_ws/src/e2e-handover
WORKDIR /catkin_ws
RUN rosdep update && rosdep install --from-paths src --ignore-src -r -y
RUN sudo apt-get update && sudo apt-get install -y python3-vcstool
WORKDIR /catkin_ws/src
RUN vcs import < /catkin_ws/src/e2e-handover/e2e.rosinstall

WORKDIR /catkin_ws
RUN catkin_make
WORKDIR /catkin_ws/src/e2e-handover

RUN echo "source /catkin_ws/devel/setup.bash" >> /etc/bash.bashrc