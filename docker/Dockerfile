FROM osrf/ros:noetic-desktop-full

# Dev tools
RUN apt-get -y update && apt-get install -y \
    x11-apps \
    python3-pip \
    python3-vcstool \
    build-essential \
    software-properties-common \
    vim \
    nano \
    git \
    wget \
    tmux

# Nvidia GPU Support (if run with nvidia-container-runtime)
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Install CUDA
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update && apt-get -y install cuda


# Make the prompt a little nicer
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\u@:\w\$ '" >> /etc/bash.bashrc  

# Project specific dependencies
ENV CATKIN_WS=/catkin_ws
ENV SOURCE_DIR=${CATKIN_WS}/src/e2e-handover
COPY ./e2e.rosinstall ${SOURCE_DIR}/e2e.rosinstall
WORKDIR ${CATKIN_WS}/src
RUN vcs import < ${SOURCE_DIR}/e2e.rosinstall

COPY ./package.xml ${SOURCE_DIR}/package.xml
WORKDIR ${CATKIN_WS}
RUN rosdep update && rosdep install --from-paths src --ignore-src -r -y

# Build the project
COPY . ${SOURCE_DIR}
RUN /bin/bash -c '. /opt/ros/$ROS_DISTRO/setup.bash; cd /${CATKIN_WS}; catkin_make'
WORKDIR ${SOURCE_DIR}

# Add source commands to bashrc
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /etc/bash.bashrc
RUN echo "source ${CATKIN_WS}/devel/setup.bash" >> /etc/bash.bashrc

ENTRYPOINT /ros_entrypoint.sh
CMD /bin/bash