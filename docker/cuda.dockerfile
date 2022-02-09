FROM ghcr.io/owenbrooks/cuda-ros:master

# Dev tools
RUN apt-get -y update && apt-get install -y \
    x11-apps \
    python3-pip \
    python3-vcstool \
    build-essential \
    libnvidia-gl-470 \
    libcanberra-gtk3-module \
    vim \
    nano \
    git \
    wget \
    tmux \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Environment variables for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

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
RUN apt-get -y update && rosdep update && rosdep install --from-paths src --ignore-src -r -y

RUN pip install wandb pyquaternion

# Build the project
COPY . ${SOURCE_DIR}
RUN /bin/bash -c '. /opt/ros/$ROS_DISTRO/setup.bash; cd /${CATKIN_WS}; catkin_make'
WORKDIR ${SOURCE_DIR}

# Add source commands to bashrc
RUN echo "source ${CATKIN_WS}/devel/setup.bash" >> /etc/bash.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]