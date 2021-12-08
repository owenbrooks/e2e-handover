FROM ghcr.io/owenbrooks/cuda-ros:master

# Dev tools
RUN apt-get -y update && apt-get install -y \
    x11-apps \
    python3-pip \
    python3-vcstool \
    build-essential \
    libnvidia-gl-470 \
    vim \
    nano \
    git \
    wget \
    tmux

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
RUN echo "source ${CATKIN_WS}/devel/setup.bash" >> /etc/bash.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]