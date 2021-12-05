FROM osrf/ros:melodic-desktop-full

# Dev tools
RUN apt-get -y update && apt-get install -y \
    # x11-apps \
    # python3-pip \
    # build-essential \
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

CMD /bin/bash
ENTRYPOINT /ros_entrypoint.sh