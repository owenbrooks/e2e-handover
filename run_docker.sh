#!/bin/bash
# Script for building and working in a docker container

attach_to_container() 
{
    # Allow docker windows to show on our current X Server
    xhost + >> /dev/null

    # Start the container in case it's stopped
    docker start e2e

    # Attach a terminal into the container
    exec docker exec -it e2e bash
}

run_with_gpu()
{
    docker run -e DISPLAY -e TERM \
        --privileged \
        -v "/dev:/dev:rw" \
        -v "$(pwd):/catkin_ws/src/e2e-handover:rw" \
        -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --runtime=nvidia \
        --net=host \
        --name e2e \
        --entrypoint /ros_entrypoint.sh \
        -d e2e /usr/bin/tail -f /dev/null
}
run_without_gpu()
{
    docker run -e DISPLAY -e TERM \
        --privileged \
        -v "/dev:/dev:rw" \
        -v "$(pwd):/catkin_ws/src/e2e-handover:rw" \
        -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --net=host \
        --name e2e \
        --entrypoint /ros_entrypoint.sh \
        -d e2e /usr/bin/tail -f /dev/null
}

case "$1" in
"build")
    docker build . -t e2e
    ;;
"--help")
    echo "Usage: run_docker.sh [command]
Available commands:
    run_docker.sh
        Attach a new terminal to the container (building, creating and starting it if necessary)
    run_docker.sh build
        Build a new image from the Dockerfile in the current directory
    run_docker.sh --help
        Show this help message    
    "
    ;;
*) # Attach a new terminal to the container (building, creating and starting it if necessary)
    if [ -z "$(docker images -f reference=e2e -q)" ]; then # if the image has not yet been built, build it
        docker build . -t e2e
    fi
    if [ -z "$(docker ps -qa -f name=e2e)" ]; then # if container has not yet been created, create it
        if [[ $(docker info | grep Runtimes) =~ nvidia ]] ; then # computer has nvidia-container-runtime, use it for GPU support
            run_with_gpu
        else # no nvidia-container-runtime
            run_without_gpu
        fi        
    fi
    attach_to_container
    ;;
esac