#!/bin/sh
# Script for building and working in a docker container

case "$1" in
"build")
    docker build . -t e2e
    ;;
"init") # with GPU support
    if [ ! -z "$(docker ps -qa -f name=e2e)" ]; then
        echo "The container is already initialized."
        echo "To use it, run "
        echo "    run_docker.sh run $2"
        echo "To remove it, run "
        echo "    docker rm -f e2e"
    else
        docker run -e DISPLAY -e TERM \
            --privileged \
            -v "/dev:/dev:rw" \
            -v "${HOME}:${HOME}:rw" \
            -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            --runtime=nvidia \
            --net=host \
            --name e2e \
            --entrypoint /ros_entrypoint.sh \
            -d e2e /usr/bin/tail -f /dev/null
    fi
    ;;
"init_no_gpu")  # without GPU support
    docker run -e DISPLAY -e TERM \
        --privileged \
        -v "/dev:/dev:rw" \
        -v "${HOME}:${HOME}:rw" \
        -v "/tmp/.X11-unix:rw" \
        -v "/dev:/dev:rw" \
        -v "${HOME}:${HOME}:rw" \
        -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --net=host \
        --name e2e \
        --entrypoint /ros_entrypoint.sh \
        -d e2e /usr/bin/tail -f /dev/null
    ;;
"run")
    # Allow docker windows to show on our current X Server
    xhost + >> /dev/null

    # Start the container in case it's stopped
    docker start e2e

    # Attach a terminal into the container
    exec docker exec -it e2e bash    
    ;;
*)
    echo "Usage: run_docker [command]
Available commands:
    build
        Build a new image from the Dockerfile in the current directory
    init
        Create a new container from the latest image
    run
        Attach a new terminal to the container (and start it if necessary)
"
    ;;
esac