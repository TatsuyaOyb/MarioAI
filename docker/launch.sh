#!/bin/bash -e

# TODO: User should be refactored instead of hard coded TatsuyaOyb

USER_NAME=TatsuyaOyb

docker run --gpus all -ti --init --rm \
        --hostname $(hostname) --name mario-ppo \
        -p 8888:8888 --shm-size=16384m \
        -v $(pwd):/home/$USER_NAME/workspace \
        -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
        mario:cu117-cudnn8-devel-ubuntu22