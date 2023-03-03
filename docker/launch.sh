#!/bin/bash -e

# TODO: User should be refactored instead of hard coded TatsuyaOyb

USER_NAME=TatsuyaOyb

docker run --gpus all -ti --init --rm \
        --hostname $(hostname) --name mario-ppo \
        -p 8888:8888 --shm-size=16384m \
        -v $(pwd):/home/$USER_NAME/workspace \
        -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
        -e PULSE_COOKIE=/tmp/pulse/cookie \
        -e PULSE_SERVER=unix:/tmp/pulse/native \
        -v /run/user/1000/pulse/native:/tmp/pulse/native \
        -v ~/.config/pulse/cookie:/tmp/pulse/cookie:ro \
        -e XMODIFIERS \
        -e GTK_IM_MODULE \
        -e QT_IM_MODULE \
        -e DefalutIMModule=fcitx \
        -e DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
        -v /run/user/1000/bus:/run/user/1000/bus \
        --privileged \
        -v ~/.cache/mesa_shader_cache:/home/$USER_NAME/.cache/mesa_shader_cache \
        mario:cu117-cudnn8-devel-ubuntu22 