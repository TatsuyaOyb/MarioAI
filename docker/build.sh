#!/bin/bash -e

docker build \
    -m 8g --shm-size 8192m \
    -t mario:cu117-cudnn8-devel-ubuntu22 \
    -f ./docker/Dockerfile \
    .