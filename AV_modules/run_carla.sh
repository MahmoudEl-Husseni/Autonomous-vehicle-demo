#!/bin/bash

# Set the environment variables for the Docker command.
DOCKER_IMAGE=carlasim/carla
PORT_RANGE="2000-2002"
CPU_SET="0-5"
GPU_LIST="all"
VOLUME_MOUNT="/dev/snd:/dev/snd"
CMD="/bin/bash ./CarlaUE4.sh -opengl $1 -RenderOffScreen"

# Create a Docker command string.
docker_cmd="docker run -p ${PORT_RANGE}:${PORT_RANGE} --cpuset-cpus=${CPU_SET} --runtime=nvidia --rm --privileged --gpus=${GPU_LIST} -v ${VOLUME_MOUNT} ${DOCKER_IMAGE} ${CMD}"

# Run the Docker command.
eval "${docker_cmd}"

