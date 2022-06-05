#!/bin/bash

if [[ "$(docker images -q cnn 2> /dev/null)" == "" ]]; then
  docker rmi cnn
fi
printf "Building the docker image...\n"
docker build -t cnn .
bash run.sh