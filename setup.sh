#!/bin/bash
read -p "Install CUDA version of PyTorch? [y/n] : " cuda

if [[ "$(docker images -q cnn 2> /dev/null)" == "" ]]; then
  docker rmi cnn
fi

printf "Building the docker image...\n"
docker build -t cnn . --build-arg cuda=$cuda
bash run.sh