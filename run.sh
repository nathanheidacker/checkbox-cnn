printf "Starting the docker container...\n"
printf "The following commands are available to you:\n"
printf "    For training:      python src/train.py <\"v1\" or \"v2\">\n"
printf "    For inference:     python src/evaluate.py <your_image_path or 'all'>\n"
printf "    For data stats:    python src/data.py <n training samples>\n"
if [["$OSTYPE" == "msys"]]: then
    winpty docker run -it --rm cnn bash
else
    docker run -it --rm cnn bash