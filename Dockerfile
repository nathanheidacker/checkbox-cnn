FROM python:3.9

ARG cuda

RUN mkdir /usr/cnn
WORKDIR /usr/cnn
COPY . .

# These are dependencies for opencv, not present by default
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

# Installing the right version of pytroch depending on GPU
RUN if ["$cuda" = "y"]; then pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113; else pip3 install torch torchvision torchaudio; fi