FROM python:3.9

RUN mkdir /usr/cnn
WORKDIR /usr/cnn

# These are dependencies for opencv, not present by default
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .