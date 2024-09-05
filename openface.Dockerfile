FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    software-properties-common \
    build-essential \
    cmake

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    python3-opencv \
    pkg-config \
    wget \
    zip \
    libatlas-base-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libswscale-dev \
    libssl-dev \
    libffi-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3.8 -m pip install --upgrade pip
#TODO:
#ADD . /root/face_recognizer
#WORKDIR /root/face_recognizer
#CD /root
#RUN git clone openface
#RUN ./models/get-models.sh && \
#    python3.8 -m pip install -r requirements.txt && \
#    python3.8 -m pip install .
##    python3 -m pip install --user --ignore-installed -r demos/web/requirements.txt && \
##    python3 -m pip install -r training/requirements.txt

#CD /root/face_recognizer/server
#RUN python3.8 -m pip install -r requirements.txt

EXPOSE 8000 9000
