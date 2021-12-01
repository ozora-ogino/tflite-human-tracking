FROM python:3.9

COPY ./src /opt/src
COPY ./requirements.txt /opt/
WORKDIR /opt/

RUN apt-get update && apt-get install -y \
	ffmpeg \
	libsm6 \
	libxext6 \
	&& rm -rf /var/lib/apt/lists/* \
	&& python -m pip install --upgrade pip \
	&& pip3 install  --no-cache-dir -r ./requirements.txt \
	&& pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime