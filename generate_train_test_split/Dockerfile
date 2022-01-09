FROM ubuntu:20.04 
RUN apt-get update && apt-get install -y \
	python3 \
	python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
	numpy \
	pandas

COPY . /
ENV INPUT="/input"
ENV OUTPUT="/output"
ENV TRAIN_TEST_RATIO="0.1"

ENTRYPOINT ["bash", "-c", "python3 -u /main.py"]

