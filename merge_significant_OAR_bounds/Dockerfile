FROM ubuntu:20.04
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install \
    simpleitk \
    numpy

ENV INPUT="/input"
ENV OUTPUT="/output"
ENV THREADS=16
RUN mkdir $INPUT $OUTPUT

COPY . /

ENTRYPOINT ["bash", "-c", "python3 -u main.py" ]