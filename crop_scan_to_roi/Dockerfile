FROM ubuntu:20.04
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    simpleitk \
    numpy

ENV SCAN_DIR="/images"
ENV MASK_DIR="/labels"
ENV OUT_DIR="/output"
ENV THREADS=16

COPY . /

CMD ["bash", "-c", "python3 -u main.py" ]
