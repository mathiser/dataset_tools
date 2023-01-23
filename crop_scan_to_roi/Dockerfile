FROM ubuntu:20.04
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

ADD requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

ENV SCAN_DIR="/images"
ENV MASK_DIR="/labels"
ENV OUT_DIR="/output"
ENV THREADS=8

COPY . /

CMD ["bash", "-c", "python3 -u main.py" ]
