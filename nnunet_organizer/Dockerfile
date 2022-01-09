# Requires the following envs to run:
# INPUT, OUTPUT, MODALITY_FILE, SEGMENTATION_FILE.
# TRAIN_PIDS_JSON and TEST_PIDS_JSON are optional, but used if you want to determine which pids end in train and in test.
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN pip3 install \
    numpy \
    simpleitk \
    batchgenerators \
    pandas

ENV INPUT="/input"
ENV OUTPUT="/output"

RUN mkdir $INPUT $OUTPUT

COPY . /

ENTRYPOINT ["bash", "-c", "python3 -u main.py" ]