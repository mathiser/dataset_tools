FROM python:3.9-slim-buster
RUN apt update && pip install \
	dcmrtstruct2nii \
    pydicom \
    && rm -rf /var/lib/apt/lists/*

ENV INPUT="/input"
ENV OUTPUT="/output"
ENV RTSTRUCT_PATHS=$OUTPUT/rtstruct_paths.json

RUN mkdir $INPUT $OUTPUT
COPY main.py /main.py
ENTRYPOINT ["bash", "-c", "python3 -u main.py $INPUT $OUTPUT $RTSTRUCT_PATHS"]

