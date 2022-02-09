docker run -e THREADS=64 -v `realpath $1`:/input:ro -v `realpath $2`:/output mathiser/dataset_tools:dcm2niix

