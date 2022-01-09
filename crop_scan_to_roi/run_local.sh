docker run -it --rm \
  -v $(realpath $1):/images:ro \
  -v $(realpath $2):/labels:ro \
  -v $(realpath $3):/output \
  -e THREADS=16 \
  -e X_PADDING=15 \
  -e Y_PADDING=15 \
  -e Z_PADDING=5 \
  mathiser/dataset_tools:crop_scan_to_roi
