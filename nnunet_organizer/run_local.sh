docker run -it --rm \
  -v $(realpath $1):/input:ro \
  -v $(realpath $2):/output:rw \
  -v $(realpath $3):/meta:ro \
  --env-file=$(realpath $4) \
  --cpus=16 \
  mathiser/dataset_tools:nnunet_organizer
