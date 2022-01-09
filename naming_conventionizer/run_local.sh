docker run -it --rm \
  -v $(realpath $1):/Task:rw \
  --cpus=8 \
  --gpus=all \
  --env-file ./env.list \
  mathiser/nnunet:train bash
