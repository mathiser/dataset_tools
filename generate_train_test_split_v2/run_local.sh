docker run --rm \
  -v $(realpath $1):/input \
  -v $(realpath $2):/output \
  mathiser/dataset_tools:generate_train_test_split
