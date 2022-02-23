docker run -v $(realpath $1):/input -v $(realpath $2):/output -e BOUNDS_TEXT=$3 mathiser/dataset_tools:merge_OAR_bounds

