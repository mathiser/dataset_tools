docker run -v $(realpath $1):/input -v $(realpath $2):/output -v $(realpath $3):/labels  -e mathiser/dataset_tools:merge_OAR_bounds

