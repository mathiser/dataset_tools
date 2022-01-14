docker run -e THREADS=8 -v $(realpath $1):/input -v $(realpath $2):/output mathiser/dataset_tools:dcmrtstruct2nii

