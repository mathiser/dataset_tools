docker run -e THREADS=8 -v $(realpath $1):/input -v $(realpath $2):/output mathiser/image_preprocessing:dcmrtstruct2nii

