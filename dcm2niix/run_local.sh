docker run --mount type=bind,source=`realpath $1`,target=/input --mount type=bind,source=`realpath $2`,target=/output mathiser/image_preprocessing:dcm2niix

