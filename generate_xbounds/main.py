import os
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool

## Author: Mathis Rasmussen
## 21-11-26
## Loops through an /input of nifti file masks, and generate an equal nifti containing only most cranial and most caudal slice and saves to file_bounds.nii.gz in /output
# In order to run properly, one must mount /input and /output as docker --mount binds.

def generate_bounds(i, out_dir):
    try:
        # Ignore patterns with following. Crude for now.
        for scan in ["CT", "PT", "T1", "T2"]:#, "T1dr", "T2dr"]:
            if scan in i:
                return "Bad boy - do not generate bounds on scans!"

        img = sitk.ReadImage(i)
        arr = sitk.GetArrayFromImage(img)

        bounds_arr = np.zeros_like(arr)

        for x in range(arr.shape[2]):
            plane = arr[:, :, x]
            if True in np.any(plane, 1):
                np.zeros_like(arr)
                bounds_arr[:, :, x] = plane
                break

        for x in range(arr.shape[2] - 1, 0, -1):
            plane = arr[:, :, x]
            if True in np.any(plane, 1):
                np.zeros_like(arr)
                bounds_arr[:, :, x] = plane
                break

        bounds_img = sitk.GetImageFromArray(bounds_arr)
        out_path = os.path.join(out_dir, os.path.basename(i).replace(".nii.gz", "_xbounds.nii.gz"))
        print(f"Generating bounds for {i} into {out_path}")
        bounds_img.CopyInformation(img)
        sitk.WriteImage(bounds_img, out_path)
    except Exception as e:
        print(e)

def iter_files(i, o):
    for fol, subs, files in os.walk(i, followlinks=True):
        for file in files:
            yield (os.path.join(fol, file), o)

def main(i, o):
    p = Pool(int(os.environ["THREADS"]))
    p.starmap(generate_bounds, iter_files(i, o))
    p.close()
    p.join()


if __name__ == "__main__":
    print(os.environ["INPUT"], os.environ["OUTPUT"])
    main(i=os.environ["INPUT"], o=os.environ["OUTPUT"])
