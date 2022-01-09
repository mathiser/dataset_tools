import os
import shutil

import SimpleITK as sitk
from multiprocessing import Pool
import numpy as np

## Author: Mathis Rasmussen
## 21-11-26
## Loops through an /input of nifti file masks, and merges where two files ending with _L.nii.gz and _R.nii.gz exists.
# Note that whenever only _L or _R exists, instance is ignored.
# In order to run properly, one must mount /input and /output as docker --mount binds.

def merge_files(file1, file2, out_dir):
    ##init arr from first file in path_to_files
    file1_img = sitk.ReadImage(file1)
    file2_img = sitk.ReadImage(file2)
    file1_arr = sitk.GetArrayFromImage(file1_img)
    print(np.unique(file1_arr))
    file1_arr = np.array(file1_arr, dtype=bool)

    file2_arr = sitk.GetArrayFromImage(file2_img)
    file2_arr = np.array(file2_arr, dtype=bool)
    file1_arr[file2_arr == 1] = 1

    merged_label_img = sitk.GetImageFromArray(file1_arr)
    merged_label_img.CopyInformation(file1_img)

    out_path = os.path.join(out_dir, os.path.basename(file1.replace("_L.nii.gz", "_merged.nii.gz")))
    print("Merging: ")
    print(file1)
    print(file2, "------- >")
    print(out_path)
    sitk.WriteImage(merged_label_img, out_path)



## Find all files containing "_L" and "_R" and return
def get_pairs(i, out_dir):
    for fol, subs, files in os.walk(i, followlinks=True):
        for file in files:
            if file.endswith("_L.nii.gz"):
                f1 = os.path.join(fol, file)
                f2 = os.path.join(fol, file.replace("_L.nii.gz", "_R.nii.gz"))
                if not os.path.isfile(f2):
                    continue

                yield (f1, f2, out_dir)


def main(i, o):
    p = Pool(int(os.environ["THREADS"]))
    p.starmap(merge_files, get_pairs(i, o))
    p.close()
    p.join()


if __name__ == "__main__":
    print(os.environ["INPUT"], os.environ["OUTPUT"])
    main(i=os.environ["INPUT"], o=os.environ["OUTPUT"])
