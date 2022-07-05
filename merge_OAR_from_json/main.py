import json
import os
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np


## Author: Mathis Rasmussen
## 21-11-26
# In order to run properly, one must mount /input and /output as docker --mount binds.
# Labels must be added as ONE file to /labels/labels.json (any name suffices, but one file that ends with .json)

def load_json(path):
    with open(path, "r") as r:
        label_dict = {k: int(v) for k, v in json.loads(r.read()).items()}
    return label_dict

## SET GLOBAL label_map
for fol, subs, files in os.walk(os.environ.get("LABELS")):
    for file in files:
        if file.endswith(".json"):
            label_map = load_json(os.path.join(fol, file))
            print(label_map)
            break

## Author: Mathis Rasmussen
## 21-11-26
# In order to run properly, one must mount /labels/, /input and /output as docker --mount binds.

def merge_files(files: dict, out_dir):
    ##init arr from first file in path_to_files
    print(f"Merging: {files}")
    ref_path = list(files.values())[0]
    ref_img = sitk.ReadImage(ref_path)
    ref_arr = sitk.GetArrayFromImage(ref_img)
    merged_array = np.zeros_like(ref_arr)

    for label, name_guess in files.items():
        img = sitk.ReadImage(name_guess)
        arr = sitk.GetArrayFromImage(img)
        max_val = np.unique(arr)[-1]
        merged_array[arr == max_val] = label_map[label]

    merged_img = sitk.GetImageFromArray(merged_array)
    merged_img.CopyInformation(ref_img)

    ## Split label away from basename and add PCM_merged instead.
    out_filename = os.path.basename(ref_path.rsplit("&", 1)[0]) + f"OARs.nii.gz"
    out_path = os.path.join(out_dir, out_filename)
    print(out_path)
    sitk.WriteImage(merged_img, out_path)

## Find all files containing "_L" and "_R" and return
def get_pairs(i, out_dir):
    pids = set()
    for fol, subs, files in os.walk(i, followlinks=True):
        for file in files:
            piddate = os.path.join(fol, file.rsplit("&", 1)[0])
            pids.add(piddate)

    for pid in pids:
        file_list = {}
        for label, i in label_map.items():
            name_guess = f"{pid}&{label}"
            #print(f"Name guess: {name_guess}")

            if os.path.exists(name_guess):
                print(f"Exists: {name_guess}")
                file_list[label] = name_guess
            #else:
                #print(f"NOT FOUND: {name_guess}")

        if len(file_list) > 0:
            yield file_list, out_dir

def main(i, o):
    p = Pool(int(os.environ["THREADS"]))
    p.starmap(merge_files, get_pairs(i, o))
    p.close()
    p.join()


if __name__ == "__main__":
    print(os.environ["INPUT"], os.environ["OUTPUT"])
    main(i=os.environ["INPUT"], o=os.environ["OUTPUT"])
