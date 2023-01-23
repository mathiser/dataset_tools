import os
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
import json

# Author: Mathis Rasmussen
# 21-12-03
# Crops nifti images in scan folder to ROIs of mask niftis with padding. There must be ONE scan and ONE mask file.
# Also, naming is assumed: the name of the mask file without .nii.gz must be present in only one scan file. Thus the folder structure corresponds to the output of
# docker.io/mathiser/dataset_tools:nnunet_organizer

def get_bound(arr, dim, padding):
    assert (dim in [0, 1, 2])

    ## Forward
    planes = arr.shape[dim]
    for i in range(planes):
        if dim == 0:
            plane = arr[i, :, :]
        elif dim == 1:
            plane = arr[:, i, :]
        elif dim == 2:
            plane = arr[:, :, i]

        if np.count_nonzero(plane) != 0:
            padded_coord = i - padding
            if padded_coord <= 0:
                coord1 = 0
            else:
                coord1 = padded_coord
            break

    ## Backwards
    planes = arr.shape[dim]
    for i in range(arr.shape[dim] - 1, 0, -1):
        if dim == 0:
            plane = arr[i, :, :]
        elif dim == 1:
            plane = arr[:, i, :]
        elif dim == 2:
            plane = arr[:, :, i]

        if np.count_nonzero(plane) != 0:
            padded_coord = i + padding
            if padded_coord >= planes:
                coord2 = planes
            else:
                coord2 = padded_coord
            break
    return (coord1, coord2)

def get_bounding_box_coordinates(mask_arr, padding):
    # Coords are reversed in arrays
    z = get_bound(mask_arr, dim=0, padding=padding[0])
    y = get_bound(mask_arr, dim=1, padding=padding[1])
    x = get_bound(mask_arr, dim=2, padding=padding[2])
    return (z, y, x)


def crop_image(img, coords):
    z, y, x = coords
    return img[x[0]:x[1], y[0]: y[1], z[0]: z[1]]


def save_img(img, cropped_img_path):
    return sitk.WriteImage(img, cropped_img_path)

def crop_imgs_to_mask(scan_paths, mask_path, scan_out_dir, mask_out_dir, padding=(7, 15, 15)):
    mask = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask)
    print(f"Loading {scan_paths}")
    bbox = get_bounding_box_coordinates(mask_arr, padding)
    print(f"Bounding box: {bbox} of {scan_paths}")

    cropped_mask = crop_image(mask, bbox)
    print(f"Saving cropped mask of {mask_path}")
    cropped_mask_path = os.path.join(mask_out_dir, os.path.basename(mask_path))
    save_img(cropped_mask, cropped_mask_path)

    for scan in scan_paths:
        img = sitk.ReadImage(scan)
        cropped_img = crop_image(img, bbox)
        print(f"Saving cropped image of {scan}")
        cropped_img_path = os.path.join(scan_out_dir, os.path.basename(scan))
        save_img(cropped_img, cropped_img_path)

    return [mask_path, bbox]
    

def data_loader(scan_dir, mask_dir, out_dir, padding):
    scan_out_dir = os.path.join(out_dir, "images")
    mask_out_dir = os.path.join(out_dir, "labels")

    if not os.path.exists(scan_out_dir):
        os.makedirs(scan_out_dir)
    if not os.path.exists(mask_out_dir):
        os.makedirs(mask_out_dir)     

    for fol, subs, files in os.walk(mask_dir, followlinks=True):
        for f_mask in files:
            mask_path = os.path.join(fol, f_mask)
            f_mask_basename = os.path.basename(mask_path)
            f_mask_basename = f_mask_basename.replace(".nii.gz", "")
            
            f_scans = []
            for f_scan in os.listdir(scan_dir):
                if f_mask_basename in f_scan:
                    f_scans.append(os.path.join(scan_dir, f_scan))
            
            if len(f_scans) == 0:
                raise Exception(f"Error in {f_mask_basename} - not found in scan_dir")
            else:
                yield [f_scans, mask_path, scan_out_dir, mask_out_dir, padding]


def main(scan_dir, mask_dir, out_dir, padding, threads):
    p = Pool(threads)
    res = p.starmap(crop_imgs_to_mask, data_loader(scan_dir, mask_dir, out_dir, padding))
    p.close()
    p.join()

    with open(os.path.join(out_dir, "bboxes.json"), "w") as f:
        f.write(json.dumps(res))

if __name__ == "__main__":
    main(scan_dir=os.environ["SCAN_DIR"],
         mask_dir=os.environ["MASK_DIR"],
         out_dir=os.environ["OUT_DIR"],
         padding=(int(os.environ["Z_PADDING"]), int(os.environ["Y_PADDING"]), int(os.environ["X_PADDING"])),
         threads=int(os.environ["THREADS"])
         )
