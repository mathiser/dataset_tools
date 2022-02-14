import os
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool


# Author: Mathis Rasmussen
# 21-12-03
# Crops nifti images in scan folder to ROIs of mask niftis with padding. There must be ONE scan and ONE mask file.
# Also, naming is assumed: the name of the mask file without .nii.gz must be present in only one scan file. Thus the folder structure corresponds to the output of
# docker.io/mathiser/dataset_tools:nnunet_organizer


def crop_array(array: np.ndarray, bounding_box):
    z, y, x = bounding_box
    cropped_arr = array[z[0]:z[1], y[0]: y[1], x[0]: x[1]]

    return cropped_arr


def get_bounding_box(array: np.ndarray, threshold_lower, threshold_upper):
    z = get_bound(array, dim=0, threshold_lower=threshold_lower, threshold_upper=threshold_upper)
    y = get_bound(array, dim=1, threshold_lower=threshold_lower, threshold_upper=threshold_upper)
    x = get_bound(array, dim=2, threshold_lower=threshold_lower, threshold_upper=threshold_upper)
    return z, y, x


def get_bound(arr: np.ndarray, dim, threshold_lower, threshold_upper):
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
        else:
            raise Exception("Cropping condition not met")
        if np.count_nonzero((threshold_lower < plane) & (plane < threshold_upper)) > 0:
            padded_coord = i - 20
            if padded_coord <= 0:
                coord1 = 0
            else:
                coord1 = padded_coord
            break
    else:
        raise Exception("Cropping condition not met")

    ## Backwards
    planes = arr.shape[dim]
    for i in range(arr.shape[dim] - 1, 0, -1):
        if dim == 0:
            plane = arr[i, :, :]
        elif dim == 1:
            plane = arr[:, i, :]
        elif dim == 2:
            plane = arr[:, :, i]
        else:
            raise Exception("Cropping condition not met")
        if np.count_nonzero((threshold_lower < plane) & (plane < threshold_upper)) > 0:
            padded_coord = i + 20
            if padded_coord >= planes:
                coord2 = planes
            else:
                coord2 = padded_coord
            break
    else:
        raise Exception("Cropping condition not met")

    return coord1, coord2


def pipeline(d):
    ct_img = sitk.ReadImage(d["ct"])
    ct_arr = sitk.GetArrayFromImage(ct_img)
    bbox = get_bounding_box(ct_arr,
                            threshold_lower=d["threshold_lower"],
                            threshold_upper=d["threshold_upper"])

    ## add ct to other to crop all
    scan_out_dir = os.path.join(d["out_dir"], "scans")
    if not os.path.exists(scan_out_dir):
        os.makedirs(scan_out_dir)
    for f in d["scans"]:
        tmp_img = sitk.ReadImage(f)
        tmp_arr = sitk.GetArrayFromImage(tmp_img)
        cropped_arr = crop_array(tmp_arr, bbox)
        cropped_img = sitk.GetImageFromArray(cropped_arr)
        cropped_img.SetSpacing(ct_img.GetSpacing())
        sitk.WriteImage(cropped_img, os.path.join(scan_out_dir, os.path.basename(f)))

    ## add ct to other to crop all
    label_out_dir = os.path.join(d["out_dir"], "labels")
    if not os.path.exists(label_out_dir):
        os.makedirs(label_out_dir)

    tmp_img = sitk.ReadImage(d["label"])
    tmp_arr = sitk.GetArrayFromImage(tmp_img)
    cropped_arr = crop_array(tmp_arr, bbox)
    cropped_img = sitk.GetImageFromArray(cropped_arr)
    cropped_img.SetSpacing(ct_img.GetSpacing())
    sitk.WriteImage(cropped_img, os.path.join(label_out_dir, os.path.basename(d["label"])))


def data_yielder(image_dir, label_dir, out_dir, threshold_lower, threshold_upper):
    for label_fol, subs, labels in os.walk(label_dir):
        for label in labels:
            d = {"ct": "",
                 "label": "",
                 "scans": [],
                 "out_dir": out_dir,
                 "threshold_lower": threshold_lower,
                 "threshold_upper": threshold_upper,
                 }

            pid = label.replace(".nii.gz", "")
            d["label"] = os.path.join(label_fol, label)

            for scan_fol, subs, scans in os.walk(image_dir):
                for scan in scans:
                    if pid in scan:
                        d["scans"].append(os.path.join(scan_fol, scan))
                        if "0000.nii.gz" in scan:
                            d["ct"] = os.path.join(scan_fol, scan)
            yield d


def main(scan_dir, mask_dir, out_dir, threshold_lower, threshold_upper, threads):
    p = Pool(threads)
    p.map(pipeline,
              data_yielder(image_dir=scan_dir,
                           label_dir=mask_dir,
                           out_dir=out_dir,
                           threshold_lower=threshold_lower,
                           threshold_upper=threshold_upper)
              )
    p.close()
    p.join()


if __name__ == "__main__":
    main(scan_dir=os.environ["SCAN_DIR"],
         out_dir=os.environ["OUT_DIR"],
         threshold_lower=int(os.environ.get("THRESHOLD_LOWER")),
         threshold_upper=int(os.environ.get("THRESHOLD_UPPER")),
         threads=int(os.environ["THREADS"])
         )