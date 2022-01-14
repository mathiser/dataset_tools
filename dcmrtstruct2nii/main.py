#!/usr/bin/env python
# coding: utf-8
import sys
import os
import dcmrtstruct2nii
import pydicom
from multiprocessing import Pool

dcm_folder = sys.argv[1]
nii_folder = sys.argv[2]
threads = int(os.environ.get("THREADS"))

print(f"RTSTRUCT Dicom folder: {dcm_folder}")
print(f"Nifti folder: {nii_folder}")

if not os.path.exists(nii_folder):
    os.makedirs(nii_folder)

def extract_to_nii(file_path, out_folder):
    if os.path.exists(out_folder):
        return
    try:
        print(f"Converting {file_path}")
        dcmrtstruct2nii.dcmrtstruct2nii(file_path, os.path.dirname(file_path), out_folder)
    except Exception as e:
        print(e)
        with open("conversion_errors.log", "a") as f:
            f.write(f"{file_path};{e}\n")

def check_if_rtstruct(f):
    try:
        with pydicom.filereader.dcmread(f, force=True) as ds:
            if ds.Modality == "RTSTRUCT":
                print(f"Found RTSTRUCT: {f}")
                return f
    except Exception as e:
        print(e)

def find_all_rtstructs(dcm):
    ## Get all subs with dicom files inside
    ## Find the shit of rtstructs
    p = Pool(threads)
    rtstructs = p.map(check_if_rtstruct, [os.path.join(fol, f) for fol, subs, files in os.walk(dcm) for f in files])
    p.close()
    p.join()

    return rtstructs

def zip_in_and_out(rtstruct_paths, out_path):
    ## Zip rtstructs with nifti_folder/pt_id
    zipped = []
    for i, r in enumerate(rtstruct_paths):
        with pydicom.filereader.dcmread(r, force=True) as ds:
            pid = ds.PatientID
        
        out = os.path.join(out_path, f"{i}_{pid}")
        zipped.append((r, out))
    yield zipped

file_paths = find_all_rtstructs(dcm_folder)
file_paths = set(file_paths)
if None in file_paths:
    file_paths.remove(None)

#zipped = zip_in_and_out(file_paths, nii_folder)

## Convert the shit out of rtstructs
p = Pool(threads)
conversion = p.starmap(extract_to_nii, zip_in_and_out(file_paths, nii_folder))
p.close()
p.join()


