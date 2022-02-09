import os
from multiprocessing import Pool
import sys

in_fol = sys.argv[1]
out_fol = sys.argv[2]

print(in_fol, out_fol)

def convert_dicom_to_nii(i, o):
    if not os.path.exists(o):
        os.makedirs(o)
    os.system(f"dcm2niix -z y -m y -o {o} {i}")

def get_sub_list(in_fol, out_fol):
    for patient_folder in os.listdir(in_fol):
        for date_folder in os.listdir(os.path.join(in_fol, patient_folder)):
            yield os.path.join(in_fol, patient_folder, date_folder), os.path.join(out_fol, patient_folder, date_folder)

if not os.path.exists(out_fol):
    os.makedirs(out_fol)

p = Pool(int(os.environ.get("THREADS")))
p.starmap(convert_dicom_to_nii, get_sub_list(in_fol, out_fol))
p.close()
p.join()