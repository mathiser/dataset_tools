#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 3.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os.path
from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk



def get_identifiers_from_splitted_files(folder: str):
    # uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    uniques = np.unique([i.rsplit("_", maxsplit=1)[0] for i in subfiles(folder, suffix='.nii.gz', join=False)])

    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities,
                          labels: dict, dataset_name: str, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 3: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []


    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = modalities
    json_dict['labels'] = labels

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = []
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))


def merge_nii(path_to_files: list, segmentations: dict, output_file: str):
    ##init arr from first file in path_to_files
    tmp_img = sitk.ReadImage(path_to_files[0])
    tmp_arr = sitk.GetArrayFromImage(tmp_img)
    merged_label_arr = np.zeros_like(tmp_arr)
    for i, label in segmentations.items():
        for file in path_to_files:
            if f"{label}.nii.gz" in os.path.basename(file):
                tmp_img = sitk.ReadImage(file)
                tmp_arr = sitk.GetArrayFromImage(tmp_img)
                tmp_arr = np.array(tmp_arr, dtype=bool)
                merged_label_arr[tmp_arr == True] = int(i)

    merged_label_img = sitk.GetImageFromArray(merged_label_arr)
    merged_label_img.CopyInformation(tmp_img)
    print(f"Write image merged_label_img to {output_file}")
    sitk.WriteImage(merged_label_img, output_file)
