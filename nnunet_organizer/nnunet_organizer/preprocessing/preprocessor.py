import os
import shutil
from multiprocessing.pool import ThreadPool

import pandas as pd

from .utils import merge_nii, generate_dataset_json


class Preprocessor:
    def __init__(self, settings):
        self.task_name = settings.task_name
        self.segmentations = settings.segmentations
        self.modality = settings.modality
        if settings.train_pids:
            self.train_pids = settings.train_pids
        else:
            self.train_pids = None
        
        if settings.test_pids:
            self.test_pids = settings.test_pids
        else:
            self.test_pids = None

        self.original_data_dir = settings.original_data_dir
        self.base_dir = settings.base_dir
        self.file_prefix = settings.file_prefix
        self.script_dir = os.path.join(self.base_dir, "scripts")

        self.nnUNet_SRC_DIR = os.path.join(self.base_dir, "nnunet")
        self.nnUNet_raw_data_base = os.path.join(self.nnUNet_SRC_DIR, "nnUNet_raw_data_base")
        self.nnUNet_raw_data = os.path.join(self.nnUNet_raw_data_base, "nnUNet_raw_data")
        self.nnUNet_preprocessed = os.path.join(self.nnUNet_SRC_DIR, "nnUNet_preprocessed")
        self.RESULTS_FOLDER = os.path.join(self.nnUNet_SRC_DIR, "nnUNet_trained_models")

        self.task_folder = os.path.join(os.path.abspath(self.nnUNet_raw_data), self.task_name)
        if not os.path.exists(self.task_folder):
            os.makedirs(self.task_folder)
        self.imagesTr = os.path.join(self.task_folder, "imagesTr")
        self.labelsTr = os.path.join(self.task_folder, "labelsTr")
        self.imagesTs = os.path.join(self.task_folder, "imagesTs")
        self.labelsTs = os.path.join(self.task_folder, "labelsTs")
        self.json_path = os.path.join(self.task_folder, "dataset.json")

        self.threads = settings.threads


    def make_dirs_of_task(self):
        for fol in [self.imagesTr, self.labelsTr, self.imagesTs, self.labelsTs]:
            if not os.path.exists(fol):
                os.makedirs(fol)

    def load_dataset(self):
        ## Loads a files of any structure as long as naming comvention HNCDL_PID_LABEL.nii.gz is honored.
        df = pd.DataFrame()

        for fol, sub, files in os.walk(self.original_data_dir):
            for file in files:
                cl, pid, label = file.replace(".nii.gz", "").split("_", maxsplit=2)
                df = df.append({"folder": fol, "file": file, "class": cl, "pid": pid, "label": label},
                               ignore_index=True)
        self.dataset = df
        self.dataset.to_pickle(os.path.join(self.base_dir, "dataset.pkl"))

    def check_file_presence(self):
        # Scans train
        train_remove = []

        print(f"Train length before: {len(self.train_pids)}")
        for pid in self.train_pids:
            tmp = self.dataset[self.dataset["pid"] == pid]
            for label in self.modality.keys():
                if label not in tmp["label"].unique():
                    train_remove.append(pid)
                    print(f"Removed patient {pid} from training set - does not have {label}")
                    break

            ## Check that at least one label to segment exists in training:
            tmp = self.dataset[(self.dataset["pid"] == pid) & (self.dataset["label"].isin(self.segmentations.values()))]
            print(tmp.size, tmp)
            if tmp.size == 0:
                train_remove.append(pid)

        [self.train_pids.remove(pid) for pid in train_remove]
        print(f"Train length after: {len(self.train_pids)}")

        print(f"Test length before: {len(self.test_pids)}")
        # Scans test
        test_remove = []
        for pid in self.test_pids:
            tmp = self.dataset[self.dataset["pid"] == pid]
            for label in self.modality.keys():
                if label not in tmp["label"].unique():
                    test_remove.append(pid)
                    print(f"Removed patient {pid} from test set - does not have {label}")
                    break

            tmp = self.dataset[(self.dataset["pid"] == pid) & (self.dataset["label"].isin(self.segmentations.values()))]
            if tmp.size == 0:
                test_remove.append(pid)

        [self.test_pids.remove(pid) for pid in test_remove]
        print(f"Test length after: {len(self.test_pids)}")

    def copy_scans_to_nnunet_folder(self):
        print("Train_pids: ", len(self.train_pids), self.train_pids)
        ## Trains first.
        ## Get included modalities.list from settings
        scan_df = self.dataset[(self.dataset["label"].isin(self.modality.keys()) &
                                     (self.dataset["pid"].isin(self.train_pids)))]
        for _, scan_row in scan_df.iterrows():
            output_file_name = "{cl}_{pid}_{int_rep}.nii.gz".format(cl=scan_row["class"], pid=scan_row["pid"],
                                                                    int_rep=self.modality[scan_row[
                                                                        "label"]])  ## label from scan_df to find appropriate scan/int_rep from project_vars
            
            fro = os.path.join(scan_row["folder"], scan_row["file"])
            to = os.path.join(self.imagesTr, output_file_name)
            shutil.copy2(fro, to)

        ## Test scans
        scan_df = self.dataset[(self.dataset["label"].isin(self.modality.keys()) &
                                (self.dataset["pid"].isin(self.test_pids)))]
        for _, scan_row in scan_df.iterrows():
            output_file_name = "{cl}_{pid}_{int_rep}.nii.gz".format(cl=scan_row["class"], pid=scan_row["pid"],
                                                                    int_rep=self.modality[scan_row[
                                                                        "label"]])  ## label from scan_df to find appropriate scan/int_rep from project_vars

            fro = os.path.join(scan_row["folder"], scan_row["file"])
            to = os.path.join(self.imagesTs, output_file_name)
            shutil.copy2(fro, to)

    def merge_labels_and_copy_to_output_for_specific_task(self):
        # Train first
        odot = []
        for pid in self.train_pids:
            label_files_to_be_merged_for_patient_df = \
                self.dataset[(self.dataset["pid"] == pid)
                        &
                        (self.dataset["label"].isin(self.segmentations.values()))]
         
            ##generate a list of paths to segments that are to be merged:
            merge_list = [os.path.abspath(os.path.join(label_file["folder"], label_file["file"]))
                          for _, label_file in label_files_to_be_merged_for_patient_df.iterrows()]

            ## make a output filename
            output_file_name = "{cls}_{pid}.nii.gz".format(cls=self.file_prefix, pid=pid)

            # Run the big merger, which outputs the merged file to output_file
            ## merge_nii() takes care of writing the merged file to output_file
            output_file_abs_path = os.path.join(self.labelsTr, output_file_name)
            odot.append((merge_list, self.segmentations, output_file_abs_path))

        t = ThreadPool(self.threads)
        t.starmap(merge_nii, odot)
        t.close()
        t.join()

        odot = []
        # test second
        for pid in self.test_pids:
            label_files_to_be_merged_for_patient_df = \
                self.dataset[(self.dataset["pid"] == pid)
                             &
                             (self.dataset["label"].isin(self.segmentations.values()))]

            ##generate a list of paths to segments that are to be merged:
            merge_list = [os.path.abspath(os.path.join(label_file["folder"], label_file["file"]))
                          for _, label_file in label_files_to_be_merged_for_patient_df.iterrows()]

            ## make a output filename
            output_file_name = "{cls}_{pid}.nii.gz".format(cls=self.file_prefix, pid=pid)

            # Run the big merger, which outputs the merged file to output_file
            ## merge_nii() takes care of writing the merged file to output_file
            output_file_abs_path = os.path.join(self.labelsTs, output_file_name)
#            merge_nii(path_to_files=merge_list,  segmentations=self.segmentations,
#                     output_file=output_file_abs_path)
            odot.append((merge_list, self.segmentations, output_file_abs_path))

        t = ThreadPool(self.threads)
        t.starmap(merge_nii, odot)
        t.close()
        t.join()

    def generate_json_for_tasks(self):
        generate_dataset_json(output_file=self.json_path,
                              imagesTr_dir=self.imagesTr,
                              imagesTs_dir=self.imagesTs,
                              modalities={v: k for k, v in self.modality.items()},
                              labels=self.segmentations,
                              dataset_name=self.task_name,
                              license='hands off!')

    def dump_vars_to_init_project_file(self):
        with open(os.path.join(self.base_dir, "init_project.sh"), "w") as f:
            f.write(
                f"""
#!/bin/bash
export nnUNet_SRC_DIR="{self.nnUNet_SRC_DIR}"
export nnUNet_raw_data_base="{self.nnUNet_raw_data_base}"
export nnUNet_raw_data="{self.nnUNet_raw_data}"
export nnUNet_preprocessed="{self.nnUNet_preprocessed}"
export RESULTS_FOLDER="{self.RESULTS_FOLDER}"

export CUDA_NVCC_EXECUTABLE="/usr/local/cuda-11.1/bin/nvcc"
export CUDA_HOME="/usr/local/cuda-11.1"
export CUDNN_INCLUDE_PATH="/usr/local/cuda-11.1/include/"
export CUDNN_LIBRARY_PATH="/usr/local/cuda-11.1/lib64/"
export LIBRARY_PATH="/usr/local/cuda-11.1/lib64"
""")

    def pipeline(self):
        self.load_dataset()
        self.check_file_presence()
        self.make_dirs_of_task()
        self.copy_scans_to_nnunet_folder()
        self.merge_labels_and_copy_to_output_for_specific_task()
        self.generate_json_for_tasks()
        self.dump_vars_to_init_project_file()
