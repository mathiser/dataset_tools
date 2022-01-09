import json

class Settings:
    def __init__(self, original_data_path, task_dir, task_name, segmentation_file, modality_file, threads, file_prefix,
                 train_pids_json=None, test_pids_json=None):
        self.original_data_dir = original_data_path
        self.base_dir = task_dir
        self.task_name = task_name
        self.train_pids = None
        self.test_pids = None
        self.threads = int(threads)
        self.file_prefix = file_prefix

        if train_pids_json:
            with open(train_pids_json, "r") as r:
                self.train_pids = json.loads(r.read())  ## Set path if you want deterministic

        if test_pids_json:
            with open(test_pids_json, "r") as r:
                self.test_pids = [i for i in json.loads(r.read())]

        with open(segmentation_file, "r") as r:
            self.segmentations = {i: v for i, v in enumerate(r.read().split("\n"))if len(v) != 0}

        with open(modality_file, "r") as r:
            self.modality = {v: str(i) for i, v in enumerate(r.read().split("\n")) if len(v) != 0}
        for k, v in self.modality.items():
            self.modality[k] = (4 - len(v)) * "0" + v

        print(self.__dict__)
