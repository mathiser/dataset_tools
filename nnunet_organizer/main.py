import os
import sys

sys.path.append("/nnunet_organizer")
from nnunet_organizer import preprocessing, settings

settings = settings.Settings(task_dir=os.path.join(os.environ["OUTPUT"], f"Task{os.environ['TASK_ID']}_{os.environ['TASK_NAME']}"),
                             original_data_path=os.environ["INPUT"],
                             modality_file=os.environ["MODALITY_FILE"],
                             segmentation_file=os.environ["SEGMENTATION_FILE"],
                             train_pids_json=os.environ["TRAIN_PIDS_JSON"],
                             test_pids_json=os.environ["TEST_PIDS_JSON"],
                             task_name="Task{}".format(os.environ["TASK_ID"]),
                             threads=os.environ["THREADS"],
                             file_prefix=os.environ["FILE_PREFIX"])

preprocessor = preprocessing.Preprocessor(settings=settings)
preprocessor.pipeline()
