import os
import sys

sys.path.append("/nnunet_organizer")
from nnunet_organizer import preprocessing, settings

settings = settings.Settings(task_dir=os.environ["OUTPUT"],
                             original_data_path=os.environ["INPUT"],
                             modality_file=os.environ["MODALITY_FILE"],
                             segmentation_file=os.environ["SEGMENTATION_FILE"],
                             train_pids_json=os.environ["TRAIN_PIDS_JSON"],
                             test_pids_json=os.environ["TEST_PIDS_JSON"],
                             task_name="Task{}_{}".format(os.environ["TASK_ID"], os.environ.get("TASK_NAME")),
                             threads=os.environ["THREADS"]
                             )
preprocessor = preprocessing.Preprocessor(settings=settings)
preprocessor.pipeline()
