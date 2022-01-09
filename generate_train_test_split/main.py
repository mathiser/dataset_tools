import os
import numpy as np
import pandas as pd
import random
import json

def load_dataset(i, o):
    ## Loads a files of any structure as long as naming comvention HNCDL_PID_LABEL.nii.gz is honored.
    df = pd.DataFrame()

    for fol, sub, files in os.walk(i):
        for file in files:
            cl, pid, label = file.replace(".nii.gz", "").split("_", maxsplit=2)
            df = df.append({"folder": fol, "file": file, "class": cl, "pid": pid, "label": label},
                           ignore_index=True)

    df.to_pickle(os.path.join(o, "dataset.pkl"))
    return df

def generate_train_and_test_datasets(df, o, train_test_ratio):
    # This function sets train_dataset, train_pids, test_pids, test_dataset.
    # If dataset pickles already exists, they will be loaded ,
    # otherwise the function will generate random split and dump new pickles.
    # To specify exact split use pickles. If pids are provided, this function will generate new
    # set from the provided pids.
    assert ([str(col) in ["folder", "file", "class", "pid", "label"] for col in df.columns])

    ## randomly split patients train:test
    # If explicitly_include is set, use these as included_pids otherwise used all pids form the dataset.
    included_pids = np.unique(df["pid"])  ## Get all pt ids
    random.shuffle(list(included_pids))  ## shuffle all pt ids

    split_int = int(len(included_pids) * float(train_test_ratio))  ## find the pt that sets the split value

    train_pids = list(included_pids[split_int:])  ## contains pids to be in train
    test_pids = list(included_pids[:split_int])

    ##contains pids to be in test
    with open(os.path.join(o, "train_pids.json"), "w") as f:
        f.write(json.dumps(list(train_pids)))

    with open(os.path.join(o, "test_pids.json"), "w") as f:
        f.write(json.dumps(list(test_pids)))

def main(i, o,  train_test_ratio):
    df = load_dataset(i,o)
    generate_train_and_test_datasets(df, o, train_test_ratio)

if __name__ == "__main__":
    i = os.environ["INPUT"]
    o = os.environ["OUTPUT"]
    ratio = os.environ["TRAIN_TEST_RATIO"]
    main(i, o, ratio)
