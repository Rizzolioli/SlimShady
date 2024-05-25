import os

import datasets.data_loader as ds
import numpy as np
import pandas as pd
import torch
from datasets.data_loader import load_preloaded

# setting up the datasets
datas = ["toxicity", "concrete", "instanbul", "ppb", "resid_build_sale_price", "energy"]

path = os.path.join(os.getcwd().split("main")[0], "datasets", "merged_data")

for dataset in datas:

    filename_tr = f"TRAINING_{1}_{dataset.upper()}.txt"

    filename_te = f"TEST_{1}_{dataset.upper()}.txt"

    df_training = pd.read_csv(
        os.path.join(
            os.getcwd().split("main")[0], "datasets", "pre_loaded_data", filename_tr
        ),
        sep=" ",
        header=None,
    ).iloc[:, :-1]

    df_testing = pd.read_csv(
        os.path.join(
            os.getcwd().split("main")[0], "datasets", "pre_loaded_data", filename_te
        ),
        sep=" ",
        header=None,
    ).iloc[:, :-1]

    merged = pd.concat([df_training, df_testing])

    merged.to_csv(
        os.path.join(path, f"{dataset}_merged.txt"), sep=" ", header=None, index=False
    )
