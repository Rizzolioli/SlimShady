import datasets.data_loader as ds
from datasets.data_loader import load_preloaded
import torch
import numpy as np
import os



# setting up the datasets
datas = ["toxicity", "concrete", "instanbul", "ppb", "resid_build_sale_price", "energy"]

path = os.path.join(os.getcwd().split("main")[0], "datasets","merged_data")

for dataset in datas:
    X_train, y_train= load_preloaded(dataset, seed=1, training=True,X_y=True)
    X_test, y_test= load_preloaded(dataset, seed=1, training=False,X_y=True)
    merged = torch.cat((X_train, X_test))
    np.savetxt(os.path.join(path,f'{dataset}_merged.txt'), merged.numpy())



