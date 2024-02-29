import logging
import os
import torch


import datasets.data_loader as ds
from datasets.data_loader import *

# creating a list with the datasets that are to be benchmarked
datas = ["ld50", "bioav", "ppb", "boston", "concrete_slump" ,"concrete_slump",
            "forest_fires", "efficiency_cooling", "diabetes", "parkinson_updrs", "efficiency_heating"]

# obtaining the data loading functions using the dataset name
data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]

# setting up the overall parameter dictionaries:
settings_dict = {"p_test": 0.2,
                 "p_val": None}
pi_init ={}
