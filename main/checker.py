from problem_instance import FUNCTIONS, get_terminals, CONSTANTS
from representations.tree import Tree
from representations.tree_utils import create_full_random_tree, create_grow_random_tree
import torch
import datasets.data_loader as ds

datas = ["ppb"]

# obtaining the data loading functions using the dataset name
data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]

TERMINALS = get_terminals(data_loaders[0])

tree = create_grow_random_tree(2, FUNCTIONS, TERMINALS, CONSTANTS)

print(tree)
