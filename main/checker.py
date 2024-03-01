from problem_instance import FUNCTIONS, get_terminals, CONSTANTS
from representations.tree import Tree
from representations.tree_utils import create_full_random_tree, create_grow_random_tree
import torch
import datasets.data_loader as ds
from operators.crossover_operators import *
from operators.mutators import *

datas = ["ppb"]

# obtaining the data loading functions using the dataset name
data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]

TERMINALS = get_terminals(data_loaders[0])

tree1 = create_full_random_tree(3, FUNCTIONS, TERMINALS, CONSTANTS)
tree2 = create_full_random_tree(3, FUNCTIONS, TERMINALS, CONSTANTS)

print(tree1)
print(tree2)

# mutation = mutate_tree_subtree(4, TERMINALS, CONSTANTS, FUNCTIONS, 0.1)
# tree3 = mutation(tree2)

xo = crossover_trees(FUNCTIONS)
tree3, tree4  = xo(tree1, tree2)

print(tree3)
print(tree4)
