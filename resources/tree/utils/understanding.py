from functions import FUNCTIONS, TERMINALS, CONSTANTS
from algorithms.GP.representations.tree import Tree
from utils import create_full_random_tree
import torch

tree = create_full_random_tree(2, FUNCTIONS, TERMINALS, CONSTANTS)

print(tree)

tree = Tree(tree, FUNCTIONS, TERMINALS, CONSTANTS)

#print("OG TENSOR FAM", torch.Tensor([[i + 10 for j in range(100)] for i in range(10)]))

og = torch.rand((10, 5))
print(og)
print(f'shape {og.shape}')

print(tree.apply_tree(og))

