from functions import FUNCTIONS, TERMINALS, CONSTANTS
from Representations.tree import Tree
from utils import create_full_random_tree, create_grow_random_tree
import torch

tree = create_grow_random_tree(4, FUNCTIONS, TERMINALS, CONSTANTS)

print(tree)

tree = Tree(tree, FUNCTIONS, TERMINALS, CONSTANTS)

#print("OG TENSOR FAM", torch.Tensor([[i + 10 for j in range(100)] for i in range(10)]))

og = torch.rand((100, 10))
print(og)

print(tree.apply_tree(og))

