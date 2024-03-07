import torch
import random



def geometric_mutation(tree, random_tree_1, random_tree_2, ms, testing): #ms needs to belong to the tree structure for reconstruction purposes

    if testing:
        return torch.add(tree.test_semantics, torch.mul(ms, torch.sub(random_tree_1.test_semantics, random_tree_2.test_semantics)))

    else:
        return torch.add(tree.train_semantics, torch.mul(ms, torch.sub(random_tree_1.train_semantics, random_tree_2.train_semantics)))
