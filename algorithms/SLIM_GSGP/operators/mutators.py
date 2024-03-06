import random
from copy import deepcopy
from algorithms.GP.representations.tree_utils import create_grow_random_tree
from algorithms.GSGP.representations.tree import Tree
import torch

def delta_tree(tr1, tr2, ms, testing):
    if testing:
        return torch.mul(ms, torch.sub(tr1.test_semantics, tr2.test_semantics))
    else:
        return torch.mul(ms, torch.sub(tr1.train_semantics, tr2.train_semantics))

def two_trees_inflate_mutation(individual, ms, X, max_depth = 8, p_c = 0.1, X_test = False):

    offs = deepcopy(individual) #TODO how to do without copy ??

    FUNCTIONS = individual.collection[0].FUNCTIONS #TODO how to retrieve better functions and etc
    TERMINALS = individual.collection[0].TERMINALS
    CONSTANTS = individual.collection[0].CONSTANTS

    random_tree1 = Tree(create_grow_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c),
                        FUNCTIONS, TERMINALS, CONSTANTS)

    random_tree2 = Tree(create_grow_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c),
                        FUNCTIONS, TERMINALS, CONSTANTS)

    random_tree1.calculate_semantics(X)
    random_tree2.calculate_semantics(X)
    if X_test != None:
        random_tree1.calculate_semantics(X_test, testing=True)
        random_tree2.calculate_semantics(X_test, testing=True)

    new_block = Tree([delta_tree, #TODO could create a function outside
                      random_tree1, random_tree2, ms],
                     individual.collection[0].FUNCTIONS,
                     individual.collection[0].TERMINALS,
                     individual.collection[0].CONSTANTS
                     )

    new_block.calculate_semantics(X, testing=False)
    if X_test != None:
        new_block.calculate_semantics(X_test, testing=True)

    offs.add_block(new_block)

    return offs



def deflate_mutation(individual):

    offs = deepcopy(individual) #TODO how to do without copy ??

    if individual.size > 1: #TODO if individual of size 1 no deflation
        offs.remove_block( random.randint(1, individual.size-1) )

    return offs