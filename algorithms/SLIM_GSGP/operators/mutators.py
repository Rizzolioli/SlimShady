import random
import torch

from algorithms.GP.representations.tree_utils import create_grow_random_tree
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.representations.individual import Individual
from utils.utils import get_random_tree


def two_trees_delta(tr1, tr2, ms, testing):
    if testing:
        return torch.mul(ms, torch.sub(tr1.test_semantics, tr2.test_semantics))
    else:
        return torch.mul(ms, torch.sub(tr1.train_semantics, tr2.train_semantics))

def one_tree_delta(tr1, ms, testing):
    if testing:
        return torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, torch.abs(tr1.test_semantics)))))
    else:
        return torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, torch.abs(tr1.train_semantics)))))


def two_trees_inflate_mutation(FUNCTIONS, TERMINALS, CONSTANTS ):

    def tt_inflate(individual, ms, X,max_depth = 8, p_c = 0.1, X_test = None, p_terminal=0.5, grow_probability=1):

        random_tree1 = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c, p_terminal=p_terminal, grow_probability=grow_probability)

        random_tree2 = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c, p_terminal=p_terminal,
                                       grow_probability=grow_probability)

        if X_test is not None:
            random_tree1.calculate_semantics(X_test, testing=True)
            random_tree2.calculate_semantics(X_test, testing=True)

        new_block = Tree([two_trees_delta,
                          random_tree1, random_tree2, ms],
                         FUNCTIONS,
                         TERMINALS,
                         CONSTANTS)

        new_block.calculate_semantics(X, testing=False)
        if X_test is not None:
            new_block.calculate_semantics(X_test, testing=True)
        offs = individual.add_block(new_block)

        if individual.train_semantics is not None:

            offs.train_semantics = torch.stack([*individual.train_semantics,
                                    (new_block.train_semantics if new_block.train_semantics.shape != torch.Size([])
                                    else new_block.train_semantics.repeat(len(X)))])

        if individual.test_semantics is not None:

            offs.test_semantics = torch.stack([*individual.test_semantics,
                                (new_block.test_semantics if new_block.test_semantics.shape != torch.Size([])
                                 else new_block.test_semantics.repeat(len(X_test)))])

        return offs

    return tt_inflate


def deflate_mutation(individual):
    # TODO: add to tournament selection the assurement that the individuals are of size 1!
    if individual.size > 1:
        mut_point = random.randint(1, individual.size - 1)

        offs = individual.remove_block(mut_point)

        if individual.train_semantics != None:
            #TODO if train semantics is a tensor of tensors this needs to change
            offs.train_semantics = torch.stack([*individual.train_semantics[:mut_point], *individual.train_semantics[mut_point+1:]])
        if individual.test_semantics != None:
            offs.test_semantics = torch.stack([*individual.test_semantics[:mut_point], *individual.test_semantics[mut_point+1:]])
    else:
        offs = Individual(individual.collection)

    return offs
