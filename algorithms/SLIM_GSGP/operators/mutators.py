import random
import torch

from algorithms.GP.representations.tree_utils import create_grow_random_tree
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.representations.individual import Individual
from utils.utils import get_random_tree


def two_trees_delta(operator='sum'):

    def tt_delta(tr1, tr2, ms, testing):

        if testing:
            return torch.mul(ms, torch.sub(tr1.test_semantics, tr2.test_semantics)) if operator == 'sum' else \
                   torch.add(1, torch.mul(ms, torch.sub(tr1.test_semantics, tr2.test_semantics)))

        else:
            return torch.mul(ms, torch.sub(tr1.train_semantics, tr2.train_semantics)) if operator == 'sum' else \
                   torch.add(1, torch.mul(ms, torch.sub(tr1.train_semantics, tr2.train_semantics)))

    tt_delta.__name__ += ('_' + operator)

    return tt_delta

def one_tree_delta(operator='sum'):

    def ot_delta(tr1, ms, testing):

        if testing:
            return torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, torch.abs(tr1.test_semantics))))) if operator == 'sum' else \
                torch.add(1, torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, torch.abs(tr1.test_semantics))))))
        else:
            return torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, torch.abs(tr1.train_semantics))))) if operator == 'sum' else \
                torch.add(1, torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, torch.abs(tr1.train_semantics))))))

    ot_delta.__name__ += ('_' + operator)

    return ot_delta


def inflate_mutation(FUNCTIONS, TERMINALS, CONSTANTS, two_trees = True, operator = 'sum'):

    def inflate(individual, ms, X, max_depth = 8, p_c = 0.1, X_test = None, p_terminal=0.5, grow_probability=1):

        # getting a random tree



        # getting another random tree if two trees are to be used
        if two_trees:
            # getting two random trees
            random_tree1 = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c,
                                           p_terminal=p_terminal, grow_probability=grow_probability)

            random_tree2 = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c, p_terminal=p_terminal,
                                       grow_probability=grow_probability)

            random_trees = [random_tree1, random_tree2]

            # getting the testing semantics of the random trees
            if X_test is not None:
                [rt.calculate_semantics(X_test, testing=True, logistic=True) for rt in random_trees]

        else:
            # getting one random tree
            random_tree1 = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c,
                                           p_terminal=p_terminal, grow_probability=grow_probability, logistic=False)

            random_trees = [random_tree1]

            if X_test is not None:
                [rt.calculate_semantics(X_test, testing=True, logistic=False) for rt in random_trees]

        # creating the mutation resulting block to be added to the individual
        new_block = Tree([(two_trees_delta(operator=operator) if two_trees else one_tree_delta(operator=operator)),
                         *random_trees, ms])

        # getting the semantics for this new block
        new_block.calculate_semantics(X, testing=False)

        if X_test is not None:

            new_block.calculate_semantics(X_test, testing=True)

        # adding the mutation block to the individual
        offs = individual.add_block(new_block)

        if individual.train_semantics is not None:

            # adding the new block's training semantics to the now mutated individual
            offs.train_semantics = torch.stack([*individual.train_semantics,
                                    (new_block.train_semantics if new_block.train_semantics.shape != torch.Size([])
                                    else new_block.train_semantics.repeat(len(X)))])

        if individual.test_semantics is not None:
            # adding the new block's testing semantics to the now mutated individual, if applicable
            offs.test_semantics = torch.stack([*individual.test_semantics,
                                (new_block.test_semantics if new_block.test_semantics.shape != torch.Size([])
                                 else new_block.test_semantics.repeat(len(X_test)))])

        return offs

    return inflate

def deflate_mutation(individual):

    if individual.size == 1:

        offs = Individual(individual.collection)
        offs.train_semantics = individual.train_semantics
        if individual.test_semantics != None:
            offs.test_semantics = individual.test_semantics

    else:

        mut_point = random.randint(1, individual.size - 1)

        offs = individual.remove_block(mut_point)

        if individual.train_semantics != None:
            offs.train_semantics = torch.stack([*individual.train_semantics[:mut_point], *individual.train_semantics[mut_point+1:]])
        if individual.test_semantics != None:
            offs.test_semantics = torch.stack([*individual.test_semantics[:mut_point], *individual.test_semantics[mut_point+1:]])


    return offs