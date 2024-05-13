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


def one_tree_delta(operator='sum', sig=False):
    def ot_delta(tr1, ms, testing):

        if sig:
            if testing:
                return torch.mul(ms, torch.sub(torch.mul(2, tr1.test_semantics), 1)) if operator == 'sum' else \
                    torch.add(1, torch.mul(ms, torch.sub(torch.mul(2, tr1.test_semantics), 1)))
            else:

                return torch.mul(ms, torch.sub(torch.mul(2, tr1.train_semantics), 1)) if operator == 'sum' else \
                    torch.add(1, torch.mul(ms, torch.sub(torch.mul(2, tr1.train_semantics), 1)))
        else:

            if testing:
                return torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, torch.abs(
                    tr1.test_semantics))))) if operator == 'sum' else \
                    torch.add(1, torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, torch.abs(tr1.test_semantics))))))
            else:
                return torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, torch.abs(
                    tr1.train_semantics))))) if operator == 'sum' else \
                    torch.add(1,
                              torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, torch.abs(tr1.train_semantics))))))

    ot_delta.__name__ += ('_' + operator + '_' + str(sig))

    return ot_delta


def inflate_mutation(FUNCTIONS, TERMINALS, CONSTANTS, two_trees=True, operator='sum', single_tree_sigmoid=False,
                     sig=False):
    def inflate(individual, ms, X, max_depth=8, p_c=0.1, X_test=None, grow_probability=1, reconstruct = True):

        # getting a random tree

        # getting another random tree if two trees are to be used
        if two_trees:
            # getting two random trees
            random_tree1 = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c,
                                            grow_probability=grow_probability, logistic=True)

            random_tree2 = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c,
                                           grow_probability=grow_probability, logistic=True)

            random_trees = [random_tree1, random_tree2]

            # getting the testing semantics of the random trees
            if X_test is not None:
                [rt.calculate_semantics(X_test, testing=True, logistic=True) for rt in random_trees]

        else:
            # getting one random tree

            # checking if we choose to apply sigmoid to a single tree:

            random_tree1 = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c,
                                           grow_probability=grow_probability,
                                           logistic=single_tree_sigmoid or sig)

            random_trees = [random_tree1]

            if X_test is not None:
                [rt.calculate_semantics(X_test, testing=True, logistic=single_tree_sigmoid or sig) for rt in random_trees]

        # creating the mutation resulting block to be added to the individual
        variator = two_trees_delta(operator=operator) if two_trees else one_tree_delta(operator=operator, sig=sig)
        new_block = Tree(
            structure = [variator, *random_trees, ms],
            train_semantics=variator( *random_trees, ms, testing= False),
            test_semantics=variator( *random_trees, ms, testing = True) if X_test is not None else None,
            reconstruct=True
        )


        # adding the mutation block to the individual
        offs = Individual(collection = [*individual.collection, new_block] if reconstruct else None,
                          train_semantics=torch.stack([*individual.train_semantics,
                                                (new_block.train_semantics
                                                 if new_block.train_semantics.shape != torch.Size([])
                                                 else new_block.train_semantics.repeat(len(X)))]),
                          test_semantics=(torch.stack([*individual.test_semantics,
                                               (new_block.test_semantics
                                                if new_block.test_semantics.shape != torch.Size([])
                                                else new_block.test_semantics.repeat(len(X_test)))]))
                                            if individual.test_semantics is not None
                                            else None,
                          reconstruct=reconstruct
                          )

        offs.size = individual.size + 1
        offs.nodes_collection = [*individual.nodes_collection, new_block.nodes]
        offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

        offs.depth_collection = [*individual.depth_collection, new_block.depth]
        offs.depth_collection[0] += 1
        offs.depth = max([depth - (i - 1) if i != 0 else depth
                          for i, depth in enumerate(offs.depth_collection)]) + (offs.size - 1)

        return offs

    return inflate


def deflate_mutation(individual, reconstruct):

    mut_point = random.randint(1, individual.size - 1)
    offs = Individual(collection = [*individual.collection[:mut_point], *individual.collection[mut_point + 1:]]
                                    if reconstruct else None,
                      train_semantics=torch.stack(
                                [*individual.train_semantics[:mut_point], *individual.train_semantics[mut_point + 1:]]
                      ),
                      test_semantics= torch.stack(
                                [*individual.test_semantics[:mut_point], *individual.test_semantics[mut_point + 1:]]
                      )
                        if individual.test_semantics is not None
                        else None,
                      reconstruct=reconstruct)
    
    offs.size = individual.size - 1
    offs.nodes_collection = [*individual.nodes_collection[:mut_point], *individual.nodes_collection[mut_point+1:] ]
    offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

    offs.depth_collection = [*individual.depth_collection[:mut_point], *individual.depth_collection[mut_point+1:] ]
    offs.depth = max([depth - (i - 1) if i != 0 else depth 
                      for i, depth in enumerate(offs.depth_collection)]) + (offs.size - 1)

    return offs
