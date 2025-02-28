import random
import torch

from algorithms.GP.representations.tree_utils import create_grow_random_tree
from algorithms.GSGP.representations.tree import Tree, find_minimum
from algorithms.SLIM_GSGP.representations.individual import Individual
from utils.utils import get_random_tree, consecutive_final_indexes, modified_abs
import numpy as np


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


def one_tree_delta(operator='sum', sig=False, adjust = False):
    def ot_delta(tr1, ms, testing):

        if sig:
            if testing:
                return torch.mul(ms, torch.sub(torch.mul(2, tr1.test_semantics), 1)) if operator == 'sum' else \
                    torch.add(1, torch.mul(ms, torch.sub(torch.mul(2, tr1.test_semantics), 1)))
            else:

                return torch.mul(ms, torch.sub(torch.mul(2, tr1.train_semantics), 1)) if operator == 'sum' else \
                    torch.add(1, torch.mul(ms, torch.sub(torch.mul(2, tr1.train_semantics), 1)))
        else:

            if adjust: #todo only use approximation
                try:
                    f_min = find_minimum(tr1.structure)
                except:
                    f_min = torch.min(tr1.train_semantics) - torch.std(tr1.train_semantics) if torch.min(tr1.train_semantics) < 0 else torch.min(tr1.train_semantics)
            else:
                f_min = None

            if testing:
                return torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, modified_abs(
                    tr1.test_semantics, f_min=f_min))))) if operator == 'sum' else \
                    torch.add(1, torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, torch.abs(tr1.test_semantics))))))
            else:
                return torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, torch.abs(
                    tr1.train_semantics))))) if operator == 'sum' else \
                    torch.add(1,
                              torch.mul(ms, torch.sub(1, torch.div(2, torch.add(1, modified_abs(tr1.train_semantics, f_min=f_min))))))

    ot_delta.__name__ += ('_' + operator + '_' + str(sig))

    return ot_delta


def inflate_mutation(FUNCTIONS, TERMINALS, CONSTANTS, two_trees=True, operator='sum', single_tree_sigmoid=False,
                     sig=False, adjust = False):
    def inflate(individual, ms, X, max_depth=8, p_c=0.1, X_test=None, grow_probability=1, reconstruct = True,
                terminals_probabilities = None):

        # getting a random tree

        # getting another random tree if two trees are to be used
        if two_trees:
            # getting two random trees
            random_tree1 = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c,
                                            grow_probability=grow_probability, logistic=True, terminals_probabilities=terminals_probabilities,
                                           adjusted_sigmoid=adjust)

            random_tree2 = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c,
                                           grow_probability=grow_probability, logistic=True, terminals_probabilities=terminals_probabilities,
                                           adjusted_sigmoid=adjust)

            random_trees = [random_tree1, random_tree2]

            # getting the testing semantics of the random trees
            if X_test is not None:
                [rt.calculate_semantics(X_test, testing=True, logistic=True,
                                           adjusted_sigmoid=adjust) for rt in random_trees]

        else:
            # getting one random tree

            # checking if we choose to apply sigmoid to a single tree:

            random_tree1 = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c,
                                           grow_probability=grow_probability,
                                           logistic=single_tree_sigmoid or sig,
                                           terminals_probabilities=terminals_probabilities,
                                           adjusted_sigmoid=adjust and (single_tree_sigmoid or sig))

            random_trees = [random_tree1]

            if X_test is not None:
                [rt.calculate_semantics(X_test, testing=True, logistic=single_tree_sigmoid or sig, adjusted_sigmoid=adjust and (single_tree_sigmoid or sig)) for rt in random_trees]

        # creating the mutation resulting block to be added to the individual
        variator = two_trees_delta(operator=operator) if two_trees else one_tree_delta(operator=operator, sig=sig, adjust = adjust)
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
                                            if X_test is not None
                                            else None,
                          reconstruct=reconstruct
                          )

        offs.size = individual.size + 1
        offs.nodes_collection = [*individual.nodes_collection, new_block.nodes]
        offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

        offs.depth_collection = [*individual.depth_collection, new_block.depth]
        offs.depth = max([depth - (i - 1) if i != 0 else depth
                          for i, depth in enumerate(offs.depth_collection)]) + (offs.size - 1)

        return offs

    return inflate


def deflate_mutation(individual, reconstruct, allow_bt = True, mut_point = None):
    
    limit = 1 if allow_bt else 2
    
    if individual.size > limit:

        mut_point = random.randint(1, individual.size - limit) if mut_point is None else mut_point

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

    
    else:
        offs = Individual(collection=individual.collection if reconstruct else None,
                          train_semantics=individual.train_semantics,
                          test_semantics=individual.test_semantics,
                          reconstruct=reconstruct
                          )
        offs.nodes_collection, offs.nodes_count, offs.depth_collection, offs.depth, offs.size = \
            individual.nodes_collection, individual.nodes_count, individual.depth_collection, individual.depth, individual.size

    return offs


def more_blocks_deflate_mutation(individual, reconstruct, allow_bt=True, blocks_to_remove = None):

    #deciding whihch points to drop is the same as deciding which points to keep

    limit = 1 if allow_bt else 2

    if individual.size > limit:

        if blocks_to_remove is None:
            points_to_keep = random.sample(range(1, individual.size-1), random.randint(1, individual.size-2))
            points_to_keep = sorted(points_to_keep)
        else:
            points_to_keep = random.sample(range(1, individual.size-1), blocks_to_remove)
            points_to_keep = sorted(points_to_keep)


        if not allow_bt:
            #check it the selected points will lead to backtracking
            safe_counter = 0
            while consecutive_final_indexes(points_to_keep, individual.size) and safe_counter < 10:
                safe_counter += 1
                points_to_keep = random.sample(range(1, individual.size - 1), random.randint(1, individual.size - 2))
                points_to_keep = sorted(points_to_keep)

        points_to_keep.insert(0, 0)

        offs = Individual(collection=[individual.collection[i] for i in points_to_keep]
        if reconstruct else None,
                          train_semantics=torch.stack(
                              [individual.train_semantics[i] for i in points_to_keep]
                          ),
                          test_semantics=torch.stack(
                              [individual.test_semantics[i] for i in points_to_keep]
                          )
                          if individual.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)

        offs.size = individual.size - 1
        offs.nodes_collection = [individual.nodes_collection[i] for i in points_to_keep]
        offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

        offs.depth_collection = [individual.depth_collection[i] for i in points_to_keep]
        offs.depth = max([depth - (i - 1) if i != 0 else depth
                          for i, depth in enumerate(offs.depth_collection)]) + (offs.size - 1)

    else:
        offs = Individual(collection=individual.collection if reconstruct else None,
                          train_semantics=individual.train_semantics,
                          test_semantics=individual.test_semantics,
                          reconstruct=reconstruct
                          )
        offs.nodes_collection, offs.nodes_count, offs.depth_collection, offs.depth, offs.size = \
            individual.nodes_collection, individual.nodes_count, individual.depth_collection, individual.depth, individual.size

    return offs



def weighted_deflate_mutation(metric, selection = np.random.choice):

    def wdm(individual, reconstruct, allow_bt=False):

        limit = 1 if allow_bt else 2

        probabilities = [metric(individual.head_signed_error, block).item() for block in individual.train_semantics[1:individual.size - limit + 1 ]]
        probabilities = [p/sum(probabilities) for p in probabilities] if sum(probabilities) > 0 else None
        idxs = [i for i in range(1,  individual.size - limit + 1 )]

        if individual.size > limit:

            mut_point = selection(idxs, p = probabilities)

            offs = Individual(collection=[*individual.collection[:mut_point], *individual.collection[mut_point + 1:]]
            if reconstruct else None,
                              train_semantics=torch.stack(
                                  [*individual.train_semantics[:mut_point], *individual.train_semantics[mut_point + 1:]]
                              ),
                              test_semantics=torch.stack(
                                  [*individual.test_semantics[:mut_point], *individual.test_semantics[mut_point + 1:]]
                              )
                              if individual.test_semantics is not None
                              else None,
                              reconstruct=reconstruct)

            offs.size = individual.size - 1
            offs.nodes_collection = [*individual.nodes_collection[:mut_point], *individual.nodes_collection[mut_point + 1:]]
            offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

            offs.depth_collection = [*individual.depth_collection[:mut_point], *individual.depth_collection[mut_point + 1:]]
            offs.depth = max([depth - (i - 1) if i != 0 else depth
                              for i, depth in enumerate(offs.depth_collection)]) + (offs.size - 1)


        else:
            offs = Individual(collection=individual.collection if reconstruct else None,
                              train_semantics=individual.train_semantics,
                              test_semantics=individual.test_semantics,
                              reconstruct=reconstruct
                              )
            offs.nodes_collection, offs.nodes_count, offs.depth_collection, offs.depth, offs.size = \
                individual.nodes_collection, individual.nodes_count, individual.depth_collection, individual.depth, individual.size

        return offs

    return wdm