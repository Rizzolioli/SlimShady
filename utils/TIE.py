import random

from algorithms.SLIM_GSGP.representations.individual import Individual
import torch
import itertools

def calculate_tie_inflate(elite, ffunction, y_train, operator, find_elit_func, mutator, X, ms_generator, neigh_size,
                        max_depth=8, p_c=0.1, grow_probability=1): # TODO: add the grow probab parameter to slim

    # obtaining the deflated neighbourhood of the individual by iterating through all of its blocks as possible
    # mutation points:

    neighbourhood = [mutator(individual=elite,
                            ms=ms_generator(),
                            X=X,
                            max_depth=max_depth,
                            p_c=p_c,
                            X_test=None,
                            grow_probability = grow_probability,
                            reconstruct = False) for _ in range(neigh_size)]

    # evaluating all the neighbours
    [neighbour.evaluate(ffunction, y=y_train, testing=False, operator=operator) for neighbour in neighbourhood]

    # determining if we are facing a minimization or a maximization problem
    comparator = compare_best_max if "max" in find_elit_func.__name__.lower() else compare_best_min

    # returning the % of neighbours that are better than the current elite (i.e., the TIE value)
    return len([neighbour for neighbour in neighbourhood if comparator(elite, neighbour) == neighbour.fitness]) / \
           len(neighbourhood) if len(neighbourhood) > 0 else 0

def calculate_tie_deflate(elite, ffunction, y_train, operator, find_elite_func, neigh_size):
    # obtaining the deflated neighbourhood of the individual by iterating through all of its blocks as possible
    # mutation points:
    neighbourhood = [Individual(collection=None,
                                train_semantics=torch.stack([*elite.train_semantics[:mut_point],
                                                             *elite.train_semantics[mut_point + 1:]]),
                                test_semantics=None,
                                reconstruct=False) for mut_point in range(1, elite.size - 1)]

    if len(neighbourhood) > neigh_size:
        neighbourhood = random.sample(neighbourhood, neigh_size)

    # evaluating all the neighbours
    [neighbour.evaluate(ffunction, y=y_train, testing=False, operator=operator) for neighbour in neighbourhood]

    # determining if we are facing a minimization or a maximization problem
    comparator = compare_best_max if "max" in find_elite_func.__name__.lower() else compare_best_min

    # returning the % of neighbours that are better than the current elite (i.e., the TIE value)
    return len([neighbour for neighbour in neighbourhood if comparator(elite, neighbour) == neighbour.fitness]) / \
           len(neighbourhood) if len(neighbourhood) > 0 else 0, len(neighbourhood)

def calculate_tie_deflate_nbt(elite, ffunction, y_train, operator, find_elite_func, neigh_size):

    # obtaining a list of all the offspring block removal combo
    blocks_to_remove = get_all_block_combos(elite)

    # removing all the back tracking block id lists (e.g., [3,4,5]) from the blocks_to_remove list:
    blocks_to_remove = [block_idxs for block_idxs in blocks_to_remove if not
                                            all(b - a == 1 for a, b in zip(block_idxs, block_idxs[1:])) or len(block_idxs) == 1]

    # obtaining all the training semantics of the neighborus in the possible semantic neighbourhood
    neighbourhood = [torch.stack([sub_training for idx, sub_training in enumerate(elite.train_semantics) if idx not in sublist])
                                        for sublist in blocks_to_remove]

    if len(neighbourhood) > neigh_size:
        neighbourhood = random.sample(neighbourhood, neigh_size)

    # creating neighbours as individuals based off of the training semantics
    neighbourhood = [Individual(collection=None,
                                train_semantics=semantics,
                                test_semantics=None,
                                reconstruct=False) for semantics in neighbourhood]
    # evaluating all the neighbours
    [neighbour.evaluate(ffunction, y=y_train, testing=False, operator=operator) for neighbour in neighbourhood]

    # determining if we are facing a minimization or a maximization problem
    comparator = compare_best_max if "max" in find_elite_func.__name__.lower() else compare_best_min

    # returning the % of neighbours that are better than the current elite (i.e., the TIE value)
    return len([neighbour for neighbour in neighbourhood if comparator(elite, neighbour) == neighbour.fitness]) /  \
           len(neighbourhood) if len(neighbourhood) > 0 else 0, len(neighbourhood)

# TODO: reformat the files so that this is in utils but no circular import issues emerge
def compare_best_max(ind1, ind2):
    return ind1.fitness if ind1.fitness > ind2.fitness else ind2.fitness

def compare_best_min(ind1, ind2):
    return ind1.fitness if ind1.fitness < ind2.fitness else ind2.fitness

def get_all_block_combos(elite):

    # determining the maximum index of the block to be eliminated
    n = elite.size - 1

    # returning all combinations of removed blocks possible
    return [list(x) for r in range(1, n + 1) for x in itertools.combinations(range(1, n + 1), r)]