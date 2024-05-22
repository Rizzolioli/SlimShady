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
    blocks_to_remove = get_all_block_combos(elite, neigh_size)

    # removing all the back tracking block id lists (e.g., [3,4,5]) from the blocks_to_remove list:
    blocks_to_remove = [block_idxs for block_idxs in blocks_to_remove if not
                        all(b - a == 1 for a, b in zip(block_idxs, block_idxs[1:])) or len(block_idxs) == 1]

    # if len(blocks_to_remove) > neigh_size:
    #     blocks_to_remove = random.sample(blocks_to_remove, neigh_size)

    # obtaining all the training semantics of the neighborus in the possible semantic neighbourhood
    neighbourhood = [torch.stack([sub_training for idx, sub_training in enumerate(elite.train_semantics) if idx not in sublist])
                                        for sublist in blocks_to_remove]



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
def calculate_tie(elite, neigh_size, ffunction, y_train, operator, find_elit_func, mutator, mut_params): # TODO: add the grow probab parameter to slim


    if "ms_generator" in mut_params.keys():

        generator = mut_params["ms_generator"]
        mut_params.pop("ms_generator")

        neighbourhood = [mutator(individual=elite,
                                reconstruct = False,
                                ms = generator(),
                                **mut_params) for _ in range(neigh_size)]
    else:
        neighbourhood = [mutator(individual=elite,
                                 reconstruct=False,
                                 **mut_params) for _ in range(neigh_size)]

    neighbourhood = list(filter(lambda x: x != None, neighbourhood))
    # neighbourhood = list(filter(lambda x: torch.equal(x.train_semantics,elite.train_semantics), neighbourhood))

    # Keep track of unique train attributes using a set comprehension

    # Filter individuals based on the unique train attribute using a list comprehension
    # if len(unique_semantics) < neigh_size:
    #     filtered_neigh = [individual for individual in neighbourhood if individual.train_semantics in unique_semantics]
    # else:
    #     filtered_neigh = neighbourhood

    if len(neighbourhood) > 0:

        # evaluating all the neighbours
        [neighbour.evaluate(ffunction, y=y_train, testing=False, operator=operator) for neighbour in neighbourhood]

        unique_fitness = {float(individual.fitness) for individual in neighbourhood}

        # determining if we are facing a minimization or a maximization problem
        # comparator = compare_best_max if "max" in find_elit_func.__name__.lower() else compare_best_min

        # returning the % of neighbours that are better than the current elite (i.e., the TIE value),
        # number of unique neighbors,
        # neighborhood size(without None and copies of the elite)
        return len([neighbour for neighbour in neighbourhood if neighbour.fitness < elite.fitness]) / \
               neigh_size , \
               len(unique_fitness), \
               len(neighbourhood)
    else:

        return 0,0,0

# TODO: reformat the files so that this is in utils but no circular import issues emerge
def compare_best_max(ind1, ind2):
    return ind1.fitness if ind1.fitness > ind2.fitness else ind2.fitness

def compare_best_min(ind1, ind2):
    return ind1.fitness if ind1.fitness < ind2.fitness else ind2.fitness

def get_all_block_combos(elite, max_):

    # determining the maximum index of the block to be eliminated
    n = elite.size - 1
    combinations = [itertools.combinations(range(1, n + 1), r) for r in range(1, n + 1)]

    if len(combinations) > max_:
        combinations = random.sample(combinations, max_)

    # returning all combinations of removed blocks possible
    return [list(x) for x in combinations]