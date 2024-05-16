from algorithms.SLIM_GSGP.representations.individual import Individual
import torch

def calculate_tie_inflate(elite, ffunction, y_train, operator, find_elit_func, mutator, X, ms_generator,
                        max_depth=8, p_c=0.1, grow_probability=1): # TODO: add the grow probab parameter to slim

    # obtaining the deflated neighbourhood of the individual by iterating through all of its blocks as possible
    # mutation points:

    neighbourhood = [mutator(individual=elite,
                            ms_=ms_generator(),
                            X=X,
                            max_depth=max_depth,
                            p_c=p_c,
                            X_test=None,
                            grow_probability = grow_probability,
                            reconstruct = False) for _ in range(1, elite.size - 1)]

    # evaluating all the neighbours
    [neighbour.evaluate(ffunction, y=y_train, testing=False, operator=operator) for neighbour in neighbourhood]

    # determining if we are facing a minimization or a maximization problem
    comparator = compare_best_max if "max" in find_elit_func.__name__.lower() else compare_best_min

    # returning the % of neighbours that are better than the current elite (i.e., the TIE value)
    return len([neighbour for neighbour in neighbourhood if comparator(elite, neighbour) == neighbour.fitness]) / len(
        neighbourhood)

def calculate_tie_deflate(elite, ffunction, y_train, operator, find_elit_func):
    # obtaining the deflated neighbourhood of the individual by iterating through all of its blocks as possible
    # mutation points:
    neighbourhood = [Individual(collection=None,
                                train_semantics=torch.stack([*elite.train_semantics[:mut_point],
                                                             *elite.train_semantics[mut_point + 1:]]),
                                test_semantics=None,
                                reconstruct=False) for mut_point in range(1, elite.size - 1)]

    # evaluating all the neighbours
    [neighbour.evaluate(ffunction, y=y_train, testing=False, operator=operator) for neighbour in neighbourhood]

    # determining if we are facing a minimization or a maximization problem
    comparator = compare_best_max if "max" in find_elit_func.__name__.lower() else compare_best_min

    # returning the % of neighbours that are better than the current elite (i.e., the TIE value)
    return len([neighbour for neighbour in neighbourhood if comparator(elite, neighbour) == neighbour.fitness]) / len(neighbourhood)


def compare_best_max(ind1, ind2):
    return ind1.fitness if ind1.fitness > ind2.fitness else ind2.fitness


def compare_best_min(ind1, ind2):
    return ind1.fitness if ind1.fitness < ind2.fitness else ind2.fitness
