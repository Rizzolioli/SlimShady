import math
import random

import numpy as np
import torch
from algorithms.GP.representations.tree_utils import (create_full_random_tree,
                                                      create_grow_random_tree)
from algorithms.GSGP.representations.tree import Tree
from datasets.data_loader import load_preloaded
from sklearn.metrics import root_mean_squared_error


def protected_div(x1, x2):
    """Implements the division protected against zero denominator

    Performs division between x1 and x2. If x2 is (or has) zero(s), the
    function returns the numerator's value(s).

    Parameters
    ----------
    x1 : torch.Tensor
        The numerator.
    x2 : torch.Tensor
        The denominator.

    Returns
    -------
    torch.Tensor
        Result of protected division between x1 and x2.
    """
    return torch.where(
        torch.abs(x2) > 0.001,
        torch.div(x1, x2),
        torch.tensor(1.0, dtype=x2.dtype, device=x2.device),
    )


def mean_(x1, x2):
    """
    Compute the mean of two tensors.

    Parameters
    ----------
    x1 : torch.Tensor
        The first tensor.
    x2 : torch.Tensor
        The second tensor.

    Returns
    -------
    torch.Tensor
        The mean of the two tensors.
    """
    return torch.div(torch.add(x1, x2), 2)


def train_test_split(X, y, p_test=0.3, shuffle=True, indices_only=False, seed=0):
    """Splits X and y tensors into train and test subsets

    This method replicates the behaviour of Sklearn's 'train_test_split'.

    Parameters
    ----------
    X : torch.Tensor
        Input data instances,
    y : torch.Tensor
        Target vector.
    p_test : float (default=0.3)
        The proportion of the dataset to include in the test split.
    shuffle : bool (default=True)
        Whether to shuffle the data before splitting.
    indices_only : bool (default=False)
        Whether to return only the indices representing training and test partition.
    seed : int (default=0)
        The seed for random numbers generators.

    Returns
    -------
    X_train : torch.Tensor
        Training data instances.
    y_train : torch.Tensor
        Training target vector.
    X_test : torch.Tensor
        Test data instances.
    y_test : torch.Tensor
        Test target vector.
    train_indices : torch.Tensor
        Indices representing the training partition.
    test_indices : torch.Tensor
        Indices representing the test partition.
    """
    torch.manual_seed(seed)
    if shuffle:
        indices = torch.randperm(X.shape[0])
    else:
        indices = torch.arange(0, X.shape[0], 1)
    split = int(math.floor(p_test * X.shape[0]))
    train_indices, test_indices = indices[split:], indices[:split]

    if indices_only:
        return train_indices, test_indices
    else:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test


def tensor_dimensioned_sum(dim):
    """
    Generate a sum function over a specified dimension.

    Parameters
    ----------
    dim : int
        The dimension to sum over.

    Returns
    -------
    function
    A function that sums tensors over the specified dimension.
    """

    def tensor_sum(input):
        return torch.sum(input, dim)

    return tensor_sum


def verbose_reporter(
    dataset, generation, pop_val_fitness, pop_test_fitness, timing, nodes
):
    """
    Prints a formatted report of generation, fitness values, timing, and node count.

    Parameters
    ----------
    generation : int
        Current generation number.
    pop_val_fitness : float
        Population's validation fitness value.
    pop_test_fitness : float
        Population's test fitness value.
    timing : float
        Time taken for the process.
    nodes : int
        Count of nodes in the population.

    Returns
    -------
    None
        Outputs a formatted report to the console.
    """
    digits_dataset = len(str(dataset))
    digits_generation = len(str(generation))
    digits_val_fit = len(str(float(pop_val_fitness)))
    if pop_test_fitness is not None:
        digits_test_fit = len(str(float(pop_test_fitness)))
        test_text_init = (
            "|"
            + " " * 3
            + str(float(pop_test_fitness))
            + " " * (23 - digits_test_fit)
            + "|"
        )
        test_text = (
            " " * 3 + str(float(pop_test_fitness)) + " " * (23 - digits_test_fit) + "|"
        )
    else:
        digits_test_fit = 4
        test_text_init = "|" + " " * 3 + "None" + " " * (23 - digits_test_fit) + "|"
        test_text = " " * 3 + "None" + " " * (23 - digits_test_fit) + "|"
    digits_timing = len(str(timing))
    digits_nodes = len(str(nodes))

    if generation == 0:
        print("Verbose Reporter")
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            "|         Dataset         |  Generation  |     Train Fitness     |       Test Fitness       |        "
            "Timing          |      Nodes       |"
        )
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            "|"
            + " " * 5
            + str(dataset)
            + " " * (20 - digits_dataset)
            + "|"
            + " " * 7
            + str(generation)
            + " " * (7 - digits_generation)
            + "|"
            + " " * 3
            + str(float(pop_val_fitness))
            + " " * (20 - digits_val_fit)
            + test_text_init
            + " " * 3
            + str(timing)
            + " " * (21 - digits_timing)
            + "|"
            + " " * 6
            + str(nodes)
            + " " * (12 - digits_nodes)
            + "|"
        )
    else:
        print(
            "|"
            + " " * 5
            + str(dataset)
            + " " * (20 - digits_dataset)
            + "|"
            + " " * 7
            + str(generation)
            + " " * (7 - digits_generation)
            + "|"
            + " " * 3
            + str(float(pop_val_fitness))
            + " " * (20 - digits_val_fit)
            + "|"
            + test_text
            + " " * 3
            + str(timing)
            + " " * (21 - digits_timing)
            + "|"
            + " " * 6
            + str(nodes)
            + " " * (12 - digits_nodes)
            + "|"
        )


def get_terminals(data_loader, seed=0):
    """
    Get terminal nodes for a dataset.

    Parameters
    ----------
    data_loader : function or str
        The data loading function or dataset name.
    seed : int, default=0
        The seed for random numbers generators.

    Returns
    -------
    dict
        Dictionary of terminal nodes.
    """
    if isinstance(data_loader, str):
        TERMINALS = {
            f"x{i}": i
            for i in range(
                len(load_preloaded(data_loader, seed, training=True, X_y=True)[0][0])
            )
        }
    else:
        TERMINALS = {f"x{i}": i for i in range(len(data_loader(True)[0][0]))}

    return TERMINALS


def get_best_min(population, n_elites):
    """
    Get the best individuals from the population with the minimum fitness.

    Parameters
    ----------
    population : Population
        The population of individuals.
    n_elites : int
        Number of elites to return.

    Returns
    -------
    list
        List of elite individuals.
    Individual
        Best individual from the elites.
    """
    if n_elites > 1:
        idx = np.argpartition(population.fit, n_elites)
        elites = [population.population[i] for i in idx[:n_elites]]
        return elites, elites[np.argmin([elite.fitness for elite in elites])]

    else:
        elite = population.population[np.argmin(population.fit)]
        return [elite], elite


def get_best_max(population, n_elites):
    """
    Get the best individuals from the population with the maximum fitness.

    Parameters
    ----------
    population : Population
        The population of individuals.
    n_elites : int
        Number of elites to return.

    Returns
    -------
    list
        List of elite individuals.
    Individual
        Best individual from the elites.
    """
    if n_elites > 1:
        idx = np.argpartition(population.fit, -n_elites)
        elites = [population.population[i] for i in idx[:-n_elites]]
        return elites, elites[np.argmax([elite.fitness for elite in elites])]

    else:
        elite = population.population[np.argmax(population.fit)]
        return [elite], elite


def get_random_tree(
    max_depth,
    FUNCTIONS,
    TERMINALS,
    CONSTANTS,
    inputs,
    p_c=0.3,
    grow_probability=1,
    logistic=True,
):
    """
    Get a random tree using either grow or full method.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    FUNCTIONS : dict
        Dictionary of functions.
    TERMINALS : dict
        Dictionary of terminals.
    CONSTANTS : dict
        Dictionary of constants.
    inputs : torch.Tensor
        Input tensor for calculating semantics.
    p_c : float, default=0.3
        Probability of choosing a constant.
    grow_probability : float, default=1
        Probability of using the grow method.
    logistic : bool, default=True
            Whether to use logistic semantics.

    Returns
    -------
    Tree
        The generated random tree.
    """
    if random.random() < grow_probability:
        tree_structure = create_grow_random_tree(
            max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c
        )
    else:
        tree_structure = create_full_random_tree(
            max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c
        )

    tree = Tree(
        structure=tree_structure,
        train_semantics=None,
        test_semantics=None,
        reconstruct=True,
    )
    tree.calculate_semantics(inputs, testing=False, logistic=logistic)
    return tree


def generate_random_uniform(lower, upper):
    """
    Generate a random number within a specified range using numpy random.uniform.

    Parameters
    ----------
    lower : float
        The lower bound of the range for generating the random number.
    upper : float
        The upper bound of the range for generating the random number.

    Returns
    -------
    function
        A function that when called, generates a random number within the specified range.
    """

    def generate_num():
        return random.uniform(lower, upper)

    generate_num.lower = lower
    generate_num.upper = upper
    return generate_num


def show_individual(tree, operator):
    """
    Display an individual's structure with a specified operator.

    Parameters
    ----------
    tree : Tree
        The tree representing the individual.
    operator : str
        The operator to display ('sum' or 'prod').

    Returns
    -------
    str
        The string representation of the individual's structure.
    """
    op = "+" if operator == "sum" else "*"

    return f" {op} ".join(
        [
            (
                str(t.structure)
                if isinstance(t.structure, tuple)
                else (
                    f"f({t.structure[1].structure})"
                    if len(t.structure) == 3
                    else f"f({t.structure[1].structure} - {t.structure[2].structure})"
                )
            )
            for t in tree.collection
        ]
    )


def gs_rmse(y_true, y_pred):
    """
    Calculate the root mean squared error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        The root mean squared error.
    """
    return root_mean_squared_error(y_true, y_pred[0])


def gs_size(y_true, y_pred):
    """
    Get the size of the predicted values.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    int
        The size of the predicted values.
    """
    return y_pred[1]
