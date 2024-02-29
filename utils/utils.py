import torch
import math
import csv
from copy import copy

"""
Taken from GPOL
"""

def train_test_split(X, y, p_test=0.3, shuffle=True, indices_only=False, seed=0):
    """ Splits X and y tensors into train and test subsets

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
    # Sets the seed before generating partition's indexes
    torch.manual_seed(seed)
    # Generates random indices
    if shuffle:
        indices = torch.randperm(X.shape[0])
    else:
        indices = torch.arange(0, X.shape[0], 1)
    # Splits indices
    split = int(math.floor(p_test * X.shape[0]))
    train_indices, test_indices = indices[split:], indices[:split]

    if indices_only:
        return train_indices, test_indices
    else:
        # Generates train/test partitions
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test

    import csv
    from copy import copy

def verbose_reporter(dataset, generation, pop_val_fitness, pop_test_fitness, timing, nodes):
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
    digits_test_fit = len(str(float(pop_test_fitness)))
    digits_timing = len(str(timing))
    digits_nodes = len(str(nodes))

    if generation == 0:
        print(
            "                                                         Verbose Reporter                                              ")
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------")
        print(
            "|         Dataset         |  Generation  |     Train Fitness     |       Test Fitness       |        Timing          |      Nodes       |")
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------")
        print("|" + " " * 5 + str(dataset) + " " * (20 - digits_dataset) + "|" +
              " " * 7 + str(generation) + " " * (7 - digits_generation) + "|"
              + " " * 3 + str(float(pop_val_fitness))
              + " " * (20 - digits_val_fit) +
              "|" + " " * 3 + str(float(pop_test_fitness)) + " " * (23 - digits_test_fit) + "|" +
              " " * 3 + str(timing) + " " * (21 - digits_timing) + "|" +
              " " * 6 + str(nodes) + " " * (12 - digits_nodes) + "|")
    else:
        print("|" + " " * 5 + str(dataset) + " " * (20 - digits_dataset) + "|" +
              " " * 7 + str(generation) + " " * (7 - digits_generation) + "|"
              + " " * 3 + str(float(pop_val_fitness))
              + " " * (20 - digits_val_fit) + "|"
              + " " * 3 + str(float(pop_test_fitness)) + " " *
              (23 - digits_test_fit) + "|" +
              " " * 3 + str(timing) + " " * (21 - digits_timing) + "|" +
              " " * 6 + str(nodes) + " " * (12 - digits_nodes) + "|")

def logger(path, generation, pop_val_fitness, timing, nodes,
           pop_test_report=None, run_info=None):
    """
        Logs information into a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        generation : int
            Current generation number.
        pop_val_fitness : float
            Population's validation fitness value.
        timing : float
            Time taken for the process.
        nodes : int
            Count of nodes in the population.
        pop_test_report : float or list, optional
            Population's test fitness value(s). Defaults to None.
        run_info : list, optional
            Information about the run. Defaults to None.

        Returns
        -------
        None
            Writes data to a CSV file as a log.
    """

    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if run_info != None:
            infos = copy(run_info)
            infos.extend([generation, float(pop_val_fitness), timing, nodes])

        else:
            infos = [generation, pop_val_fitness, timing]
        if pop_test_report != None and isinstance(pop_test_report, list):
            infos.extend(pop_test_report)
        elif pop_test_report != None:
            infos.extend([pop_test_report])
        writer.writerow(infos)
