import random
import torch
import math
from copy import copy
import numpy as np
from sklearn.metrics import root_mean_squared_error

from algorithms.GP.representations.tree_utils import create_full_random_tree, create_grow_random_tree
from algorithms.GSGP.representations.tree import Tree
from datasets.data_loader import load_preloaded

"""
Taken from GPOL
"""


def protected_div(x1, x2):
    """ Implements the division protected against zero denominator

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
    # if  torch.is_tensor(x2):
    return torch.where(torch.abs(x2) > 0.001, torch.div(x1, x2), torch.tensor(1.0, dtype=x2.dtype, device=x2.device))

    # else:
    #     if x2 < 0:
    #         return 0
    #     else:
    #
    #         return x1/x2


def mean_(x1, x2):
    return torch.div(torch.add(x1, x2), 2)


# def w_mean_(x1, x2):
#
#     r = random.random()
#
#     return torch.add(torch.mul(x1, r), torch.mul(x2, r))


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


"""

Not taken from GPOL

"""


def tensor_dimensioned_sum(dim):
    def tensor_sum(input):
        return torch.sum(input, dim)

    return tensor_sum


def verbose_reporter(first_col, second_col, third_col, fourth_col, fifth_col, sixth_col,
                     first_col_name = 'Dataset', second_col_name = 'Generation', third_col_name = 'Train Fitness',
                     fourth_col_name = 'Test Fitness', fifth_col_name = 'Timing', sixth_col_name = 'Nodes',
                     first_call = False):
    """
        Prints a formatted report of second_col, fitness values, fifth_col, and node count.

        Parameters
        ----------
        second_col : int
            Current second_col number.
        third_col : float
            Population's validation fitness value.
        fourth_col : float
            Population's test fitness value.
        fifth_col : float
            Time taken for the process.
        sixth_col : int
            Count of sixth_col in the population.

        Returns
        -------
        None
            Outputs a formatted report to the console.
    """
    digits_first_col = len(str(first_col))
    digits_second_col = len(str(second_col))
    digits_val_fit = len(str(float(third_col))) if not isinstance(third_col, str) else len(third_col)
    if fourth_col is not None:
        digits_test_fit = len(str(float(fourth_col)))
        test_text_init = "|" + " " * 3 + str(float(fourth_col)) + " " * (23 - digits_test_fit) + "|"
        test_text = " " * 3 + str(float(fourth_col)) + " " * (23 - digits_test_fit) + "|"
    else:
        digits_test_fit = 4
        test_text_init = "|" + " " * 3 + "None" + " " * (23 - digits_test_fit) + "|"
        test_text = " " * 3 + "None" + " " * (23 - digits_test_fit) + "|"
    digits_fifth_col = len(str(fifth_col))
    digits_sixth_col = len(str(sixth_col))

    if first_call:
        print(
            "                                                         Verbose Reporter                                              ")
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------")
        print(
            f"|         {first_col_name}         |  {second_col_name}  |     {third_col_name}     |       {fourth_col_name}       |        {fifth_col_name}          |      {sixth_col_name}       |")
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------")
        print("|" + " " * 5 + str(first_col) + " " * (20 - digits_first_col) + "|" +
              " " * 7 + str(second_col) + " " * (7 - digits_second_col) + "|"
              + " " * 3 + (str(float(third_col)) if not isinstance(third_col, str) else third_col)
              + " " * (20 - digits_val_fit) +
              test_text_init +
              " " * 3 + str(fifth_col) + " " * (21 - digits_fifth_col) + "|" +
              " " * 6 + str(sixth_col) + " " * (12 - digits_sixth_col) + "|")
    else:
        print("|" + " " * 5 + str(first_col) + " " * (20 - digits_first_col) + "|" +
              " " * 7 + str(second_col) + " " * (7 - digits_second_col) + "|"
              + " " * 3 + (str(float(third_col)) if not isinstance(third_col, str) else third_col)
              + " " * (20 - digits_val_fit) + "|"
              + test_text +
              " " * 3 + str(fifth_col) + " " * (21 - digits_fifth_col) + "|" +
              " " * 6 + str(sixth_col) + " " * (12 - digits_sixth_col) + "|")


def get_terminals(data_loader, seed=0):
    if isinstance(data_loader, str):
        TERMINALS = {f"x{i}": i for i in range(len(load_preloaded(data_loader, seed, training=True, X_y=True)[0][0]))}
    else:
        TERMINALS = {f"x{i}": i for i in range(len(data_loader(True)[0][0]))}

    return TERMINALS


def get_best_min(population, n_elites):
    # if more than one elite is to be saved
    if n_elites > 1:
        # getting the indexes of the lower n_elites fitnesses in the population
        idx = np.argpartition(population.fit, n_elites)

        # getting the best n_elites individuals
        elites = [population.population[i] for i in idx[:n_elites]]

        # returning the elites and the best elite from among them
        return elites, elites[np.argmin([elite.fitness for elite in elites])]

    # if only the best individual is to be obtained
    else:

        elite = population.population[np.argmin(population.fit)]

        # returning the elite as the list of elites and the elite as the best in population
        return [elite], elite


def get_best_max(population, n_elites):
    # if more than one elite is to be saved
    if n_elites > 1:
        # getting the indexes of the higher n_elites fitnesses in the population
        idx = np.argpartition(population.fit, -n_elites)

        # getting the best n_elites individuals
        elites = [population.population[i] for i in idx[:-n_elites]]

        # returning the elites and the best elite from among them
        return elites, elites[np.argmax([elite.fitness for elite in elites])]

    # if only the best individual is to be obtained
    else:
        elite = population.population[np.argmax(population.fit)]

        # returning the elite as the list of elites and the elite as the best in population
        return [elite], elite


def get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs, p_c=0.3, grow_probability=1,
                    logistic=True, terminals_probabilities = None):
    # choose between grow and full
    if random.random() < grow_probability:

        # creating a tree using grow
        tree = create_grow_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c, terminals_probabilities = terminals_probabilities)

        #reconstruct set to true to calculate the s
        tree = Tree(structure=tree,
                    train_semantics=None,
                    test_semantics=None,
                    reconstruct=True)

        # calculating the tree semantics
        tree.calculate_semantics(inputs, testing=False, logistic=logistic)

    else:
        # creating a full tree
        tree = create_full_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c)

        tree = Tree(structure=tree,
                    train_semantics=None,
                    test_semantics=None,
                    reconstruct=True)

        # calculating the tree semantics
        tree.calculate_semantics(inputs, testing=False, logistic=logistic)

    return tree


def generate_random_uniform(lower, upper):
    """
    Generate a random number within a specified range using numpy random.uniform.

    Parameters:
    lower (float): The lower bound of the range for generating the random number.
    upper (float): The upper bound of the range for generating the random number.

    Returns:
    function: A function that when called, generates a random number within the specified range.
    """

    def generate_num():
        return random.uniform(lower, upper)
        # return 1.5
    generate_num.lower = lower
    generate_num.upper = upper
    return generate_num


def show_individual(tree, operator):
    op = "+" if operator == "sum" else "*"

    return f" {op} ".join([str(t.structure) if isinstance(t.structure,
                                                          tuple) else f'f({t.structure[1].structure})' if len(
        t.structure) == 3
    else f'f({t.structure[1].structure} - {t.structure[2].structure})' for t in tree.collection])


def gs_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred[0])


def gs_size (y_true, y_pred):
    return y_pred[1]

def consecutive_final_indexes(sampled_indexes, original_list_length):

    # Check if the list is consecutive
    for i in range(1, len(sampled_indexes)):
        if sampled_indexes[i] != sampled_indexes[i - 1] + 1:
            return True  # Found a non-consecutive pair

    # If all elements are consecutive, check if they are at the end
    if sampled_indexes[-1] == original_list_length - 1:
        return False

    return True

def replace_with_nan(tensor, percentage):
    """
    Replace a given percentage of values in a tensor with NaN.

    Args:
    tensor (torch.Tensor): The input tensor.
    percentage (float): The percentage of values to replace with NaN (0 to 100).

    Returns:
    torch.Tensor: The tensor with the specified percentage of values replaced by NaN.
    """
    # Ensure the percentage is between 0 and 100
    if not (0 <= percentage <= 1):
        raise ValueError("Percentage must be between 0 and 100")

    # Calculate the number of elements to replace
    total_elements = tensor.numel()
    num_elements_to_replace = int(total_elements * percentage )

    # Generate random indices to replace
    indices = np.random.choice(total_elements, num_elements_to_replace, replace=False)

    # Flatten the tensor, replace the values, then reshape it back
    flat_tensor = tensor.flatten()
    flat_tensor[indices] = float('nan')

    return flat_tensor.view(tensor.shape)

def replace_extreme_values(tensor, extremity_level):
    """
    Replace values in a 2D tensor that are extreme based on the specified extremity level.

    Args:
        tensor (torch.Tensor): A 2D tensor of shape (m, n).
        extremity_level (float): A value between 0 and 1 indicating the level of extremity.
                                 For example, a value of 0.1 will consider values below the 10th percentile
                                 and above the 90th percentile as extreme.

    Returns:
        torch.Tensor: A tensor with extreme values replaced by NaN.
    """
    if not (0 < extremity_level < 1):
        raise ValueError("Extremity level must be between 0 and 1.")

    # Make a copy of the tensor to avoid modifying the original tensor
    tensor_copy = tensor.clone()

    # Get the number of rows and columns
    num_rows, num_cols = tensor_copy.shape

    # Calculate the lower and upper percentile thresholds for extreme values
    lower_percentile = extremity_level
    upper_percentile = 1 - extremity_level

    # Loop through each column
    for col in range(num_cols):
        # Get the lower and upper thresholds for extreme values
        lower_threshold = torch.quantile(tensor_copy[:, col], lower_percentile)
        upper_threshold = torch.quantile(tensor_copy[:, col], upper_percentile)

        # Replace extreme values with NaN
        tensor_copy[tensor_copy[:, col] < lower_threshold, col] = float('nan')
        tensor_copy[tensor_copy[:, col] > upper_threshold, col] = float('nan')

    return tensor_copy


def add_noise(X, n):
    """
    Adds n random columns to the input tensor X.

    Parameters:
    X (torch.Tensor): The input tensor (matrix).
    n (int): The number of random columns to add.

    Returns:
    torch.Tensor: The new tensor with added random columns.
    """
    # Get the number of rows in X
    rows = X.size(0)



    # Generate n random columns with the same number of rows
    random_columns = torch.randn(rows, n)

    # Concatenate the original tensor with the random columns along the second dimension (columns)
    X_new = torch.cat((X, random_columns), dim=1)

    return X_new

def add_noise_to_random_columns(X, num_columns=1, noise_std=1.0):
    """
    Adds num_columns noisy copies of random columns from the input tensor X.

    Parameters:
    X (torch.Tensor): The input tensor (matrix).
    num_columns (int): The number of noisy columns to add.
    noise_std (float): The standard deviation of the Gaussian noise to be added.

    Returns:
    torch.Tensor: The new tensor with the noisy columns appended.
    """
    # Initialize the new tensor as the original one
    X_new = X.clone()

    for _ in range(num_columns):
        # Randomly select a column index
        col_idx = torch.randint(0, X.size(1), (1,)).item()

        # Copy the selected column
        column = X[:, col_idx].clone().unsqueeze(1)  # Keep it as a 2D tensor

        # Compute the standard deviation of the selected column
        column_std = column.std()

        # Generate Gaussian noise proportional to the column's standard deviation
        noise = torch.randn_like(column) * column_std * noise_std

        # Add noise to the copied column
        noisy_column = column + noise

        # Concatenate the noisy column to the original tensor
        X_new = torch.cat((X_new, noisy_column), dim=1)

    return X_new
