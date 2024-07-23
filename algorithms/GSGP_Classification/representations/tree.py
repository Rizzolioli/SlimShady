from utils.utils import get_labels, get_columns, generate_percentiles
from main.parametrization import LOGICAL_OPERATORS
import torch
import numpy as np

# For testing, delete later
import pandas as pd


class Tree:
    LABELS = None
    COLUMNS = None
    PERCENTILES = None

    def __init__(self, structure, train_semantics, test_semantics, reconstruct=True):
        if structure is not None and reconstruct:
            self.structure = structure  # either repr_ from gp(tuple) or list of pointers

        self.train_semantics = train_semantics
        self.test_semantics = test_semantics

        self.fitness = None
        self.test_fitness = None

        # TODO: Depth calculation in utils and call it here

    def calculate_semantics(self, inputs, testing= False, probabilities=False):
        # Calculate semantics manually (otherwise calculate them on crossover/mutation)
        # Need help to include batch processing of the input
        pass


def create_full_random_tree(depth, LABELS, COLUMNS, PERCENTILES, LOGICAL_OPERATORS):
    col = np.random.choice(list(COLUMNS.keys()))
    # log_op = np.random.choice(list(LOGICAL_OPERATORS.keys()))
    log_op = "between"
    perc_column = PERCENTILES[:, COLUMNS[col]]
    chosen_perc = np.random.randint(1, 99)
    interval = [perc_column[chosen_perc - 1].item(), perc_column[chosen_perc + 1].item()]

    node = (col, log_op, interval)

    if depth <= 2:
        left_subtree, right_subtree = np.random.choice(LABELS, size = 2, replace=False)
    else:
        left_subtree = create_full_random_tree(depth - 1, LABELS, COLUMNS, PERCENTILES, LOGICAL_OPERATORS)
        right_subtree = create_full_random_tree(depth - 1, LABELS, COLUMNS, PERCENTILES, LOGICAL_OPERATORS)

    node = (node, left_subtree, right_subtree)
    return node




if __name__ == '__main__':
    data = pd.read_csv(r"../../../datasets/classification_data/parkinsons.txt", delim_whitespace=True, names = range(22))
    X_train = torch.from_numpy(data.iloc[:,:-1].values)
    y_train = torch.from_numpy(data.iloc[:,-1].values)

    Tree.LABELS = get_labels(y_train)
    Tree.COLUMNS = get_columns(X_train)
    Tree.PERCENTILES = generate_percentiles(X_train)

    print(create_full_random_tree(depth=6,
                                  LABELS=Tree.LABELS,
                                  COLUMNS=Tree.COLUMNS,
                                  PERCENTILES=Tree.PERCENTILES,
                                  LOGICAL_OPERATORS=LOGICAL_OPERATORS))

