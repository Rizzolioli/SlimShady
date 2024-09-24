from utils.utils import get_labels, get_columns, generate_percentiles, gs_accuracy
from main.parametrization import LOGICAL_OPERATORS
import torch
import numpy as np

# For testing, delete later
import pandas as pd

class Tree:
    LABELS = None
    COLUMNS = None
    PERCENTILES = None

    def __init__(self, structure, train_semantics = None, test_semantics = None, reconstruct=True):
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
        if testing and self.test_semantics is None:
            self.test_semantics = apply_tree(self.structure, inputs)
        elif self.train_semantics is None:
            self.train_semantics = apply_tree(self.structure, inputs)


def create_full_random_tree(depth, step, LABELS, COLUMNS, PERCENTILES, LOGICAL_OPERATORS):
    col = np.random.choice(list(COLUMNS.keys()))
    # log_op = np.random.choice(list(LOGICAL_OPERATORS.keys()))
    log_op = "between"
    perc_column = PERCENTILES[:, COLUMNS[col]]
    chosen_perc = np.random.randint(step, len(PERCENTILES) - step - 1) # TODO: Add sanity check to step
    interval = torch.FloatTensor([perc_column[chosen_perc - step].item(), perc_column[chosen_perc + step].item()])

    if depth <= 2:
        left_subtree, right_subtree = np.random.choice(LABELS, size = 2, replace=False)
    else:
        left_subtree = create_full_random_tree(depth - 1, step, LABELS, COLUMNS, PERCENTILES, LOGICAL_OPERATORS)
        right_subtree = create_full_random_tree(depth - 1, step, LABELS, COLUMNS, PERCENTILES, LOGICAL_OPERATORS)

    node = (col, log_op, interval, left_subtree, right_subtree)
    return node

def full(size, depth, LABELS, COLUMNS, PERCENTILES, LOGICAL_OPERATORS):
    return [create_full_random_tree(depth, LABELS, COLUMNS, PERCENTILES, LOGICAL_OPERATORS)
            for i in range(size)]


def apply_tree(tree: tuple, inputs: torch.Tensor) -> torch.Tensor:
    if isinstance(tree, tuple):  # If it is a function node

        if inputs.dim() == 1:
            x = inputs = inputs.unsqueeze(0)
        else:
            x = inputs

        function_name = tree[1]
        chosen_branch = LOGICAL_OPERATORS[function_name]['function'](
            x[:, int(tree[0][1:])],  # Getting column name and then index
            tree[2]  # Getting the interval
        )
        result = [apply_tree(tree[3], x[i]) if chosen_branch[i]
                  else apply_tree(tree[4], x[i])
                  for i in range(len(chosen_branch))]
        return result
    else:  # If it is a label node
        return tree


# if __name__ == '__main__':
#     data = pd.read_csv(r"../../../datasets/classification_data/parkinsons.txt", delim_whitespace=True, names = range(22))
#     X_train = torch.from_numpy(data.iloc[:,:-1].values)
#     y_train = torch.from_numpy(data.iloc[:,-1].values)
#
#     Tree.LABELS = get_labels(y_train)
#     Tree.COLUMNS = get_columns(X_train)
#     Tree.PERCENTILES = generate_percentiles(X_train)
#
#     test_indiv = Tree(create_full_random_tree(depth=3, step = 5,
#                                           LABELS=Tree.LABELS,
#                                           COLUMNS=Tree.COLUMNS,
#                                           PERCENTILES=Tree.PERCENTILES,
#                                           LOGICAL_OPERATORS=LOGICAL_OPERATORS)
#                       )
#
#     test_indiv2 = Tree(create_full_random_tree(depth=3, step=5,
#                                                LABELS=Tree.LABELS,
#                                                COLUMNS=Tree.COLUMNS,
#                                                PERCENTILES=Tree.PERCENTILES,
#                                                LOGICAL_OPERATORS=LOGICAL_OPERATORS)
#                        )
#
#     test_indiv.calculate_semantics(inputs=X_train)
#     test_indiv2.calculate_semantics(inputs=X_train)
#     test_offspring = standard_mutation(test_indiv, test_indiv2, ms=10, testing=False, COLUMNS=Tree.LABELS,
#                                        PERCENTILES=Tree.PERCENTILES, LOGICAL_OPERATORS=LOGICAL_OPERATORS,
#                                        inputs=X_train)
#
#
#     print(test_indiv.structure)
#     print(test_indiv2.structure)
#     print(test_offspring.structure)
#     print(test_indiv.train_semantics)
#     print(test_indiv2.train_semantics)
#     print(test_offspring.train_semantics)
#
#     print(gs_accuracy(y_true=y_train, y_pred=test_indiv.train_semantics))

