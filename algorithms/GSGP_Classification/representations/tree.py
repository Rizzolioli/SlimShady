from main.parametrization import LOGICAL_OPERATORS
import torch
import numpy as np

class Tree:
    LABELS = None
    COLUMNS = None
    PERCENTILES = None

    def __init__(self, structure: tuple, train_semantics: list = None, test_semantics: list = None, reconstruct: bool =True):
        if structure is not None and reconstruct:
            self.structure = structure  # either repr_ from gp(tuple) or list of pointers

        self.train_semantics = train_semantics
        self.test_semantics = test_semantics

        self.fitness = None
        self.test_fitness = None

        # TODO: Depth calculation in utils and call it here

    def calculate_semantics(self, inputs: torch.Tensor, testing: bool = False, probabilities=False):
        # Calculate semantics manually (otherwise calculate them on crossover/mutation)
        # Need help to include batch processing of the input
        if testing and self.test_semantics is None:
            self.test_semantics = torch.Tensor(apply_tree(self.structure, inputs))
        elif self.train_semantics is None:
            self.train_semantics = torch.Tensor(apply_tree(self.structure, inputs))


def create_full_random_tree(depth: int, step: int, LABELS: torch.Tensor, COLUMNS: dict, PERCENTILES: torch.Tensor, LOGICAL_OPERATORS: dict) -> tuple:
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

def full(size: int, step:int, depth: int, LABELS: torch.Tensor, COLUMNS:dict, PERCENTILES: torch.Tensor) -> list[tuple]:
    return [create_full_random_tree(depth, step, LABELS, COLUMNS, PERCENTILES, LOGICAL_OPERATORS)
            for _ in range(size)]


def apply_tree(tree: tuple | int, inputs: torch.Tensor) -> int | list[int]:
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


