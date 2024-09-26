import torch
import numpy as np
from algorithms.GSGP_Classification.representations.tree import Tree


def standard_mutation(parent, random_indiv, ms, testing, COLUMNS, PERCENTILES, LOGICAL_OPERATORS, inputs):
    ms = ms // 2

    col = np.random.choice(list(COLUMNS.keys()))
    # log_op = np.random.choice(list(LOGICAL_OPERATORS.keys()))
    log_op = "between"
    perc_column = PERCENTILES[:, COLUMNS[col]]
    chosen_perc = np.random.randint(ms, len(PERCENTILES) - ms)  # TODO: Add sanity check to ms
    interval = torch.FloatTensor([perc_column[chosen_perc - ms].item(), perc_column[chosen_perc + ms].item()])

    # New structure
    node = (col, log_op, interval, parent.structure, random_indiv.structure)

    # New Semantics

    function_name = log_op
    chosen_branch = LOGICAL_OPERATORS[function_name]['function'](
        inputs[:, int(col[1:])],  # Getting column name and then index
        interval  # Getting the interval
    )

    # REMINDER: The condition is meant to change only a small portion
    # FALSE means they keep the semantic of the original parent
    # TRUE means that semantic changes to the random indiv
    if testing is False:
        new_semantics = [parent.train_semantics[i] if not chosen_branch[i] else random_indiv.train_semantics[i]
                         for i in range(len(chosen_branch))]

        new_indiv = Tree(structure=node, train_semantics=new_semantics)
    else:
        new_semantics = [parent.test_semantics[i] if not chosen_branch[i] else random_indiv.test_semantics[i]
                         for i in range(len(chosen_branch))]

        new_indiv = Tree(structure=node, test_semantics=new_semantics)

    return new_indiv


if __name__ == '__main__':
    pass
