from utils.utils import get_labels, get_columns, generate_percentiles, gs_accuracy
from main.parametrization import LOGICAL_OPERATORS
import torch
from representations.tree import Tree, create_full_random_tree

# For testing, delete later
import pandas as pd
from algorithms.GSGP_Classification.operators.mutators import standard_mutation

if __name__ == '__main__':
    data = pd.read_csv(r"../../datasets/classification_data/parkinsons.txt", delim_whitespace=True, names = range(22))
    X_train = torch.from_numpy(data.iloc[:,:-1].values)
    y_train = torch.from_numpy(data.iloc[:,-1].values)

    Tree.LABELS = get_labels(y_train)
    Tree.COLUMNS = get_columns(X_train)
    Tree.PERCENTILES = generate_percentiles(X_train)

    test_indiv = Tree(create_full_random_tree(depth=2, step = 5,
                                              LABELS=Tree.LABELS,
                                              COLUMNS=Tree.COLUMNS,
                                              PERCENTILES=Tree.PERCENTILES,
                                              LOGICAL_OPERATORS=LOGICAL_OPERATORS)
                      )

    test_indiv2 = Tree(create_full_random_tree(depth=2, step=5,
                                               LABELS=Tree.LABELS,
                                               COLUMNS=Tree.COLUMNS,
                                               PERCENTILES=Tree.PERCENTILES,
                                               LOGICAL_OPERATORS=LOGICAL_OPERATORS)
                       )

    test_indiv.calculate_semantics(inputs=X_train)
    test_indiv2.calculate_semantics(inputs=X_train)
    test_offspring = standard_mutation(test_indiv, test_indiv2, ms=10,
                                       testing=False, COLUMNS=Tree.COLUMNS,
                                       PERCENTILES=Tree.PERCENTILES,
                                       LOGICAL_OPERATORS=LOGICAL_OPERATORS,
                                       inputs=X_train)

    print(type(Tree.COLUMNS))
    print(type(Tree.PERCENTILES))
    print(type(Tree.LABELS))
    print(type(LOGICAL_OPERATORS))


    print(test_indiv.structure)
    print(test_indiv2.structure)
    print(test_offspring.structure)
    print(test_indiv.train_semantics)
    print(test_indiv2.train_semantics)
    print(test_offspring.train_semantics)

    print(gs_accuracy(y_true=y_train, y_pred=test_indiv.train_semantics))
    print(gs_accuracy(y_true=y_train, y_pred=test_offspring.train_semantics))
