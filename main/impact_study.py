from main.functions_impact_study import create_random_slim_ind, generate_random_uniform
from datasets.data_loader import *
from utils.utils import get_terminals, train_test_split, protected_div
import torch


FUNCTIONS = {
    'add': {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide': {'function': lambda x, y: protected_div(x, y), 'arity': 2}
}

CONSTANTS = {}

data_loaders = [load_yatch, load_airfoil, load_concrete_slump, load_concrete_strength, load_ppb,
                load_bioav, load_ld50]

slim_dataset_params = {"toxicity": {"p_inflate": 0.1, "ms": generate_random_uniform(0, 0.1)},
                        "ld50": {"p_inflate": 0.1, "ms": generate_random_uniform(0, 0.1)},
                       "concrete_strength": {"p_inflate": 0.5, "ms": generate_random_uniform(0, 0.3)},
                       "other": {"p_inflate": 0.3, "ms": generate_random_uniform(0, 1)}}

for seed in range(100):
    for algo in [ (True, 'sum', False), (False,  "mul", True), (False,  "mul", False)]: #
        for loader in data_loaders:

            X, y = loader(X_y=True)

            # getting the name of the dataset
            dataset = loader.__name__.split("load_")[-1]

            # getting the terminals and defining the terminal-dependant parameters
            TERMINALS = get_terminals(loader)
            # Performs train/test split
            X_train, X_test, y_train, y_test = train_test_split(X=X, y=y,
                                                                p_test=0.2,
                                                                seed=seed)

            create_random_slim_ind( blocks = 1000,
                                       X_train = X_train,
                                       X_test = X_test,
                                       FUNCTIONS = FUNCTIONS,
                                       TERMINALS = TERMINALS,
                                       CONSTANTS = CONSTANTS,
                                       deflate_on = 1000,
                                       dataset_name = dataset,
                                       y_train = None,
                                       y_test = None,
                                       algorithm = algo,
                                       mutation_step = slim_dataset_params[dataset]["ms"] if dataset in slim_dataset_params.keys() else  slim_dataset_params["other"]["ms"],
                                       initial_depth = 6, seed = seed,
                                       log = 1, log_path = 'log/deep_mutation_impact_n.csv', verbose = 1)