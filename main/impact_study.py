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

for seed in range(10):
    for loader in data_loaders:

        X, y = loader(X_y=True)

        # getting the name of the dataset
        dataset = loader.__name__.split("load_")[-1]

        # getting the terminals and defining the terminal-dependant parameters
        TERMINALS = get_terminals(loader)
        # Performs train/test split
        X_train, X_test, y_train, y_test = train_test_split(X=X, y=y,
                                                            p_test=0.2,
                                                            seed=42)

        create_random_slim_ind( blocks = 1000,
                                   X_train = X_train,
                                   X_test = X_test,
                                   FUNCTIONS = FUNCTIONS,
                                   TERMINALS = TERMINALS,
                                   CONSTANTS = CONSTANTS,
                                   dataset_name = dataset,
                                   y_train = None,
                                   y_test = None,
                                   algorithm = (True, 'sum', False),
                                   mutation_step = generate_random_uniform(0,1), initial_depth = 6, seed = seed,
                                   log = 1, log_path = 'log/deep_mutation_impact.csv', verbose = 1)