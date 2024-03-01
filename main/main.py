import logging
import os
import torch

from problem_instance import FUNCTIONS, CONSTANTS, get_terminals
from algorithms.genetic_programming import GP
from evaluators.fitness_functions import rmse
from operators.initializers import rhh
from operators.crossover_operators import crossover_trees
from operators.mutators import mutate_tree_node
from operators.selection_algorithms import tournament_selection_min
import datasets.data_loader as ds
from datasets.data_loader import *

# creating a list with the datasets that are to be benchmarked
datas = ["ld50", "bioav", "ppb", "boston", "concrete_slump", "concrete_slump",
            "forest_fires", "efficiency_cooling", "diabetes", "parkinson_updrs", "efficiency_heating"]

# obtaining the data loading functions using the dataset name
data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]

# setting up the overall parameter dictionaries:
settings_dict = {"p_test": 0.2}

for loader in data_loaders:

    TERMINALS = get_terminals(loader)

    pi_init = {'size': 25,
               'depth': 8,
               'FUNCTIONS': FUNCTIONS,
               'TERMINALS': TERMINALS,
               'CONSTANTS': CONSTANTS,
               "p_c": 0.3}

    GP_parameters = {"initializer": rhh,
                  "selector": tournament_selection_min(2),
                  "mutator": mutate_tree_node(8, TERMINALS, CONSTANTS, FUNCTIONS, p_c=pi_init["p_c"]),
                  "crossover": crossover_trees(FUNCTIONS),
                  "p_m": 0.2,
                  "p_c": 0.8,
                  "pop_size": 100,
                  "settings_dict": settings_dict,

    }

    solve_parameters = {"elitism": True,
                        "log": 1,
                        "verbose": 1,
                        "test_elite": True,
                        "log_path": os.path.join(os.getcwd(), "log","bla.csv"),
                        "run_info": None,
                        "max_depth": 17,
                        "max_": False,
                        "ffunction": rmse,
                        "n_iter": 100
                        }

    for seed in [1, 2]:
        optimizer = GP(pi_init=pi_init, **GP_parameters, seed=seed)
        optimizer.solve(dataset_loader=loader, **solve_parameters)
