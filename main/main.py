import logging
import os
import torch

from problem_instance import FUNCTIONS, CONSTANTS, get_terminals
from algorithms.gp import GP
from evaluators.fitness_functions import rmse
from operators.initializers import rhh
from operators.crossover_operators import crossover_trees
from operators.mutators import mutate_tree_node, mutate_tree_subtree
from operators.selection_algorithms import tournament_selection_min
import datasets.data_loader as ds
from datasets.data_loader import *

########################################################################################################################

                                            # DATASETS & ALGORITHMS

########################################################################################################################

# creating a list with the datasets that are to be benchmarked

datas = ["ld50", "bioav", "ppb", "boston", "concrete_slump", "concrete_slump", "forest_fires", \
"efficiency_cooling", "diabetes", "parkinson_updrs", "efficiency_heating"]

# datas = ["ppb"]

# obtaining the data loading functions using the dataset name
data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]

# defining the names of the algorithms to be run

algos = ["StandardGP"]
########################################################################################################################

                                            # SETTING PARAMETERS

########################################################################################################################

# setting up the overall parameter dictionaries:
n_runs = 30
settings_dict = {"p_test": 0.2}

solve_parameters = {"elitism": True,
                    "log": 1,
                    "verbose": 1,
                    "test_elite": True,
                    "log_path": os.path.join(os.getcwd(), "log", "logger.csv"),
                    "run_info": None,
                    "max_depth": 17,
                    "max_": False,
                    "ffunction": rmse,
                    "n_iter": 100
                    }

GP_parameters = {"initializer": rhh,
                  "selector": tournament_selection_min(2),
                  "crossover": crossover_trees(FUNCTIONS),
                  "p_xo": 0,
                  "pop_size": 100,
                  "settings_dict": settings_dict,
    }
GP_parameters["p_m"] = 1 - GP_parameters["p_xo"]

pi_init = {'size': GP_parameters["pop_size"],
           'depth': 8,
           'FUNCTIONS': FUNCTIONS,
           'CONSTANTS': CONSTANTS,
           "p_c": 0.3}

########################################################################################################################

                                            # RUNNING THE ALGORITHM & DEFINING
                                            #    DATA-DEPENDANT PARAMETERS

########################################################################################################################

# for each dataset
for loader in data_loaders:
    # getting the name of the dataset
    dataset = loader.__name__.split("load_")[-1]

    # getting the terminals and defining the terminal-dependant parameters
    TERMINALS = get_terminals(loader)
    pi_init["TERMINALS"] = TERMINALS
    GP_parameters["mutator"] = mutate_tree_subtree(pi_init['depth'], TERMINALS, CONSTANTS, FUNCTIONS,
                                                   p_c=pi_init['p_c'])

    # for each dataset, run all the planned algorithms
    for algo in algos:
        # adding the dataset name and algorithm name to the run info for the logger
        solve_parameters['run_info'] = [dataset, algo]

        # running each dataset + algo configuration n_runs times
        for seed in range(n_runs):
            optimizer = GP(pi_init=pi_init, **GP_parameters, seed=seed)
            optimizer.solve(dataset_loader=loader, **solve_parameters)
