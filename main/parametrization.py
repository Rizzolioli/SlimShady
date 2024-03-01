import torch
from utils.utils import protected_div, mean_


from evaluators.fitness_functions import rmse
from operators.initializers import rhh
from operators.crossover_operators import crossover_trees
from operators.selection_algorithms import tournament_selection_min
from datasets.data_loader import *

########################################################################################################################

                                            # TREE PARAMETERS

########################################################################################################################

FUNCTIONS = {
    'add': {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide': {'function': lambda x, y: protected_div(x, y), 'arity': 2},
    'mean': {'function': lambda x, y: mean_(x, y), 'arity': 2},
    'tan': {'function': lambda x: torch.tan(x), 'arity': 1},
    'sin': {'function': lambda x: torch.sin(x), 'arity': 1},
    'cos': {'function': lambda x: torch.cos(x), 'arity': 1},
}

CONSTANTS = {
    'constant_2': lambda x: torch.tensor(2).float(),
    'constant_3': lambda x: torch.tensor(3).float(),
    'constant_4': lambda x: torch.tensor(4).float(),
    'constant_5': lambda x: torch.tensor(5).float(),
    'constant__1': lambda x: torch.tensor(-1).float()
}

########################################################################################################################

                                            # GP & RUN PARAMETERS

########################################################################################################################

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

