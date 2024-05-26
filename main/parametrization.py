from algorithms.GP.operators.initializers import rhh
from algorithms.GP.operators.selection_algorithms import \
    tournament_selection_min
from algorithms.GSGP.operators.crossover_operators import geometric_crossover
from algorithms.GSGP.operators.mutators import standard_geometric_mutation
from algorithms.SLIM_GSGP.operators.mutators import (deflate_mutation,
                                                     inflate_mutation)
from algorithms.SLIM_GSGP.operators.selection_algorithms import \
    tournament_selection_min_slim
from datasets.data_loader import *
from evaluators.fitness_functions import rmse
from utils.utils import (generate_random_uniform, get_best_max, get_best_min,
                         mean_, protected_div)

# Define functions and constants
# todo use a notation coherent (gp or GP). use only one dictionary for the parameters of each algorithm

FUNCTIONS = {
    'add': {'function': torch.add, 'arity': 2},
    'subtract': {'function': torch.sub, 'arity': 2},
    'multiply': {'function': torch.mul, 'arity': 2},
    'divide': {'function': protected_div, 'arity': 2}
}

CONSTANTS = {
    'constant_2': lambda _: torch.tensor(2.0),
    'constant_3': lambda _: torch.tensor(3.0),
    'constant_4': lambda _: torch.tensor(4.0),
    'constant_5': lambda _: torch.tensor(5.0),
    'constant__1': lambda _: torch.tensor(-1.0)
}

# Set parameters
settings_dict = {"p_test": 0.2}


# GSGP solve parameters
gsgp_solve_parameters = {
    "elitism": True,
    "log": 0,
    "verbose": 1,
    "test_elite": True,
    "log_path": os.path.join(os.getcwd(), "log", "gsgp.csv"),
    "run_info": None,
    "ffunction": rmse,
    "n_iter": 2000,
    "reconstruct": False,
    "n_elites": 1,
}

# GSGP parameters
GSGP_parameters = {
    "initializer": rhh,
    "selector": tournament_selection_min(2),
    "crossover": geometric_crossover,
    "ms": generate_random_uniform(0, 1),
    "mutator": standard_geometric_mutation,
    "p_xo": 0,
    "pop_size": 100,
    "settings_dict": settings_dict,
    "find_elit_func": get_best_min
}
GSGP_parameters["p_m"] = 1 - GSGP_parameters["p_xo"]

gsgp_pi_init = {
    'init_pop_size': GSGP_parameters["pop_size"],
    'init_depth': 8,
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0
}

# SLIM GSGP solve parameters
slim_gsgp_solve_parameters = {
    "elitism": True,
    "log": 1,
    "verbose": 1,
    "test_elite": True,
    "log_path": os.path.join(os.getcwd(), "log", "slim_gsgp.csv"),
    "run_info": None,
    "ffunction": rmse,
    "n_iter": 2000,
    "max_depth": None,
    "n_elites": 1,
    "reconstruct": False
}

# SLIM GSGP parameters
slim_GSGP_parameters = {
    "initializer": rhh,
    "selector": tournament_selection_min_slim(2),
    "crossover": geometric_crossover,
    "ms": None,
    "inflate_mutator": None,
    "deflate_mutator": deflate_mutation,
    "p_xo": 0,
    "pop_size": 100,
    "settings_dict": settings_dict,
    "find_elit_func": get_best_min,
    "p_inflate": None,
    "copy_parent": None,
    "operator": None
}
slim_GSGP_parameters["p_m"] = 1 - slim_GSGP_parameters["p_xo"]

slim_gsgp_pi_init = {
    'init_pop_size': slim_GSGP_parameters["pop_size"],
    'init_depth': 6,
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0
}

# Define all parameters
all_params = {
    "SLIM_GSGP": [
        slim_gsgp_solve_parameters,
        slim_GSGP_parameters,
        slim_gsgp_pi_init,
        settings_dict
    ],
}

# Dataset parameters for SLIM GSGP
slim_dataset_params = {
    "toxicity": {
        "p_inflate": 0.1,
        "ms": generate_random_uniform(0, 0.1)
    },
    "concrete": {
        "p_inflate": 0.5,
        "ms": generate_random_uniform(0, 0.3)
    },
    "other": {
        "p_inflate": 0.3,
        "ms": generate_random_uniform(0, 1)
    }
}
