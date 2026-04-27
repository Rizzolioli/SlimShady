from utils.utils import protected_div, get_best_min
from evaluators.fitness_functions import rmse
from algorithms.GP.operators.initializers import rhh
from algorithms.GSGP.operators.crossover_operators import geometric_crossover
from algorithms.SLIM_GSGP.operators.selection_algorithms import tournament_selection_min_slim
from datasets.data_loader import *
from algorithms.SLIM_GSGP.operators.mutators import *
from utils.utils import generate_random_uniform

########################################################################################################################

# TREE PARAMETERS

########################################################################################################################

FUNCTIONS = {
    'add':      {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide':   {'function': lambda x, y: protected_div(x, y), 'arity': 2},
}

CONSTANTS = {
    'constant_2':  lambda x: torch.tensor(2).float(),
    'constant_3':  lambda x: torch.tensor(3).float(),
    'constant_4':  lambda x: torch.tensor(4).float(),
    'constant_5':  lambda x: torch.tensor(5).float(),
    'constant__1': lambda x: torch.tensor(-1).float(),
}

########################################################################################################################

# RUN PARAMETERS

########################################################################################################################

n_runs = 30
settings_dict = {"p_test": 0.2}

########################################################################################################################

# SLIM-GSGP PARAMETERS

########################################################################################################################

slim_gsgp_solve_parameters = {
    "elitism":    True,
    "log":        0,        # 0 = disabled, 1 = enabled
    "verbose":    1,
    "test_elite": True,
    "log_path":   os.path.join(os.getcwd(), "log", "results.csv"),
    "run_info":   None,
    "ffunction":  rmse,
    "n_iter":     2000,
    "max_depth":  None,
    "n_elites":   1,
    "reconstruct": True,
}

slim_GSGP_parameters = {
    "initializer":     rhh,
    "selector":        tournament_selection_min_slim(2),
    "crossover":       geometric_crossover,
    "ms":              None,    # set per-dataset via slim_dataset_params
    "inflate_mutator": None,    # set in main before each run
    "deflate_mutator": deflate_mutation,
    "p_xo":            0,
    "pop_size":        100,
    "settings_dict":   settings_dict,
    "find_elit_func":  get_best_min,
    "p_inflate":       None,    # set per-dataset via slim_dataset_params
    "p_deflate":       None,    # derived as 1 - p_inflate in main
    "copy_parent":     None,
    "operator":        None,    # set per-run: "sum" or "mul"
    "two_trees":       False,
}

slim_GSGP_parameters["p_m"] = 1 - slim_GSGP_parameters["p_xo"]

slim_gsgp_pi_init = {
    'init_pop_size': slim_GSGP_parameters["pop_size"],
    'init_depth':    6,
    'FUNCTIONS':     FUNCTIONS,
    'CONSTANTS':     CONSTANTS,
    "p_c":           0,
}

inflate_mutator = inflate_mutation

all_params = {
    "SLIM_GSGP": ["slim_gsgp_solve_parameters", "slim_GSGP_parameters", "slim_gsgp_pi_init", "settings_dict"],
}

# Dataset-specific inflate probability and mutation step size.
# Datasets not listed here fall back to "other".
slim_dataset_params = {
    "toxicity": {"p_inflate": 0.1, "ms": generate_random_uniform(0, 0.1)},
    "concrete": {"p_inflate": 0.5, "ms": generate_random_uniform(0, 0.3)},
    "other":    {"p_inflate": 0.3, "ms": generate_random_uniform(0, 1)},
}
