from utils.utils import protected_div, mean_, get_best_min, get_best_max
from evaluators.fitness_functions import rmse
from algorithms.GP.operators.initializers import rhh
from algorithms.GP.operators.crossover_operators import crossover_trees
from algorithms.GSGP.operators.crossover_operators import geometric_crossover
from algorithms.GSGP.operators.mutators import *
from algorithms.GP.operators.selection_algorithms import tournament_selection_min
from datasets.data_loader import *
from algorithms.SLIM_GSGP.operators.mutators import *

########################################################################################################################

                                            # TREE PARAMETERS

########################################################################################################################
# TODO: add the settings to logger - Liah

# TODO: add grid-search code to main - DIOGO

# TODO: Davide add geno and pheno diversity --> logger level

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

                                            # RUN PARAMETERS & settings

########################################################################################################################

n_runs = 30
settings_dict = {"p_test": 0.2}


########################################################################################################################

                                            # GP PARAMETERS

########################################################################################################################

gp_solve_parameters = {"elitism": True,
                    "log": 1,
                    "verbose": 1,
                    "test_elite": True,
                    "log_path": os.path.join(os.getcwd(), "log", "logger.csv"),
                    "run_info": None,
                    "max_depth": 17,
                    "max_": False,
                    "ffunction": rmse,
                    "n_iter": 100,
                    "n_elites": 1
                    }


GP_parameters = {"initializer": rhh,
                  "selector": tournament_selection_min(2),
                  "crossover": crossover_trees(FUNCTIONS),
                  "p_xo": 0.2,
                  "pop_size": 100,
                  "settings_dict": settings_dict,
                 "find_elit_func": get_best_max if gp_solve_parameters["max_"] else get_best_min
    }
GP_parameters["p_m"] = 1 - GP_parameters["p_xo"]

gp_pi_init = {'init_pop_size': GP_parameters["pop_size"], # assuming that the initial population size is the same as the GP pop size
           'init_depth': 8,
           'FUNCTIONS': FUNCTIONS,
           'CONSTANTS': CONSTANTS,
           "p_c": 0.1,
           "p_terminals": 0.5}



########################################################################################################################

                                            # GSGP PARAMETERS

########################################################################################################################


gsgp_solve_parameters = {"elitism": True,
                    "log": 1,
                    "verbose": 1,
                    "test_elite": True,
                    "log_path": os.path.join(os.getcwd(), "log", "logger.csv"),
                    "run_info": None,
                    "max_": False,
                    "ffunction": rmse,
                    "n_iter": 100,
                    "reconstruct": False
                    }

GSGP_parameters = {"initializer": rhh,
                  "selector": tournament_selection_min(2),
                  "crossover": geometric_crossover,
                   "ms" : torch.arange(0.25, 5.25, 0.25, device='cpu'),
                 "mutator" : standard_one_tree_geometric_mutation,
                  "p_xo": 0.8,
                  "pop_size": 100,
                  "settings_dict": settings_dict,
                "find_elit_func": get_best_max if gsgp_solve_parameters["max_"] else get_best_min
    }
GSGP_parameters["p_m"] = 1 - GP_parameters["p_xo"]

gsgp_pi_init = {'init_pop_size': GSGP_parameters["pop_size"],
           'init_depth': 8,
           'FUNCTIONS': FUNCTIONS,
           'CONSTANTS': CONSTANTS,
           "p_c": 0.1}

########################################################################################################################

                                            # SLIM GSGP PARAMETERS

########################################################################################################################

slim_gsgp_solve_parameters = {"elitism": True,
                    "log": 1,
                    "verbose": 1,
                    "test_elite": True,
                    "log_path": os.path.join(os.getcwd(), "log", "logger.csv"),
                    "run_info": None,
                    "max_": False,
                    "ffunction": rmse,
                    "n_iter": 100,
                    "max_depth": 17,
                    "n_elites": 1
                    }

slim_GSGP_parameters = {"initializer": rhh,
                  "selector": tournament_selection_min(2),
                  "crossover": geometric_crossover,
                   "ms" : torch.arange(0.25, 5.25, 0.25, device='cpu'),
                 "inflate_mutator" : None,
                  "deflate_mutator": deflate_mutation,
                  "p_xo": 0,
                  "pop_size": 100,
                  "settings_dict": settings_dict,
                "find_elit_func": get_best_max if slim_gsgp_solve_parameters["max_"] else get_best_min,
                "p_inflate": 0.3
    }

inflate_mutator = two_trees_inflate_mutation
slim_GSGP_parameters['p_deflate'] = 1 - slim_GSGP_parameters['p_inflate']
slim_GSGP_parameters["p_m"] = 1 - GP_parameters["p_xo"]

slim_gsgp_pi_init = {'init_pop_size': GSGP_parameters["pop_size"],
           'init_depth': 8,
           'FUNCTIONS': FUNCTIONS,
           'CONSTANTS': CONSTANTS,
           "p_c": 0.1}

all_params = {"SLIM_GSGP": ["slim_gsgp_solve_parameters", "slim_GSGP_parameters", "slim_gsgp_pi_init", "settings_dict"],
              "GSGP": ["gsgp_solve_parameters", "GSGP_parameters", "gsgp_pi_init", "settings_dict"],
              "GP": ["gp_solve_parameters", "GP_parameters", "gp_pi_init", "settings_dict"]}