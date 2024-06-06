import random
import time
import uuid

from parametrization import *
from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
import datasets.data_loader as ds
from utils.utils import get_terminals, train_test_split
from algorithms.SLIM_GSGP.operators.mutators import *
from utils.logger import log_settings
from utils.utils import show_individual
from utils.data_creation import create_dataset

########################################################################################################################

# DATASETS & ALGORITHMS

########################################################################################################################

algo_name = "SlimGSGP"

# data_loaders = [ "airfoil", "concrete_slump", "concrete_strength", "ppb", "ld50", "bioavalability", "yatch"]
data_loaders = [load_yatch, load_airfoil, load_concrete_slump, load_concrete_strength, load_ppb,
                load_bioav, load_ld50]

########################################################################################################################

# RUNNING THE ALGORITHM & DEFINING
#    DATA-DEPENDANT PARAMETERS

########################################################################################################################

# saving the elites looks:

elites = {}

# attibuting a unique id to the run
unique_run_id = uuid.uuid1()

n_runs = 10
settings_dict = {"p_test": 0.2}

FUNCTIONS = {
    'add': {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide': {'function': lambda x, y: protected_div(x, y), 'arity': 2}
}

CONSTANTS = {
    'constant_2': lambda x: torch.tensor(2).float(),
    'constant_3': lambda x: torch.tensor(3).float(),
    'constant_4': lambda x: torch.tensor(4).float(),
    'constant_5': lambda x: torch.tensor(5).float(),
    'constant__1': lambda x: torch.tensor(-1).float()
}

slim_gsgp_solve_parameters = {"elitism": True,
                              "log": 1,
                              "verbose": 1,
                              "test_elite": True,
                              "log_path": os.path.join(os.getcwd(), "log", "mut_step.csv"),
                              "run_info": None,
                              "ffunction": rmse,
                              "n_iter": 1000,
                              "max_depth": None,
                              "n_elites": 1,
                              "reconstruct" : False
                              }

slim_GSGP_parameters = {"initializer": rhh,
                        "selector": tournament_selection_min_slim(2),
                        "crossover": None,
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

mutation_parameters ={
"sig": None,
"two_trees": None
}

inflate_mutator = inflate_mutation

slim_GSGP_parameters["p_m"] = 1 - slim_GSGP_parameters["p_xo"]

slim_gsgp_pi_init = {'init_pop_size': slim_GSGP_parameters["pop_size"],
                     'init_depth': 6,
                     'FUNCTIONS': FUNCTIONS,
                     'CONSTANTS': CONSTANTS,
                     "p_c": 0}

all_params = {"SLIM_GSGP": ["slim_gsgp_solve_parameters", "slim_GSGP_parameters", "slim_gsgp_pi_init", "settings_dict"],
              "GSGP": ["gsgp_solve_parameters", "GSGP_parameters", "gsgp_pi_init", "settings_dict"],
              "GP": ["gp_solve_parameters", "GP_parameters", "gp_pi_init", "settings_dict"]}

slim_dataset_params = {"toxicity": {"p_inflate": 0.1, "ms": generate_random_uniform(0, 0.1)},
                        "ld50": {"p_inflate": 0.1, "ms": generate_random_uniform(0, 0.1)},
                       "concrete_strength": {"p_inflate": 0.5, "ms": generate_random_uniform(0, 0.3)},
                       "other": {"p_inflate": 0.3, "ms": generate_random_uniform(0, 1)}}


# saving the elites looks:

elites = {}

# attibuting a unique id to the run
unique_run_id = uuid.uuid1()



for data_function in ['rastrigin', 'sphere', 'rosenbrock' ]:

    for (input_scale, output_scale) in[((0,1), (0,1)), ((0,10), (0,1)), ((0,100), (0,1)), ((0,1000), (0,1)),
                                       ((0,1), (0,10)), ((0,1), (0,100)), ((0,1), (0,1000))]:

        for mut_step in [(0,0.1), (0,1), (0,10), (0, 100), (0, 1000)]:

            for (sig, ttress, op) in [(True, False, "mul"), (False, False, "mul"), (True, True, "sum")]:


                slim_GSGP_parameters["two_trees"] = ttress
                slim_GSGP_parameters["operator"] = op
                # getting the log file name according to the used parameters:

                if (sig, ttress, op) == (True, False, "mul"):
                    algo = 'SLIM*1SIG'
                elif (sig, ttress, op) == (False, False, "mul"):
                    algo = 'SLIM*1NORM'
                else:
                    algo = 'SLIM+2SIG'

                # algo += f'_{str(mut_step)}'

                # algo = f'{algo_name}_{1 + slim_GSGP_parameters["two_trees"] * 1}_{slim_GSGP_parameters["operator"]}' \
                #        f'_{sig}'
                print(sig)
                print(ttress)
                print(op)
                # running each dataset + algo configuration n_runs times
                for seed in range(n_runs):
                    start = time.time()

                    curr_dataset = f'{data_function}_{str(input_scale)}_{str(output_scale)}'
                    X, y = create_dataset(100, 10, input_scale, output_scale, data_function, seed = seed)
                    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()

                    TERMINALS = {f"x{i}": i for i in range(10)}

                    X_train, X_test, y_train, y_test = train_test_split(X=X, y=y,
                                                                        p_test=settings_dict['p_test'],
                                                                        seed=seed)


                    slim_GSGP_parameters["ms"] = generate_random_uniform(*mut_step)
                    slim_GSGP_parameters['p_inflate'] = slim_dataset_params["other"]["p_inflate"]

                    slim_GSGP_parameters['p_deflate'] = 1 - slim_GSGP_parameters['p_inflate']

                    # setting up the dataset related parameters:
                    slim_gsgp_pi_init["TERMINALS"] = TERMINALS

                    slim_GSGP_parameters["inflate_mutator"] = inflate_mutator(FUNCTIONS=FUNCTIONS,
                                                                              TERMINALS=TERMINALS,
                                                                              CONSTANTS=CONSTANTS,
                                                                              two_trees=slim_GSGP_parameters[
                                                                                  'two_trees'],
                                                                              operator=slim_GSGP_parameters[
                                                                                  'operator'],
                                                                              sig=sig)

                    # adding the dataset name and algorithm name to the run info for the logger
                    slim_gsgp_solve_parameters['run_info'] = [algo, str(mut_step), unique_run_id, curr_dataset]

                    optimizer = SLIM_GSGP(pi_init=slim_gsgp_pi_init, **slim_GSGP_parameters, seed=seed)

                    optimizer.solve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                    curr_dataset=curr_dataset,
                                    **slim_gsgp_solve_parameters)

                    print(time.time() - start)

                    # elites[seed] = {"structure": optimizer.elite.structure,
                    #                 "looks": show_individual(optimizer.elite,
                    #                                          operator=slim_GSGP_parameters['operator']),
                    #                 "collection": optimizer.elite.collection}
                    print("THE USED SEED WAS", seed)

log_settings(path=os.path.join(os.getcwd(), "log", "settings.csv"),
             settings_dict=[globals()[d] for d in all_params["SLIM_GSGP"]], unique_run_id=unique_run_id)

# elite_saving_path = os.path.join(os.getcwd(), "log", "elite_looks.txt")
# with open(elite_saving_path, 'w+') as file:
#     file.write(str(elites))
