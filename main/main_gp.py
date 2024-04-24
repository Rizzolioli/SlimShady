from parametrization import *
from algorithms.GP.gp import GP
from algorithms.GP.operators.mutators import mutate_tree_subtree
import datasets.data_loader as ds
from utils.utils import get_terminals, train_test_split
from utils.logger import logger, log_settings
from algorithms.GP.representations.tree_utils import tree_pruning, tree_depth
import uuid
import time


########################################################################################################################

                                            # DATASETS & ALGORITHMS

########################################################################################################################

# creating a list with the datasets that are to be benchmarked

datas = ["ld50", "bioav", "ppb", "boston", "concrete_slump", "concrete_slump", "forest_fires", \
"efficiency_cooling", "diabetes", "parkinson_updrs", "efficiency_heating"]

# datas = ["ppb"]

# obtaining the data loading functions using the dataset name
# data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]
# data_loaders = ["toxicity", "concrete", "instanbul", "ppb", "resid_build_sale_price"]
data_loaders = ['toxicity']

# defining the names of the algorithms to be run

algos = ["StandardGP"]

########################################################################################################################

                                            # RUNNING THE ALGORITHM & DEFINING
                                            #    DATA-DEPENDANT PARAMETERS

########################################################################################################################
# attibuting a unique id to the run
unique_run_id = uuid.uuid1()

# for each dataset
for loader in data_loaders:
    # getting the name of the dataset
    #dataset = loader.__name__.split("load_")[-1]

    # getting the terminals and defining the terminal-dependant parameters

    # for each dataset, run all the planned algorithms
    for algo in algos:
        # adding the dataset name and algorithm name to the run info for the logger
        gp_solve_parameters['run_info'] = [algo, unique_run_id ,loader]

        # running each dataset + algo configuration n_runs times
        for seed in range(2):

            start = time.time()

            if isinstance(loader, str):
                # getting the name of the dataset
                dataset = loader

                curr_dataset = f"load_{dataset}"

                TERMINALS = get_terminals(loader, seed+1)

                X_train, y_train = load_preloaded(loader, seed=seed+1, training=True, X_y=True)

                X_test, y_test = load_preloaded(loader, seed=seed+1, training=False, X_y=True)

            gp_pi_init["TERMINALS"] = TERMINALS
            GP_parameters["mutator"] = mutate_tree_subtree(gp_pi_init['init_depth'], TERMINALS, CONSTANTS, FUNCTIONS,
                                                           p_c=gp_pi_init['p_c'])
            gp_solve_parameters["tree_pruner"] = tree_pruning(TERMINALS=TERMINALS, CONSTANTS=CONSTANTS,
                                                              FUNCTIONS=FUNCTIONS,
                                                              p_c=gp_pi_init["p_c"])
            gp_solve_parameters['depth_calculator'] = tree_depth(FUNCTIONS=FUNCTIONS)

            optimizer = GP(pi_init=gp_pi_init, **GP_parameters, seed=seed)
            optimizer.solve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, curr_dataset=curr_dataset, **gp_solve_parameters)

            print(time.time() - start)

log_settings(path=os.path.join(os.getcwd(), "log", "settings.csv"), settings_dict=[globals()[d] for d in all_params["GP"]], unique_run_id=unique_run_id)

