from parametrization import *
from algorithms.GP.gp import GP
from algorithms.GP.operators.mutators import mutate_tree_subtree
import datasets.data_loader as ds
from utils.utils import get_terminals, train_test_split
from utils.logger import logger, log_settings
import uuid

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

                                            # RUNNING THE ALGORITHM & DEFINING
                                            #    DATA-DEPENDANT PARAMETERS

########################################################################################################################
# attibuting a unique id to the run
unique_run_id = uuid.uuid1()

# for each dataset
for loader in data_loaders:
    # getting the name of the dataset
    dataset = loader.__name__.split("load_")[-1]

    # getting the terminals and defining the terminal-dependant parameters
    TERMINALS = get_terminals(loader)
    gp_pi_init["TERMINALS"] = TERMINALS
    GP_parameters["mutator"] = mutate_tree_subtree(gp_pi_init['init_depth'], TERMINALS, CONSTANTS, FUNCTIONS,
                                                   p_c=gp_pi_init['p_c'])

    # for each dataset, run all the planned algorithms
    for algo in algos:
        # adding the dataset name and algorithm name to the run info for the logger
        gp_solve_parameters['run_info'] = [algo, unique_run_id ,dataset]

        # running each dataset + algo configuration n_runs times
        for seed in range(1):

            # Loads the data via the dataset loader
            X, y = loader(X_y=True)

            # getting the name of the dataset:
            curr_dataset = loader.__name__

            # Performs train/test split
            X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, p_test=settings_dict['p_test'],
                                                                seed=seed)

            optimizer = GP(pi_init=gp_pi_init, **GP_parameters, seed=seed)
            optimizer.solve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, curr_dataset=curr_dataset, **gp_solve_parameters)


log_settings(path=os.path.join(os.getcwd(), "log", "settings.csv"), settings_dict=[globals()[d] for d in all_params["GP"]], unique_run_id=unique_run_id)

