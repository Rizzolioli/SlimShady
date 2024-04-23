import uuid
from parametrization import *
from algorithms.GSGP.gsgp import GSGP
import datasets.data_loader as ds
from utils.utils import get_terminals, train_test_split
from utils.logger import log_settings

########################################################################################################################

                                            # DATASETS & ALGORITHMS

########################################################################################################################

# creating a list with the datasets that are to be benchmarked

#datas = ["ld50", "bioav", "ppb", "boston", "concrete_slump", "concrete_slump", "forest_fires", \
#"efficiency_cooling", "diabetes", "parkinson_updrs", "efficiency_heating"]

# datas = ["ppb"]

# obtaining the data loading functions using the dataset name
#data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]

data_loaders = ["instanbul"]

# defining the names of the algorithms to be run

algos = ["StandardGSGP"]

########################################################################################################################

                                            # RUNNING THE ALGORITHM & DEFINING
                                            #    DATA-DEPENDANT PARAMETERS

########################################################################################################################
# attibuting a unique id to the run
unique_run_id = uuid.uuid1()

# for each dataset
for loader in data_loaders:

    # getting the name of the dataset
    dataset = loader

    # getting the terminals and defining the terminal-dependant parameters

    # for each dataset, run all the planned algorithms
    for algo_name in algos:
        # adding the dataset name and algorithm name to the run info for the logger
        gsgp_solve_parameters['run_info'] = [algo_name, unique_run_id ,dataset]

        algo_name = f'{algo_name}'

        # running each dataset + algo configuration n_runs times
        for seed in range(n_runs):

            if isinstance(loader, str):
                # getting the name of the dataset
                dataset = loader

                curr_dataset = f"load_{dataset}"

                TERMINALS = get_terminals(loader, seed + 1)

                X_train, y_train = load_preloaded(loader, seed=seed + 1, training=True, X_y=True)

                X_test, y_test = load_preloaded(loader, seed=seed + 1, training=False, X_y=True)

            gsgp_pi_init["TERMINALS"] = TERMINALS
            optimizer = GSGP(pi_init=gsgp_pi_init, **GSGP_parameters, seed=seed)

            optimizer.solve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, curr_dataset=curr_dataset, **gsgp_solve_parameters)

log_settings(path=os.path.join(os.getcwd(), "log", "settings.csv"), settings_dict=[globals()[d] for d in all_params["GSGP"]], unique_run_id=unique_run_id)

