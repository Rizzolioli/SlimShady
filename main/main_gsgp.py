from parametrization import *
from algorithms.GSGP.gsgp import GSGP
import datasets.data_loader as ds
from utils.utils import get_terminals

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

algos = ["StandardGSGP"]

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
    gsgp_pi_init["TERMINALS"] = TERMINALS

    # for each dataset, run all the planned algorithms
    for algo in algos:
        # adding the dataset name and algorithm name to the run info for the logger
        gsgp_solve_parameters['run_info'] = [algo, dataset]

        # running each dataset + algo configuration n_runs times
        for seed in range(n_runs):
            optimizer = GSGP(pi_init=gsgp_pi_init, **GSGP_parameters, seed=seed)
            optimizer.solve(dataset_loader=loader, **gsgp_solve_parameters)
