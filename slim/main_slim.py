"""
This script runs the SLIM_GSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import time
import uuid

from slim.algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from slim.config.slim_config import *
from slim.utils.logger import log_settings
from slim.utils.utils import get_terminals, train_test_split, check_slim_version, validate_inputs

ELITES = {}
UNIQUE_RUN_ID = uuid.uuid1()


# todo: would not be better to first log the settings and then perform the algorithm?
# todo:update how the name is saved and make it coherent with the paper
# todo: in slim I am not using a crossover probability, is this right?
# todo: the user can customize the p_inflate and ms parameters
def slim(datasets: list, slim_version: str = "SLIM+SIG2", n_runs: int = 30, pop_size: int = 100, n_iter: int = 100,
         elitism: bool = True, n_elites: int = 1, init_depth: int = 6,
         log_path: str = os.path.join(os.getcwd(), "log", "slim.csv")):
    """
    Main function to execute the SLIM GSGP algorithm on specified datasets

    Parameters
    ----------
    datasets : list
        A list of dataset loaders. Each loader can be a string representing the dataset name or another appropriate type.
    slim_version : list
    n_runs : int, optional
        The number of runs to execute for each dataset and algorithm combination (default is 30).
    pop_size : int, optional
        The population size for the genetic programming algorithm (default is 100).
    n_iter : int, optional
        The number of iterations for the genetic programming algorithm (default is 100).
    elitism : bool, optional
        Indicate the presence or absence of elitism.
    n_elites : int, optional
        The number of elites.
    init_depth : int, optional
        The depth value for the initial GP trees population.
    log_path : str, optional
        The path where is created the log directory where results are saved.

    Returns
    -------
    None
        This function does not return any values. It performs the execution of the StandardGP algorithm and logs the results.
    """
    op, sig, trees = check_slim_version(slim_version=slim_version)
    if not op:
        return

    validate_inputs(datasets=datasets, n_runs=n_runs, pop_size=pop_size, n_iter=n_iter, elitism=elitism,
                    n_elites=n_elites, init_depth=init_depth, log_path=log_path)

    for loader in datasets:
        slim_GSGP_parameters["two_trees"] = trees
        slim_GSGP_parameters["operator"] = op

        # if ttress and sig:
        #     continue

        algo = f'{slim_version}'

        for seed in range(n_runs):

            if isinstance(loader, str):
                dataset = loader
                curr_dataset = f"load_{dataset}"
                TERMINALS = get_terminals(loader, seed + 1)

                X_train, y_train = load_preloaded(loader, seed=seed + 1, training=True, X_y=True)
                X_test, y_test = load_preloaded(loader, seed=seed + 1, training=False, X_y=True)
            else:
                X, y = loader(X_y=True)
                curr_dataset = loader.__name__
                dataset = loader.__name__.split("load_")[-1]
                TERMINALS = get_terminals(loader)
                X_train, X_test, y_train, y_test = train_test_split(X=X, y=y,
                                                                    p_test=settings_dict['p_test'],
                                                                    seed=seed)

            if dataset in slim_dataset_params.keys():
                slim_GSGP_parameters["ms"] = slim_dataset_params[dataset]["ms"]
                slim_GSGP_parameters['p_inflate'] = slim_dataset_params[dataset]["p_inflate"]
            else:
                slim_GSGP_parameters["ms"] = slim_dataset_params["other"]["ms"]
                slim_GSGP_parameters['p_inflate'] = slim_dataset_params["other"]["p_inflate"]

            slim_GSGP_parameters['p_deflate'] = 1 - slim_GSGP_parameters['p_inflate']

            slim_gsgp_pi_init["TERMINALS"] = TERMINALS
            slim_gsgp_pi_init["init_pop_size"] = pop_size
            slim_gsgp_pi_init["init_depth"] = init_depth

            slim_GSGP_parameters["p_m"] = 1 - slim_GSGP_parameters["p_xo"]
            slim_GSGP_parameters["pop_size"] = pop_size
            slim_GSGP_parameters["inflate_mutator"] = inflate_mutation(
                FUNCTIONS=FUNCTIONS,
                TERMINALS=TERMINALS,
                CONSTANTS=CONSTANTS,
                two_trees=slim_GSGP_parameters['two_trees'],
                operator=slim_GSGP_parameters['operator'],
                sig=sig
            )

            slim_gsgp_solve_parameters["log_path"] = log_path
            slim_gsgp_solve_parameters["elitism"] = elitism
            slim_gsgp_solve_parameters["n_elites"] = n_elites
            slim_gsgp_solve_parameters["n_iter"] = n_iter
            slim_gsgp_solve_parameters['run_info'] = [algo, UNIQUE_RUN_ID, dataset]

            optimizer = SLIM_GSGP(
                pi_init=slim_gsgp_pi_init,
                **slim_GSGP_parameters,
                seed=seed
            )

            optimizer.solve(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                curr_dataset=curr_dataset,
                **slim_gsgp_solve_parameters
            )

    log_settings(
        path=os.path.join(os.getcwd(), "log", "slim_settings.csv"),
        settings_dict=[slim_gsgp_solve_parameters,
                       slim_GSGP_parameters,
                       slim_gsgp_pi_init,
                       settings_dict],
        unique_run_id=UNIQUE_RUN_ID
    )


if __name__ == "__main__":
    datasets = [
        "instanbul"
    ]
    n_runs = 1

    slim(datasets=datasets, slim_version="SLIM+SIG2", n_runs=n_runs, pop_size=100, n_iter=50)
