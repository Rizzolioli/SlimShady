"""
This script runs the SLIM_GSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import time
import uuid

from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from config.slim_config import *
from utils.logger import log_settings
from utils.utils import get_terminals, train_test_split

ELITES = {}
UNIQUE_RUN_ID = uuid.uuid1()


# todo update how the name is saved and make it coherent with the paper
def slim(datasets: list, ops: list = None, sigs: list = None, ttreess: list = None, n_runs: int = 30,
         pop_size: int = 100, n_iter: int = 100, p_xo: float = 0.0):
    """
    Main function to execute the SLIM GSGP algorithm on specified datasets

    Parameters
    ----------
    datasets : list
        A list of dataset loaders. Each loader can be a string representing the dataset name or another appropriate type.
    ops : list
        A list containing the version of slim that need to be run (additive or multiplicative)
    sigs : list
        A list containing boolean value indicating if the user want to use the SIG version of the algorithm or the NORM
    ttreess : list
        A list containing the boolean value indicating if the users want to use the 2 version of the algorithm or the 1
    n_runs : int, optional
        The number of runs to execute for each dataset and algorithm combination (default is 30).
    pop_size : int, optional
        The population size for the genetic programming algorithm (default is 100).
    n_iter : int, optional
        The number of iterations for the genetic programming algorithm (default is 100).
    p_xo : float, optional
        The probability of crossover in the genetic programming algorithm. Must be a number between 0 and 1 (default is 0.8).

    Returns
    -------
    None
        This function does not return any values. It performs the execution of the StandardGP algorithm and logs the results.    """

    if ttreess is None:
        ttreess = [True, False]
    if sigs is None:
        sigs = [True, False]
    if ops is None:
        ops = ["sum", "mul"]
    assert isinstance(datasets, list), "Input must be a list"
    assert isinstance(n_runs, int), "Input must be a int"
    assert isinstance(pop_size, int), "Input must be a int"
    assert isinstance(n_iter, int), "Input must be a int"
    assert 0 <= p_xo <= 1, "p_xo must be a number between 0 and 1"

    for loader in datasets:
        for algo_name in ["SlimGSGP"]:
            for sig in sigs:
                for ttress in ttreess:
                    slim_GSGP_parameters["two_trees"] = ttress

                    for op in ops:
                        slim_GSGP_parameters["operator"] = op

                        if ttress and sig:
                            continue

                        algo = f'{algo_name}_{1 + slim_GSGP_parameters["two_trees"] * 1}_{slim_GSGP_parameters["operator"]}_{sig}'

                        for seed in range(n_runs):
                            start = time.time()

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

                            slim_GSGP_parameters["p_xo"] = p_xo
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
        path=os.path.join(os.getcwd(), "log", "gsgp_settings.csv"),
        settings_dict=[slim_gsgp_solve_parameters,
                       slim_GSGP_parameters,
                       slim_gsgp_pi_init,
                       settings_dict],
        unique_run_id=UNIQUE_RUN_ID
    )


if __name__ == "__main__":
    datasets = [
        "toxicity"
    ]
    n_runs = 1

    slim(datasets=datasets, n_runs=n_runs, pop_size=1)
