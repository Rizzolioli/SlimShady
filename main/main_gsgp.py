"""
This script runs the StandardGSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""

import uuid

from algorithms.GSGP.gsgp import GSGP
# from config.gsgp_config import *
from parametrization import *
from utils.logger import log_settings
from utils.utils import get_terminals


def gsgp(datasets: list, n_runs: int = 30, pop_size: int = 100, n_iter: int = 100, p_xo: float = 0.8):
    """
    Main function to execute the Standard GSGP algorithm on specified datasets

    Parameters
    ----------
    datasets : list
        A list of dataset loaders. Each loader can be a string representing the dataset name or another appropriate type.
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
        This function does not return any values. It performs the execution of the StandardGP algorithm and logs the results.
    """

    assert isinstance(datasets, list), "Input must be a list"
    assert isinstance(n_runs, int), "Input must be a int"
    assert isinstance(pop_size, int), "Input must be a int"
    assert isinstance(n_iter, int), "Input must be a int"
    assert 0 <= p_xo <= 1, "p_xo must be a number between 0 and 1"

    unique_run_id = uuid.uuid1()

    for loader in datasets:
        dataset = loader

        for algo_name in ["StandardGSGP"]:
            gsgp_solve_parameters["run_info"] = [algo_name, unique_run_id, dataset]

            for seed in range(n_runs):
                if isinstance(loader, str):
                    dataset = loader
                    curr_dataset = f"load_{dataset}"
                    TERMINALS = get_terminals(loader, seed + 1)

                    X_train, y_train = load_preloaded(
                        loader, seed=seed + 1, training=True, X_y=True
                    )
                    X_test, y_test = load_preloaded(
                        loader, seed=seed + 1, training=False, X_y=True
                    )

                gsgp_pi_init["TERMINALS"] = TERMINALS
                gsgp_pi_init["init_pop_size"] = pop_size

                GSGP_parameters["p_xo"] = p_xo
                GSGP_parameters["p_m"] = 1 - GSGP_parameters["p_xo"]
                GSGP_parameters["pop_size"] = pop_size

                gsgp_solve_parameters["n_iter"] = n_iter

                optimizer = GSGP(pi_init=gsgp_pi_init, **GSGP_parameters, seed=seed)

                optimizer.solve(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    curr_dataset=curr_dataset,
                    **gsgp_solve_parameters,
                )

    log_settings(
        path=os.path.join(os.getcwd(), "log", "gsgp_settings.csv"),
        settings_dict=[gsgp_solve_parameters,
                       GSGP_parameters,
                       gsgp_pi_init,
                       settings_dict],
        unique_run_id=unique_run_id,
    )


if __name__ == "__main__":
    datasets = [
        "toxicity"
    ]
    n_runs = 1

    gsgp(datasets=datasets, n_runs=n_runs, pop_size=1)
