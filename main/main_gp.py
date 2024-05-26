"""
This script runs the StandardGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import time
import uuid

from main.algorithms.GP.gp import GP
from main.algorithms.GP.operators.mutators import mutate_tree_subtree
from main.algorithms.GP.representations.tree_utils import tree_depth, tree_pruning
from main.config.gp_config import *
from main.utils.logger import log_settings
from main.utils.utils import get_terminals


def gp(datasets: list, n_runs: int = 30, pop_size: int = 100, n_iter: int = 1000, p_xo: float = 0.8):
    """
    Main function to execute the StandardGP algorithm on specified datasets

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
        for algo in ["StandardGP"]:
            gp_solve_parameters['run_info'] = [algo, unique_run_id, loader]

            for seed in range(n_runs):
                start = time.time()

                if isinstance(loader, str):
                    dataset = loader
                    curr_dataset = f"load_{dataset}"
                    TERMINALS = get_terminals(loader, seed + 1)

                    X_train, y_train = load_preloaded(loader, seed=seed + 1, training=True, X_y=True)
                    X_test, y_test = load_preloaded(loader, seed=seed + 1, training=False, X_y=True)

                gp_pi_init["TERMINALS"] = TERMINALS
                gp_pi_init["init_pop_size"] = pop_size

                GP_parameters["p_xo"] = p_xo
                GP_parameters["p_m"] = 1 - GP_parameters["p_xo"]
                GP_parameters["pop_size"] = pop_size
                GP_parameters["mutator"] = mutate_tree_subtree(
                    gp_pi_init['init_depth'], TERMINALS, CONSTANTS, FUNCTIONS, p_c=gp_pi_init['p_c']
                )

                gp_solve_parameters["n_iter"] = n_iter
                gp_solve_parameters["tree_pruner"] = tree_pruning(
                    TERMINALS=TERMINALS, CONSTANTS=CONSTANTS, FUNCTIONS=FUNCTIONS, p_c=gp_pi_init["p_c"]
                )
                gp_solve_parameters['depth_calculator'] = tree_depth(FUNCTIONS=FUNCTIONS)

                optimizer = GP(pi_init=gp_pi_init, **GP_parameters, seed=seed)
                optimizer.solve(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    curr_dataset=curr_dataset,
                    **gp_solve_parameters
                )

                print(time.time() - start)

    log_settings(
        path=os.path.join(os.getcwd(), "log", "gp_settings.csv"),
        settings_dict=[gp_solve_parameters,
                       GP_parameters,
                       gp_pi_init,
                       settings_dict],
        unique_run_id=unique_run_id,
    )


if __name__ == "__main__":
    datasets = [
        "toxicity"
    ]
    n_runs = 1

    gp(datasets=datasets, n_runs=n_runs, pop_size=100)
