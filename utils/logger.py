import csv
import os
from copy import copy
import pandas as pd


def merge_task_log(tmp_log_path, final_log_path):
    """
    Append a single task's private log file onto the shared experiment log,
    then delete it. If a `{tmp_log_path}_sem_gen.csv` companion exists, it is
    folded into `{final_log_path}_sem_gen.csv` the same way.

    Must be called only from the single orchestrating process (never from a
    worker), so the shared log file only ever has one writer and concurrent
    appends from multiple processes can no longer tear/interleave lines.
    """
    def _append(src, dst):
        if not os.path.exists(src):
            return
        with open(src, 'rb') as fsrc, open(dst, 'ab') as fdst:
            fdst.write(fsrc.read())
        os.remove(src)

    _append(tmp_log_path, final_log_path)

    tmp_sem_gen = (tmp_log_path[:-4] if tmp_log_path.endswith('.csv') else tmp_log_path) + '_sem_gen.csv'
    final_sem_gen = (final_log_path[:-4] if final_log_path.endswith('.csv') else final_log_path) + '_sem_gen.csv'
    _append(tmp_sem_gen, final_sem_gen)

def log_settings(path, settings_dict, unique_run_id):

    settings_dict = merge_settings(*settings_dict)

    del settings_dict['TERMINALS']

    infos = [unique_run_id, settings_dict]

    with open(path, 'a', newline='') as file:

        writer = csv.writer(file)
        writer.writerow(infos)

def merge_settings(sd1, sd2, sd3, sd4):
    return {**sd1, **sd2, **sd3, **sd4}

def logger(path, generation, pop_val_fitness, timing, nodes,
           additional_infos=None, run_info=None,  seed=0):
    """
        Logs information into a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        generation : int
            Current generation number.
        pop_val_fitness : float
            Population's validation fitness value.
        timing : float
            Time taken for the process.
        nodes : int
            Count of nodes in the population.
        additional_infos : list, optional
            Contains population's test fitness value(s) and may contain diversity measurements. Defaults to None.
        run_info : list, optional
            Information about the run. Defaults to None.
        seed: int, optional
            The seed that was used in random, numpy and torch libraries.

        Returns
        -------
        None
            Writes data to a CSV file as a log.
    """

    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if run_info != None:
            infos = copy(run_info)
            infos.extend([seed, generation, float(pop_val_fitness), timing, nodes])

        else:
            infos = [seed, generation, float(pop_val_fitness), timing, nodes]

        if additional_infos != None:
            try:
                additional_infos[0] = float(additional_infos[0])
            except:
                additional_infos[0] = "None"
            infos.extend(additional_infos)

        writer.writerow(infos)

def logger_sem_gen(path, generation, tree_repr, train_sem, test_sem, run_info=None, seed=0):
    """Logs elite genotype and semantics to a separate CSV."""
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if run_info is not None:
            infos = copy(run_info)
            infos.extend([seed, generation, tree_repr, train_sem, test_sem])
        else:
            infos = [seed, generation, tree_repr, train_sem, test_sem]
        writer.writerow(infos)


def drop_experiment_from_logger(experiment_id, log_path):

    """
    Eliminates an experiment from the logger csv file. If the given experiment_id is -1, the lastly saved experiment is
    eliminated

    Parameters
    ----------
    experiment_id : str or int
        the expriment id that is to be eliminated. If -1, the most recent experiment is eliminated

    log_path : str
        the path to the file that contains the logging information

    Returns
    -------
    None
        deletes the rows with the corresponding experiment id from the logger csv file.

    """

    logger_data = pd.read_csv(log_path)

    # if we choose the remove the lastly stored experiment
    if experiment_id == -1:
        # finding the experiment id of the last row in the csv file
        experiment_id = logger_data.iloc[-1][1]

    # excluding the logger data with the chosen id
    to_keep = logger_data[logger_data[1] != experiment_id]
    # saving the new excluded dataset
    logger_data.to_csv(log_path, index=False, header=None)

