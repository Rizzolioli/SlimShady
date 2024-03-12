import csv
from copy import copy


def logger(path, generation, pop_val_fitness, timing, nodes,
           pop_test_report=None, run_info=None,  seed=0):
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
        pop_test_report : float or list, optional
            Population's test fitness value(s). Defaults to None.
        run_info : list, optional
            Information about the run. Defaults to None.

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

        if pop_test_report != None:
            infos.extend([float(pop_test_report)])

        writer.writerow(infos)

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
            infos.extend(additional_infos)

        writer.writerow(infos)