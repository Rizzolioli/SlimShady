import pandas as pd
import numpy as np
import os
from copy import copy
import seaborn as sb
import matplotlib.pyplot as plt

columns = ["algo", "experiment_id", "dataset", "seed", "generation", "training_fitness", "timing", "pop_node_count"]


def get_column_names(log_level=1, base_cols=columns):
    base_cols = copy(base_cols)

    log_level = int(log_level)

    if log_level == 1:
        base_cols.extend(["test_fitness", "nodes_count", "log_level"])

    elif log_level == 2:

        base_cols.extend(["test_fitness", "genotypic_diversity", "phenotipic_diversity",
                          "nodes_count", "log_level"])
    elif log_level == 3:
        base_cols.extend(["test_fitness", "pop_nodes", "pop_fitnesses",
                          "nodes_count", "log_level"])
    else:
        base_cols.extend(["test_fitness", "genotypic_diversity", "phenotipic_diversity", "pop_nodes", "pop_fitnesses",
                          "nodes_count", "log_level"])

    return base_cols


def get_experiment_results(experiment_id=None, logger_name="logger_checking.csv", base_cols=columns,
                           experiment_id_index=1, log_level=1):
    # getting the path to the logger file
    logger = os.path.join(os.getcwd().split("utils")[0], "main", "log", logger_name)

    # seeing what the maximum number of columns in the logger is, as to avoid different logger level errors:
    with open(os.path.join(os.getcwd().split("utils")[0], "main", "log", logger_name), 'r') as temp_f:
        # Read the lines
        lines = temp_f.readlines()

        use_cols = max([len(line.split(",")) for line in lines])

    # loading logger data into a pandas dataframe
    results = pd.read_csv(logger, header=None, index_col=None, names=range(use_cols))

    # getting the experiment id of the last row in the logger data, if -1 is given as the experiment id
    if experiment_id == -1:

        # getting the experiment id of the last experiment
        experiment_id = results[experiment_id_index].iloc[-1]

        # filtering the results to only contain the required experiment_id
        results = results[results[experiment_id_index] == experiment_id].dropna(axis=1)

    # if a specific expriment id was given
    elif isinstance(experiment_id, str):
        results = results[results[experiment_id_index] == experiment_id].dropna(axis=1)

    # if a list of experiment_ids was given
    elif isinstance(experiment_id, list):

        # filtering the results to only contain the required experiment_ids
        results = results[results[experiment_id_index].isin(experiment_id)].dropna(axis=1)

    # if experiment_id is none, return the entire logger dataset
    else:

        # getting the column names from the inffered log level:
        colnames = get_column_names(log_level=log_level, base_cols=base_cols)

        results.columns = colnames

        # returning the results
        return results.drop(columns=["log_level"])

    # getting the column names from the inffered log level:
    colnames = get_column_names(log_level=log_level, base_cols=base_cols)

    results.columns = colnames

    # returning the results
    return results.drop(columns=["log_level"])


def show_results(x_var="generation", y_var="training_fitness", experiment_id=-1, logger_name="logger_checking.csv",
                 colnames=columns, log_level=2, dataset=None):
    # getting the results dataframe
    df = get_experiment_results(experiment_id=experiment_id, logger_name=logger_name, log_level=log_level)

    if y_var == "training_fitness":

        # obtaining the results only for the specific dataset
        if dataset is not None:
            plotting = df[df["dataset"] == dataset]

            # performing a groupby on the variables of interest
            tr_plotting = pd.DataFrame(plotting.groupby([x_var, "algo"])[y_var].median())
            te_plotting = pd.DataFrame(plotting.groupby([x_var, "algo"])['test_fitness'].median())

            # plotting training and testing side by side
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))

            num_algos = len(set([val[-1] for val in tr_plotting.index]))

            sb.lineplot(data=tr_plotting, x=x_var, y=y_var, hue="algo", ax=ax[0],
                        palette=["red", "green", "blue", "gold", "black", "gray"][:num_algos], linewidth=3)
            sb.lineplot(data=te_plotting, x=x_var, y='test_fitness', hue="algo", ax=ax[1]
                        , palette=["red", "green", "blue", "gold", "black", "gray"][:num_algos], linewidth=3)

            ax[0].set_xlabel("generation")
            ax[0].set_ylabel("training fitness")

            ax[1].set_xlabel("generation")
            ax[1].set_ylabel("testing fitness")

            ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            fig.tight_layout()
            fig.subplots_adjust(top=0.8)
            fig.suptitle(f'{dataset.capitalize()}', fontsize=16)
            fig.show()

        else:
            for ds in df.dataset.unique():
                # keeping only one dataset at a time
                plotting = df[df["dataset"] == ds]

                # performing a groupby on the variables of interest
                tr_plotting = pd.DataFrame(plotting.groupby([x_var, "algo"])[y_var].median())
                te_plotting = pd.DataFrame(plotting.groupby([x_var, "algo"])['test_fitness'].median())

                num_algos = len(set([val[-1] for val in tr_plotting.index]))

                # plotting training and testing side by side
                fig, ax = plt.subplots(1, 2, figsize=(14, 5))
                sb.lineplot(data=tr_plotting, x=x_var, y=y_var, hue="algo", ax=ax[0],
                            palette=["red", "green", "blue", "gold", "black", "gray"][:num_algos], linewidth=3)
                sb.lineplot(data=te_plotting, x=x_var, y='test_fitness', hue="algo", ax=ax[1],
                            palette=["red", "green", "blue", "gold", "black", "gray"][:num_algos], linewidth=3)

                ax[0].set_xlabel("generation")
                ax[0].set_ylabel("training fitness")

                ax[1].set_xlabel("generation")
                ax[1].set_ylabel("testing fitness")
                fig.tight_layout()
                fig.subplots_adjust(top=0.8)
                fig.suptitle(f'{ds.capitalize()}', fontsize=16)
                plt.show()

                # added here to see the final median:
                print(tr_plotting[y_var])
                # added here to see the final median:
                print(te_plotting["test_fitness"])

    elif y_var == "nodes_count":

        # obtaining the results only for the specific dataset
        if dataset is not None:
            plotting = df[df["dataset"] == dataset]

            # performing a groupby on the variables of interest
            plotting = pd.DataFrame(plotting.groupby([x_var, "algo"])[y_var].median())
            num_algos = len(set([val[-1] for val in plotting.index]))
            sb.lineplot(data=plotting, x=x_var, y=y_var, hue="algo",
                        palette=["red", "green", "blue", "gold", "black", "gray"][:num_algos], linewidth=3)
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.ylim([0, 1000])
            plt.title(f'{dataset.capitalize()}')
            plt.show()

        else:
            for ds in df.dataset.unique():
                # keeping only one dataset at a time
                plotting = df[df["dataset"] == ds]

                # performing a groupby on the variables of interest
                plotting = pd.DataFrame(plotting.groupby([x_var, "algo"])[y_var].median())
                num_algos = len(set([val[-1] for val in plotting.index]))
                sb.lineplot(data=plotting, x=x_var, y=y_var, hue="algo",
                            palette=["red", "green", "blue", "gold", "black", "gray"][:num_algos], linewidth=3)
                plt.xlabel(x_var)
                plt.ylabel(y_var)
                plt.ylim([0, 1000])
                plt.title(f'{ds.capitalize()}')
                plt.show()

                # added here to see the final median:
                print(plotting[y_var])


    else:
        # obtaining the results only for the specific dataset
        if dataset is not None:
            plotting = df[df["dataset"] == dataset]

            # performing a groupby on the variables of interest
            plotting = pd.DataFrame(plotting.groupby([x_var, "algo"])[y_var].median())
            num_algos = len(set([val[-1] for val in plotting.index]))
            sb.lineplot(data=plotting, x=x_var, y=y_var, hue="algo",
                        palette=["red", "green", "blue", "gold", "black", "gray"][:num_algos], linewidth=3)
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.title(f'{dataset.capitalize()}')
            plt.show()

        else:
            for ds in df.dataset.unique():
                # keeping only one dataset at a time
                plotting = df[df["dataset"] == ds]

                # performing a groupby on the variables of interest
                plotting = pd.DataFrame(plotting.groupby([x_var, "algo"])[y_var].median())
                num_algos = len(set([val[-1] for val in plotting.index]))
                sb.lineplot(data=plotting, x=x_var, y=y_var, hue="algo",
                            palette=["red", "green", "blue", "gold", "black", "gray"][:num_algos], linewidth=3)
                plt.xlabel(x_var)
                plt.ylabel(y_var)
                plt.title(f'{ds.capitalize()}')
                plt.show()


def verify_integrity(df):
    for a in df.algo.unique():
        for s in range(len(df.seed.unique())):
            temp = len(df[(df.algo == a) & (df.seed == s)])
            print(f'for algo {a} and seed {s} we have {temp}')
