from slim.main_gp import gp  # import the slim library
from datasets.data_loader import load_ppb  # import the loader for the dataset PPB
from slim.evaluators.fitness_functions import rmse  # import the rmse fitness metric
from slim.utils.utils import train_test_split  # import the train-test split function
from datasets.data_loader import *


data_loaders = [load_yatch, load_airfoil, load_concrete_slump, load_concrete_strength, load_ppb,
                load_bioav, load_ld50]

for loader in data_loaders:

    for i in range(i):
        # Load the PPB dataset
        X, y = loader(X_y=True)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2, seed = i)

        # Split the test set into validation and test sets
        # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

        # Apply the GP algorithm
        final_tree = gp(X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                        dataset_name=loader.__name__.split("load_")[-1], pop_size=100, n_iter=1000,
                        log_path = os.path.join(os.getcwd(), "log", "gp_ieee.csv"), seed = i)
