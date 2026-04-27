import time

from datasets.data_loader import *
import csv

from sklearn.metrics import matthews_corrcoef, accuracy_score, mean_squared_error, root_mean_squared_error
# from sklearn.model_selection import train_test_split as tts_sklearn
from utils.utils import get_terminals, train_test_split, protected_div
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import datetime




models = {'DecisionTree' : DecisionTreeRegressor,
          'SupportVectorMachine' : SVR,
          'MLP' : MLPRegressor,
          'LinRegression' : LinearRegression}



now = datetime.datetime.now()
day = now.strftime("%Y%m%d")


########################################################################################################################

# DATASETS & ALGORITHMS

########################################################################################################################

algos = ["SlimGSGP"]

data_loaders = [load_yatch, load_airfoil, load_concrete_slump, load_concrete_strength, load_ppb,
                load_bioav, load_ld50]

results = {}

for loader in data_loaders:

    # Loads the data via the dataset loader
    X, y = loader(X_y=True)

    # getting the name of the dataset
    dataset = loader.__name__.split("load_")[-1]

    # Performs train/test split

    print(dataset)


    for model in models.keys():

        print(model)


        for seed in range(30):
            start = time.time()

            reg = models[model]()

            X_train, X_test, y_train, y_test = train_test_split(X=X, y=y,
                                                                p_test=0.2,
                                                                seed=seed)


            # X_train, X_val, y_train, y_val = tts_sklearn(X_test, y_test,
            #                                                stratify= y_test,
            #                                                test_size=0.25,
            #                                                shuffle = True,
            #                                                random_state=seed)


            reg.fit(X_train, y_train)

            train_pred = reg.predict(X_train)
            train_corr = root_mean_squared_error(y_train, train_pred)


            test_pred = reg.predict(X_test)
            test_corr = root_mean_squared_error(y_test, test_pred)

            print(train_corr)
            print(test_corr)


            with open(os.path.join(os.getcwd(), "log", f"models_{day}.csv"), 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [model, seed, dataset, train_corr,  test_corr])