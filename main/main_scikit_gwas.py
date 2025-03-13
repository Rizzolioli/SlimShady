import time

from datasets.data_loader import *
import csv

from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split as tts_sklearn
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import datetime
import numpy as np



models = {'DecisionTree' : DecisionTreeClassifier,
          'SupportVectorMachine' : SVC,
          'NaiveBayes' : GaussianNB,
          'LogisticRegression' : LogisticRegression
          }

# parameters = {'DecisionTree': {'2w_1000a_0.05her': {'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2},
#                                '2w_1000a_0.1her': {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 10},
#                                '2w_10a_0.4her': {'criterion': 'gini', 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 10},
#                                '2w_100a_0.1her': {'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 10},
#                                '2w_10a_0.2her': {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5},
#                                '2w_5000a_0.1her': {'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 5},
#                                '2w_5000a_0.4her': {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 5},
#                                '2w_1000a_0.2her': {'criterion': 'gini', 'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 5},
#                                '2w_100a_0.4her': {'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 5},
#                                '2w_100a_0.05her': {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 2},
#                                '2w_5000a_0.05her': {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10},
#                                '2w_10a_0.1her': {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2},
#                                '2w_100a_0.2her': {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 5},
#                                '2w_5000a_0.2her': {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2},
#                                '2w_1000a_0.4her': {'criterion': 'gini', 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 5},
#                                '2w_10a_0.05her': {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10}},
#               'SupportVectorMachine': {'2w_1000a_0.05her': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'},
#                                        '2w_1000a_0.1her': {'C': 1, 'gamma': 'scale', 'kernel': 'poly'},
#                                        '2w_10a_0.4her': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
#                                        '2w_100a_0.1her': {'C': 10, 'gamma': 'auto', 'kernel': 'poly'},
#                                        '2w_10a_0.2her': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
#                                        '2w_5000a_0.1her': {'C': 10, 'gamma': 'auto', 'kernel': 'sigmoid'},
#                                        '2w_5000a_0.4her': {'C': 10, 'gamma': 'scale', 'kernel': 'sigmoid'},
#                                        '2w_1000a_0.2her': {'C': 100, 'gamma': 'auto', 'kernel': 'sigmoid'},
#                                        '2w_100a_0.4her': {'C': 10, 'gamma': 'scale', 'kernel': 'poly'},
#                                        '2w_100a_0.05her': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'},
#                                        '2w_5000a_0.05her': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'},
#                                        '2w_10a_0.1her': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
#                                        '2w_100a_0.2her': {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'},
#                                        '2w_5000a_0.2her': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
#                                        '2w_1000a_0.4her': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
#                                        '2w_10a_0.05her': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}},
#               'NaiveBayes': {'2w_1000a_0.05her': {'var_smoothing': 1e-09},
#                              '2w_1000a_0.1her': {'var_smoothing': 1e-09},
#                              '2w_10a_0.4her': {'var_smoothing': 1e-09},
#                              '2w_100a_0.1her': {'var_smoothing': 1e-09},
#                              '2w_10a_0.2her': {'var_smoothing': 1e-09},
#                              '2w_5000a_0.1her': {'var_smoothing': 1e-09},
#                              '2w_5000a_0.4her': {'var_smoothing': 1e-09},
#                              '2w_1000a_0.2her': {'var_smoothing': 1e-09},
#                              '2w_100a_0.4her': {'var_smoothing': 1e-09},
#                              '2w_100a_0.05her': {'var_smoothing': 1e-09},
#                              '2w_5000a_0.05her': {'var_smoothing': 1e-09},
#                              '2w_10a_0.1her': {'var_smoothing': 1e-09},
#                              '2w_100a_0.2her': {'var_smoothing': 1e-09},
#                              '2w_5000a_0.2her': {'var_smoothing': 1e-09},
#                              '2w_1000a_0.4her': {'var_smoothing': 1e-09},
#                              '2w_10a_0.05her': {'var_smoothing': 1e-09}},
#               'LogisticRegression': {'2w_1000a_0.05her': {'C': 10, 'max_iter': 300, 'penalty': 'l2', 'solver': 'sag'},
#                                      '2w_1000a_0.1her': {'C': 0.1, 'max_iter': 100, 'penalty': None, 'solver': 'saga'},
#                                      '2w_10a_0.4her': {'C': 0.1, 'max_iter': 100, 'penalty': 'l1', 'solver': 'saga'},
#                                      '2w_100a_0.1her': {'C': 0.1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'},
#                                      '2w_10a_0.2her': {'C': 0.1, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'},
#                                      '2w_5000a_0.1her': {'C': 1, 'max_iter': 200, 'penalty': 'l2', 'solver': 'sag'},
#                                      '2w_5000a_0.4her': {'C': 0.01, 'max_iter': 100, 'penalty': None, 'solver': 'lbfgs'},
#                                      '2w_1000a_0.2her': {'C': 0.01, 'max_iter': 100, 'penalty': None, 'solver': 'lbfgs'},
#                                      '2w_100a_0.4her': {'C': 0.1, 'max_iter': 100, 'penalty': 'l1', 'solver': 'saga'},
#                                      '2w_100a_0.05her': {'C': 0.1, 'max_iter': 300, 'penalty': 'l2', 'solver': 'saga'},
#                                      '2w_5000a_0.05her': {'C': 1, 'max_iter': 200, 'penalty': 'l2', 'solver': 'sag'},
#                                      '2w_10a_0.1her': {'C': 0.1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'newton-cg'},
#                                      '2w_100a_0.2her': {'C': 0.1, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'},
#                                      '2w_5000a_0.2her': {'C': 100, 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'},
#                                      '2w_1000a_0.4her': {'C': 0.1, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'},
#                                      '2w_10a_0.05her': {'C': 0.1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'}}}

parameters_gwas = {'DecisionTree': {'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 5},
                   'SupportVectorMachine': {'C': 100, 'gamma': 'scale', 'kernel': 'sigmoid'},
                   'NaiveBayes': {'var_smoothing': 1e-09},
                   'LogisticRegression': {'C': 0.01, 'max_iter': 100, 'penalty': 'l2', 'solver': 'newton-cg'}}



now = datetime.datetime.now()
day = now.strftime("%Y%m%d")


########################################################################################################################

# DATASETS & ALGORITHMS

########################################################################################################################

algos = ["SlimGSGP"]

# data = pd.read_csv('../../../gwas_cleaned_ordered.csv')
# data = pd.read_csv('../../../Bicocca/GWAS/data/gwas_cleaned_ordered.csv')
# getting the name of the dataset
dataset = 'GWAS'

# data = pd.read_csv('../../../Bicocca/GWAS/data/gwas_cleaned_ordered.csv')
# dataset = 'GWAS'
# Loads the data via the dataset loader
data = pd.read_csv('../../../Bicocca/GWAS/data/gwas_FINAL_cleaned_ordered.csv')
dataset = 'GWAS_ANALYZED'

X = data.values[:, :-1]
y = data.values[:, -1]

#



X = data.values[:, :-1]
y = data.values[:, -1]


for model in models.keys():


    for seed in range(30):
        start = time.time()

        clf = models[model](**parameters_gwas[model])

        X_train, X_test, y_train, y_test = tts_sklearn(X, y,
                                                       stratify=y,
                                                       test_size=0.2,
                                                       shuffle=True,
                                                       random_state=seed)

        # X_train, X_val, y_train, y_val = tts_sklearn(X_test, y_test,
        #                                                stratify= y_test,
        #                                                test_size=0.25,
        #                                                shuffle = True,
        #                                                random_state=seed)


        clf.fit(X_train, y_train)

        train_pred = clf.predict(X_train)
        train_corr = matthews_corrcoef(y_train, train_pred)

        # val_pred = clf.predict(X_val)
        # val_corr = matthews_corrcoef(y_val, val_pred)

        test_pred = clf.predict(X_test)
        test_corr = matthews_corrcoef(y_test, test_pred)


        # with open(os.path.join(os.getcwd(), "log", f"fixed_tuned_models_gwas_{day}.csv"), 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(
        #         [model, seed, dataset, train_corr,  test_corr])

        with open(os.path.join(os.getcwd(), "log", f"dt_feat_imp_gwas_{day}.csv"), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                np.argpartition(clf.feature_importances_, -20)[-20:])