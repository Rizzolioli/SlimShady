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




models = {'DecisionTree' : DecisionTreeClassifier(),
          'SupportVectorMachine' : SVC(),
          'NaiveBayes' : GaussianNB(),
          'LogisticRegression' : LogisticRegression()}

# Parameter Grids
param_grids = {
    'DecisionTree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SupportVectorMachine': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    },
    'NaiveBayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    },
    'LogisticRegression': {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 200, 300]
    }
}

results = {}

# path = '../../GAMETES dataset/data'
# data_loaders = os.listdir(path)

data_results = {}

data = pd.read_csv('../../gwas_cleaned_ordered.csv')

X = data.values[:, :-1]
y = data.values[:, -1]

#
#         # getting the name of the dataset
data_name = 'GWAS'

for model in models.keys():

    print(model)



    X_train, X_test, y_train, y_test = tts_sklearn(X, y, test_size=0.4,  stratify=y)

    estimator = models[model]

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grids[model],
        scoring='f1_weighted',
        n_jobs=10,
        cv=3,
        verbose=True
    )


    grid_search.fit(X_train, y_train)

    print(data_name)
    print(grid_search.best_params_)

    results[model] = grid_search.best_params_

    # data_results

print(results)