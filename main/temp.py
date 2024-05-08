
from parametrization import FUNCTIONS, CONSTANTS
from utils.utils import get_terminals, get_best_min, get_best_max
from algorithms.GP.representations.tree_utils import create_full_random_tree
import datasets.data_loader as ds
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.representations.individual import Individual
from algorithms.GSGP.operators.crossover_operators import *
from algorithms.GSGP.operators.mutators import *
from evaluators.fitness_functions import rmse
from algorithms.SLIM_GSGP.representations.population import Population
from datasets.data_loader import *

from utils.utils import train_test_split
import numpy as np
X, y = load_parkinson_updrs(X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X=X, y=y,
                                                    p_test=0.9,
                                                    seed=74)

X_train = X_train.detach().cpu().numpy().tolist()
y_train = y_train.detach().cpu().numpy().tolist()

print(len(X_train))
for i, row in enumerate(X_train):
    row.append(y_train[i])



