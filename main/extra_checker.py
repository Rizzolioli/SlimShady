import random
import time
import uuid

from parametrization import *
from algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
import datasets.data_loader as ds
from utils.utils import get_terminals, train_test_split
from algorithms.SLIM_GSGP.operators.mutators import *
from utils.logger import log_settings
from utils.utils import show_individual

dataset = "energy"

curr_dataset = f"load_{dataset}"

TERMINALS = get_terminals(dataset, 0 + 1)

X_train, y_train = load_preloaded(dataset, seed=0 + 1, training=True, X_y=True)

X_test, y_test = load_preloaded(dataset, seed=0 + 1, training=False, X_y=True)

print(X_test)
