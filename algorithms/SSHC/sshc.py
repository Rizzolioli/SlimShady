import time

from algorithms.SLIM_GSGP.representations.individual import Individual
from algorithms.SLIM_GSGP.representations.population import Population
from algorithms.GP.representations.tree_utils import create_grow_random_tree
from algorithms.GP.representations.tree import Tree
from utils.utils import verbose_reporter
from utils.logger import logger
import numpy as np

import torch
import random

class SSHC():

    def __init__(self,
                 X_train,
                 y_train,
                 ffunction,
                 eval_operator,
                 neigh_operator,
                 FUNCTIONS,
                 TERMINALS,
                 CONSTANTS,
                 seed = 42,
                 reconstruct = True,
                 curr_dataset = None,
                 X_test = None,
                 y_test = None,
                 log = 0,
                 log_path = None,
                 run_info = None,
                 verbose = 0,
                 initial_depth = None,
                 individual = None):

        self.X_train = X_train
        self.y_train = y_train
        self.ffunction = ffunction
        self.eval_operator = eval_operator
        self.neigh_operator = neigh_operator
        self.FUNCTIONS = FUNCTIONS
        self.TERMINALS = TERMINALS
        self.CONSTANTS = CONSTANTS

        self.seed = seed
        self.reconstruct = reconstruct
        self.curr_dataset = curr_dataset
        self.X_test  = X_test
        self.y_test = y_test
        self.log = log
        self.log_path = log_path
        self.run_info = run_info
        self.verbose = verbose

        start = time.time()


        if individual is not None:
            self.individual = individual
        else:
            self.individual = Individual([
                Tree(create_grow_random_tree(initial_depth, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS),
                                               self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)])

        if self.individual.train_semantics is None:
            self.individual.calculate_semantics(X_train, testing=False)
        if X_test is not None and self.individual.test_semantics is None:
            self.individual.calculate_semantics(X_test, testing=True)


        if self.individual.fitness is None:
            self.individual.evaluate(ffunction, y=y_train, testing=False, operator=self.eval_operator)
        if X_test is not None and self.individual.test_fitness is None:
            self.individual.evaluate(ffunction, y=y_test, testing=True, operator=self.eval_operator)

        end = time.time()

        if self.verbose != 0:
            verbose_reporter(curr_dataset.split("load_")[-1], 0,  self.individual.fitness, self.individual.test_fitness, end-start,
                             self.individual.nodes_count)

        # if self.log != 0:
        #
        #     add_info = [self.individual.test_fitness, self.individual.nodes_count, self.log]
        #
        #     logger(log_path, 0, self.elite.fitness, end - start, 0,
        #            additional_infos=add_info, run_info=self.run_info, seed=self.seed)

    def solve(self, neighborhood_size, generations, start_gen = 0, early_stopping = None):

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        counter = 0
        for i in range(start_gen, start_gen+generations):

            start = time.time()
            
            neighboors = [self.neigh_operator(self.individual, self.reconstruct) for _ in range(neighborhood_size)]
            neighboors = Population(neighboors)
            neighboors.calculate_semantics(self.X_train)

            neighboors.evaluate(self.ffunction, y=self.y_train, operator=self.eval_operator)

            counter += 1
            if min(neighboors.fit) <= self.individual.fitness:

                counter = 0
                self.individual = neighboors[np.argmin(neighboors.fit)]
                if self.X_test is not None and self.individual.test_fitness is None:
                    self.individual.evaluate(self.ffunction, y=self.y_test, testing=True, operator=self.eval_operator)

            end = time.time()

            if self.verbose != 0:
                verbose_reporter(self.curr_dataset.split("load_")[-1], i, self.individual.fitness,
                                 self.individual.test_fitness, end - start,
                                 self.individual.nodes_count)

            if self.log != 0:
                add_info = [self.individual.test_fitness, self.individual.nodes_count, self.log]

                logger(self.log_path, i, self.individual.fitness, end - start, neighboors.nodes_count,
                       additional_infos=add_info, run_info=self.run_info, seed=self.seed)

            if early_stopping is not None and counter >= early_stopping:
                break








