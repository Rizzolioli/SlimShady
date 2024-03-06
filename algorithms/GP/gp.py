import time
import random
import torch
import numpy as np

from utils.utils import verbose_reporter, logger, train_test_split
from algorithms.GP.representations.population import Population
from algorithms.GP.representations.tree import Tree
from algorithms.GP.representations.tree_utils import tree_pruning, tree_depth


# small fixes - Liah
# TODO: confirm that the mutated tree cannot be used for crossover! CONFIRMED

# TODO: make crossover random subtree uniform distribution (use depth; kill recursion)!
# TODO: change the name of pi_init depth and size to init_depth and init_size
# TODO: make elitism parametrized
# TODO: change pop to population
# TODO: make get best min and get best max
# TODO: change tournament selection names
# TODO: change to only choose one parent if mutation and two parents if xover --> first operator then parents

class GP:

    def __init__(self, pi_init, initializer, selector, mutator, crossover,
                 p_m=0.2, p_xo=0.8, pop_size=100, seed=0,
                 settings_dict=None):

        #other initial parameters, tipo dataset
        self.pi_init = pi_init  # dictionary with all the parameters needed for evaluation
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.mutator = mutator
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed

        self.settings_dict = settings_dict

    def solve(self, n_iter=20, elitism=True, log=0, verbose=0,
              test_elite=False, log_path=None, run_info=None,
              max_depth=None, max_=False, dataset_loader=None,
              ffunction=None):

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        # Loads the data via the dataset loader
        X, y = dataset_loader(X_y=True)

        # getting the name of the dataset:
        curr_dataset = dataset_loader.__name__

        # Performs train/test split
        X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, p_test=self.settings_dict['p_test'], seed=self.seed)

        ################################################################################################################

                                                        # INITIALIZATION #

        ################################################################################################################

        # initializing the population
        pop = Population([Tree(tree,  self.pi_init['FUNCTIONS'], self.pi_init['TERMINALS'], self.pi_init['CONSTANTS'])
                          for tree in self.initializer(**self.pi_init)])
        pop.evaluate(ffunction, X=X_train, y=y_train)

        end = time.time()

        # obtaining the initial population elite
        if max_:
            self.elite = pop.pop[np.argmax(pop.fit)]
        else:
            self.elite = pop.pop[np.argmin(pop.fit)]

        # testing the elite on validation/testing, if applicable
        if test_elite:
            self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

        if log != 0:

            if max_:
                logger(log_path, 0, max(pop.fit), end-start, float(pop.nodes_count),
                    pop_test_report = self.elite.test_fitness, run_info=run_info, seed=self.seed)

            else:
                logger(log_path, 0, min(pop.fit), end - start, float(pop.nodes_count),
                       pop_test_report=self.elite.test_fitness, run_info=run_info, seed=self.seed)

        if verbose != 0:
            if max_:
                verbose_reporter(curr_dataset.split("load_")[-1], 0, max(pop.fit), self.elite.test_fitness, end-start, pop.nodes_count)
            else:
                verbose_reporter(curr_dataset.split("load_")[-1], 0, min(pop.fit), self.elite.test_fitness, end - start, pop.nodes_count)

        ################################################################################################################

                                                        # GP EVOLUTION #

        ################################################################################################################

        for it in range(1, n_iter +1, 1):

            offs_pop, start = [], time.time()

            if elitism:

                offs_pop.append(self.elite)

            while len(offs_pop) < self.pop_size:

                p1, p2 = self.selector(pop), self.selector(pop)

                while p1 == p2:
                    p1, p2 = self.selector(pop), self.selector(pop)

                # choosing between crossover and mutation
                if random.random() < self.p_xo:

                    offs1, offs2 = self.crossover(p1.repr_, p2.repr_)

                else:

                    offs1, offs2 = self.mutator(p1.repr_), self.mutator(p2.repr_)

                if max_depth != None:

                    if tree_depth(offs1, self.pi_init["FUNCTIONS"]) > max_depth: # TODO change this to keep one of the parents instead ?? offs1 = p1 FASTER less DIVERSITY

                        offs1 = tree_pruning(offs1, max_depth,
                                                  self.pi_init["TERMINALS"], self.pi_init["CONSTANTS"],
                                                  self.pi_init["FUNCTIONS"], self.pi_init["p_c"])

                    if tree_depth(offs2, self.pi_init["FUNCTIONS"]) > max_depth:
                        offs2 = tree_pruning(offs2, max_depth,
                                             self.pi_init["TERMINALS"], self.pi_init["CONSTANTS"],
                                             self.pi_init["FUNCTIONS"], self.pi_init["p_c"])

                offs_pop.extend([Tree(offs1,  self.pi_init["FUNCTIONS"],
                                                  self.pi_init["TERMINALS"], self.pi_init["CONSTANTS"]),
                                 Tree(offs2, self.pi_init["FUNCTIONS"],
                                                  self.pi_init["TERMINALS"], self.pi_init["CONSTANTS"])])


            if len(offs_pop) > pop.size:

                offs_pop = offs_pop[:pop.size]

            offs_pop = Population(offs_pop)
            offs_pop.evaluate(ffunction, X=X_train, y=y_train)
            pop = offs_pop

            end = time.time()

            if max_:
                self.elite = pop.pop[np.argmax(pop.fit)]
            else:
                self.elite = pop.pop[np.argmin(pop.fit)]

            if test_elite:
                self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

            if log != 0:
                if max_:
                    logger(log_path, it, max(pop.fit), end - start, float(pop.nodes_count),
                           pop_test_report=[
                                       self.elite.test_fitness,
                                       ], run_info=run_info, seed=self.seed)
                else:
                    logger(log_path, it, min(pop.fit), end - start, float(pop.nodes_count),
                           pop_test_report=self.elite.test_fitness, run_info=run_info, seed=self.seed)

            if verbose != 0:
                if max_:
                    verbose_reporter(run_info[-1], it, max(pop.fit), self.elite.test_fitness, end - start, pop.nodes_count)
                else:
                    verbose_reporter(run_info[-1],it, min(pop.fit), self.elite.test_fitness, end - start, pop.nodes_count)

