import time
import random

import torch

from tree.population import Population
import numpy as np
from tree.utils.utils_info import logger, verbose_reporter
from tree.trees import Tree
from tree.utils.utils import tree_pruning, tree_depth
from tree.utils.gp_settings import set_gp, set_gsgp
from gpol.utils.utils import train_test_split


class GPDP():


    def __init__(self, pi_eval, pi_init, initializer, selector, mutator, crossover,
                 p_m=0.2, p_c=0.8, pop_size=100, elitism=True, seed = 0, pi_test = None):
        #other initial parameters, tipo dataset
        self.pi_eval = pi_eval #dictionary with all the parameters needed for evaluation
        self.pi_init = pi_init  # dictionary with all the parameters needed for evaluation
        self.pi_test = pi_test
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.mutator = mutator
        self.p_c = p_c
        self.elitism = elitism
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed


    def solve(self, n_iter=20, elitism = True, log = 0, verbose = 0,
              test_elite = False, log_path = None, run_info = None,
              max_depth = None, max_ = False, datasets = None,
              base_algo = 'gp'):

        if base_algo not in ['gp', 'gsgp']:

            raise Exception('The base algorithm needs to be either gp or gsgp')

        if base_algo == 'gp':
            setter = set_gp

        else:
            setter = set_gsgp


        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        start = time.time()

        # Loads the data
        X, y = datasets[0](X_y=True, scaled = True)
        curr_dataset = '_'.join(str(datasets[0].__name__).split('_')[1:])
        # Performs train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.3, seed=self.seed)
        self.pi_eval['solver'] = setter(X_train, y_train, seed=self.seed, dataset_name=curr_dataset,
                                        pop_size=self.pi_eval['std_params']['max_pop'])

        pop = Population(self.initializer(**self.pi_init))
        pop.evaluate(**self.pi_eval)

        end = time.time()

        if max_:
            self.elite = pop.pop[np.argmax(pop.fit)]
        else:
            self.elite = pop.pop[np.argmin(pop.fit)]


        if test_elite and self.pi_test != None:
            self.pi_test['solver'] = setter(X_test, y_test, seed=self.seed,  pop_size=self.pi_test['std_params']['max_pop'])
            self.elite.evaluate_tree(**self.pi_test, test = True)
        else:
            self.elite.test_fitness = None


        if log != 0:
            if run_info != None:
                run_info.append(curr_dataset)

            if max_:
                logger(log_path, 0, max(pop.fit), end-start, float(pop.nodes_count),
                    pop_test_report = [float(self.elite.test_fitness),
                                       ], run_info = run_info)

            else:
                logger(log_path, 0, min(pop.fit), end - start, float(pop.nodes_count),
                       pop_test_report=float(self.elite.test_fitness), run_info=run_info)
        if verbose != 0:
            if max_:
                verbose_reporter(curr_dataset, 0, max(pop.fit), self.elite.test_fitness, end-start, pop.nodes)
            else:
                verbose_reporter(curr_dataset, 0, min(pop.fit), self.elite.test_fitness, end - start, pop.nodes)


        for it in range(1, n_iter +1, 1):

            offs_pop, start = [], time.time()

            if elitism:

                offs_pop.append(self.elite)

            while len(offs_pop) < pop.size:


                p1, p2 = self.selector(pop), self.selector(pop)

                while p1 == p2:
                    p1, p2 = self.selector(pop), self.selector(pop)

                if random.random() < self.p_c:

                    offs1, offs2 = self.crossover(p1.repr_, p2.repr_)


                else:

                    offs1, offs2 = self.mutator(p1.repr_), self.mutator(p2.repr_)

                if max_depth != None:

                    if tree_depth(offs1, self.pi_init["FUNCTIONS"]) > max_depth:

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

            # Loads the data
            X, y = datasets[it%len(datasets)](X_y=True, scaled = True)
            curr_dataset = '_'.join(str(datasets[it%len(datasets)].__name__).split('_')[1:])
            # Performs train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.3, seed=self.seed)
            self.pi_eval['solver'] = setter(X_train, y_train, seed=self.seed, dataset_name=curr_dataset,
                                            pop_size=self.pi_eval['std_params']['max_pop']
                                            )

            offs_pop = Population(offs_pop)
            offs_pop.evaluate(**self.pi_eval)

            pop = offs_pop

            end = time.time()

            if max_:
                self.elite = pop.pop[np.argmax(pop.fit)]
            else:
                self.elite = pop.pop[np.argmin(pop.fit)]

            if test_elite and self.pi_test != None:
                self.pi_test['solver'] = setter(X_test, y_test, seed=self.seed, pop_size=self.pi_test['std_params']['max_pop'])
                self.elite.evaluate_tree(**self.pi_test, test = True)
            else:
                self.elite.test_fitness = None

            if log != 0:

                if run_info != None:
                    run_info[-1] = curr_dataset

                if max_:
                    logger(log_path, it, max(pop.fit), end - start, float(pop.nodes_count),
                           pop_test_report=[
                                       float(self.elite.test_fitness),
                                       ], run_info=run_info)
                else:
                    logger(log_path, it, min(pop.fit), end - start, float(pop.nodes_count),
                           pop_test_report=float(self.elite.test_fitness), run_info=run_info)

            if verbose != 0:
                if run_info != None:
                    run_info[-1] = curr_dataset

                if max_:
                    verbose_reporter(curr_dataset, it, max(pop.fit), self.elite.test_fitness, end - start, pop.nodes)
                else:
                    verbose_reporter(curr_dataset, it, min(pop.fit), self.elite.test_fitness, end - start, pop.nodes)

