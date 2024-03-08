import time
import random
import torch
import numpy as np

from utils.utils import verbose_reporter, logger, train_test_split
from algorithms.SLIM_GSGP.representations.population import Population
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.representations.individual import Individual



class SLIM_GSGP:

    def __init__(self, pi_init, initializer, selector, inflate_mutator, deflate_mutator, ms, crossover,
                 p_m=1, p_xo=0, p_inflate = 0.3, p_deflate = 0.7, pop_size=100, seed=0,
                 settings_dict=None):

        #other initial parameters, tipo dataset
        self.pi_init = pi_init  # dictionary with all the parameters needed for evaluation
        self.selector = selector
        self.p_m = p_m
        self.p_inflate = p_inflate
        self.p_deflate = p_deflate
        self.crossover = crossover
        self.inflate_mutator = inflate_mutator
        self.deflate_mutator = deflate_mutator
        self.ms = ms
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed
        #TODO check wheter to include max_depth(Leo)

        self.settings_dict = settings_dict

    def solve(self, X_train, X_test, y_train, y_test, curr_dataset, n_iter=20, elitism=True, log=0, verbose=0,
              test_elite=False, log_path=None, run_info=None,
              max_=False,
              ffunction=None):

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        ################################################################################################################

                                                        # INITIALIZATION #

        ################################################################################################################

        # initializing the population
        pop = Population([Individual([
                          Tree(tree, self.pi_init['FUNCTIONS'], self.pi_init['TERMINALS'], self.pi_init['CONSTANTS']) ])
                          for tree in self.initializer(**self.pi_init)])


        pop.calculate_semantics(X_train)
        pop.evaluate(ffunction,  y=y_train)

        end = time.time()

        # obtaining the initial population elite
        if max_:
            self.elite = pop.population[np.argmax(pop.fit)]
        else:
            self.elite = pop.population[np.argmin(pop.fit)]

        # testing the elite on validation/testing, if applicable

        if test_elite:
            self.elite.calculate_semantics(X_test, testing=True)
            self.elite.evaluate(ffunction, y=y_test, testing=True)

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



                # choosing between crossover and mutation
                if random.random() < self.p_xo:

                    p1, p2 = self.selector(pop), self.selector(pop)

                    while p1 == p2:
                        p1, p2 = self.selector(pop), self.selector(pop)

                    pass # implement crossover

                else:

                    p1 = self.selector(pop)

                    if random.random() < self.p_deflate and it != 1: #TODO can't deflate at the first generation(Leo)

                        off1 = self.deflate_mutator(p1)

                    else:

                        ms_ = self.ms if len(self.ms) == 1 else self.ms[random.randint(0, len(self.ms) - 1)]

                        off1 = self.inflate_mutator(p1, ms_, X_train, max_depth = self.pi_init["init_depth"]
                                                    , p_c = self.pi_init["p_c"], X_test = X_test)

                    offs_pop.append(off1)



            if len(offs_pop) > pop.size:

                offs_pop = offs_pop[:pop.size]

            offs_pop = Population(offs_pop)
            offs_pop.calculate_semantics(X_train)

            offs_pop.evaluate(ffunction, y=y_train)
            pop = offs_pop

            end = time.time()

            if max_:
                self.elite = pop.population[np.argmax(pop.fit)]
            else:
                self.elite = pop.population[np.argmin(pop.fit)]

            if test_elite:
                self.elite.calculate_semantics(X_test, testing=True)
                self.elite.evaluate(ffunction, y=y_test, testing=True)

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

