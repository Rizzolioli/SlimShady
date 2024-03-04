import time
import random
import torch
import numpy as np

from utils.utils import verbose_reporter, logger, train_test_split
from algorithms.GSGP.representations.population import Population
from algorithms.GSGP.representations.tree import Tree



class GSGP:

    def __init__(self, pi_init, initializer, selector, mutator, ms, crossover,
                 p_m=0.8, p_xo=0.2, pop_size=100, seed=0,
                 settings_dict=None):

        #other initial parameters, tipo dataset
        self.pi_init = pi_init  # dictionary with all the parameters needed for evaluation
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.mutator = mutator
        self.ms = ms
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed
        #TODO check wheter to include max_depth

        self.settings_dict = settings_dict

    def solve(self, n_iter=20, elitism=True, log=0, verbose=0,
              test_elite=False, log_path=None, run_info=None,
              max_=False, dataset_loader=None,
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

        # and the population of random trees
        random_trees = Population([Tree(tree, self.pi_init['FUNCTIONS'], self.pi_init['TERMINALS'], self.pi_init['CONSTANTS'])
                          for tree in self.initializer(**self.pi_init)]) #TODO rn same size as initial population, decide wheter to create new rt at each variation
        random_trees.calculate_semantics(X_train)
        if test_elite:
            random_trees.calculate_semantics(X_test, testing=True)

        # initializing the population
        pop = Population([Tree(tree, self.pi_init['FUNCTIONS'], self.pi_init['TERMINALS'], self.pi_init['CONSTANTS'])
                          for tree in self.initializer(**self.pi_init)])
        pop.calculate_semantics(X_train)
        if test_elite:
            pop.calculate_semantics(X_test, testing=True)

        pop.evaluate(ffunction,  y=y_train)

        end = time.time()

        # obtaining the initial population elite
        if max_:
            self.elite = pop.pop[np.argmax(pop.fit)]
        else:
            self.elite = pop.pop[np.argmin(pop.fit)]

        # testing the elite on validation/testing, if applicable

        if test_elite:
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

        ancestry = []

        for it in range(1, n_iter +1, 1):

            offs_pop, start = [], time.time()

            if elitism:

                offs_pop.append(self.elite)

            while len(offs_pop) < self.pop_size:

                p1, p2 = self.selector(pop), self.selector(pop)

                while p1 == p2:
                    p1, p2 = self.selector(pop), self.selector(pop)

                ancestry.extend([p1, p2])

                # choosing between crossover and mutation
                if random.random() < self.p_xo:

                    offs1 = Tree([self.crossover, p1, p2, random.choice(random_trees)],
                                 p1.FUNCTIONS, p1.TERMINALS, p1.CONSTANTS)

                    offs_pop.append(offs1)

                else:

                    ms_ = self.ms if len(self.ms) == 1 else self.ms[random.randint(0, len(self.ms) - 1)]

                    offs1 = Tree([self.mutator, p1, random.choice(random_trees), random.choice(random_trees), ms_],
                                 p1.FUNCTIONS, p1.TERMINALS, p1.CONSTANTS)
                    offs2 = Tree([self.mutator, p2, random.choice(random_trees),  random.choice(random_trees), ms_],
                                 p1.FUNCTIONS, p1.TERMINALS, p1.CONSTANTS)

                    offs_pop.extend([offs1, offs2])


            if len(offs_pop) > pop.size:

                offs_pop = offs_pop[:pop.size]

            offs_pop = Population(offs_pop)
            offs_pop.calculate_semantics(X_train)
            if test_elite:
                offs_pop.calculate_semantics(X_test, testing=True)
            offs_pop.evaluate(ffunction, y=y_train)
            pop = offs_pop

            end = time.time()

            if max_:
                self.elite = pop.pop[np.argmax(pop.fit)]
            else:
                self.elite = pop.pop[np.argmin(pop.fit)]

            if test_elite:
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

