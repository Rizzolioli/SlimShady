import time
import random
import torch
import numpy as np

from utils.utils import verbose_reporter, logger, train_test_split
from algorithms.GSGP.representations.population import Population
from algorithms.GSGP.representations.tree import Tree


# TODO: not to create a population of random trees, just add a tree to a table when we need them
class GSGP:

    def __init__(self, pi_init, initializer, selector, mutator, ms, crossover, find_elit_func,
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
        # TODO check whether to include max_depth

        self.settings_dict = settings_dict
        self.find_elit_func = find_elit_func

    def solve(self, n_iter=20, elitism=True, log=0, verbose=0,
              test_elite=False, log_path=None, run_info=None,
              max_=False, dataset_loader=None,
              ffunction=None):

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        # TODO move outisde the gsgp code

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
                          for tree in self.initializer(**self.pi_init)])
        # TODO rn same size as initial population, decide whether to create new rt at each variation
        random_trees.calculate_semantics(X_train)
        if test_elite:
            random_trees.calculate_semantics(X_test, testing=True)

        # initializing the population
        population = Population([Tree(tree, self.pi_init['FUNCTIONS'], self.pi_init['TERMINALS'], self.pi_init['CONSTANTS'])
                          for tree in self.initializer(**self.pi_init)])
        population.calculate_semantics(X_train)
        if test_elite:
            population.calculate_semantics(X_test, testing=True)

        population.evaluate(ffunction,  y=y_train)

        end = time.time()

        # obtaining the initial population elite
        if max_:
            self.elite = population.population[np.argmax(population.fit)]
        else:
            self.elite = population.population[np.argmin(population.fit)]

        # testing the elite on validation/testing, if applicable

        if test_elite:
            self.elite.evaluate(ffunction, y=y_test, testing=True)

        if log != 0:

            if max_:
                logger(log_path, 0, max(population.fit), end-start, float(population.nodes_count),
                    pop_test_report = self.elite.test_fitness, run_info=run_info, seed=self.seed)

            else:
                logger(log_path, 0, min(population.fit), end - start, float(population.nodes_count),
                       pop_test_report=self.elite.test_fitness, run_info=run_info, seed=self.seed)

        if verbose != 0:
            if max_:
                verbose_reporter(curr_dataset.split("load_")[-1], 0, max(population.fit), self.elite.test_fitness, end-start, population.nodes_count)
            else:
                verbose_reporter(curr_dataset.split("load_")[-1], 0, min(population.fit), self.elite.test_fitness, end - start, population.nodes_count)

        ################################################################################################################

                                                        # GP EVOLUTION #

        ################################################################################################################

        #TODO add reconstruct bool
        ancestry = []

        for it in range(1, n_iter +1, 1):

            offs_pop, start = [], time.time()

            if elitism:

                offs_pop.append(self.elite)

            while len(offs_pop) < self.pop_size:

                # choosing between crossover and mutation
                if random.random() < self.p_xo:

                    # if crossover, select two parents
                    p1, p2 = self.selector(population), self.selector(population)

                    # the two parents generate one offspring
                    offs1 = Tree([self.crossover, p1, p2, random.choice(random_trees)],
                                 p1.FUNCTIONS, p1.TERMINALS, p1.CONSTANTS)

                    # adding the offspring to the population
                    offs_pop.append(offs1)

                    # adding the parent information to the ancestry
                    ancestry.extend([p1, p2])

                else:
                    # if mutation choose one parent
                    p1 = self.selector(population)

                    # determining the mutation step
                    ms_ = self.ms if len(self.ms) == 1 else self.ms[random.randint(0, len(self.ms) - 1)]
                    # mutating the individual
                    offs1 = Tree([self.mutator, p1, random.choice(random_trees), random.choice(random_trees), ms_],
                                 p1.FUNCTIONS, p1.TERMINALS, p1.CONSTANTS)

                    # adding the individual to the population
                    offs_pop.append(offs1)

                    # adding the parent information to the ancestry
                    ancestry.append(p1)

            if len(offs_pop) > population.size:

                offs_pop = offs_pop[:population.size]

            offs_pop = Population(offs_pop)
            offs_pop.calculate_semantics(X_train)
            if test_elite:
                offs_pop.calculate_semantics(X_test, testing=True)
            offs_pop.evaluate(ffunction, y=y_train)
            population = offs_pop

            end = time.time()

            if max_:
                self.elite = population.population[np.argmax(population.fit)]
            else:
                self.elite = population.population[np.argmin(population.fit)]

            if test_elite:
                self.elite.evaluate(ffunction, y=y_test, testing=True)

            if log != 0:
                if max_:
                    logger(log_path, it, max(population.fit), end - start, float(population.nodes_count),
                           pop_test_report=[
                                       self.elite.test_fitness,
                                       ], run_info=run_info, seed=self.seed)
                else:
                    logger(log_path, it, min(population.fit), end - start, float(population.nodes_count),
                           pop_test_report=self.elite.test_fitness, run_info=run_info, seed=self.seed)

            if verbose != 0:
                if max_:
                    verbose_reporter(run_info[-1], it, max(population.fit), self.elite.test_fitness, end - start, population.nodes_count)
                else:
                    verbose_reporter(run_info[-1],it, min(population.fit), self.elite.test_fitness, end - start, population.nodes_count)

