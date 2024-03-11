import time
import random
import torch
import numpy as np

from utils.utils import verbose_reporter, logger, get_random_tree
from algorithms.GSGP.representations.population import Population
from algorithms.GSGP.representations.tree import Tree

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

        self.settings_dict = settings_dict
        self.find_elit_func = find_elit_func

    def solve(self, X_train, X_test, y_train, y_test , curr_dataset, n_iter=20, elitism=True, log=0, verbose=0,
              test_elite=False, log_path=None, run_info=None,
              max_=False, ffunction=None, reconstruct=False, n_elites=1):

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()


        ################################################################################################################

                                                        # INITIALIZATION #

        ################################################################################################################

        # initializing the population
        population = Population([Tree(tree, self.pi_init['FUNCTIONS'], self.pi_init['TERMINALS'], self.pi_init['CONSTANTS'])
                          for tree in self.initializer(**self.pi_init)])

        # getting the individuals' semantics
        population.calculate_semantics(X_train)

        if test_elite:
            population.calculate_semantics(X_test, testing=True)

        # getting individuals' fitness
        population.evaluate(ffunction,  y=y_train)

        end = time.time()

        # obtaining the initial population elites
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        # testing the elite on validation/testing, if applicable
        if test_elite:
            self.elite.evaluate(ffunction, y=y_test, testing=True)

        # logging the results for the population initialization
        if log != 0:

            logger(log_path, 0, self.elite.fitness, end-start, float(population.nodes_count),
                pop_test_report = self.elite.test_fitness, run_info=run_info, seed=self.seed)

        # displaying the results for the population initialization on console
        if verbose != 0:

            verbose_reporter(curr_dataset.split("load_")[-1], 0,  self.elite.fitness, self.elite.test_fitness, end-start, population.nodes_count)

        # initializing a random tree list table
        random_trees = []

        ################################################################################################################

                                                        # GP EVOLUTION #

        ################################################################################################################

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

                    while p1 == p2:
                        p1, p2 = self.selector(population), self.selector(population)

                    # getting a random tree
                    r_tree = get_random_tree(max_depth=self.pi_init['init_depth'], FUNCTIONS=self.pi_init['FUNCTIONS'], TERMINALS=self.pi_init['TERMINALS'],
                                             CONSTANTS=self.pi_init['CONSTANTS'], inputs=X_train)

                    # calculating its semantics on testing, if applicable
                    if test_elite:
                        r_tree.calculate_semantics(X_test, testing=True)

                    # adding the random tree to the random tree list
                    random_trees.append(r_tree)

                    # the two parents generate one offspring
                    offs1 = Tree([self.crossover, p1, p2, r_tree],
                                 p1.FUNCTIONS, p1.TERMINALS, p1.CONSTANTS)

                    # adding the offspring to the population
                    offs_pop.append(offs1)

                    # adding the parent information to the ancestry
                    if reconstruct:
                        ancestry.extend([p1, p2])

                else:
                    # if mutation choose one parent
                    p1 = self.selector(population)

                    # determining the mutation step
                    ms_ = self.ms if len(self.ms) == 1 else self.ms[random.randint(0, len(self.ms) - 1)]

                    # getting two random trees
                    r_tree1 = get_random_tree(max_depth=self.pi_init['init_depth'], FUNCTIONS=self.pi_init['FUNCTIONS'],
                                             TERMINALS=self.pi_init['TERMINALS'],
                                             CONSTANTS=self.pi_init['CONSTANTS'], inputs=X_train)

                    r_tree2 = get_random_tree(max_depth=self.pi_init['init_depth'],
                                              FUNCTIONS=self.pi_init['FUNCTIONS'],
                                              TERMINALS=self.pi_init['TERMINALS'],
                                              CONSTANTS=self.pi_init['CONSTANTS'], inputs=X_train)

                    # calculating random trees' semantics on testing, if applicable
                    if test_elite:
                        r_tree1.calculate_semantics(X_test, testing=True)
                        r_tree2.calculate_semantics(X_test, testing=True)

                    # adding the random trees to the random tree list
                    random_trees.extend([r_tree1, r_tree2])

                    # mutating the individual
                    offs1 = Tree([self.mutator, p1, r_tree1, r_tree2, ms_],
                                 p1.FUNCTIONS, p1.TERMINALS, p1.CONSTANTS)

                    # adding the individual to the population
                    offs_pop.append(offs1)

                    # adding the parent information to the ancestry
                    if reconstruct:
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

            # getting the population elite
            self.elites, self.elite = self.find_elit_func(population, n_elites)

            if test_elite:
                self.elite.evaluate(ffunction, y=y_test, testing=True)

            # logging the results for the current generation
            if log != 0:
                logger(log_path, it, self.elite.fitness, end - start, float(population.nodes_count),
                       pop_test_report=self.elite.test_fitness, run_info=run_info, seed=self.seed)

            # displaying the results for the current generation on console
            if verbose != 0:
                verbose_reporter(run_info[-1], it, self.elite.fitness, self.elite.test_fitness, end - start,
                                 population.nodes_count)



