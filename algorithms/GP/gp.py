import time
import random
import torch
import numpy as np

from utils.utils import verbose_reporter
from algorithms.GP.representations.population import Population
from algorithms.GP.representations.tree import Tree
from algorithms.GP.representations.tree_utils import tree_depth
from utils.diversity import niche_entropy
from utils.logger import logger


# TODO handling of TERMINALS FUNCTIONS etc in all scripts diogo "is handling it"


class GP:

    def __init__(self, pi_init, initializer, selector, mutator, crossover, find_elit_func,
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

        self.find_elit_func = find_elit_func
        self.settings_dict = settings_dict

        Tree.FUNCTIONS = pi_init['FUNCTIONS']
        Tree.TERMINALS = pi_init['TERMINALS']
        Tree.CONSTANTS = pi_init['CONSTANTS']

    def solve(self, X_train, X_test, y_train, y_test, curr_dataset,n_iter=20, elitism=True, log=0, verbose=0,
              test_elite=False, log_path=None, run_info=None,
              max_depth=None, max_=False,
              ffunction=None, n_elites = 1, tree_pruner=None, depth_calculator = None):

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # starting the timer
        start = time.time()

        ################################################################################################################

                                                        # INITIALIZATION #

        ################################################################################################################

        # initializing the population
        population = Population([Tree(tree)
                          for tree in self.initializer(**self.pi_init)])

        population.evaluate(ffunction, X=X_train, y=y_train)

        # ending the timer
        end = time.time()

        # obtaining the initial population elites
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        # testing the elite on validation/testing, if applicable
        if test_elite:
            self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

        # logging the population initialization
        if log != 0:

            if log == 2:
                add_info = [self.elite.test_fitness, self.elite.node_count, float(niche_entropy([ind.repr_ for ind in population.population])),
                            np.std(population.fit), log]

            # log level 3 saves the number of nodes and fitness of all the individuals in the population
            elif log == 3:

                add_info = [self.elite.test_fitness,self.elite.node_count,
                        " ".join([str(ind.node_count) for ind in population.population]),
                        " ".join([str(f) for f in population.fit]), log]

            elif log == 4:

                add_info = [self.elite.test_fitness,self.elite.node_count,
                            float(niche_entropy([ind.repr_ for ind in population.population])),
                            np.std(population.fit),
                            " ".join([str(ind.node_count) for ind in population.population]),
                            " ".join([str(f) for f in population.fit]), log
                            ]

            else:

                add_info = [self.elite.test_fitness,self.elite.node_count, log]


            logger(log_path, 0, self.elite.fitness, end-start, float(population.nodes_count),
                additional_infos = add_info, run_info=run_info, seed=self.seed)


        if verbose != 0:
          # displaying the results on console
          verbose_reporter(curr_dataset.split("load_")[-1], 0, self.elite.fitness, self.elite.test_fitness,
                           end-start, population.nodes_count)

        ################################################################################################################

                                                        # GP EVOLUTION #

        ################################################################################################################

        for it in range(1, n_iter +1, 1):

            offs_pop, start = [], time.time()

            if elitism:

                offs_pop.extend(self.elites)

            while len(offs_pop) < self.pop_size:

                # choosing between crossover and mutation
                if random.random() < self.p_xo:

                    # if crossover, choose two parents
                    p1, p2 = self.selector(population), self.selector(population)

                    while p1 == p2:
                        p1, p2 = self.selector(population), self.selector(population)

                    # getting the offspring
                    offs1, offs2 = self.crossover(p1.repr_, p2.repr_, tree1_n_nodes=p1.node_count, tree2_n_nodes=p2.node_count)

                    # saving the offspring in an offspring list
                    offspring = [offs1, offs2]
                else:
                    # if mutation, choose one parent
                    p1 = self.selector(population)

                    # obtain one offspring
                    offs1 = self.mutator(p1.repr_, num_of_nodes=p1.node_count)

                    # saving the offspring in an offspring list
                    offspring = [offs1]

                if max_depth is not None:

                    # pruning all the offspring that are too big:
                    offspring = [tree_pruner(child, max_depth)
                                 if depth_calculator(child) > max_depth else child for child in offspring]

                # adding the offspring to the offspring population
                offs_pop.extend([Tree(child) for child in offspring])


            # keeping only the amount of offspring that is equal to the population size
            if len(offs_pop) > population.size:

                offs_pop = offs_pop[:population.size]

            # overriding the current population as the offspring population
            offs_pop = Population(offs_pop)
            offs_pop.evaluate(ffunction, X=X_train, y=y_train)
            population = offs_pop

            end = time.time()

            # getting the population elites
            self.elites, self.elite = self.find_elit_func(population, n_elites)

            # testing the elite if test_elite is True
            if test_elite:
                self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

            # logging the results for the current generation
            if log != 0:

                # logging the population initialization
                if log != 0:

                    if log == 2:
                        add_info = [self.elite.test_fitness,self.elite.node_count,
                                    float(niche_entropy([ind.repr_ for ind in population.population])),
                                    np.std(population.fit), log]

                    # log level 3 saves the number of nodes and fitness of all the individuals in the population
                    elif log == 3:

                        add_info = [self.elite.test_fitness,self.elite.node_count,
                                    " ".join([str(ind.node_count) for ind in population.population]),
                                    " ".join([str(f) for f in population.fit]), log]

                    elif log == 4:

                        add_info = [self.elite.test_fitness,self.elite.node_count,
                                    float(niche_entropy([ind.repr_ for ind in population.population])),
                                    np.std(population.fit),
                                    " ".join([str(ind.node_count) for ind in population.population]),
                                    " ".join([str(f) for f in population.fit]), log
                                    ]

                    else:

                        add_info = [self.elite.test_fitness,self.elite.node_count, log]

                logger(log_path, it, self.elite.fitness, end - start, float(population.nodes_count),
                           additional_infos=add_info, run_info=run_info, seed=self.seed)

            # displaying the results for the current generation on console
            if verbose != 0:
                verbose_reporter(run_info[-1], it, self.elite.fitness, self.elite.test_fitness, end - start, population.nodes_count)

