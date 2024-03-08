import time
import random
import torch
import numpy as np

from utils.utils import verbose_reporter, logger, train_test_split
from algorithms.GP.representations.population import Population
from algorithms.GP.representations.tree import Tree
from algorithms.GP.representations.tree_utils import tree_pruning, tree_depth


# small fixes - Liah
# TODO: consider logger levels (pickel population)
# TODO: make elitism parametrized

# Diogo
# TODO handling of TERMINALS FUNCTIONS etc in all scripts


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

    def solve(self, n_iter=20, elitism=True, log=0, verbose=0,
              test_elite=False, log_path=None, run_info=None,
              max_depth=None, max_=False, dataset_loader=None,
              ffunction=None):

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # starting the timer
        start = time.time()

        #TODO move outisde the gp code(also for gsgp and slim)

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
        population = Population([Tree(tree,  self.pi_init['FUNCTIONS'], self.pi_init['TERMINALS'], self.pi_init['CONSTANTS'])
                          for tree in self.initializer(**self.pi_init)])
        population.evaluate(ffunction, X=X_train, y=y_train)

        # ending the timer
        end = time.time()

        # obtaining the initial population elite
        self.elite = self.find_elit_func(population)

        # testing the elite on validation/testing, if applicable
        if test_elite:
            self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

        if log != 0:

        # logging the results

            logger(log_path, 0, self.elite.fitness, end-start, float(population.nodes_count),
                pop_test_report = self.elite.test_fitness, run_info=run_info, seed=self.seed)


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

                offs_pop.append(self.elite)

            while len(offs_pop) < self.pop_size:

                # choosing between crossover and mutation
                if random.random() < self.p_xo:

                    # if crossover, choose two parents
                    p1, p2 = self.selector(population), self.selector(population)

                    # getting the offspring
                    offs1, offs2 = self.crossover(p1.repr_, p2.repr_)

                    # saving the offspring in an offspring list
                    offspring = [offs1, offs2]
                else:
                    # if mutation, choose one parent
                    p1 = self.selector(population)

                    # obtain one offspring
                    offs1 = self.mutator(p1.repr_)

                    # saving the offspring in an offspring list
                    offspring = [offs1]

                if max_depth != None:

                    # pruning all the offspring that are too big:
                    offspring = [tree_pruning(child, max_depth,
                                                      self.pi_init["TERMINALS"], self.pi_init["CONSTANTS"],
                                                      self.pi_init["FUNCTIONS"], self.pi_init["p_c"])
                                 if tree_depth(child, self.pi_init["FUNCTIONS"]) > max_depth else child for child in offspring]

                # adding the offspring to the offspring population
                offs_pop.extend([Tree(child,self.pi_init["FUNCTIONS"],
                                                  self.pi_init["TERMINALS"], self.pi_init["CONSTANTS"])
                                 for child in offspring])


            # keeping only the amount of offspring that is equal to the population size
            if len(offs_pop) > population.size:

                offs_pop = offs_pop[:population.size]

            # overriding the current population as the offspring population
            offs_pop = Population(offs_pop)
            offs_pop.evaluate(ffunction, X=X_train, y=y_train)
            population = offs_pop

            end = time.time()

            # getting the population elite
            self.elite = self.find_elit_func(population)

            # testing the elite if test_elite is True
            if test_elite:
                self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

            # logging the results
            if log != 0:
               logger(log_path, it, self.elite.fitness, end - start, float(population.nodes_count),
                           pop_test_report=self.elite.test_fitness, run_info=run_info, seed=self.seed)

            if verbose != 0:
                verbose_reporter(run_info[-1], it, self.elite.fitness, self.elite.test_fitness, end - start, population.nodes_count)

