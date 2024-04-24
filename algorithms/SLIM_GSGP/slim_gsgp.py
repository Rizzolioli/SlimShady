import time
import random
import torch
import numpy as np

from utils.utils import verbose_reporter, train_test_split
from utils.logger import logger
from algorithms.SLIM_GSGP.representations.population import Population
from algorithms.GSGP.representations.tree import Tree
from algorithms.GP.representations.tree import Tree as GP_Tree
from algorithms.SLIM_GSGP.representations.individual import Individual

from utils.diversity import gsgp_pop_div_from_vectors



class SLIM_GSGP:
    # TODO: implement improvement rate
    # TODO: implement TIE & convex hull

    def __init__(self, pi_init, initializer, selector, inflate_mutator, deflate_mutator, ms, crossover, find_elit_func,
                 p_m=1, p_xo=0, p_inflate = 0.3, p_deflate = 0.7, pop_size=100, seed=0, operator = 'sum',
                 two_trees=True, settings_dict=None):

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
        self.operator = operator
        self.two_trees = two_trees

        self.settings_dict = settings_dict
        self.find_elit_func = find_elit_func
        
        Tree.FUNCTIONS = pi_init['FUNCTIONS']
        Tree.TERMINALS = pi_init['TERMINALS']
        Tree.CONSTANTS = pi_init['CONSTANTS']

        GP_Tree.FUNCTIONS = pi_init['FUNCTIONS']
        GP_Tree.TERMINALS = pi_init['TERMINALS']
        GP_Tree.CONSTANTS = pi_init['CONSTANTS']

    def solve(self, X_train, X_test, y_train, y_test, curr_dataset, run_info ,n_iter=20, elitism=True, log=0, verbose=0,
              test_elite=False, log_path=None,
              max_=False, ffunction=None, max_depth=17, n_elites=1):

        # TO REMOVE:

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        ################################################################################################################

                                                        # INITIALIZATION #

        ################################################################################################################

        # initializing the population
        population = Population([Individual([Tree(tree)])for tree in self.initializer(**self.pi_init)])


        population.calculate_semantics(X_train)
        population.evaluate(ffunction,  y=y_train, operator=self.operator)

        end = time.time()

        # obtaining the initial population elites
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        # testing the elite on validation/testing, if applicable
        if test_elite:
            self.elite.calculate_semantics(X_test, testing=True)
            self.elite.evaluate(ffunction, y=y_test, testing=True, operator=self.operator)

        # logging the population initialization
        if log != 0:

            if log == 2:
                gen_diversity = gsgp_pop_div_from_vectors(torch.stack([torch.sum(ind.train_semantics, dim=0)
                                                                        for ind in population.population]),
                                                           ) \
                    if self.operator == 'sum' else \
                    gsgp_pop_div_from_vectors(torch.stack([torch.prod(ind.train_semantics, dim=0)
                                                           for ind in population.population]))
                add_info = [self.elite.test_fitness,
                            self.elite.nodes_count,
                            float(gen_diversity),
                            np.std(population.fit), log]

            # log level 3 saves the number of nodes and fitness of all the individuals in the population
            elif log == 3:

                add_info = [self.elite.test_fitness,
                            self.elite.nodes_count,
                        " ".join([str(ind.nodes_count) for ind in population.population]),
                        " ".join([str(f) for f in population.fit]), log]

            elif log == 4:

                gen_diversity = gsgp_pop_div_from_vectors(torch.stack([torch.sum(ind.train_semantics, dim=0)
                                                                       for ind in population.population]),
                                                          ) \
                    if self.operator == 'sum' else \
                    gsgp_pop_div_from_vectors(torch.stack([torch.prod(ind.train_semantics, dim=0)
                                                           for ind in population.population]))
                add_info = [self.elite.test_fitness,
                            self.elite.nodes_count,
                            float(gen_diversity),
                            np.std(population.fit),
                            " ".join([str(ind.nodes_count) for ind in population.population]),
                            " ".join([str(f) for f in population.fit]), log
                            ]


            else:

                add_info = [self.elite.test_fitness, self.elite.nodes_count, log]

            logger(log_path, 0, self.elite.fitness, end - start, float(population.nodes_count),
                   additional_infos=add_info, run_info=run_info, seed=self.seed)

        # displaying the results of the population initialization on console
        if verbose != 0:
            verbose_reporter(curr_dataset.split("load_")[-1], 0,  self.elite.fitness, self.elite.test_fitness, end-start,
                             self.elite.nodes_count)

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

                    # if crossover selecting two parents
                    p1, p2 = self.selector(population), self.selector(population)

                    # making sure the parents aren't the same
                    while p1 == p2:
                        p1, p2 = self.selector(population), self.selector(population)

                    pass # implement crossover

                else:


                    # choose between deflating or inflating the individual
                    if random.random() < self.p_deflate:



                        # getting a list with the valid population
                        # valid_pop = [ind for ind in population.population if ind.size > 1]

                        # if the valid population list is enough for the tournament size:

                        # if deflate mutation, pick one individual that is of size > 1
                        # p1 = self.selector(population, deflate=True)

                        # if a valid deflatable individual was found and selected
                        # if p1 is not None:
                        #
                        #
                        #     off1 = self.deflate_mutator(p1)
                        #     print('deflating')
                        #
                        # # if there arent enough valid individuals for deflating, we inflate instead
                        # else:
                            # selecting  a random individual with no restrictions
                            # p1 = self.selector(population, deflate=False)

                        p1 = self.selector(population, deflate=False)

                        off1 = self.deflate_mutator(p1)

                        # off1 = Individual(p1.collection)
                        # off1.train_semantics = p1.train_semantics
                        # if p1.test_semantics != None:
                        #     off1.test_semantics = p1.test_semantics



                    else:
                        # if inflate mutation, pick a random individual with no restrictions

                        p1 = self.selector(population, deflate=False)

                        # obtaining the random mutation step
                        ms_ = self.ms()

                        if max_depth is not None and p1.depth == max_depth:

                            off1 = self.deflate_mutator(p1)

                        else:

                            off1 = self.inflate_mutator(p1, ms_, X_train, max_depth = self.pi_init["init_depth"]
                                                    , p_c = self.pi_init["p_c"], X_test = X_test)


                        if max_depth is not None and off1.depth > max_depth: #TODO if offspring too big return parent (Koza)

                            off1 = self.deflate_mutator(p1)


                    offs_pop.append(off1)



            if len(offs_pop) > population.size:

                offs_pop = offs_pop[:population.size]

            offs_pop = Population(offs_pop)
            offs_pop.calculate_semantics(X_train)

            offs_pop.evaluate(ffunction, y=y_train, operator=self.operator)
            population = offs_pop

            self.population = population

            end = time.time()

            # obtaining the initial population elites
            self.elites, self.elite = self.find_elit_func(population, n_elites)

            if test_elite:
                self.elite.calculate_semantics(X_test, testing=True)
                self.elite.evaluate(ffunction, y=y_test, testing=True, operator=self.operator)

            # logging the population initialization
            if log != 0:

                if log == 2:
                    gen_diversity = gsgp_pop_div_from_vectors(torch.stack([torch.sum(ind.train_semantics, dim=0)
                                                                           for ind in population.population]),
                                                              ) \
                        if self.operator == 'sum' else \
                        gsgp_pop_div_from_vectors(torch.stack([torch.prod(ind.train_semantics, dim=0)
                                                               for ind in population.population]))
                    add_info = [self.elite.test_fitness,
                                self.elite.nodes_count,
                                float(gen_diversity),
                                np.std(population.fit), log]

                # log level 3 saves the number of nodes and fitness of all the individuals in the population
                elif log == 3:

                    add_info = [self.elite.test_fitness,
                                self.elite.nodes_count,
                                " ".join([str(ind.nodes_count) for ind in population.population]),
                                " ".join([str(f) for f in population.fit]), log]

                elif log == 4:

                    gen_diversity = gsgp_pop_div_from_vectors(torch.stack([torch.sum(ind.train_semantics, dim=0)
                                                                           for ind in population.population]),
                                                              ) \
                        if self.operator == 'sum' else \
                        gsgp_pop_div_from_vectors(torch.stack([torch.prod(ind.train_semantics, dim=0)
                                                               for ind in population.population]))
                    add_info = [self.elite.test_fitness,
                                self.elite.nodes_count,
                                float(gen_diversity),
                                np.std(population.fit),
                                " ".join([str(ind.nodes_count) for ind in population.population]),
                                " ".join([str(f) for f in population.fit]), log
                                ]

                else:

                    add_info = [self.elite.test_fitness, self.elite.nodes_count, log]

                # logging the desired results
                logger(log_path, it, self.elite.fitness, end - start, float(population.nodes_count),
                               additional_infos=add_info, run_info=run_info, seed=self.seed)

            if verbose != 0:
                verbose_reporter(run_info[-1], it, self.elite.fitness, self.elite.test_fitness, end - start,
                                         self.elite.nodes_count)
