import time
import random
import torch
import numpy as np

from utils.utils import verbose_reporter, train_test_split
from utils.logger import logger
from algorithms.SLIM_GSGP.representations.population import Population
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.representations.individual import Individual

from utils.diversity import gsgp_pop_div_from_vectors



class SLIM_GSGP:
    # TODO: implement improvement rate
    # TODO: implement TIE & convex hull

    def __init__(self, pi_init, initializer, selector, inflate_mutator, deflate_mutator, ms, crossover, find_elit_func,
                 p_m=1, p_xo=0, p_inflate = 0.3, p_deflate = 0.7, pop_size=100, seed=0, operator = 'sum',
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
        self.operator = operator

        self.settings_dict = settings_dict
        self.find_elit_func = find_elit_func

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
        population = Population([Individual([
                          Tree(tree, self.pi_init['FUNCTIONS'], self.pi_init['TERMINALS'], self.pi_init['CONSTANTS']) ])
                          for tree in self.initializer(**self.pi_init)])


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
                            float(gen_diversity),
                            np.std(population.fit), log]

            # log level 3 saves the number of nodes and fitness of all the individuals in the population
            elif log == 3:

                add_info = [self.elite.test_fitness,
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
                            float(gen_diversity),
                            np.std(population.fit),
                            " ".join([str(ind.nodes_count) for ind in population.population]),
                            " ".join([str(f) for f in population.fit]), log
                            ]


            else:

                add_info = [self.elite.test_fitness, log]

            logger(log_path, 0, self.elite.fitness, end - start, float(population.nodes_count),
                   additional_infos=add_info, run_info=run_info, seed=self.seed)

        # displaying the results of the population initialization on console
        if verbose != 0:
            verbose_reporter(curr_dataset.split("load_")[-1], 0,  self.elite.fitness, self.elite.test_fitness, end-start, population.nodes_count)

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
                    if random.random() < self.p_deflate and it != 1:

                        # if deflate mutation, pick one individual that is of size > 1
                        p1 = self.selector(population, deflate=True)

                        off1 = self.deflate_mutator(p1)

                    else:
                        # if inlate mutation, pick a random individual with no restrictions

                        p1 = self.selector(population, deflate=False)

                        ms_ = self.ms if len(self.ms) == 1 else self.ms[random.randint(0, len(self.ms) - 1)]

                        if p1.depth == max_depth:

                            off1 = self.deflate_mutator(p1) #TODO if parent depth == max depth, apply deflate instead of deflate, ok?

                        else:

                            off1 = self.inflate_mutator(p1, ms_, X_train, max_depth = self.pi_init["init_depth"]
                                                    , p_c = self.pi_init["p_c"], X_test = X_test)


                        # if off1.depth < max_depth: #TODO if offspring too big return parent (Koza)
                        #
                            # off1 = self.deflate_mutator(p1)

                    offs_pop.append(off1)



            if len(offs_pop) > population.size:

                offs_pop = offs_pop[:population.size]

            offs_pop = Population(offs_pop)
            offs_pop.calculate_semantics(X_train)

            offs_pop.evaluate(ffunction, y=y_train, operator=self.operator)
            population = offs_pop

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
                                float(gen_diversity),
                                np.std(population.fit), log]

                # log level 3 saves the number of nodes and fitness of all the individuals in the population
                elif log == 3:

                    add_info = [self.elite.test_fitness,
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
                                float(gen_diversity),
                                np.std(population.fit),
                                " ".join([str(ind.nodes_count) for ind in population.population]),
                                " ".join([str(f) for f in population.fit]), log
                                ]


                else:

                    add_info = [self.elite.test_fitness, log]

                # logging the desired results
                logger(log_path, it, self.elite.fitness, end - start, float(population.nodes_count),
                               additional_infos=add_info, run_info=run_info, seed=self.seed)

            if verbose != 0:
                verbose_reporter(run_info[-1], it, self.elite.fitness, self.elite.test_fitness, end - start,
                                         population.nodes_count)
