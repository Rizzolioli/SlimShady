import time
import random
import torch
import numpy as np

from utils.utils import verbose_reporter, get_random_tree
from utils.logger import logger

from algorithms.GSGP.representations.population import Population
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.representations.individual import Individual

from evaluators.fitness_functions import weibull_cindex_loss
from algorithms.GSGP.operators.mutators import standard_geometric_mutation


class GSGPWeibullTwoTree:

    def __init__(self, pi_init, initializer, selector, ms, crossover, find_elit_func,
                 p_m=0.8, p_xo=0.2, pop_size=100, seed=0, settings_dict=None):

        self.pi_init = pi_init
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.ms = ms
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed

        self.settings_dict = settings_dict
        self.find_elit_func = find_elit_func

        Tree.FUNCTIONS = pi_init['FUNCTIONS']
        Tree.TERMINALS = pi_init['TERMINALS']
        Tree.CONSTANTS = pi_init['CONSTANTS']

    def _make_two_tree_individual(self, t1, t2):
        return Individual(collection=[Tree(structure=t1, train_semantics=None, test_semantics=None, reconstruct=True),
                                      Tree(structure=t2, train_semantics=None, test_semantics=None, reconstruct=True)],
                          train_semantics=None,
                          test_semantics=None,
                          reconstruct=True)

    def _mutate_tree(self, base_tree, X_train, X_test=None, reconstruct=True):
        r1 = get_random_tree(max_depth=self.pi_init['init_depth'],
                             FUNCTIONS=self.pi_init['FUNCTIONS'],
                             TERMINALS=self.pi_init['TERMINALS'],
                             CONSTANTS=self.pi_init['CONSTANTS'],
                             inputs=X_train,
                             logistic=False,
                             p_c=self.pi_init['p_c'])
        r2 = get_random_tree(max_depth=self.pi_init['init_depth'],
                             FUNCTIONS=self.pi_init['FUNCTIONS'],
                             TERMINALS=self.pi_init['TERMINALS'],
                             CONSTANTS=self.pi_init['CONSTANTS'],
                             inputs=X_train,
                             logistic=False,
                             p_c=self.pi_init['p_c'])
        step = self.ms()

        off = Tree(structure=[standard_geometric_mutation, base_tree, r1, r2, step] if reconstruct else None,
                   train_semantics=standard_geometric_mutation(base_tree, r1, r2, step, testing=False),
                   test_semantics=None,
                   reconstruct=reconstruct)
        if X_test is not None:
            r1.calculate_semantics(X_test, testing=True, logistic=False)
            r2.calculate_semantics(X_test, testing=True, logistic=False)
            base_tree.calculate_semantics(X_test, testing=True, logistic=False)
            off.test_semantics = standard_geometric_mutation(base_tree, r1, r2, step, testing=True)
        return off

    def solve(self, X_train, X_test, y_train, y_test, curr_dataset, n_iter=20, elitism=True, log=0, verbose=0,
              test_elite=False, log_path=None, run_info=None, reconstruct=True, n_elites=1):

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        init_structs_1 = self.initializer(**self.pi_init)
        init_structs_2 = self.initializer(**self.pi_init)
        size = min(len(init_structs_1), len(init_structs_2), self.pop_size)
        population = Population([
            self._make_two_tree_individual(init_structs_1[i], init_structs_2[i])
            for i in range(size)
        ])

        population.calculate_semantics(X_train)
        population.evaluate(weibull_cindex_loss, y=y_train)

        end = time.time()
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        if test_elite:
            population.calculate_semantics(X_test, testing=True)
            self.elite.evaluate(weibull_cindex_loss, y=y_test, testing=True)

        if log != 0:
            add_info = [self.elite.test_fitness, self.elite.nodes_count, log]
            logger(log_path, 0, self.elite.fitness, end - start, float(population.nodes_count),
                   additional_infos=add_info, run_info=run_info, seed=self.seed)

        if verbose != 0:
            verbose_reporter(curr_dataset.split("load_")[-1], 0, self.elite.fitness, self.elite.test_fitness, end - start,
                             self.elite.nodes_count)

        for it in range(1, n_iter + 1, 1):
            offs_pop, start = [], time.time()

            if elitism:
                offs_pop.extend(self.elites)

            while len(offs_pop) < self.pop_size:
                p = self.selector(population)

                if random.random() < self.p_m:
                    base_t1, base_t2 = p.collection[0], p.collection[1]
                    off_t1 = self._mutate_tree(base_t1, X_train, X_test if test_elite else None, reconstruct=reconstruct)
                    off_t2 = self._mutate_tree(base_t2, X_train, X_test if test_elite else None, reconstruct=reconstruct)

                    off = Individual(collection=[off_t1, off_t2] if reconstruct else None,
                                     train_semantics=torch.stack([off_t1.train_semantics, off_t2.train_semantics]),
                                     test_semantics=torch.stack([off_t1.test_semantics, off_t2.test_semantics]) if test_elite else None,
                                     reconstruct=reconstruct)
                    if not reconstruct:
                        off.size = 2
                else:
                    off = p

                offs_pop.append(off)

            if len(offs_pop) > population.size:
                offs_pop = offs_pop[:population.size]

            offs_pop = Population(offs_pop)
            offs_pop.calculate_semantics(X_train)
            offs_pop.evaluate(weibull_cindex_loss, y=y_train)
            population = offs_pop
            self.population = population

            end = time.time()
            self.elites, self.elite = self.find_elit_func(population, n_elites)

            if test_elite:
                self.elite.calculate_semantics(X_test, testing=True)
                self.elite.evaluate(weibull_cindex_loss, y=y_test, testing=True)

            if log != 0:
                add_info = [self.elite.test_fitness, self.elite.nodes_count, log]
                logger(log_path, it, self.elite.fitness, end - start, float(population.nodes_count),
                       additional_infos=add_info, run_info=run_info, seed=self.seed)

            if verbose != 0:
                verbose_reporter(run_info[-1], it, self.elite.fitness, self.elite.test_fitness, end - start,
                                 self.elite.nodes_count)


