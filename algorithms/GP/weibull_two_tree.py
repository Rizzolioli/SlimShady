import time
import random
import torch
import numpy as np

from utils.utils import verbose_reporter
from utils.logger import logger

from algorithms.GP.representations.population import Population
from algorithms.GP.representations.tree import Tree

from evaluators.fitness_functions import weibull_cindex_loss, weibull_expected_time


class GPTwoTreeIndividual:
    def __init__(self, tree1: Tree, tree2: Tree):
        self.tree1 = tree1
        self.tree2 = tree2
        self.node_count = tree1.node_count + tree2.node_count
        self.fitness = None
        self.test_fitness = None

    def evaluate(self, ffunction, X, y, testing=False):
        if testing:
            p1 = self.tree1.apply_tree(X)
            p2 = self.tree2.apply_tree(X)
            params = torch.stack([p1, p2])
            self.test_fitness = ffunction(y, params)
        else:
            p1 = self.tree1.apply_tree(X)
            p2 = self.tree2.apply_tree(X)
            params = torch.stack([p1, p2])
            self.fitness = ffunction(y, params)


class GPWeibullTwoTree:

    def __init__(self, pi_init, initializer, selector, mutator, crossover, find_elit_func,
                 p_m=0.2, p_xo=0.8, pop_size=100, seed=0, settings_dict=None):

        self.pi_init = pi_init
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

    def _make_two_tree_individual(self, t1, t2):
        return GPTwoTreeIndividual(Tree(t1), Tree(t2))

    def solve(self, X_train, X_test, y_train, y_test, curr_dataset, n_iter=20, elitism=True, log=0, verbose=0,
              test_elite=False, log_path=None, run_info=None, max_depth=None, max_=False,
              ffunction=weibull_cindex_loss, n_elites=1, tree_pruner=None, depth_calculator=None):

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

        population.evaluate(ffunction, X=X_train, y=y_train)
        end = time.time()

        self.elites, self.elite = self.find_elit_func(population, n_elites)
        if test_elite:
            self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

        if log != 0:
            add_info = [self.elite.test_fitness, self.elite.node_count, log]
            logger(log_path, 0, self.elite.fitness, end-start, float(population.nodes_count),
                   additional_infos=add_info, run_info=run_info, seed=self.seed)

        if verbose != 0:
            verbose_reporter(curr_dataset.split("load_")[-1], 0, self.elite.fitness, self.elite.test_fitness,
                             end-start, self.elite.node_count)

        for it in range(1, n_iter + 1, 1):
            offs_pop, start = [], time.time()

            if elitism:
                offs_pop.extend(self.elites)

            while len(offs_pop) < self.pop_size:
                if random.random() < self.p_xo:
                    p1, p2 = self.selector(population), self.selector(population)
                    while p1 == p2:
                        p1, p2 = self.selector(population), self.selector(population)

                    # crossover both trees independently
                    c11, c12 = self.crossover(p1.tree1.repr_, p2.tree1.repr_, tree1_n_nodes=p1.tree1.node_count, tree2_n_nodes=p2.tree1.node_count)
                    c21, c22 = self.crossover(p1.tree2.repr_, p2.tree2.repr_, tree1_n_nodes=p1.tree2.node_count, tree2_n_nodes=p2.tree2.node_count)

                    off = [self._make_two_tree_individual(c11, c21), self._make_two_tree_individual(c12, c22)]
                else:
                    p = self.selector(population)
                    m1 = self.mutator(p.tree1.repr_, num_of_nodes=p.tree1.node_count)
                    m2 = self.mutator(p.tree2.repr_, num_of_nodes=p.tree2.node_count)
                    off = [self._make_two_tree_individual(m1, m2)]

                offs_pop.extend(off)

            if len(offs_pop) > population.size:
                offs_pop = offs_pop[:population.size]

            offs_pop = Population(offs_pop)
            offs_pop.evaluate(ffunction, X=X_train, y=y_train)
            population = offs_pop

            end = time.time()
            self.elites, self.elite = self.find_elit_func(population, n_elites)
            if test_elite:
                self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

            if log != 0:
                add_info = [self.elite.test_fitness, self.elite.node_count, log]
                logger(log_path, it, self.elite.fitness, end - start, float(population.nodes_count),
                       additional_infos=add_info, run_info=run_info, seed=self.seed)

            if verbose != 0:
                verbose_reporter(run_info[-1], it, self.elite.fitness, self.elite.test_fitness, end - start, self.elite.node_count)


