import random
import time

import numpy as np
import torch
from algorithms.GP.representations.tree import Tree as GP_Tree
from algorithms.GSGP.representations.population import Population
from algorithms.GSGP.representations.tree import Tree
from algorithms.GSGP.representations.tree_utils import (
    apply_tree, nested_depth_calculator, nested_nodes_calculator)
from utils.diversity import gsgp_pop_div_from_vectors
from utils.logger import logger
from utils.utils import get_random_tree, verbose_reporter


class GSGP:

    def __init__(
        self,
        pi_init,
        initializer,
        selector,
        mutator,
        ms,
        crossover,
        find_elit_func,
        p_m=0.8,
        p_xo=0.2,
        pop_size=100,
        seed=0,
        settings_dict=None,
    ):

        # other initial parameters, tipo dataset
        self.pi_init = (
            pi_init  # dictionary with all the parameters needed for evaluation
        )
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

        Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        Tree.TERMINALS = pi_init["TERMINALS"]
        Tree.CONSTANTS = pi_init["CONSTANTS"]
        GP_Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        GP_Tree.TERMINALS = pi_init["TERMINALS"]
        GP_Tree.CONSTANTS = pi_init["CONSTANTS"]

    def solve(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        curr_dataset,
        n_iter=20,
        elitism=True,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        run_info=None,
        ffunction=None,
        reconstruct=False,
        n_elites=1,
    ):

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        ################################################################################################################

        # INITIALIZATION #

        ################################################################################################################

        # initializing the population
        population = Population(
            [
                Tree(
                    structure=tree,
                    train_semantics=None,
                    test_semantics=None,
                    reconstruct=True,
                )
                for tree in self.initializer(**self.pi_init)
            ]
        )  # reconstruct set as true to calculate the initial pop semantics

        # getting the individuals' semantics
        population.calculate_semantics(X_train)

        if test_elite:
            population.calculate_semantics(X_test, testing=True)

        # getting individuals' fitness
        population.evaluate(ffunction, y=y_train)

        end = time.time()

        # obtaining the initial population elites
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        # testing the elite on validation/testing, if applicable
        if test_elite:
            self.elite.evaluate(ffunction, y=y_test, testing=True)

        # logging the results for the population initialization
        if log != 0:

            if log == 2:
                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes,
                    gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                (
                                    ind.train_semantics
                                    if ind.train_semantics.shape != torch.Size([])
                                    else ind.train_semantics.repeat(len(X_train))
                                )
                                for ind in population.population
                            ]
                        )
                    ),
                    np.std(population.fit),
                    log,
                ]

            # log level 3 saves the number of nodes and fitness of all the individuals in the population
            elif log == 3:

                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes,
                    " ".join([str(ind.nodes_count) for ind in population.population]),
                    " ".join([str(f) for f in population.fit]),
                    log,
                ]

            elif log == 4:

                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes,
                    gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                (
                                    ind.train_semantics
                                    if ind.train_semantics.shape != torch.Size([])
                                    else ind.train_semantics.repeat(len(X_train))
                                )
                                for ind in population.population
                            ]
                        )
                    ),
                    np.std(population.fit),
                    " ".join([str(ind.nodes) for ind in population.population]),
                    " ".join([str(f) for f in population.fit]),
                    log,
                ]

            else:

                add_info = [self.elite.test_fitness, self.elite.nodes, log]

            logger(
                log_path,
                0,
                self.elite.fitness,
                end - start,
                float(population.nodes_count),
                additional_infos=add_info,
                run_info=run_info,
                seed=self.seed,
            )

        # displaying the results for the population initialization on console
        if verbose != 0:
            verbose_reporter(
                curr_dataset.split("load_")[-1],
                0,
                self.elite.fitness,
                self.elite.test_fitness,
                end - start,
                self.elite.nodes,
            )

        ################################################################################################################

        # GP EVOLUTION #

        ################################################################################################################

        for it in range(1, n_iter + 1, 1):

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
                    r_tree = get_random_tree(
                        max_depth=self.pi_init["init_depth"],
                        FUNCTIONS=self.pi_init["FUNCTIONS"],
                        TERMINALS=self.pi_init["TERMINALS"],
                        CONSTANTS=self.pi_init["CONSTANTS"],
                        inputs=X_train,
                        logistic=True,
                        p_c=self.pi_init["p_c"],
                    )

                    # calculating its semantics on testing, if applicable
                    if test_elite:
                        r_tree.calculate_semantics(X_test, testing=True, logistic=True)

                    # the two parents generate one offspring
                    offs1 = Tree(
                        structure=(
                            [self.crossover, p1, p2, r_tree] if reconstruct else None
                        ),
                        train_semantics=self.crossover(p1, p2, r_tree, testing=False),
                        test_semantics=(
                            self.crossover(p1, p2, r_tree, testing=True)
                            if test_elite
                            else None
                        ),
                        reconstruct=reconstruct,
                    )
                    if not reconstruct:
                        offs1.nodes = nested_nodes_calculator(
                            self.crossover, [p1.nodes, p2.nodes, r_tree.nodes]
                        )
                        offs1.depth = nested_depth_calculator(
                            self.crossover, [p1.depth, p2.depth, r_tree.depth]
                        )

                    # adding the offspring to the population
                    offs_pop.append(offs1)

                else:
                    # if mutation choose one parent
                    p1 = self.selector(population)

                    # determining the mutation step
                    ms_ = self.ms()

                    # checking if one or two trees are required for mutation
                    if self.mutator.__name__ in [
                        "standard_geometric_mutation",
                        "product_two_trees_geometric_mutation",
                    ]:

                        r_tree1 = get_random_tree(
                            max_depth=self.pi_init["init_depth"],
                            FUNCTIONS=self.pi_init["FUNCTIONS"],
                            TERMINALS=self.pi_init["TERMINALS"],
                            CONSTANTS=self.pi_init["CONSTANTS"],
                            inputs=X_train,
                            p_c=self.pi_init["p_c"],
                        )

                        r_tree2 = get_random_tree(
                            max_depth=self.pi_init["init_depth"],
                            FUNCTIONS=self.pi_init["FUNCTIONS"],
                            TERMINALS=self.pi_init["TERMINALS"],
                            CONSTANTS=self.pi_init["CONSTANTS"],
                            inputs=X_train,
                            p_c=self.pi_init["p_c"],
                        )

                        mutation_trees = [r_tree1, r_tree2]

                        # calculating random trees' semantics on testing, if applicable
                        if test_elite:
                            [
                                rt.calculate_semantics(
                                    X_test, testing=True, logistic=True
                                )
                                for rt in mutation_trees
                            ]

                    else:
                        # if only one tree is used, no logistic function is needed
                        r_tree1 = get_random_tree(
                            max_depth=self.pi_init["init_depth"],
                            FUNCTIONS=self.pi_init["FUNCTIONS"],
                            TERMINALS=self.pi_init["TERMINALS"],
                            CONSTANTS=self.pi_init["CONSTANTS"],
                            inputs=X_train,
                            logistic=False,
                            p_c=self.pi_init["p_c"],
                        )

                        mutation_trees = [r_tree1]

                        # calculating random trees' semantics on testing, if applicable
                        if test_elite:
                            r_tree1.calculate_semantics(
                                X_test, testing=True, logistic=False
                            )

                    # mutating the individual
                    offs1 = Tree(
                        structure=(
                            [self.mutator, p1, *mutation_trees, ms_]
                            if reconstruct
                            else None
                        ),
                        train_semantics=self.mutator(
                            p1, *mutation_trees, ms_, testing=False
                        ),
                        test_semantics=(
                            self.mutator(p1, *mutation_trees, ms_, testing=True)
                            if test_elite
                            else None
                        ),
                        reconstruct=reconstruct,
                    )

                    # adding the individual to the population
                    offs_pop.append(offs1)
                    if not reconstruct:
                        offs1.nodes = nested_nodes_calculator(
                            self.mutator,
                            [p1.nodes, *[rt.nodes for rt in mutation_trees]],
                        )
                        offs1.depth = nested_depth_calculator(
                            self.mutator,
                            [p1.depth, *[rt.depth for rt in mutation_trees]],
                        )

            if len(offs_pop) > population.size:
                offs_pop = offs_pop[: population.size]

            offs_pop = Population(offs_pop)
            # offs_pop.calculate_semantics(X_train)
            #
            # if test_elite:
            #     offs_pop.calculate_semantics(X_test, testing=True)

            offs_pop.evaluate(ffunction, y=y_train)

            population = offs_pop

            end = time.time()

            # getting the population elite
            self.elites, self.elite = self.find_elit_func(population, n_elites)

            if test_elite:
                self.elite.evaluate(ffunction, y=y_test, testing=True)

            # logging the results for the current generation
            if log != 0:

                if log == 2:
                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes,
                        gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    (
                                        ind.train_semantics
                                        if ind.train_semantics.shape != torch.Size([])
                                        else ind.train_semantics.repeat(len(X_train))
                                    )
                                    for ind in population.population
                                ]
                            )
                        ),
                        np.std(population.fit),
                        log,
                    ]

                # log level 3 saves the number of nodes and fitness of all the individuals in the population
                elif log == 3:

                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes,
                        " ".join(
                            [str(ind.nodes_count) for ind in population.population]
                        ),
                        " ".join([str(f) for f in population.fit]),
                        log,
                    ]

                elif log == 4:

                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes,
                        gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    (
                                        ind.train_semantics
                                        if ind.train_semantics.shape != torch.Size([])
                                        else ind.train_semantics.repeat(len(X_train))
                                    )
                                    for ind in population.population
                                ]
                            )
                        ),
                        np.std(population.fit),
                        " ".join([str(ind.nodes) for ind in population.population]),
                        " ".join([str(f) for f in population.fit]),
                        log,
                    ]

                else:

                    add_info = [self.elite.test_fitness, self.elite.nodes, log]

                logger(
                    log_path,
                    it,
                    self.elite.fitness,
                    end - start,
                    float(population.nodes_count),
                    additional_infos=add_info,
                    run_info=run_info,
                    seed=self.seed,
                )

            # displaying the results for the current generation on console
            if verbose != 0:
                verbose_reporter(
                    run_info[-1],
                    it,
                    self.elite.fitness,
                    self.elite.test_fitness,
                    end - start,
                    self.elite.nodes,
                )
