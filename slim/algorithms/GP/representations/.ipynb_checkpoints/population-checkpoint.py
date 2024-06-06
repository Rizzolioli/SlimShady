"""
Population class implementation for evaluating genetic programming individuals.
"""
from joblib import Parallel, delayed
from slim.algorithms.GP.representations.tree_utils import _execute_tree, _evaluate_pop, _evaluate_pop_2

import time # todo: remove
import csv # todo: remove

class Population:
    def __init__(self, pop):
        """
        Initializes the Population object.

        Parameters
        ----------
        pop : list
            List of individual objects that make up the population.
        """
        self.population = pop
        self.size = len(pop)
        self.nodes_count = sum(ind.node_count for ind in pop)

    def evaluate(self, ffunction, X, y, n_jobs = 1):
        #
        # todo
        # move n_jobs to GP run definitions
        #
        """
        Evaluates the population given a certain fitness function, input data (X), and target data (y).

        Parameters
        ----------
        ffunction : function
            Fitness function to evaluate the individual.
        X : torch.Tensor
            The input data (which can be training or testing).
        y : torch.Tensor
            The expected output (target) values.

        Returns
        -------
        None
            Attributes a fitness tensor to the population.
        """
        # -------------------------------
        start_time = time.time()
        for individual in self.population:
            individual.evaluate(ffunction, X, y)

        self.fit = [individual.fitness for individual in self.population]
        # print(self.fit)
        et_loop = time.time() - start_time
        # print('\nLoop execution_time:\t\t\t\t{}'.format(et_loop))
        # -------------------------------
        
        # todo
        # is there a bettter why for using functions, terminals and constants here?
        # the import throws a circular import error
        #
        start_time = time.time()
        # Evaluates individuals semantics
        y_pred = Parallel(n_jobs=n_jobs)(
            delayed(_execute_tree)(
                individual.repr_, X, 
                individual.FUNCTIONS, individual.TERMINALS, individual.CONSTANTS
            ) for individual in self.population
        )
        # print('y pred {} (len {})'.format(y_pred[:5], len(y_pred)))
        # print('y {}'.format(y[:5]))

        # Evaluate fitnesses
        self.fit = [ffunction(y, y_pred_ind) for y_pred_ind in y_pred]
        # fits = Parallel(n_jobs=n_jobs)(
        #     delayed(_evaluate_pop)(
        #         ffunction, y, y_pred_ind
        #     ) for y_pred_ind in y_pred
        # )
        # print('fits {}'.format(fits))
        

        # Assign individuals' fitness
        [self.population[i].__setattr__('fit', f) for i, f in enumerate(self.fit)]

        et_parallel_1 = time.time() - start_time
        # print('_execute_tree Parallel execution_time:\t\t{}'.format(et_parallel_1))

        # -------------------------------
        
        start_time = time.time()
        # Evaluates individuals fits
        self.fit = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_pop_2)(
                ffunction, y,
                individual.repr_, X, 
                individual.FUNCTIONS, individual.TERMINALS, individual.CONSTANTS
            ) for individual in self.population
        )
        # print('y pred {} (len {})'.format(y_pred[:5], len(y_pred)))
        # print('y {}'.format(y[:5]))

        # Evaluate fitnesses
        # fits = [ffunction(y, y_pred_ind) for y_pred_ind in y_pred]
        # fits = Parallel(n_jobs=n_jobs)(
        #     delayed(_evaluate_pop)(
        #         ffunction, y, y_pred_ind
        #     ) for y_pred_ind in y_pred
        # )
        # print('fits {}'.format(fits))

        # Assign individuals' fitness
        [self.population[i].__setattr__('fit', f) for i, f in enumerate(self.fit)]

        et_parallel_2 = time.time() - start_time
        # print('_evaluate_pop_2 Parallel execution_time:\t{}\n'.format(et_parallel_2))

        # -------------------------------
        with open('execution_times_pop.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([et_loop, et_parallel_1, et_parallel_2])
        



