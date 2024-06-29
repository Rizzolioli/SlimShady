from algorithms.SLIM_GSGP.representations.individual import Individual
from algorithms.SLIM_GSGP.representations.population import Population
from algorithms.GP.representations.tree_utils import create_grow_random_tree
from algorithms.GP.representations.tree import Tree
import numpy as np

class SSHC():

    def __init__(self,
                 X_train,
                 y_train,
                 ffunction,
                 eval_operator,
                 neigh_operator,
                 FUNCTIONS,
                 TERMINALS,
                 CONSTANTS,
                 X_test = None,
                 y_test = None,
                 log = 0,
                 log_path = None,
                 verbose = 0,
                 initial_depth = None,
                 individual = None):

        self.X_train  = X_train
        self.y_train = y_train
        self.ffunction = ffunction
        self.eval_operator = eval_operator
        self.neigh_operator = neigh_operator
        self.FUNCTIONS = FUNCTIONS
        self.TERMINALS = TERMINALS
        self.CONSTANTS = CONSTANTS

        self.X_test  = X_test
        self.y_test = y_test
        self.log = log
        self.log_path = log_path
        self.verbose = verbose


        if individual is not None:
            self.individual = individual
        else:
            self.individual = Individual([
                Tree(create_grow_random_tree(initial_depth, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS),
                                               self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)])

        if self.individual.train_semantics is None:
            self.individual.calculate_semantics(X_train, testing=False)
        if X_test is not None and self.individual.test_semantics is None:
            self.individual.calculate_semantics(X_test, testing=True)


        if self.individual.fitness is None:
            self.individual.evaluate(ffunction, y=y_train, testing=False, operator=self.eval_operator)
        if X_test is not None and self.individual.test_fitness is None:
            self.individual.evaluate(ffunction, y=y_test, testing=True, operator=self.eval_operator)

        #todo add logger
        #todo add verbose

    def solve(self, neighborhood_size, generations, early_stopping = None):

        for i in range(generations):

            neighboors = []
            counter = 0
            while len(neighboors) < neighborhood_size:
                neighboors.append(self.neigh_operator(self.individual))

            neighboors = Population(neighboors)
            neighboors.calculate_semantics(self.X_train)

            neighboors.evaluate(self.ffunction, y=self.y_train, operator=self.eval_operator)

            counter += 1
            if min(self.population.fit) <= self.individual.fitness:

                counter = 0
                self.individual = np.argmin(self.population.fit)

            # todo add logger
            # todo add verbose

            if early_stopping is not None and counter >= early_stopping:
                break








