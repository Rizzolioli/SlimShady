from algorithms.GSGP.representations.tree_utils import apply_tree
from algorithms.GP.representations.tree_utils import flatten
from algorithms.GP.representations import tree
import torch


class Tree:
    FUNCTIONS = None
    TERMINALS = None
    CONSTANTS = None

    def __init__(self, structure):
        self.FUNCTIONS = Tree.FUNCTIONS
        self.TERMINALS = Tree.TERMINALS
        self.CONSTANTS = Tree.CONSTANTS

        self.structure = structure # either repr_ from gp(tuple) or list of pointers

        if isinstance(structure, tuple):
            self.depth = len(structure)
        else:
            self.depth = max([tree.depth for tree in self.structure[1:] if isinstance(tree, Tree)]) + 1

        if isinstance(structure, tuple):
                    self.nodes = len(list(flatten(structure)))
        else:
            # operator_nodes = [5, self.structure[-1].nodes] if self.structure[0].__name__ == 'geometric_crossover' else [4]
            self.nodes = sum([*[tree.nodes for tree in self.structure[1:] if isinstance(tree, Tree)],
                              *([5, self.structure[-1].nodes] if self.structure[0].__name__ == 'geometric_crossover' else (
                                 [10] if self.structure[0].__name__ == 'ot_delta_mul' else
                                 ([6] if self.structure[0].__name__ == 'tt_delta_mul' else
                                  ([8] if self.structure[0].__name__ == 'ot_delta_sum' else [4])))
                                #TODO change checking performance aganist leo's
                                )])

        self.fitness = None
        self.test_fitness = None

    def calculate_semantics(self, inputs, testing = False, logistic=False):

        # checking if the individual is part of the initial population (table) or is a random tree (table)
        if isinstance(self.structure, tuple):
            if testing:
                self.test_semantics = torch.sigmoid(apply_tree(self, inputs)) if logistic else apply_tree(self, inputs)

            else:
                self.train_semantics = torch.sigmoid(apply_tree(self, inputs)) if logistic else apply_tree(self, inputs)

        # if the individual is a result of GSGP evolution
        else:
            if testing:
                # self.structure[0] is the operator (mutation or xo) while the remaining of the structure dare pointers to the trees
                self.test_semantics = self.structure[0](*self.structure[1:], testing = True)
            else:
                self.train_semantics = self.structure[0](*self.structure[1:], testing = False)

    def evaluate(self, ffunction, y, testing=False):

        """
        evaluates the tree given a certain fitness function (ffunction) x) and target data (y).

        This evaluation is performed by applying ffunction to the semantics of self and the expected output y. The
        optional parameter testing is used to control whether the training y or test y is being used as well as which
        semantics should be used. The result is stored as an attribute of self.

        Parameters
        ----------
        ffunction: function
            fitness function to evaluate the individual
        y: torch tensor
            the expected output (target) values
        testing: bool
            Flag symbolizing if the y is from testing

        Returns
        -------
        None
            attributes a fitness value to the tree
        """

        # attributing the tree fitness
        if testing:
            self.test_fitness = ffunction(y, self.test_semantics)
        else:
            self.fitness = ffunction(y, self.train_semantics)
