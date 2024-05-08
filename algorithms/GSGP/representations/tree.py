from algorithms.GSGP.representations.tree_utils import apply_tree, nested_depth_calculator, nested_nodes_calculator
from algorithms.GP.representations.tree_utils import flatten, tree_depth
import torch


class Tree:
    FUNCTIONS = None
    TERMINALS = None
    CONSTANTS = None

    def __init__(self, structure, train_semantics, test_semantics, reconstruct):
        self.FUNCTIONS = Tree.FUNCTIONS
        self.TERMINALS = Tree.TERMINALS
        self.CONSTANTS = Tree.CONSTANTS
        
        if structure is not None and reconstruct:
            self.structure = structure # either repr_ from gp(tuple) or list of pointers

        self.train_semantics = train_semantics
        self.test_semantics = test_semantics

        if isinstance(structure, tuple):
            self.depth = tree_depth(Tree.FUNCTIONS)(structure)
            self.nodes = len(list(flatten(structure)))
        elif reconstruct:
            self.depth = nested_depth_calculator(self.structure[0],
                                                  [tree.depth for tree in self.structure[1:] if isinstance(tree, Tree)])
            # operator_nodes = [5, self.structure[-1].nodes] if self.structure[0].__name__ == 'geometric_crossover' else [4]
            self.nodes = nested_nodes_calculator(self.structure[0],
                                                 [tree.nodes for tree in self.structure[1:] if isinstance(tree, Tree)])



        self.fitness = None
        self.test_fitness = None

    def calculate_semantics(self, inputs, testing = False, logistic=False):

        if testing and self.test_semantics is None:
            # checking if the individual is part of the initial population (table) or is a random tree (table)
            if isinstance(self.structure, tuple):
                self.test_semantics = torch.sigmoid(apply_tree(self, inputs)) if logistic else apply_tree(self, inputs)
            else:
                # self.structure[0] is the operator (mutation or xo) while the remaining of the structure dare pointers to the trees
                self.test_semantics = self.structure[0](*self.structure[1:], testing=True)
        elif self.train_semantics is None:
            if isinstance(self.structure, tuple):
                self.train_semantics = torch.sigmoid(apply_tree(self, inputs)) if logistic else apply_tree(self, inputs)
            else:
                self.train_semantics = self.structure[0](*self.structure[1:], testing=False)


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
