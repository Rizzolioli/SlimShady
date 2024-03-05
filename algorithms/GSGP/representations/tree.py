from algorithms.GSGP.representations.tree_utils import apply_tree
from algorithms.GP.representations.tree_utils import flatten

class Tree:
    #TODO add sigmoid

    def __init__(self, structure, FUNCTIONS, TERMINALS, CONSTANTS):

        self.structure = structure #either repr_ from gp(tuple) or list of pointers
        self.FUNCTIONS = FUNCTIONS
        self.TERMINALS = TERMINALS
        self.CONSTANTS = CONSTANTS

        if isinstance(structure, tuple):
            self.depth = len(structure)
        else:
            self.depth = [max(tree.depth) + 1 for tree in self.structure[1:] if isinstance(tree, Tree)] #TODO not the sum, and put inside function

        if isinstance(structure, tuple):
            self.nodes = len(list(flatten(structure)))
        else:
            self.nodes = sum([tree.nodes for tree in self.structure[1:] if isinstance(tree, Tree)] + [2]) #TODO not exactly the sum, and put inside function

        self.fitness = None
        self.test_fitness = None

    def calculate_semantics(self, inputs, testing = False):

        if isinstance(self.structure, tuple):
            # will be done only for initial population (table) and random trees (table)
            if testing:
                self.test_semantics = apply_tree(self, inputs)
            else:
                self.train_semantics = apply_tree(self, inputs)
        else:
            if testing:
                # self.structure[0] is the operator (mutation or xo) while the remaining of the structure dare pointers to the trees
                self.test_semantics = self.structure[0](*self.structure[1:], testing = True)
            else:
                self.train_semantics = self.structure[0](*self.structure[1:], testing = False)

    def evaluate(self, ffunction, y, testing=False): # TODO: fix documentation

        """
        evaluates the tree given a certain fitness function, input data(x) and target data (y)
        Parameters
        ----------
        ffunction: function
            fitness function to evaluate the individual
        y: torch tensor
            the expected output (target) values

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