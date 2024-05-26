"""
Tree class implementation for representing tree structures in genetic programming.
"""

from main.algorithms.GP.representations.tree_utils import (bound_value, flatten,
                                                           tree_depth)


class Tree:
    """
    Represents a tree structure for genetic programming.

    Attributes
    ----------
    repr_ : object
        Representation of the tree structure.

    FUNCTIONS : dict
        Dictionary of allowed functions in the tree.

    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.

    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.

    depth : int
        Depth of the tree structure.

    fitness : float
        Fitness value of the tree.

    test_fitness : float
        Test fitness value of the tree.

    node_count : int
        Number of nodes in the tree.

    Methods
    -------
    __init__(repr_)
        Initializes a Tree object.

    apply_tree(inputs)
        Evaluates the tree on input vectors.

    evaluate(ffunction, X, y, testing=False)
        Evaluates the tree given a fitness function and data.

    print_tree_representation(indent="")
        Prints the tree representation with indentation.
    """

    TERMINALS = None
    FUNCTIONS = None
    CONSTANTS = None

    def __init__(self, repr_):
        """
        Initializes a Tree object.

        Parameters
        ----------
        repr_ : object
            Representation of the tree structure.
        """
        self.FUNCTIONS = Tree.FUNCTIONS
        self.TERMINALS = Tree.TERMINALS
        self.CONSTANTS = Tree.CONSTANTS

        self.repr_ = repr_
        self.depth = tree_depth(Tree.FUNCTIONS)(repr_)
        self.fitness = None
        self.test_fitness = None
        self.node_count = len(list(flatten(self.repr_)))

    def apply_tree(self, inputs):
        """
        Evaluates the tree on input vectors.

        Parameters
        ----------
        inputs : tuple
            Input vectors.

        Returns
        -------
        float
            Output of the evaluated tree.
        """
        if isinstance(self.repr_, tuple):  # If it's a function node
            function_name = self.repr_[0]
            if Tree.FUNCTIONS[function_name]["arity"] == 2:
                left_subtree, right_subtree = self.repr_[1], self.repr_[2]
                left_result = Tree(left_subtree).apply_tree(inputs)
                right_result = Tree(right_subtree).apply_tree(inputs)
                output = Tree.FUNCTIONS[function_name]["function"](
                    left_result, right_result
                )
            else:
                left_subtree = self.repr_[1]
                left_result = Tree(left_subtree).apply_tree(inputs)
                output = Tree.FUNCTIONS[function_name]["function"](left_result)

            return bound_value(output, -1e12, 1e12)
        else:  # If it's a terminal node
            if self.repr_ in self.TERMINALS:
                return inputs[:, self.TERMINALS[self.repr_]]
            elif self.repr_ in self.CONSTANTS:
                return self.CONSTANTS

    def evaluate(self, ffunction, X, y, testing=False):
        """
        Evaluates the tree given a fitness function, input data (X), and target data (y).

        Parameters
        ----------
        ffunction : function
            Fitness function to evaluate the individual.
        X : torch.Tensor
            The input data (which can be training or testing).
        y : torch.Tensor
            The expected output (target) values.
        testing : bool, optional
            Flag indicating if the data is testing data. Default is False.

        Returns
        -------
        None
            Attributes a fitness value to the tree.
        """
        preds = self.apply_tree(X)
        if testing:
            self.test_fitness = ffunction(y, preds)
        else:
            self.fitness = ffunction(y, preds)

    def print_tree_representation(self, indent=""):
        """
        Prints the tree representation with indentation.

        Parameters
        ----------
        indent : str, optional
            Indentation for tree structure representation. Default is an empty string.
        """
        if isinstance(self.repr_, tuple):  # If it's a function node
            function_name = self.repr_[0]
            print(indent + f"{function_name}(")
            if Tree.FUNCTIONS[function_name]["arity"] == 2:
                left_subtree, right_subtree = self.repr_[1], self.repr_[2]
                Tree(left_subtree).print_tree_representation(indent + "  ")
                Tree(right_subtree).print_tree_representation(indent + "  ")
            else:
                left_subtree = self.repr_[1]
                Tree(left_subtree).print_tree_representation(indent + "  ")
            print(indent + ")")
        else:  # If it's a terminal node
            print(indent + f"{self.repr_}")
