from algorithms.GP.representations.tree_utils import bound_value
from algorithms.GP.representations.tree_utils import flatten, tree_depth


_APPLY_MARKER = object()  # sentinel that separates tree nodes from deferred fn-apply items


def _apply(structure, inputs, FUNCTIONS, TERMINALS, CONSTANTS):
    """Iterative tree evaluation — no recursion, no Tree object creation per node."""
    operands = []
    work = [structure]
    while work:
        item = work.pop()
        if isinstance(item, tuple):
            if isinstance(item[0], str):
                # GP tree node: (fn_name, left[, right])
                fn_info = FUNCTIONS[item[0]]
                arity = fn_info['arity']
                # Push sentinel first (executed last, after children are on operands stack)
                work.append((_APPLY_MARKER, fn_info['function'], arity))
                if arity == 2:
                    work.append(item[2])  # right — popped second
                    work.append(item[1])  # left  — popped first
                else:
                    work.append(item[1])
            else:
                # Deferred function application: (_APPLY_MARKER, fn, arity)
                _, fn, arity = item
                if arity == 2:
                    r, l = operands.pop(), operands.pop()
                    operands.append(bound_value(fn(l, r), -1000000000000.0, 10000000000000.0))
                else:
                    operands.append(bound_value(fn(operands.pop()), -1000000000000.0, 10000000000000.0))
        elif item in TERMINALS:
            operands.append(inputs[:, TERMINALS[item]])
        elif item in CONSTANTS:
            operands.append(CONSTANTS[item](1))
    return operands[0]


class Tree:

    """
            Represents a tree structure for genetic programming.

            Attributes
            ----------
            repr_ : object
                Representation of the tree structure.

            functions : dict
                Dictionary of allowed functions in the tree.

            terminals : dict
                Dictionary of terminal symbols allowed in the tree.

            constants : dict
                Dictionary of constant values allowed in the tree.

            depth : int
                Depth of the tree structure.

            Methods
            -------
            __init__(repr_, FUNCTIONS, TERMINALS, CONSTANTS)
                Initializes a Tree object.

            apply_tree(inputs)
                Evaluates the tree on input vectors x and y.

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

                functions : dict
                    Dictionary of allowed functions in the tree.

                terminals : dict
                    Dictionary of terminal symbols allowed in the tree.

                constants : dict
                    Dictionary of constant values allowed in the tree.
        """
        self.FUNCTIONS = Tree.FUNCTIONS
        self.TERMINALS = Tree.TERMINALS
        self.CONSTANTS = Tree.CONSTANTS

        self.repr_ = repr_
        self.depth = tree_depth(Tree.FUNCTIONS)(repr_)
        self.fitness = None
        self.test_fitness = None
        self.node_count = len(list(flatten(self.repr_)))
    # Function to evaluate a tree on input vectors x and y.
    def apply_tree(self, inputs):
        return _apply(self.repr_, inputs, Tree.FUNCTIONS, Tree.TERMINALS, Tree.CONSTANTS)

    def evaluate(self, ffunction, X, y, testing=False):

        """
        evaluates the tree given a certain fitness function, input data(x) and target data (y).

        The result of this evaluation (given the output of ffunction) will be stored as a parameter of self.
        The testing and validation optional parameters specify which partition of the data will the fitness be
        attributed to. If both are False the data is considered training data.

        Parameters
        ----------
        ffunction: function
            fitness function to evaluate the individual
        X: torch tensor
            the input data (which can be training or testing)
        y: torch tensor
            the expected output (target) values
        testing: bool
            Flag symbolizing if the data is testing data.

        Returns
        -------
        None
            attributes a fitness value to the tree
        """
        # obtaining the output of the tree from input data
        preds = self.apply_tree(X)

        # attributing the tree fitness
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
                    Indentation for tree structure representation.
        """

        if isinstance(self.repr_, tuple):  # If it's a function node
            function_name = self.repr_[0]

            print(indent + f"{function_name}(")
            if Tree.FUNCTIONS[function_name]['arity'] == 2:
                left_subtree, right_subtree = self.repr_[1], self.repr_[2]
                left_subtree = Tree(left_subtree)
                right_subtree = Tree(right_subtree)
                left_subtree.print_tree_representation(indent + "  ")
                right_subtree.print_tree_representation(indent + "  ")
            else:
                left_subtree = self.repr_[1]
                left_subtree = Tree(left_subtree)
                left_subtree.print_tree_representation(indent + "  ")
            print(indent + ")")
        else:  # If it's a terminal node
            print(indent + f"{self.repr_}")



