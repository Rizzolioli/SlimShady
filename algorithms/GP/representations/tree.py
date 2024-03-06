from algorithms.GP.representations.tree_utils import bound_value
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

    def __init__(self, repr_, FUNCTIONS, TERMINALS, CONSTANTS):

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

        self.repr_ = repr_
        self.FUNCTIONS = FUNCTIONS
        self.TERMINALS = TERMINALS
        self.CONSTANTS = CONSTANTS
        self.depth = len(repr_)
        self.fitness = None
        self.test_fitness = None

    # Function to evaluate a tree on input vectors x and y.
    def apply_tree(self, inputs):

        """
                Evaluates the tree on input vectors x and y.

                Parameters
                ----------
                inputs : tuple
                    Input vectors x and y.

                Returns
                -------
                float
                    Output of the evaluated tree.
        """

        if isinstance(self.repr_, tuple):  # If it's a function node
            function_name = self.repr_[0]
            if self.FUNCTIONS[function_name]['arity'] == 2:
                left_subtree, right_subtree = self.repr_[1], self.repr_[2]
                left_subtree = Tree(left_subtree, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)
                right_subtree = Tree(right_subtree, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)
                left_result = left_subtree.apply_tree(inputs)
                right_result = right_subtree.apply_tree(inputs)
                output = self.FUNCTIONS[function_name]['function'](left_result, right_result)
            else:
                left_subtree = self.repr_[1]
                left_subtree = Tree(left_subtree, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)
                # right_subtree = Tree(right_subtree, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)
                left_result = left_subtree.apply_tree(inputs)
                # right_result = right_subtree.apply_tree(inputs)
                output = self.FUNCTIONS[function_name]['function'](left_result)

            return bound_value(output, -1000000000000.0, 10000000000000.0)

        else:  # If it's a terminal node
            # if self.repr_ == '_':
            #     output = '_'
            if self.repr_ in list(self.TERMINALS.keys()):
                output = inputs[:, self.TERMINALS[self.repr_]]

                return output

            elif self.repr_ in list(self.CONSTANTS.keys()):

                output = self.CONSTANTS[self.repr_](1)

                return output

    def evaluate(self, ffunction, X, y, testing=False, validation=False):
        # TODO: Do we need validation? GSGP only uses testing and we agreed we wouldn't do a 3 part split
        #  inside the algorithm.

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
        validation: bool
            Flag symbolizing if the data is validation data.

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
        elif validation:
            self.validation_fitness = ffunction(y, preds)
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
            if self.FUNCTIONS[function_name]['arity'] == 2:
                left_subtree, right_subtree = self.repr_[1], self.repr_[2]
                left_subtree = Tree(left_subtree, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)
                right_subtree = Tree(right_subtree, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)
                left_subtree.print_tree_representation(indent + "  ")
                right_subtree.print_tree_representation(indent + "  ")
            else:
                left_subtree = self.repr_[1]
                left_subtree = Tree(left_subtree, self.FUNCTIONS, self.TERMINALS, self.CONSTANTS)
                left_subtree.print_tree_representation(indent + "  ")
            print(indent + ")")
        else:  # If it's a terminal node
            print(indent + f"{self.repr_}")



