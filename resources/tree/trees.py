from tree.utils.utils import bound_value


class Tree():
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

        Methods
        -------
        __init__(repr_, FUNCTIONS, TERMINALS, CONSTANTS)
            Initializes a Tree object.

        apply_tree(inputs)
            Evaluates the tree on input vectors x and y.

        print_tree_representation(indent="")
            Prints the tree representations with indentation.
        """

    def __init__(self, repr_, FUNCTIONS, TERMINALS, CONSTANTS):

        """
                Initializes a Tree object.

                Parameters
                ----------
                repr_ : object
                    Representation of the tree structure.

                FUNCTIONS : dict
                    Dictionary of allowed functions in the tree.

                TERMINALS : dict
                    Dictionary of terminal symbols allowed in the tree.

                CONSTANTS : dict
                    Dictionary of constant values allowed in the tree.
        """

        self.repr_ = repr_
        self.FUNCTIONS = FUNCTIONS
        self.TERMINALS = TERMINALS
        self.CONSTANTS = CONSTANTS
        self.depth = len(repr_)


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

        x1, x2, x3, x4, x5, x6, x7 = inputs[0], inputs[1], inputs[2], inputs[3], \
                                    inputs[4], inputs[5], inputs[6]
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
                output = self.TERMINALS[self.repr_](x1, x2, x3, x4, x5, x6, x7)
                return output
            elif self.repr_ in list(self.CONSTANTS.keys()):
                output = self.CONSTANTS[self.repr_](x1, x2, x3, x4, x5, x6, x7)
                return output

    def evaluate_tree(self, solver, std_params, test = False):

        pass




    def print_tree_representation(self, indent=""):

        """
                Prints the tree representations with indentation.

                Parameters
                ----------
                indent : str, optional
                    Indentation for tree structure representations.
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














