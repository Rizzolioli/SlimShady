
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

    def __init__(self, repr_, functions, terminals, constants):

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
        self.functions = functions
        self.terminals = terminals
        self.constants = constants
        self.depth = len(repr_)

