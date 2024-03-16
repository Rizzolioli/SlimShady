from algorithms.GP.representations.tree_utils import random_subtree, substitute_subtree

# Function to perform crossover between two trees.
def crossover_trees(FUNCTIONS):

    """
    Performs crossover between two trees.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of allowed functions in the trees.

    Returns
    -------
    function
        Function to perform crossover between two trees.
    """

    def inner_xo(tree1, tree2, tree1_n_nodes, tree2_n_nodes): #todo: finish this liah

        if isinstance(tree1, tuple) and isinstance(tree2, tuple):
            # Randomly select crossover points in both trees
            crossover_point_tree1 = random_subtree(tree1, FUNCTIONS, num_of_nodes=tree1_n_nodes)
            crossover_point_tree2 = random_subtree(tree2, FUNCTIONS, num_of_nodes=tree2_n_nodes)

            # Swap subtrees at the crossover points
            new_tree1 = substitute_subtree(tree1, crossover_point_tree1, crossover_point_tree2, FUNCTIONS)
            new_tree2 = substitute_subtree(tree2, crossover_point_tree2, crossover_point_tree1, FUNCTIONS)

            return new_tree1, new_tree2
        else:
            # If either tree is a terminal node, return them as they are (no crossover)
            return tree1, tree2

    return inner_xo