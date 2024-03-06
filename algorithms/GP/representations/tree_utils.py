import random
import numpy as np
import torch
# from gp4os.utils.functions import TERMINALS

def bound_value(vector, min_val, max_val):
    """
        Constrains the values within a specific range.

        Parameters
        ----------
        vector : torch.Tensor
            Input tensor to be bounded.
        min_val : float
            Minimum value for bounding.
        max_val : float
            Maximum value for bounding.

        Returns
        -------
        torch.Tensor
            Tensor with values bounded between min_val and max_val.
    """

    return torch.clamp(vector, min_val, max_val)

def flatten(data):
    """
        Flattens a nested tuple structure.

        Parameters
        ----------
        data : tuple
            Input nested tuple data structure.

        Yields
        ------
        object
            Flattened data element by element.
    """

    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data

# Function to create a random grow tree.
def create_grow_random_tree(depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c = 0.3, first_call=True):
    """
        Generates a random tree using the Grow method with a specified depth.

        Parameters
        ----------
        depth : int
            Maximum depth of the tree to be created.

        FUNCTIONS : dict
            Dictionary of functions allowed in the tree.

        TERMINALS : dict
            Dictionary of terminal symbols allowed in the tree.

        CONSTANTS : dict
            Dictionary of constant values allowed in the tree.

        p_c : float, optional
            Probability of choosing a constant node. Default is 0.3.

        Returns
        -------
        tuple
            The generated tree according to the specified parameters.
        """

    num_functions = len(list(FUNCTIONS.keys()))
    num_constants = len(list(CONSTANTS.keys()))
    num_terminals = len(list(TERMINALS.keys()))
    p_t = (num_terminals + num_constants) / (num_terminals + num_constants + num_functions)

    if (depth <= 1 or random.random() < p_t) and not first_call:

        # Choose a terminal node (input or constant)
        if random.random() > p_c:
            node = np.random.choice(list(TERMINALS.keys()))
        else:
            node = np.random.choice(list(CONSTANTS.keys()))
    else:
        # Choose a function node
        node = np.random.choice(list(FUNCTIONS.keys()))
        if FUNCTIONS[node]['arity'] == 2:
            # Recursively create left and right subtrees
            left_subtree = create_grow_random_tree(depth - 1,  FUNCTIONS, TERMINALS, CONSTANTS, first_call=False)
            right_subtree = create_grow_random_tree(depth - 1,  FUNCTIONS, TERMINALS, CONSTANTS, first_call=False)
            node = (node, left_subtree, right_subtree)
        else:
            # Recursively create left and right subtrees
            left_subtree = create_grow_random_tree(depth - 1,  FUNCTIONS, TERMINALS, CONSTANTS, first_call=False)
            node = (node, left_subtree)

    return node

# Function to create a random full tree.
def create_full_random_tree(depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c = 0.3):
    """
        Generates a full random tree with a specified depth.

        Parameters
        ----------
        depth : int
            Maximum depth of the tree to be created.

        FUNCTIONS : dict
            Dictionary of functions allowed in the tree.

        TERMINALS : dict
            Dictionary of terminal symbols allowed in the tree.

        CONSTANTS : dict
            Dictionary of constant values allowed in the tree.

        p_c : float, optional
            Probability of choosing a function node. Default is 0.3.

        Returns
        -------
        tuple
            The generated full tree based on the specified parameters.
        """
    if depth <= 1:
        # Choose a terminal node (input or constant)
        if random.random() > p_c:
            node = np.random.choice(list(TERMINALS.keys()))
        else:
            node = np.random.choice(list(CONSTANTS.keys()))
    else:
        # Choose a function node
        node = np.random.choice(list(FUNCTIONS.keys()))
        if FUNCTIONS[node]['arity'] == 2:
            # Recursively create left and right subtrees
            left_subtree = create_full_random_tree(depth - 1,  FUNCTIONS, TERMINALS, CONSTANTS)
            right_subtree = create_full_random_tree(depth - 1,  FUNCTIONS, TERMINALS, CONSTANTS)
            node = (node, left_subtree, right_subtree)
        else:
            # Recursively create left and right subtrees
            left_subtree = create_full_random_tree(depth - 1,  FUNCTIONS, TERMINALS, CONSTANTS)
            node = (node, left_subtree)
    return node

# Helper function to select a random subtree from a tree.
def random_subtree(tree, FUNCTIONS, first_call = True):
    """
        Selects a random subtree from a given tree.

        Parameters
        ----------
        tree : tuple
            The tree to select the subtree from.

        FUNCTIONS : dict
            Dictionary of functions allowed in the tree.

        Returns
        -------
        tuple or terminal node
            The randomly selected subtree from the input tree.
        """



    if isinstance(tree, tuple):
        # Randomly choose to explore left or right or return the current subtree

        #calculating the number of nodes of the current tree
        current_number_of_nodes = len(list(flatten(tree)))
        if FUNCTIONS[tree[0]]['arity'] == 2:
            if first_call:
                # if it's the first time, 0 (the root node) cannot be returned
                # normalizing the probability of choosing left or right based on the number of nodes in each side
                subtree_exploration = 1 if random.random() < len(list(flatten(tree[1]))) / ( current_number_of_nodes -1) else 2
            else:
                p = random.random()
                subtree_exploration = 0 if p < 1/current_number_of_nodes else \
                                        (1 if p < len(list(flatten(tree[1]))) / current_number_of_nodes else 2)

        elif FUNCTIONS[tree[0]]['arity'] == 1:
            if first_call:
                subtree_exploration = 1
            else:
                subtree_exploration = 0 if random.random() < 1/current_number_of_nodes else 1

        if subtree_exploration == 0:
            return tree
        elif subtree_exploration == 1:
            return random_subtree(tree[1], FUNCTIONS, first_call = False) if isinstance(tree[1], tuple) else tree[1]
        elif subtree_exploration == 2:
            return random_subtree(tree[2], FUNCTIONS, first_call = False) if isinstance(tree[2], tuple) else tree[2]
    else:
        # If the tree is a terminal node, return it as is
        return tree


# Helper function to substitute a subtree in a tree.
def substitute_subtree(tree, target_subtree, new_subtree, FUNCTIONS):
    """
        Substitutes a specific subtree in a tree with a new subtree.

        Parameters
        ----------
        tree : tuple
            The tree where substitution occurs.

        target_subtree : tuple or terminal node
            The subtree to be replaced.

        new_subtree : tuple or terminal node
            The new subtree for replacement.

        FUNCTIONS : dict
            Dictionary of functions allowed in the tree.

        Returns
        -------
        tuple
            The tree after the subtree substitution.
        """

    if tree == target_subtree:
        return new_subtree
    elif isinstance(tree, tuple):
        if FUNCTIONS[tree[0]]['arity'] == 2:
            return (tree[0], substitute_subtree(tree[1], target_subtree, new_subtree, FUNCTIONS),
                    substitute_subtree(tree[2], target_subtree, new_subtree, FUNCTIONS))
        elif FUNCTIONS[tree[0]]['arity'] == 1:
            return (tree[0], substitute_subtree(tree[1], target_subtree, new_subtree, FUNCTIONS))
    else:
        return tree

# Function to reduce both sides of a tree to a specific depth.
def tree_pruning(tree, target_depth, TERMINALS, CONSTANTS, FUNCTIONS, p_c = 0.3):
    """
       Reduces both sides of a tree to a specific depth.

       Parameters
       ----------
       tree : tuple
           The tree to be pruned.

       target_depth : int
           The depth to reduce the tree to.

       TERMINALS : dict
           Dictionary of terminal symbols allowed in the tree.

       CONSTANTS : dict
           Dictionary of constant values allowed in the tree.

       FUNCTIONS : dict
           Dictionary of functions allowed in the tree.

       p_c : float, optional
           Probability of choosing a constant node. Default is 0.3.

       Returns
       -------
       tuple
           The pruned tree according to the specified depth.
       """

    if target_depth <= 1 and not tree in list(TERMINALS.keys()):
        # If the target depth is 1 or less, return a terminal node
        if random.random() > p_c:
            return np.random.choice(list(TERMINALS.keys()))
        else:
            return np.random.choice(list(CONSTANTS.keys()))
    elif not isinstance(tree, tuple):
        # If the tree is already a terminal node, return it
        return tree
    # Recursively reduce the left and right subtrees
    if FUNCTIONS[tree[0]]['arity'] == 2:
        new_left_subtree = tree_pruning(tree[1], target_depth - 1, TERMINALS, CONSTANTS, FUNCTIONS, p_c)
        new_right_subtree = tree_pruning(tree[2], target_depth - 1, TERMINALS, CONSTANTS, FUNCTIONS, p_c)
        return (tree[0], new_left_subtree, new_right_subtree)
    elif FUNCTIONS[tree[0]]['arity'] == 1:
        new_left_subtree = tree_pruning(tree[1], target_depth - 1, TERMINALS, CONSTANTS, FUNCTIONS, p_c)
        # new_right_subtree = tree_pruning(tree[2], target_depth - 1, TERMINALS, CONSTANTS, p_c)
        return (tree[0], new_left_subtree)


# Function to calculate the depth of a tree.
def tree_depth(tree, FUNCTIONS):
    """
        Calculates the depth of a given tree.

        Parameters
        ----------
        tree : tuple
            The tree to calculate the depth of.

        FUNCTIONS : dict
            Dictionary of functions allowed in the tree.

        Returns
        -------
        int
            The depth of the input tree.
        """

    if not isinstance(tree, tuple):
        # If it's a terminal node, the depth is 1
        return 1
    else:
        # Recursively calculate the depth of the left and right subtrees
        if FUNCTIONS[tree[0]]['arity'] == 2:
            left_depth = tree_depth(tree[1], FUNCTIONS)
            right_depth = tree_depth(tree[2], FUNCTIONS)
        elif FUNCTIONS[tree[0]]['arity'] == 1:
            left_depth = tree_depth(tree[1], FUNCTIONS)
            right_depth = 0
        # The depth of the tree is one more than the maximum depth of its subtrees
        return 1 + max(left_depth, right_depth)