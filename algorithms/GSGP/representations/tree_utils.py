from algorithms.GP.representations.tree import Tree
from algorithms.GP.representations.tree_utils import bound_value

def apply_tree(tree, inputs):
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

    if isinstance(tree.structure, tuple):  # If it's a function node
        function_name = tree.structure[0]
        if tree.FUNCTIONS[function_name]['arity'] == 2:
            left_subtree, right_subtree = tree.structure[1], tree.structure[2]
            left_subtree = Tree(left_subtree)
            right_subtree = Tree(right_subtree, )
            left_result = left_subtree.apply_tree(inputs)
            right_result = right_subtree.apply_tree(inputs)
            output = tree.FUNCTIONS[function_name]['function'](left_result, right_result)
        else:
            left_subtree = tree.structure[1]
            left_subtree = Tree(left_subtree)
            # right_subtree = Tree(right_subtree, tree.FUNCTIONS, tree.TERMINALS, tree.CONSTANTS)
            left_result = left_subtree.apply_tree(inputs)
            # right_result = right_subtree.apply_tree(inputs)
            output = tree.FUNCTIONS[function_name]['function'](left_result)

        return bound_value(output, -1000000000000.0, 10000000000000.0)

    else:  # If it's a terminal node
        # if tree.structure == '_':
        #     output = '_'
        if tree.structure in list(tree.TERMINALS.keys()):
            output = inputs[:, tree.TERMINALS[tree.structure]]

            return output

        elif tree.structure in list(tree.CONSTANTS.keys()):

            output = tree.CONSTANTS[tree.structure](1)

            return output

def nested_depth_calculator(operator, depths):

    if operator.__name__ == 'tt_delta_sum':
        depths[0] += 2
        depths[1] += 2

    elif operator.__name__ == 'tt_delta_mul':
        depths[0] += 3
        depths[1] += 3
        
    elif operator.__name__  == 'ot_delta_sum_True' :
        depths[0] += 3
        
    elif operator.__name__ in ['ot_delta_sum_False', 'ot_delta_mul_True']:
        depths[0] += 4
        
    elif operator.__name__ == 'ot_delta_mul_False':
        depths[0] += 5
        
    elif operator.__name__ == 'geometric_crossover':
        depths[:] += 2
        depths.append(depths[-1] + 1)
        

    return max(depths)



def nested_nodes_calculator(operator, nodes):
    extra_operators_nodes = [5, nodes[-1]] if operator.__name__ == 'geometric_crossover' \
        else (
            [7] if operator.__name__ == 'ot_delta_sum_True' else
            ([11] if operator.__name__ == 'ot_delta_mul_False' else
            ([9] if operator.__name__ == ['ot_delta_sum_False', 'ot_delta_mul_True'] else
            ([6] if operator.__name__ == 'tt_delta_mul' else
            ([4] if operator.__name__ == 'tt_delta_sum' else [0]
            )))))

    return sum([*nodes, *extra_operators_nodes])