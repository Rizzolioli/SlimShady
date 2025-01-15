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

    if operator.__name__ == 'standard_geometric_mutation':
        depths[0] += 1
        depths[1:] = [d + 3 for d in depths[1:]]

    elif operator.__name__ == 'tt_delta_sum':
        depths[0] += 3
        
    elif operator.__name__ in ['tt_delta_mul', 'ot_delta_sum_True']:
        depths[0] += 4
        
    elif operator.__name__ in ['ot_delta_sum_False', 'ot_delta_mul_True']:
        depths[0] += 5
        
    elif operator.__name__ == 'ot_delta_mul_False':
        depths[0] += 6
        
    elif operator.__name__ in ['geometric_crossover', 'stdxo_delta']:
        depths = [d + 2 for d in depths]
        # depths[:] += 2
        depths.append(depths[-1] + 1)

    elif operator.__name__ == 'stdxo_ot_delta_first':
        depths[:] += 1

    elif operator.__name__ == 'stdxo_ot_delta_second':
        depths = [d + 1 for d in depths]
        # depths[0] += 1
        depths[1] += 2

    elif operator.__name__ == 'stdxo_a_delta':
        # depths[:] += 2
        depths = [d + 2 for d in depths]

    elif operator.__name__ in ['stdxo_ot_a_delta_first', 'stdxo_ot_a_delta_second']:
        depths = [d + 1 for d in depths]
        # depths[:] += 1

    elif operator.__name__ == 'combined_geometric_crossover':

        depths[:3] = [d +3 for d in depths[:3] ]
        depths[4:] = [d + 4 for d in depths[4:]]

    else:

        raise Exception('Invalid Operator')
        

    return max(depths)


def nested_nodes_calculator(operator, nodes):
    if operator.__name__ not in ['standard_geometric_mutation', 'tt_delta_sum',
                                'tt_delta_mul', 'ot_delta_sum_True',
                                'ot_delta_sum_False', 'ot_delta_mul_True',
                                'ot_delta_mul_False',
                                'geometric_crossover', 'stdxo_delta',
                                'stdxo_ot_delta_first',
                                'stdxo_ot_delta_second',
                                'stdxo_a_delta',
                                'stdxo_ot_a_delta_first', 'stdxo_ot_a_delta_second',
                                'combined_geometric_crossover',
                                ]:
        raise Exception("Invalid Operator")

    extra_operators_nodes = \
         \
         [5, nodes[-1]] if operator.__name__ in ['geometric_crossover', 'stdxo_delta'] else \
         \
         ([7] if operator.__name__ in ['ot_delta_sum_True', 'stdxo_a_delta'] else

         ([1] if operator.__name__ == 'stdxo_ot_delta_first' else

         ([3] if operator.__name__ == 'stdxo_ot_delta_second' else

         ([2] if operator.__name__ in ['stdxo_a_second', 'stdxo_ot_a_delta_first'] else

         ([11] if operator.__name__ == 'ot_delta_mul_False' else

         ([9] if operator.__name__ == ['ot_delta_sum_False', 'ot_delta_mul_True'] else

         ([6] if operator.__name__ == 'tt_delta_mul' else

         ([4] if operator.__name__ in ['stdxo_ot_a_delta_second','tt_delta_sum', 'standard_geometric_mutation'] else

          ([13] if operator.__name__ == 'combined_geometric_crossover' else [0]

           )))))))))

    return sum([*nodes, *extra_operators_nodes])