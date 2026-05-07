from algorithms.GP.representations.tree_utils import bound_value


def _apply_struct(structure, inputs, FUNCTIONS, TERMINALS, CONSTANTS):
    """Evaluate a GP tree structure recursively without creating Tree objects."""
    if isinstance(structure, tuple):
        fn = FUNCTIONS[structure[0]]
        if fn['arity'] == 2:
            l = _apply_struct(structure[1], inputs, FUNCTIONS, TERMINALS, CONSTANTS)
            r = _apply_struct(structure[2], inputs, FUNCTIONS, TERMINALS, CONSTANTS)
            return bound_value(fn['function'](l, r), -1000000000000.0, 10000000000000.0)
        else:
            l = _apply_struct(structure[1], inputs, FUNCTIONS, TERMINALS, CONSTANTS)
            return bound_value(fn['function'](l), -1000000000000.0, 10000000000000.0)
    elif structure in TERMINALS:
        return inputs[:, TERMINALS[structure]]
    elif structure in CONSTANTS:
        return CONSTANTS[structure](1)


def apply_tree(tree, inputs):
    return _apply_struct(tree.structure, inputs, tree.FUNCTIONS, tree.TERMINALS, tree.CONSTANTS)

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