from algorithms.GP.representations.tree_utils import bound_value


_APPLY_MARKER = object()  # sentinel for deferred function-apply stack items


def _apply_struct(structure, inputs, FUNCTIONS, TERMINALS, CONSTANTS):
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