import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from sympy.utilities.lambdify import lambdify
from sympy.printing import srepr
import zlib
from algorithms.GP.representations.tree_utils import flatten
from math import isclose
from typing import List, Optional

def sigmoid_tree(x):
    # Returns full sigmoid tree: 1 / (1 + exp(-x))
    return (
        'divide', 1,
        (
            'add', 1,
            ('exp', ('neg', x))
        )
    )

def mutate_tree(tree, mutation_type='sigmoid', ms='ms', t2=None):
    """
    Apply a mutation to the given GP tree.

    Parameters:
    - tree: tuple, the original GP tree (t1)
    - mutation_type: str, one of 'sigmoid', 'abs', 'sigmoid2'
    - ms: str or float, scalar multiplier
    - t2: tuple, optional second GP tree for 'sigmoid2' mutation

    Returns:
    - mutated_tree: tuple representing the mutated GP tree
    """

    if mutation_type == 'sigmoid':
        # ms * (2 * sigmoid(tree) - 1)
        return (
            'multiply', ms,
            (
                'subtract',
                (
                    'multiply', 2,
                    sigmoid_tree(tree)
                ),
                1
            )
        )

    elif mutation_type == 'abs':
        # ms * (1 - 2 / (1 + abs(tree)))
        return (
            'multiply', ms,
            (
                'subtract', 1,
                (
                    'divide', 2,
                    (
                        'add', 1,
                        ('abs', tree)
                    )
                )
            )
        )

    elif mutation_type == 'sigmoid2':
        if t2 is None:
            raise ValueError("t2 must be provided for 'sigmoid2' mutation")

        # ms * (sigmoid(t1) - sigmoid(t2))
        return (
            'multiply', ms,
            (
                'subtract',
                sigmoid_tree(tree),
                sigmoid_tree(t2)
            )
        )

    else:
        raise ValueError("Unsupported mutation_type. Choose 'sigmoid', 'abs', or 'sigmoid2'.")



# Recursive converter
import sympy as sp

def tree_to_sympy(tree, x_symbols=None):
    """
    Recursively converts a GP tree (in tuple form) to a sympy expression.

    Parameters:
        tree (tuple | str | float): GP tree node.
        x_symbols (dict): Optional mapping from variable names to sympy symbols.

    Returns:
        sympy.Expr: Converted symbolic expression.
    """
    if x_symbols is None:
        x_symbols = {f'x{i}': sp.Symbol(f'x{i}') for i in range(100)}

    if isinstance(tree, (int, float)):
        return sp.sympify(tree)
    elif isinstance(tree, str):
        return x_symbols.get(tree, sp.Symbol(tree))

    op, *args = tree

    if op == 'add':
        return tree_to_sympy(args[0], x_symbols) + tree_to_sympy(args[1], x_symbols)
    elif op == 'subtract':
        return tree_to_sympy(args[0], x_symbols) - tree_to_sympy(args[1], x_symbols)
    elif op == 'multiply':
        return tree_to_sympy(args[0], x_symbols) * tree_to_sympy(args[1], x_symbols)
    elif op == 'divide':
        return tree_to_sympy(args[0], x_symbols) / tree_to_sympy(args[1], x_symbols)
    elif op == 'neg':
        return -tree_to_sympy(args[0], x_symbols)
    elif op == 'abs':
        return sp.Abs(tree_to_sympy(args[0], x_symbols))
    elif op == 'exp':
        return sp.exp(tree_to_sympy(args[0], x_symbols))
    else:
        raise ValueError(f"Unknown operator: {op}")


def sympy_to_tree(expr):
    # Atoms: numbers and symbols
    if expr.is_Number:
        return float(expr)
    if expr.is_Symbol:
        return str(expr)

    op = expr.func
    args = expr.args

    if op == sp.Add:
        # sympy Add can have multiple args; fold pairwise
        tree = sympy_to_tree(args[0])
        for arg in args[1:]:
            tree = ('add', tree, sympy_to_tree(arg))
        return tree

    elif op == sp.Mul:
        # handle unary minus: Mul(-1, x)
        args_list = list(args)
        if len(args_list) == 2 and args_list[0] == -1:
            return ('neg', sympy_to_tree(args_list[1]))

        # Separate numerator and denominator parts
        numerator = []
        denominator = []

        for a in args_list:
            if a.func == sp.Pow and len(a.args) == 2 and a.args[1] == -1:
                denominator.append(a.args[0])
            else:
                numerator.append(a)

        # build numerator tree
        if not numerator:
            # just denominator? then numerator is 1
            num_tree = 1.0
        elif len(numerator) == 1:
            num_tree = sympy_to_tree(numerator[0])
        else:
            num_tree = ('multiply', sympy_to_tree(numerator[0]), sympy_to_tree(numerator[1]))
            for arg in numerator[2:]:
                num_tree = ('multiply', num_tree, sympy_to_tree(arg))

        # build denominator tree
        if not denominator:
            return num_tree
        elif len(denominator) == 1:
            denom_tree = sympy_to_tree(denominator[0])
        else:
            denom_tree = ('multiply', sympy_to_tree(denominator[0]), sympy_to_tree(denominator[1]))
            for arg in denominator[2:]:
                denom_tree = ('multiply', denom_tree, sympy_to_tree(arg))

        return ('divide', num_tree, denom_tree)

    elif op == sp.Pow:
        base, exponent = args
        if exponent == -1:
            # handled in Mul, but just in case:
            return ('divide', 1.0, sympy_to_tree(base))
        else:
            return ('exp', sympy_to_tree(base), exponent)

    elif op == sp.exp:
        return ('exp', sympy_to_tree(args[0]))

    elif op == sp.Abs:
        return ('abs', sympy_to_tree(args[0]))

    elif op == sp.log:
        return ('log',  sympy_to_tree(args[0]))

    elif op == sp.core.numbers.Exp1:
        return 2.71828

    else:
        raise ValueError(f"Unsupported operator: {op}")




def kolmogorov_complexity(tree) -> int:
    """
    Approximate Kolmogorov complexity of a GP tree structure by compressing
    its serialized tuple representation.
    """
    tree_str = repr(tree)  # or str(tree) if preferred
    compressed = zlib.compress(tree_str.encode('utf-8'))
    return len(compressed)


# Compute slope-based complexity
def slope_complexity(tree, data_points) -> float:
    """
    Compute slope-based complexity from a GP tree without simplification.
    - data_points: ndarray of shape (n, m).
    """
    m = data_points.shape[1]
    n = data_points.shape[0]
    total_complexity = 0.0

    for j in range(m):
        sorted_idx = np.argsort(data_points[:, j])
        sorted_points = data_points[sorted_idx]
        f_vals = eval_tree(tree, sorted_points)

        slopes = []
        for i in range(n - 1):
            dx = sorted_points[i + 1, j] - sorted_points[i, j]
            slope = 0 if dx == 0 else (f_vals[i + 1] - f_vals[i]) / dx
            slopes.append(slope)

        total_complexity += sum(abs(slopes[i + 1] - slopes[i])
                                for i in range(len(slopes) - 1))

    return total_complexity / m


def eval_tree(tree, X):
    if isinstance(tree, str):  # terminal
        if tree.startswith('x'):
            return X[:, int(tree[1:])]  # variable
        else:
            return float(tree)  # constant
    elif isinstance(tree, (float, int)):  # numeric constant
        return tree
    elif isinstance(tree, tuple):
        op = tree[0]

        # Unary operators
        if op == 'neg':
            return -eval_tree(tree[1], X)
        if op == 'abs':
            return np.abs(eval_tree(tree[1], X))
        if op == 'exp':
            return np.exp(eval_tree(tree[1], X))

        # Binary operators
        left, right = tree[1], tree[2]
        a, b = eval_tree(left, X), eval_tree(right, X)
        if op == 'add':
            return a + b
        if op == 'subtract':
            return a - b
        if op == 'multiply':
            return a * b
        if op == 'divide':
            return np.where(b != 0, a / b, np.inf)

        raise ValueError(f"Unknown op {op}")
    else:
        raise ValueError(f"Invalid tree node type: {type(tree)}")

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else -np.inf

def sam_s(tree, X, eps=0.1, n=10):
    sigma = np.std(X, axis=0)
    idx = np.random.choice(len(X), n, replace=False)
    Xs = X[idx].copy()
    Xs_eps = Xs + np.random.uniform(-eps * sigma, eps * sigma)

    y_orig = eval_tree(tree, Xs)
    y_pert = eval_tree(tree, Xs_eps)

    return r2_score(y_orig, y_pert)



def get_info(tree):

    len_before_simpl = len(list(flatten(tree)))

    x_symbols = {f'x{i}': sp.Symbol(f'x{i}') for i in range(20)}

    s_tree = tree_to_sympy(tree, x_symbols)
    ss_tree = sp.simplify(s_tree)
    ss_tree = ss_tree.rewrite(sp.exp)

    len_after_simpl = len(list(flatten(sympy_to_tree(ss_tree))))

    data = np.random.uniform(-5, 5, size=(100, 20))
    data[data == 0] = -5

    k_complexity = kolmogorov_complexity(tree)
    s_complexity = slope_complexity(tree, data)
    sharpness = sam_s(tree, data)

    return [len_before_simpl, len_after_simpl, k_complexity, s_complexity, sharpness]


def compute_partial_complexity(p, semantics):
    unique_p = np.unique(p)

    n = len(unique_p)
    if n < 3:
        return 0.0

    median_g = np.array([np.median(semantics[p == val]) for val in unique_p])

    sorted_indices = np.argsort(unique_p)

    p_sorted = unique_p[sorted_indices]
    g_sorted = median_g[sorted_indices]

    pcFinal = 0.0

    for i in range(n - 2):
        pc = 0.0

        num1 = g_sorted[i + 1] - g_sorted[i]
        den1 = p_sorted[i + 1] - p_sorted[i]
        num2 = g_sorted[i + 2] - g_sorted[i + 1]
        den2 = p_sorted[i + 2] - p_sorted[i + 1]

        pc = abs((num1 / den1) - (num2 / den2))
        pcFinal += pc

    return pcFinal


def compute_complexity(X, semantics):
    fc = 0

    n, m = X.shape

    for feature in range(m):
        pc = compute_partial_complexity(X[:, feature], np.array(semantics))
        fc = fc + pc

    return fc / m

