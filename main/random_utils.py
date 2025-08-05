import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from sympy.utilities.lambdify import lambdify
from sympy.printing import srepr
import zlib


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


import sympy as sp


import sympy as sp

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
            raise ValueError("Unsupported power operator other than exponent -1 (division)")

    elif op == sp.exp:
        return ('exp', sympy_to_tree(args[0]))

    elif op == sp.Abs:
        return ('abs', sympy_to_tree(args[0]))

    else:
        raise ValueError(f"Unsupported operator: {op}")





def tree_to_sympy_basic(tree, x_symbols=None):
    """
    Converts a GP tree to a SymPy expression using +, -, *, / (as Pow), and exp.
    Disables simplification by using evaluate=False.
    """
    if x_symbols is None:
        x_symbols = {f'x{i}': sp.Symbol(f'x{i}') for i in range(100)}

    if isinstance(tree, (int, float)):
        return sp.sympify(tree, rational=True)
    elif isinstance(tree, str):
        return x_symbols.get(tree, sp.Symbol(tree))

    op, *args = tree

    if op == 'add':
        return sp.Add(
            tree_to_sympy_basic(args[0], x_symbols),
            tree_to_sympy_basic(args[1], x_symbols),
            evaluate=False
        )
    elif op == 'subtract':
        return sp.Add(
            tree_to_sympy_basic(args[0], x_symbols),
            -tree_to_sympy_basic(args[1], x_symbols),
            evaluate=False
        )
    elif op == 'multiply':
        return sp.Mul(
            tree_to_sympy_basic(args[0], x_symbols),
            tree_to_sympy_basic(args[1], x_symbols),
            evaluate=False
        )
    elif op == 'divide':
        numerator = tree_to_sympy_basic(args[0], x_symbols)
        denominator = tree_to_sympy_basic(args[1], x_symbols)
        return sp.Mul(
            numerator,
            sp.Pow(denominator, -1, evaluate=False),
            evaluate=False
        )
    elif op == 'neg':
        return sp.Mul(-1, tree_to_sympy_basic(args[0], x_symbols), evaluate=False)
    elif op == 'exp':
        return sp.exp(tree_to_sympy_basic(args[0], x_symbols), evaluate=False)
    else:
        raise ValueError(f"Unsupported operator: {op}")


def count_nodes(expr):
    """
    Count all operator and operand nodes in a SymPy expression,
    excluding trivial internal wrappers.
    """
    # Base case: if expr is a symbol or a number (Atom)
    if expr.is_Atom:
        return 1

    # For some expressions like Pow with exponent -1 (which SymPy uses for division),
    # treat it as division operator node instead of separate Pow node.
    # So, let's consider 'Pow' with exponent -1 as division.

    op = expr.func
    args = expr.args

    # Special case: count division as 1 operator node for 'Pow' with exponent -1
    if op == sp.Pow and len(args) == 2 and args[1] == -1:
        # count 1 node for division + count numerator nodes
        numerator_nodes = count_nodes(args[0])
        # division operator node itself counts as 1
        return 1 + numerator_nodes

    # Otherwise, count 1 node for this operator plus nodes in children
    return 1 + sum(count_nodes(arg) for arg in args)

def kolmogorov_complexity(expr):
    """
    Approximates Kolmogorov Complexity using compression.
    """
    expr_str = srepr(expr)  # Full string representation of expression tree
    compressed = zlib.compress(expr_str.encode('utf-8'))
    return len(compressed)


def chebyshev_order(expr, var, interval=[-10, 10], tol=1e-6):
    from numpy.polynomial.chebyshev import Chebyshev
    func = lambdify(var, expr, 'numpy')
    x = np.linspace(interval[0], interval[1], 500)

    try:
        y = func(x)
        y = np.asarray(y, dtype=np.float64)  # Convert and catch issues
        mask = np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) < 10:
            return float('inf')  # Not enough valid points

        for deg in range(1, 50):
            cheb_fit = Chebyshev.fit(x, y, deg, domain=interval)
            if np.max(np.abs(cheb_fit(x) - y)) < tol:
                return deg
        return 50  # Max degree tested
    except Exception as e:
        print(f"Chebyshev fitting error: {e}")
        return float('inf')  # Use a large value to indicate failure


def chebyshev_order_multivariate(expr, vars, interval=[-10, 10], tol=1e-6):
    degrees = []
    for i, var in enumerate(vars):
        other_vars = [v for j, v in enumerate(vars) if j != i]
        fixed_values = [0.1] * len(other_vars)  # Avoid 0 if it causes singularities

        expr_slice = expr
        for ov, val in zip(other_vars, fixed_values):
            expr_slice = expr_slice.subs(ov, val)

        deg = chebyshev_order(expr_slice, var, interval, tol)
        degrees.append(deg)
    return max(degrees)



def holderian_regularity(expr, var, interval=[-10, 10], num_points=500):
    if expr.has(sp.zoo) or expr.has(sp.oo) or expr.has(sp.nan):
        return float('-inf')  # Lower regularity = more erratic

    func = lambdify(var, expr, 'numpy')
    x = np.linspace(interval[0], interval[1], num_points)

    try:
        y = func(x)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) < 20:
            return float('-inf')

        epsilons = np.logspace(-3, -0.1, 20)
        osc = []

        for eps in epsilons:
            osc_eps = []
            for xi in x:
                x_left = max(xi - eps, interval[0])
                x_right = min(xi + eps, interval[1])
                idx = (x >= x_left) & (x <= x_right)
                if np.any(idx):
                    osc_eps.append(np.max(y[idx]) - np.min(y[idx]))
            if osc_eps:
                osc.append(np.mean(osc_eps))

        log_eps = np.log(epsilons[:len(osc)])
        log_osc = np.log(osc)
        coeffs = np.polyfit(log_eps, log_osc, 1)
        alpha = coeffs[0]
        return alpha
    except Exception as e:
        print(f"Hölderian error: {e}")
        return float('-inf')


def holderian_regularity_multivariate(expr, variables, interval=[-10, 10], num_points=500):
    alphas = []
    for i, var in enumerate(variables):
        other_vars = [v for j, v in enumerate(variables) if j != i]
        fixed_values = [0] * len(other_vars)

        expr_slice = expr
        for ov, val in zip(other_vars, fixed_values):
            expr_slice = expr_slice.subs(ov, val)

        alpha = holderian_regularity(expr_slice, var, interval, num_points)
        alphas.append(alpha)
    return min(alphas)  # Conservative estimate of regularity


def slope_complexity(expr, vars, data_points):
    """
    Compute the slope-based functional complexity for a SymPy expression.

    Parameters:
        expr        : sympy expression
        vars        : list of sympy.Symbol, e.g. [x1, x2, ..., xm]
        data_points : numpy.ndarray of shape (n, m), n points in m-dimensional space

    Returns:
        complexity: float
    """
    # Convert the SymPy expression to a Python callable
    f = sp.lambdify(vars, expr, modules=["numpy"])

    m = data_points.shape[1]
    n = data_points.shape[0]

    total_complexity = 0.0

    for j in range(m):  # Loop over each input dimension
        # Sort data points by the j-th variable
        sorted_idx = np.argsort(data_points[:, j])
        sorted_points = data_points[sorted_idx]

        # Evaluate function at each sorted point
        f_vals = np.array([f(*point) for point in sorted_points])

        # Compute slopes
        slopes = []
        for i in range(n - 1):
            delta_x = sorted_points[i + 1, j] - sorted_points[i, j]
            if delta_x == 0:
                slope = 0
            else:
                slope = (f_vals[i + 1] - f_vals[i]) / delta_x
            slopes.append(slope)

        # Compute sum of absolute differences of consecutive slopes
        partial_complexity = sum(abs(slopes[i + 1] - slopes[i]) for i in range(len(slopes) - 1))
        total_complexity += partial_complexity

    return total_complexity / m
