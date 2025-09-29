
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


def chebyshev_order(expr, var,interval=[-10, 10], tol=1e-6):
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

def sam_in_semantic(expr: sp.Expr,
                    vars: List[sp.Symbol],
                    X: np.ndarray,
                    n: int = 10,
                    eps: float = 0.2,
                    random_state: Optional[int] = None) -> float:
    """
    Compute SAM-S (Semantic Sharpness) for a given SymPy expression.

    This measures the stability of the output semantics of an individual under
    small perturbations of both the input features and the numeric constants
    inside the expression.

    Parameters
    ----------
    expr : sympy.Expr
        The symbolic expression representing the model.
    vars : list of sympy.Symbol
        The variables (features) in the same order as columns in X.
    X : np.ndarray, shape (n_samples, n_features)
        The dataset of feature values.
    n : int, optional (default=20)
        Number of random samples to take from X for the perturbation test.
        If n >= len(X), the whole dataset is used.
    eps : float, optional (default=0.1)
        Perturbation magnitude:
        - Inputs are perturbed by uniform noise in [-eps * sigma_j, eps * sigma_j]
          where sigma_j is the std deviation of feature j.
        - Constants in the expression are perturbed by uniform noise in [-eps, eps].
    random_state : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    semantic_diff : float
        RMSE between the original outputs on the subset and outputs after
        perturbing both inputs and constants.
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    if n_samples == 0:
        raise ValueError("X must contain at least one sample")
    if n <= 0:
        raise ValueError("n must be > 0")

    # choose subset indices
    if n >= n_samples:
        idx = np.arange(n_samples)
    else:
        idx = rng.choice(n_samples, size=n, replace=False)
    Ds = X[idx, :].copy()

    # compute sigma (std) per feature
    sigma = np.std(X, axis=0, ddof=0)

    # build perturbed Ds+eps
    noise = np.zeros_like(Ds)
    for j in range(Ds.shape[1]):
        scale = eps * (sigma[j] if not isclose(sigma[j], 0.0) else 1.0)
        noise[:, j] = rng.uniform(-scale, scale, size=Ds.shape[0])
    Ds_eps = Ds + noise

    # evaluate original expr on Ds
    f_callable = sp.lambdify(vars, expr, modules=["numpy"])
    try:
        y_pred = np.array(f_callable(*[Ds[:, j] for j in range(Ds.shape[1])]), dtype=float)
    except Exception:
        y_pred = np.array([float(f_callable(*row)) for row in Ds], dtype=float)

    # perturb constants in the expression
    nums = [n for n in expr.atoms(sp.Number) if n.is_real]
    mapping = {}
    for number in nums:
        delta = rng.uniform(-eps, eps)
        new_val = float(number.evalf()) + float(delta)
        if number > 0 and new_val <= 0:
            new_val = abs(new_val) if new_val != 0 else eps
        elif number < 0 and new_val >= 0:
            new_val = -abs(new_val) if new_val != 0 else -eps
        mapping[number] = sp.Float(new_val)
    expr_perturbed_constants = expr.xreplace(mapping) if mapping else expr

    # evaluate perturbed expression on Ds_eps
    f_callable_pert = sp.lambdify(vars, expr_perturbed_constants, modules=["numpy"])
    try:
        y_pred_eps_consts = np.array(f_callable_pert(*[Ds_eps[:, j] for j in range(Ds_eps.shape[1])]), dtype=float)
    except Exception:
        y_pred_eps_consts = np.array([float(f_callable_pert(*row)) for row in Ds_eps], dtype=float)

    # semantic difference (RMSE between outputs)
    semantic_diff = float(np.sqrt(np.mean((y_pred - y_pred_eps_consts) ** 2)))
    return semantic_diff
