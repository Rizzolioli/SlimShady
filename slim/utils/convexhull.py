import cvxpy as cp
import numpy as np
import torch
from slim.evaluators.fitness_functions import signed_errors


def calculate_signed_errors(semantics, y_true, operator):
    """
    Calculate signed errors based on the given operator.

    Args:
        semantics (torch.Tensor): The predicted values.
        y_true (torch.Tensor): The true values.
        operator (str): The operator to be used, either "sum" or "prod".

    Returns:
        torch.Tensor: The signed errors.
    """
    if operator == "sum":
        operator_func = torch.sum
    else:
        operator_func = torch.prod

    return signed_errors(y_true, operator_func(semantics, dim=0))


def global_optimum_in_ch(errors):
    """
    Determine whether the global optimum (0,0) is inside the convex hull of the errors.

    Args:
        errors (torch.Tensor): The errors tensor.

    Returns:
        bool: True if the global optimum is inside the convex hull, False otherwise.
    """
    A = errors.T.numpy()
    b = np.zeros(A.shape[0])

    A_par = cp.Parameter(A.shape)
    b_par = cp.Parameter(b.shape)

    a = cp.Variable(A.shape[1])
    objective = cp.Minimize(cp.sum_squares(A_par @ a - b_par))
    constraints = [0 <= a, cp.sum(a) == 1]
    prob = cp.Problem(objective, constraints)

    A_par.value = A
    b_par.value = b

    try:
        prob.solve(solver="MOSEK")
        return (
            prob.status == "optimal"
            and np.isclose(np.sum(prob.variables()[0].value), 1)
            and np.all(prob.variables()[0].value >= 0)
        )
    except cp.error.SolverError:
        print("Optimization not completed")
        return False


def distance_from_chull(errors):
    """
    Calculate the distance of the convex hull from the global optimum (0,0).

    Args:
        errors (torch.Tensor): The errors tensor.

    Returns:
        float: The distance of the convex hull from the global optimum.
    """
    A = errors.T.numpy()

    m, n = A.shape

    e = cp.Variable(m)
    e_ = cp.Variable(m)
    a = cp.Variable(n)

    A_par = cp.Parameter(A.shape)

    objective = cp.Minimize(cp.sum(e + e_))
    constraints = [0 <= a, cp.sum(a) == 1, 0 <= e, 0 <= e_, A_par @ a + e - e_ == 0]
    prob = cp.Problem(objective, constraints)

    A_par.value = A

    try:
        prob.solve(solver=cp.CLARABEL, ignore_dpp=True)
        return np.sum(prob.variables()[0].value) + np.sum(prob.variables()[1].value)
    except cp.error.SolverError:
        print("Distance calculation not completed")
        return np.inf
