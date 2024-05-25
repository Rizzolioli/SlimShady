import cvxpy as cp
import numpy as np
import torch
from evaluators.fitness_functions import signed_errors


def calculate_signed_errors(semantics, y_true, operator):

    if operator == "sum":
        operator = torch.sum
    else:
        operator = torch.prod

    return signed_errors(y_true, operator(semantics, dim=0))


def global_optimum_in_ch(errors):
    # function that returns wheter or not the global optimum (0,0) is inside
    # the convexhull of the errors

    A = errors.T.numpy()
    b = np.zeros(A.shape[0])

    A_par = cp.Parameter(A.shape)
    b_par = cp.Parameter(b.shape)

    a = cp.Variable(A.shape[1])
    objective = cp.Minimize(cp.sum_squares(A_par @ a - b_par))
    constraints = [0 <= a, sum(a) == 1]
    prob = cp.Problem(objective, constraints)

    prob.parameters()[0].value = A
    prob.parameters()[1].value = b
    try:
        prob.solve(solver="MOSEK")
        # print(f'Sum equal to 1 {np.isclose(sum(prob.variables()[0].value),1)}')
        # print(f'All positive: {all(prob.variables()[0].value >= 0)}')
        return (
            prob.status == "optimal"
            and np.isclose(sum(prob.variables()[0].value), 1)
            and all(prob.variables()[0].value >= 0)
        )
    except BaseException:
        print("Optimization not completed")
        return False


def distance_from_chull(errors):
    # function that returns the distance of the convexhull from the global
    # optimum (0,0)

    A = errors.T.numpy()

    m, n = A.shape[0], A.shape[1]

    e = cp.Variable(m)
    e_ = cp.Variable(m)
    a = cp.Variable(n)

    A_par = cp.Parameter(A.shape)

    objective = cp.Minimize(cp.sum(e + e_))
    constraints = [
        0 <= a,
        sum(a) == 1,
        0 <= e,
        0 <= e_,
        A_par @ a +
        e -
        e_ == 0]
    prob = cp.Problem(objective, constraints)

    prob.parameters()[0].value = A

    try:
        prob.solve(solver=cp.CLARABEL, ignore_dpp=True)

        return sum(prob.variables()[0].value) + sum(prob.variables()[1].value)

    except BaseException:

        return np.inf
