import scipy.optimize
from scipy.optimize import rosen
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

def rastrigin(matrix):
    """
    Calculate the Rastrigin function value for each row in the matrix.
    """
    n_rows, n_cols = matrix.shape
    fitness_values = np.zeros(n_rows)
    for i in range(n_rows):
        position = matrix[i]
        fitness_val = 0.0
        for xi in position:
            fitness_val += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
        fitness_values[i] = fitness_val
    return fitness_values

def sphere(matrix):
    n_rows, n_cols = matrix.shape
    fitness_values = np.zeros(n_rows)
    for i in range(n_rows):
        position = matrix[i]
        fitness_val = 0.0
        for xi in position:
            fitness_val += (xi * xi)
        fitness_values[i] = fitness_val
    return fitness_values

def create_random_matrix(n, m, ranges):
    if len(ranges) != m:
        raise ValueError("Length of ranges list must be equal to the number of columns (m).")
    matrix = np.zeros((n, m))
    for i in range(m):
        min_val, max_val = ranges[i]
        matrix[:, i] = np.random.uniform(min_val, max_val, n)
    return matrix


def create_dataset(rows, columns, scale_inputs, scale_output, function, seed):

    np.random.seed(seed)

    functions = {'rastrigin' : rastrigin,
                 'sphere' : sphere,
                 'rosenbrock' : scipy.optimize.rosen}

    if function not in functions.keys():
        raise ValueError('Invalid function')

    X = create_random_matrix(rows, columns, [scale_inputs for _ in range(columns)])
    if function == 'rosenbrock':
        X = X.T

    scaler = MinMaxScaler(scale_output)
    y = scaler.fit_transform(functions[function](X).reshape(-1, 1))

    return X, y.flatten()


