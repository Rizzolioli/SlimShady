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

def ackley(matrix):
    """
    Calculate the Ackley function value for each row in the matrix.
    """
    n_rows, n_cols = matrix.shape
    fitness_values = np.zeros(n_rows)
    for i in range(n_rows):
        position = matrix[i]
        sum1 = 0.0
        sum2 = 0.0
        for xi in position:
            sum1 += xi * xi
            sum2 += math.cos(2 * math.pi * xi)
        avg1 = sum1 / n_cols
        avg2 = sum2 / n_cols
        fitness_val = -20 * math.exp(-0.2 * math.sqrt(avg1)) - math.exp(avg2) + 20 + math.e
        fitness_values[i] = fitness_val
    return fitness_values

def alpine_1(matrix):
    """
    Calculate the Alpine 1 function value for each row in the matrix.
    """
    n_rows, n_cols = matrix.shape
    fitness_values = np.zeros(n_rows)
    for i in range(n_rows):
        position = matrix[i]
        fitness_val = 0.0
        for xi in position:
            fitness_val += abs(xi * math.sin(xi) + 0.1 * xi)
        fitness_values[i] = fitness_val
    return fitness_values

def alpine_2(matrix):
    """
    Calculate the Alpine 2 function value for each row in the matrix.
    """
    n_rows, n_cols = matrix.shape
    fitness_values = np.zeros(n_rows)
    for i in range(n_rows):
        position = matrix[i]
        fitness_val = 1.0
        for xi in position:
            fitness_val *= math.sqrt(xi) * math.sin(xi)
        fitness_values[i] = fitness_val
    return fitness_values


def michalewicz(matrix, m=10):
    """
    Calculate the Michalewicz function value for each row in the matrix.
    """
    n_rows, n_cols = matrix.shape
    fitness_values = np.zeros(n_rows)
    for i in range(n_rows):
        position = matrix[i]
        fitness_val = 0.0
        for idx, xi in enumerate(position):
            fitness_val -= math.sin(xi) * (math.sin((idx + 1) * xi ** 2 / math.pi)) ** (2 * m)
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
                 'rosenbrock' : scipy.optimize.rosen,
                 'ackley' : ackley,
                 'alpine1' : alpine_1,
                 'alpine2' : alpine_2,
                 'michalewicz' : michalewicz
                 }

    if function not in functions.keys():
        raise ValueError('Invalid function')

    if function == 'rastrigin' or function == 'alpine2':
        X = create_random_matrix(rows, columns, [(0, 10) for _ in range(columns)])
    else:
        X = create_random_matrix(rows, columns, [(-10,10) for _ in range(columns)])
    scaler_input = MinMaxScaler(scale_inputs)
    scaler_output = MinMaxScaler(scale_output)

    if function == 'rosenbrock':
        y = scaler_output.fit_transform(functions[function](X.T).reshape(-1, 1))
    else:
        y = scaler_output.fit_transform(functions[function](X).reshape(-1, 1))

    return scaler_input.fit_transform(X), y.flatten()


