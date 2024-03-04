import random
import csv
import torch
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd

def protected_div(x1, x2):
    """ Implements the division protected against zero denominator

    Performs division between x1 and x2. If x2 is (or has) zero(s), the
    function returns the numerator's value(s).

    Parameters
    ----------
    x1 : torch.Tensor
        The numerator.
    x2 : torch.Tensor
        The denominator.

    Returns
    -------
    torch.Tensor
        Result of protected division between x1 and x2.
    """
    # if  torch.is_tensor(x2):

    return torch.where(torch.abs(x2) > 0.001, torch.div(x1, x2), torch.tensor(1.0, dtype=x2.dtype, device=x2.device))

    # else:
    #     if x2 < 0:
    #         return 0
    #     else:
    #
    #         return x1/x2



def mean_(x1, x2):

    return torch.div(torch.add(x1, x2), 2)

# def w_mean_(x1, x2):
#
#     r = random.random()
#
#     return torch.add(torch.mul(x1, r), torch.mul(x2, r))


FUNCTIONS = {
    'add': {'function': lambda x, y: torch.add(x,y), 'arity':2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity':2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity':2},
    'divide': {'function': lambda x, y: protected_div(x, y), 'arity':2},
    'mean' : {'function': lambda x, y:  mean_(x, y), 'arity':2},
    'tan' : {'function': lambda x: torch.tan(x), 'arity':1},
    'sin' : {'function': lambda x: torch.sin(x), 'arity':1},
'cos' : {'function':lambda x: torch.cos(x), 'arity':1},
}

"""


    
TERMINALS = {
    'input_gen': lambda x1, x2, x3, x4, x5, x6, x7: x1,
    'input_fit_elite': lambda x1, x2, x3, x4, x5, x6, x7: x2,
    'input_avg_fit': lambda x1, x2, x3, x4, x5, x6, x7: x3,
    'input_phen_div': lambda x1, x2, x3, x4, x5, x6, x7: x4,
    'input_gen_div': lambda x1, x2, x3, x4, x5, x6, x7: x5,
    'input_pop_size': lambda x1, x2, x3, x4, x5, x6, x7: x6,
    'input_fit_imp': lambda x1, x2, x3, x4, x5, x6, x7: x7,
}
"""

TERMINALS = {f"x{i}": i for i in range(5)}

CONSTANTS = {
    'constant_2': lambda x: torch.tensor(2).float(),
    'constant_3': lambda x: torch.tensor(3).float(),
    'constant_4': lambda x: torch.tensor(4).float(),
    'constant_5': lambda x: torch.tensor(5).float(),
    'constant__1': lambda x: torch.tensor(-1).float()
}

def scale_dataset(data_loader):

    data = data_loader()
    data_scaled = MinMaxScaler().fit_transform(data)

    data_scaled = pd.DataFrame(data_scaled)

    path = 'C:\\Users\\utente\\OneDrive - NOVAIMS\\dottorato\\GPDP\\general-purpose-optimization-library' \
           '\\gpol\\utils'

    data_scaled.to_csv(os.path.join(path, \
                                    "data", '_'.join(str(data_loader.__name__).split('_')[1:]) \
                                    + "_scaled.txt"))

