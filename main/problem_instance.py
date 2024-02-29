import torch
from utils.utils import protected_div, mean_

FUNCTIONS = {
    'add': {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide': {'function': lambda x, y: protected_div(x, y), 'arity': 2},
    'mean': {'function': lambda x, y: mean_(x, y), 'arity': 2},
    'tan': {'function': lambda x: torch.tan(x), 'arity': 1},
    'sin': {'function': lambda x: torch.sin(x), 'arity': 1},
    'cos': {'function': lambda x: torch.cos(x), 'arity': 1},
}

# TODO: make this dataset dependent
TERMINALS = {f"x{i}": i for i in range(5)}

CONSTANTS = {
    'constant_2': lambda x: torch.tensor(2).float(),
    'constant_3': lambda x: torch.tensor(3).float(),
    'constant_4': lambda x: torch.tensor(4).float(),
    'constant_5': lambda x: torch.tensor(5).float(),
    'constant__1': lambda x: torch.tensor(-1).float()
}
