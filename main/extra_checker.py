import torch
from datasets.data_loader import load_preloaded

x, y = load_preloaded("ppb", 1, training=True, X_y=True)

print(x)

print(y)