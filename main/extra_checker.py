import torch, random
from datasets.data_loader import load_preloaded
from utils.utils import generate_random_uniform

"""x, y = load_preloaded("bioavailability", 1, training=False, X_y=True)

print(x)

print(y)"""

r = generate_random_uniform(0, 0.1)

print(r())