from datasets.data_loader import *
from utils.utils import get_terminals
import itertools
import random

def get_all_block_combos(n):

    # determining the maximum index of the block to be eliminated
    #n = elite.size - 1

    # returning all combinations of removed blocks possible
    return [list(x) for r in range(1, n + 1) for x in itertools.combinations(range(1, n + 1), r)]

print(get_all_block_combos(5))

