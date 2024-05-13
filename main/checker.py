import random

from parametrization import FUNCTIONS, CONSTANTS
from utils.utils import get_terminals, get_best_min, get_best_max
from algorithms.GP.representations.tree_utils import create_full_random_tree
import datasets.data_loader as ds
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.representations.individual import Individual
from algorithms.GSGP.operators.crossover_operators import *
from algorithms.GSGP.operators.mutators import *
from evaluators.fitness_functions import rmse
from algorithms.SLIM_GSGP.representations.population import Population
from datasets.data_loader import *

datas = ["ppb"]

# obtaining the data loading functions using the dataset name
data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]
X, y = data_loaders[0](X_y = True)

TERMINALS = get_terminals(data_loaders[0])

tree1 = Tree(create_full_random_tree(3, FUNCTIONS, TERMINALS, CONSTANTS), FUNCTIONS, TERMINALS, CONSTANTS) #for base trees need to calculate the semantics
tree2 = Tree(create_full_random_tree(3, FUNCTIONS, TERMINALS, CONSTANTS), FUNCTIONS, TERMINALS, CONSTANTS)

tree1.calculate_semantics(X)
tree2.calculate_semantics(X)

random_tree1 = Tree(create_full_random_tree(3, FUNCTIONS, TERMINALS, CONSTANTS), FUNCTIONS, TERMINALS, CONSTANTS)
random_tree2 = Tree(create_full_random_tree(3, FUNCTIONS, TERMINALS, CONSTANTS), FUNCTIONS, TERMINALS, CONSTANTS)

random_tree1.calculate_semantics(X)
random_tree2.calculate_semantics(X)

print(random_tree1.structure)

tree3 = Tree([geometric_crossover, tree1, tree2, random_tree1], tree1.FUNCTIONS, tree1.TERMINALS, tree1.CONSTANTS)

ms = torch.arange(0.25, 5.25, 0.25, device='cpu')

tree4 = Tree([geometric_mutation, tree1, tree2, random_tree1, ms[random.randint(0, len(ms) - 1)]], tree1.FUNCTIONS, tree1.TERMINALS, tree1.CONSTANTS)

ind1 = Individual([tree1, tree2, tree3, random_tree2])

print(ind1.structure)

ind2 = Individual([tree2, tree3, tree4, random_tree2])

# print(ind2.structure)

# ind1.calculate_semantics(X)
# print(ind1.train_semantics)

pop = Population([ind1, ind2, ind1, ind2])
pop.calculate_semantics(X)

pop.evaluate(rmse, y)

res, elite = get_best_max(pop, 2)


ran = random.choice(pop)

print("pop", pop.population)
print("ran",ran)
#ind3 = two_trees_inflate_mutation(ind1, 0.1, X)

#ind4 = deflate_mutation(ind1)

#print(ind1)


