from parametrization import FUNCTIONS, CONSTANTS
from utils.utils import get_terminals
from algorithms.GP.representations.tree_utils import create_full_random_tree
import datasets.data_loader as ds
from algorithms.GSGP.representations.tree import Tree
from algorithms.GSGP.operators.crossover_operators import *
from algorithms.GSGP.operators.mutators import *
from evaluators.fitness_functions import rmse
from algorithms.GSGP.representations.population import Population

datas = ["ppb"]

# obtaining the data loading functions using the dataset name
data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]

TERMINALS = get_terminals(data_loaders[0])

tree1 = Tree(create_full_random_tree(3, FUNCTIONS, TERMINALS, CONSTANTS), FUNCTIONS, TERMINALS, CONSTANTS)
tree2 = Tree(create_full_random_tree(3, FUNCTIONS, TERMINALS, CONSTANTS), FUNCTIONS, TERMINALS, CONSTANTS)

random_tree1 = Tree(create_full_random_tree(3, FUNCTIONS, TERMINALS, CONSTANTS), FUNCTIONS, TERMINALS, CONSTANTS)
random_tree2 = Tree(create_full_random_tree(3, FUNCTIONS, TERMINALS, CONSTANTS), FUNCTIONS, TERMINALS, CONSTANTS)

print(random_tree1.structure)


X, y = data_loaders[0](X_y = True)

pop = Population([tree1, tree2, random_tree1, random_tree2])
pop.calculate_semantics(X)
pop.evaluate(rmse, y)

tree1.calculate_semantics(X)
tree2.calculate_semantics(X)
random_tree1.calculate_semantics(X)
random_tree2.calculate_semantics(X)

tree1.calculate_semantics(X, testing = True)
tree2.calculate_semantics(X, testing = True)
random_tree1.calculate_semantics(X, testing = True)
random_tree2.calculate_semantics(X, testing = True)

print(tree1)
print(tree2)
print(random_tree1)

tree3 = Tree([geometric_crossover, tree1, tree2, random_tree1], tree1.FUNCTIONS, tree1.TERMINALS, tree1.CONSTANTS)
tree3.calculate_semantics(X)
tree3.calculate_semantics(X, testing=True)

ms = torch.arange(0.25, 5.25, 0.25, device='cpu')

tree4 = Tree([geometric_mutation, tree1, tree2, random_tree1, ms[random.randint(0, len(ms) - 1)]], tree1.FUNCTIONS, tree1.TERMINALS, tree1.CONSTANTS)
tree4.calculate_semantics(X)
tree4.calculate_semantics(X, testing=True)

pop = Population([tree1, tree2, tree3, tree4, random_tree1, random_tree2])

pop.evaluate(rmse, y)


