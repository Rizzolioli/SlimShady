from algorithms.SLIM_GSGP.representations.individual import Individual
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.operators.mutators import one_tree_delta
import datasets.data_loader as ds
from utils.utils import get_terminals, train_test_split
from parametrization import FUNCTIONS, CONSTANTS
from algorithms.GP.representations.tree import Tree as GP_Tree
import torch
from evaluators.fitness_functions import rmse
from utils.utils import show_individual
from algorithms.GP.operators.initializers import grow
from algorithms.GP.representations.tree_utils import tree_depth, flatten

datas = ["ppb"]

data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]
X, y = data_loaders[0](X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, p_test=0.4, seed=42)
TERMINALS = get_terminals( data_loaders[0])

Tree.FUNCTIONS = FUNCTIONS
Tree.TERMINALS = TERMINALS
Tree.CONSTANTS = CONSTANTS

GP_Tree.FUNCTIONS = FUNCTIONS
GP_Tree.TERMINALS = TERMINALS
GP_Tree.CONSTANTS = CONSTANTS

operator_ = 'sum'

initial_tree = Individual([Tree(('add', 'x1', 'x2'))])
initial_tree.calculate_semantics(X_train)
initial_tree.calculate_semantics(X_test, testing=True)


random_trees = [Tree(('add', 'x3', 'x4')),
                Tree(('divide', 'x5', 'x9')),
                Tree(('multiply', 'x6', 'x10')),
                Tree(('subtract', 'x7', 'x11')),
                Tree(('subtract', 'x8', 'x12'))
                ]

[rt.calculate_semantics(X_train) for rt in random_trees]
[rt.calculate_semantics(X_test, testing=True) for rt in random_trees]

mut_steps = [0.1, 0.2, 0.3, 0.4, 0.5]

deflation_points = [5, 4, 3, 2, 1]

otd = one_tree_delta(operator_)

for i in range(len(random_trees)):

    new_block = Tree([otd, random_trees[i], mut_steps[i]])
    new_block.calculate_semantics(X_train, testing=False)
    new_block.calculate_semantics(X_test, testing=True)
    offspring = initial_tree.add_block( new_block )

    offspring.train_semantics = torch.stack([*initial_tree.train_semantics,
                                        (new_block.train_semantics if new_block.train_semantics.shape != torch.Size([])
                                         else new_block.train_semantics.repeat(len(X_train)))])

    offspring.test_semantics = torch.stack([*initial_tree.test_semantics,
                                       (new_block.test_semantics if new_block.test_semantics.shape != torch.Size([])
                                        else new_block.test_semantics.repeat(len(X_test)))])

    offspring.evaluate(rmse, y_train, testing= False, operator= operator_)
    offspring.evaluate(rmse, y_test, testing= True, operator= operator_)

    print('STRUCTURE:', show_individual(offspring, operator_))
    print('TRAIN FITNESS:', float(offspring.fitness))
    print('TEST FITNESS :', float(offspring.test_fitness))
    print('SIZE (number of blocks):', offspring.size)
    print('DEPTH :', offspring.depth)
    print('NODES :', offspring.nodes_count)
    print('\n')

    initial_tree = offspring

for i in range(len(deflation_points)):

    offspring = initial_tree.remove_block( deflation_points[i])


    offspring.train_semantics = torch.stack(
        [*initial_tree.train_semantics[:deflation_points[i]], *initial_tree.train_semantics[deflation_points[i] + 1:]])
    offspring.test_semantics = torch.stack(
        [*initial_tree.test_semantics[:deflation_points[i]], *initial_tree.test_semantics[deflation_points[i] + 1:]])

    offspring.evaluate(rmse, y_train, testing= False, operator= operator_)
    offspring.evaluate(rmse, y_test, testing= True, operator= operator_)

    print('STRUCTURE:', show_individual(offspring, operator_))
    print('TRAIN FITNESS:', float(offspring.fitness))
    print('TEST FITNESS :', float(offspring.test_fitness))
    print('SIZE (number of blocks):', offspring.size)
    print('DEPTH :', offspring.depth)
    print('NODES :', offspring.nodes_count)
    print('\n')

    initial_tree = offspring


grown_pop = grow(2000, 6, FUNCTIONS, TERMINALS, CONSTANTS, p_c=0)

depths = [tree_depth(tree) for tree in grown_pop]
sizes = [len(flatten(tree)) for tree in grown_pop]





