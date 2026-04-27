from algorithms.SLIM_GSGP.representations.individual import Individual
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.operators.mutators import one_tree_delta
import datasets.data_loader as ds
from datasets.data_loader import load_dummy_test, load_dummy_train, load_preloaded
from utils.utils import get_terminals, train_test_split
from parametrization import FUNCTIONS, CONSTANTS
from algorithms.GP.representations.tree import Tree as GP_Tree
import torch
from evaluators.fitness_functions import rmse
from utils.utils import show_individual
from algorithms.GP.operators.initializers import grow
from algorithms.GP.representations.tree_utils import tree_depth, flatten
import numpy as np

datas = ["ppb"]

X_train, y_train = load_preloaded(datas[0], seed= 1, training=True, X_y=True)

X_test, y_test = load_preloaded(datas[0], seed= 1, training=False, X_y=True)

# data_loaders = [getattr(ds, func) for func in dir(ds) for dts in datas if "load_" + dts in func]
# X, y = load_preloaded[0](datas[0], seed = 1, X_y = True)
# X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, p_test=0.4, seed=42)
# X_train, y_train = load_dummy_train()
# X_test, y_test = load_dummy_test()
# TERMINALS = get_terminals( load_dummy_train)
TERMINALS = get_terminals('ppb', seed = 1)


# Tree.FUNCTIONS = FUNCTIONS
# Tree.TERMINALS = TERMINALS
# Tree.CONSTANTS = CONSTANTS
#
# GP_Tree.FUNCTIONS = FUNCTIONS
# GP_Tree.TERMINALS = TERMINALS
# GP_Tree.CONSTANTS = CONSTANTS
#
# operator_ = 'mul'
#
# initial_tree = Individual([Tree(('divide', 'x536', 'x625'))])
# initial_tree.calculate_semantics(X_train)
# initial_tree.calculate_semantics(X_test, testing=True)
#
# initial_tree.evaluate(rmse, y_train, testing=False, operator=operator_)
# initial_tree.evaluate(rmse, y_test, testing=True, operator=operator_)
#
# print('INITIAL TREE :')
# print('STRUCTURE:', show_individual(initial_tree, operator_))
# print('TRAIN FITNESS:', float(initial_tree.fitness))
# print('TEST FITNESS :', float(initial_tree.test_fitness))
# print('SIZE (number of blocks):', initial_tree.size)
# print('DEPTH :', initial_tree.depth)
# print('NODES :', initial_tree.nodes_count)
# print('\n')
#
#
# random_trees = [[Tree(('multiply', 'x533', 'x487'))],
#                 [Tree(('divide', 'x442', 'x104'))],
# [Tree(('add', 'x226', 'x404'))],
# [Tree(('divide', 'x126', 'x81'))],
# [Tree(('add', 'x126', 'x244'))],
#                 ]
#
# [[rt.calculate_semantics(X_train) for rt in rts] for rts in random_trees]
# [[rt.calculate_semantics(X_test, testing=True) for rt in rts] for rts in random_trees]
#
# mut_steps = [0.08023122565519353, 0.06361369735162661, 0.040869470433096126, 0.0015551416005836205, 0.04942270774722046]
#
# deflation_points = [ 4, 3, 2, 1]
#
# otd = one_tree_delta(operator_)
#
# for i in range(len(random_trees)):
#
#     print('INITIAL STRUCTURE:', show_individual(initial_tree, operator_))
#     print(f'INFLATING WITH {[rt.structure for rt in random_trees[i]]}')
#
#     new_block = Tree([otd, *random_trees[i], mut_steps[i]])
#     new_block.calculate_semantics(X_train, testing=False)
#     new_block.calculate_semantics(X_test, testing=True)
#     offspring = initial_tree.add_block( new_block )
#
#     offspring.train_semantics = torch.stack([*initial_tree.train_semantics,
#                                         (new_block.train_semantics if new_block.train_semantics.shape != torch.Size([])
#                                          else new_block.train_semantics.repeat(len(X_train)))])
#
#     offspring.test_semantics = torch.stack([*initial_tree.test_semantics,
#                                        (new_block.test_semantics if new_block.test_semantics.shape != torch.Size([])
#                                         else new_block.test_semantics.repeat(len(X_test)))])
#
#     offspring.evaluate(rmse, y_train, testing= False, operator= operator_)
#     offspring.evaluate(rmse, y_test, testing= True, operator= operator_)
#
#     print('FINAL STRUCTURE:', show_individual(offspring, operator_))
#     print('TRAIN FITNESS:', float(offspring.fitness))
#     print('TEST FITNESS :', float(offspring.test_fitness))
#     print('SIZE (number of blocks):', offspring.size)
#     print('DEPTH :', offspring.depth)
#     print('NODES :', offspring.nodes_count)
#     print('\n')
#
#     initial_tree = offspring
#
# for i in range(len(deflation_points)):
#
#     print('INITIAL STRUCTURE:', show_individual(initial_tree, operator_))
#     print(f'DEFLATING at {deflation_points[i]}')
#
#     offspring = initial_tree.remove_block( deflation_points[i])
#
#
#     offspring.train_semantics = torch.stack(
#         [*initial_tree.train_semantics[:deflation_points[i]], *initial_tree.train_semantics[deflation_points[i] + 1:]])
#     offspring.test_semantics = torch.stack(
#         [*initial_tree.test_semantics[:deflation_points[i]], *initial_tree.test_semantics[deflation_points[i] + 1:]])
#
#     offspring.evaluate(rmse, y_train, testing= False, operator= operator_)
#     offspring.evaluate(rmse, y_test, testing= True, operator= operator_)
#
#     print('FINAL STRUCTURE:', show_individual(offspring, operator_))
#     print('TRAIN FITNESS:', float(offspring.fitness))
#     print('TEST FITNESS :', float(offspring.test_fitness))
#     print('SIZE (number of blocks):', offspring.size)
#     print('DEPTH :', offspring.depth)
#     print('NODES :', offspring.nodes_count)
#     print('\n')
#
#     initial_tree = offspring
#

grown_pop = grow(2000, 6, FUNCTIONS, TERMINALS, CONSTANTS, p_c=0)

depths = [tree_depth(FUNCTIONS)(tree) for tree in grown_pop]
sizes = [len(list(flatten(tree))) for tree in grown_pop]

print('MEAN NODES :', np.mean(sizes))
print('MEDIAN NODES :', np.median(sizes))
print('STD NODES :', np.std(sizes))
print('\n')
print('MEAN DEPTH :', np.mean(depths))
print('MEDIAN DEPTH :', np.median(depths))
print('STD DEPTH :', np.std(depths))





