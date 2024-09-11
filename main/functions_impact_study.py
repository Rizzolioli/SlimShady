from algorithms.SLIM_GSGP.representations.individual import Individual
from algorithms.GP.representations.tree import Tree as GPTree
from algorithms.GSGP.representations.tree import Tree as GSGPTree
from algorithms.GP.representations.tree_utils import create_grow_random_tree
from utils.utils import generate_random_uniform
from algorithms.SLIM_GSGP.operators.mutators import inflate_mutation, deflate_mutation
import torch
import numpy as np
import random
import csv
from utils.utils import verbose_reporter
import time

csim = torch.nn.CosineSimilarity(dim = 0)
cdist = torch.nn.PairwiseDistance(p=2)

def create_random_slim_ind(blocks,
                           X_train,
                           FUNCTIONS, TERMINALS, CONSTANTS,
                           deflate_on,
                            X_test = None,
                           dataset_name = None,
                           y_train = None,
                           y_test = None,
                           algorithm = (True, 'sum', False),
                           mutation_step = generate_random_uniform(0,1), initial_depth = 6, seed = 42,
                           log = 0, log_path = None, verbose = 0):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    GSGPTree.FUNCTIONS = FUNCTIONS
    GSGPTree.TERMINALS = TERMINALS
    GSGPTree.CONSTANTS = CONSTANTS

    GPTree.FUNCTIONS = FUNCTIONS
    GPTree.TERMINALS = TERMINALS
    GPTree.CONSTANTS = CONSTANTS

    if algorithm == (False,  "mul", True):
        algorithm_name = 'SLIM*1SIG'
    elif algorithm == (False,  "mul", False):
        algorithm_name = 'SLIM*1NORM'
    else:
        algorithm_name = 'SLIM+2SIG'

    inflator = inflate_mutation(FUNCTIONS=FUNCTIONS, TERMINALS=TERMINALS, CONSTANTS=CONSTANTS,
                                two_trees=algorithm[0], operator=algorithm[1],
                                sig=algorithm[2])

    individual = Individual(collection=[GSGPTree(create_grow_random_tree(initial_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c = 0),
                                                 reconstruct=True,
                                                 train_semantics=None,
                                                 test_semantics=None
                                                 )],
                           reconstruct=True,
                           train_semantics=None,
                           test_semantics=None)

    individual.calculate_semantics(X_train, testing = False)
    if X_test is not None:
        individual.calculate_semantics(X_test, testing=True)

    individual.full_train_semantics = torch.reshape(individual.train_semantics, (individual.train_semantics.shape[1],))
    if X_test is not None:
        individual.full_test_semantics = torch.reshape(individual.test_semantics, (individual.test_semantics.shape[1],))

    for _ in range(blocks-1):

        start = time.time()

        inflated_individual = inflator(individual,
                            ms = mutation_step(),
                            X = X_train,
                            max_depth=initial_depth,
                            p_c=0,
                            X_test=X_test,
                            reconstruct = False)

        if algorithm[-2] == 'sum':
            inflated_individual.full_train_semantics = torch.sum(inflated_individual.train_semantics, dim=0)
            if X_test is not None:
                inflated_individual.full_test_semantics = torch.sum(inflated_individual.test_semantics, dim=0)
        else:
            inflated_individual.full_train_semantics = torch.prod(inflated_individual.train_semantics, dim=0)
            if X_test is not None:
                inflated_individual.full_test_semantics = torch.prod(inflated_individual.test_semantics, dim=0)

        train_var= torch.sum(individual.full_train_semantics != inflated_individual.full_train_semantics).item() / individual.full_train_semantics.shape[0]
        if X_test is not None:
            test_var = torch.sum(individual.full_test_semantics != inflated_individual.full_test_semantics).item()  /individual.full_test_semantics.shape[0]
        else:
            test_var = 0

        end = time.time()


        if verbose > 0:
            verbose_reporter(dataset_name, -1, 'inflate', inflated_individual.size, end-start, train_var,
                             first_col_name='Dataset', second_col_name='Index', third_col_name='Operation',
                             fourth_col_name='OffSpring Size', fifth_col_name='Timing', sixth_col_name='Variation',
                             first_call= _ == 0)

        if log > 0:
            log_row = [dataset_name, algorithm_name, seed, 'inflate', inflated_individual.size,
                       inflated_individual.nodes_count, inflated_individual.depth, -1,
                       train_var, test_var, individual.full_train_semantics[0]]
    
            with open(log_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(log_row)

        if any(inflated_individual.size == def_ for def_ in deflate_on):
            for j in range(1, inflated_individual.size-1):

                start = time.time()

                deflated_individual = deflate_mutation(inflated_individual, reconstruct=False, mut_point=j)

                if algorithm[-2] == 'sum':
                    deflated_individual.full_train_semantics = torch.sum(deflated_individual.train_semantics, dim=0)
                    if X_test is not None:
                        deflated_individual.full_test_semantics = torch.sum(deflated_individual.test_semantics, dim=0)
                else:
                    deflated_individual.full_train_semantics = torch.prod(deflated_individual.train_semantics, dim=0)
                    if X_test is not None:
                        deflated_individual.full_test_semantics = torch.prod(deflated_individual.test_semantics, dim=0)


                train_var = torch.sum(
                    deflated_individual.full_train_semantics != inflated_individual.full_train_semantics).item() / \
                            individual.full_train_semantics.shape[0]
                if X_test is not None:
                    test_var = torch.sum(deflated_individual.full_test_semantics != inflated_individual.full_test_semantics).item() / \
                               individual.full_test_semantics.shape[0]
                else:
                    test_var = 0

                end = time.time()

                if verbose > 0:
                    verbose_reporter(dataset_name, j, 'deflate', deflated_individual.size, end - start, train_var,
                                     first_col_name='Dataset', second_col_name='Index', third_col_name='Operation',
                                     fourth_col_name='OffSpring Size', fifth_col_name='Timing',
                                     sixth_col_name='Variation')

                if log > 0:
                    log_row = [dataset_name, algorithm_name, seed, 'deflate', deflate_on,
                               deflated_individual.nodes_count, deflated_individual.depth, j,
                               train_var, test_var, deflated_individual.full_train_semantics[0]]

                    with open(log_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(log_row)

        individual = inflated_individual
