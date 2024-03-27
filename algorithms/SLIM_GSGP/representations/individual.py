import torch
import numpy as np
from algorithms.GSGP.representations.tree_utils import apply_tree

class Individual():

    def __init__(self, collection):
        # defining the list (block) of pointers
        self.collection = collection
        # keeping the structure of the trees in the block
        self.structure = [tree.structure for tree in collection]
        self.size = len(collection) # size == number of blocks
        self.nodes_count = sum([tree.nodes for tree in collection]) + (self.size-1)

        self.depth = max([tree.depth - (i-1) if i != 0 else tree.depth for i, tree in enumerate(collection) ]) + (self.size-1)

        # starting the individual with empty semantics and fitnesses
        self.train_semantics = None
        self.test_semantics = None
        self.fitness =  None
        self.test_fitness = None

    def calculate_semantics(self, inputs, testing=False):

        [tree.calculate_semantics(inputs, testing) for tree in self.collection]


        if testing:

            self.test_semantics = torch.stack([tree.test_semantics if tree.test_semantics.shape != torch.Size([]) \
                                                    else tree.test_semantics.repeat(len(inputs)) for tree in self.collection])


        else:

            self.train_semantics = torch.stack([tree.train_semantics if tree.train_semantics.shape != torch.Size([]) \
                                                    else tree.train_semantics.repeat(len(inputs)) for tree in self.collection])



    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.collection[item]

    def remove_block(self, index):

        new_collection = [*self.collection[:index], *self.collection[index+1:]]

        new_individual = Individual(new_collection)

        return new_individual


    def add_block(self, tree):

        new_collection = [*self.collection, tree]
        new_individual = Individual(new_collection)

        return new_individual



    def evaluate(self, ffunction, y, testing = False, operator = 'sum'):
        """
                evaluates the population given a certain fitness function, input data(x) and target data (y)
                Parameters
                ----------
                ffunction: function
                    fitness function to evaluate the individual
                X: torch tensor
                    the input data (which can be training or testing)
                y: torch tensor
                    the expected output (target) values

                Returns
                -------
                None
                    attributes a fitness tensor to the population
                """
        if operator == 'sum':
            operator = torch.sum
        else:
            operator = torch.prod

        if testing:
            self.test_fitness = ffunction(y, operator(self.test_semantics, dim = 0))

        else:
            self.fitness = ffunction(y, operator(self.train_semantics, dim = 0))

    def apply_individual(self, data, operator="sum"): # TODO Davide or Liah verify. I get slightly different RMSE on the elite using this
        return [apply_tree(tree, data) for tree in self.collection][0]

