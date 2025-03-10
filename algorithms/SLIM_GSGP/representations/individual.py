import torch
import numpy as np
from algorithms.GSGP.representations.tree_utils import apply_tree
from utils.utils import show_individual

class Individual():

    def __init__(self, collection, train_semantics, test_semantics, reconstruct):
        # defining the list (block) of pointers
        if collection is not None and reconstruct:
            self.collection = collection
            # keeping the structure of the trees in the block
            self.structure = [tree.structure for tree in collection]
            self.size = len(collection) # size == number of blocks

            self.nodes_collection = [tree.nodes for tree in collection]
            self.nodes_count = sum(self.nodes_collection) + (self.size-1)
            self.depth_collection = [tree.depth for tree in collection]
            self.depth = max([depth - (i-1) if i != 0 else depth for i, depth in enumerate(self.depth_collection)]) + (self.size-1)

        # starting the individual with empty semantics and fitnesses
        self.train_semantics = train_semantics
        self.test_semantics = test_semantics
        self.fitness =  None
        self.test_fitness = None

    def calculate_semantics(self, inputs, testing=False):

        if testing and self.test_semantics is None:
            [tree.calculate_semantics(inputs, testing) for tree in self.collection]
            self.test_semantics = torch.stack([tree.test_semantics if tree.test_semantics.shape != torch.Size([]) \
                                                    else tree.test_semantics.repeat(len(inputs)) for tree in self.collection])


        elif self.train_semantics is None:
            [tree.calculate_semantics(inputs, testing) for tree in self.collection]
            self.train_semantics = torch.stack([tree.train_semantics if tree.train_semantics.shape != torch.Size([]) \
                                                    else tree.train_semantics.repeat(len(inputs)) for tree in self.collection])



    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.collection[item]


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
            self.test_fitness = ffunction(y, torch.clamp(operator(self.test_semantics, dim = 0), -1000000000000.0, 1000000000000.0))

        else: # note: clamping in case the operator results in very big semantics
            self.fitness = ffunction(y, torch.clamp(operator(self.train_semantics, dim = 0), -1000000000000.0, 1000000000000.0))

def apply_individual_fixed(tree, data, operator = "sum", sig = False):

    semantics = []

    for t in tree.collection:
        # checking if the individual is part of the initial population (table) or is a random tree (table)

        if isinstance(t.structure, tuple):
            semantics.append(apply_tree(t, data))
        # if the tree structure is a list, checking if we are using one or two trees with our operator
        else:

            if len(t.structure) == 3: # one tree
                if sig:
                    # saving the old training semantics in case its needed
                    t.structure[1].previous_training = t.train_semantics
                    # obtaining the new train_semantics for the new unseen data
                    t.structure[1].train_semantics = torch.sigmoid(apply_tree(t.structure[1], data))
                else:
                    # saving the old training semantics in case its needed
                    t.structure[1].previous_training = t.train_semantics
                    # obtaining the new train_semantics for the new unseen data
                    t.structure[1].train_semantics = apply_tree(t.structure[1], data)

            elif len(t.structure) == 4: # two tree
                t.structure[1].previous_training = t.train_semantics
                t.structure[1].train_semantics = torch.sigmoid(apply_tree(t.structure[1], data))

                t.structure[2].previous_training = t.train_semantics
                t.structure[2].train_semantics = torch.sigmoid(apply_tree(t.structure[2], data))

            semantics.append(t.structure[0](*t.structure[1:], testing=False))

    operator = torch.sum if operator == "sum" else torch.prod

    print("WORK PLEASE", show_individual(tree, operator=operator))

    return torch.clamp(operator(torch.stack(semantics), dim=0), -1000000000000.0, 1000000000000.0)

