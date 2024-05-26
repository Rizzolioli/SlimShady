"""
Individual Class and Utility Functions for Genetic Programming using PyTorch.
"""

import torch
from main.algorithms.GSGP.representations.tree_utils import apply_tree


class Individual:
    """
    Initialize an Individual with a collection of trees and semantics.

    Args:
        collection: List of trees representing the individual.
        train_semantics: Training semantics associated with the individual.
        test_semantics: Testing semantics associated with the individual.
        reconstruct: Boolean indicating if the individual should be reconstructed.
    """

    def __init__(self, collection, train_semantics, test_semantics, reconstruct):
        if collection is not None and reconstruct:
            self.collection = collection
            self.structure = [tree.structure for tree in collection]
            self.size = len(collection)

            self.nodes_collection = [tree.nodes for tree in collection]
            self.nodes_count = sum(self.nodes_collection) + (self.size - 1)
            self.depth_collection = [tree.depth for tree in collection]
            self.depth = max(
                [
                    depth - (i - 1) if i != 0 else depth
                    for i, depth in enumerate(self.depth_collection)
                ]
            ) + (self.size - 1)

        self.train_semantics = train_semantics
        self.test_semantics = test_semantics
        self.fitness = None
        self.test_fitness = None

    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for the individual.

        Args:
            inputs: Input data for calculating semantics.
            testing: Boolean indicating if the calculation is for testing semantics.

        Returns:
            None
        """

        if testing and self.test_semantics is None:
            [tree.calculate_semantics(inputs, testing) for tree in self.collection]
            self.test_semantics = torch.stack(
                [
                    (
                        tree.test_semantics
                        if tree.test_semantics.shape != torch.Size([])
                        else tree.test_semantics.repeat(len(inputs))
                    )
                    for tree in self.collection
                ]
            )

        elif self.train_semantics is None:
            [tree.calculate_semantics(inputs, testing) for tree in self.collection]
            self.train_semantics = torch.stack(
                [
                    (
                        tree.train_semantics
                        if tree.train_semantics.shape != torch.Size([])
                        else tree.train_semantics.repeat(len(inputs))
                    )
                    for tree in self.collection
                ]
            )

    def __len__(self):
        """
        Return the size of the individual.

        Returns:
            int: Size of the individual.
        """
        return self.size

    def __getitem__(self, item):
        """Get a tree from the individual by index.

        Args:
            item: Index of the tree to retrieve.

        Returns:
            Tree: The tree at the specified index.
        """
        return self.collection[item]

    def evaluate(self, ffunction, y, testing=False, operator="sum"):
        """
        Evaluate the individual using a fitness function.

        Args:
            ffunction: Fitness function to evaluate the individual.
            y: Expected output (target) values as a torch tensor.
            testing: Boolean indicating if the evaluation is for testing semantics.
            operator: Operator to apply to the semantics ("sum" or "prod").

        Returns:
            None
        """
        if operator == "sum":
            operator = torch.sum
        else:
            operator = torch.prod

        if testing:
            self.test_fitness = ffunction(
                y,
                torch.clamp(
                    operator(self.test_semantics, dim=0),
                    -1000000000000.0,
                    1000000000000.0,
                ),
            )

        else:
            self.fitness = ffunction(
                y,
                torch.clamp(
                    operator(self.train_semantics, dim=0),
                    -1000000000000.0,
                    1000000000000.0,
                ),
            )


def apply_individual_fixed(tree, data, operator="sum", sig=False):

    semantics = []

    for t in tree.collection:
        if isinstance(t.structure, tuple):
            semantics.append(apply_tree(t, data))
        else:

            if len(t.structure) == 3:  # one tree
                if sig:
                    t.structure[1].previous_training = t.train_semantics
                    t.structure[1].train_semantics = torch.sigmoid(
                        apply_tree(t.structure[1], data)
                    )
                else:
                    t.structure[1].previous_training = t.train_semantics
                    t.structure[1].train_semantics = apply_tree(t.structure[1], data)

            elif len(t.structure) == 4:  # two tree
                t.structure[1].previous_training = t.train_semantics
                t.structure[1].train_semantics = torch.sigmoid(
                    apply_tree(t.structure[1], data)
                )

                t.structure[2].previous_training = t.train_semantics
                t.structure[2].train_semantics = torch.sigmoid(
                    apply_tree(t.structure[2], data)
                )

            semantics.append(t.structure[0](*t.structure[1:], testing=False))

    operator = torch.sum if operator == "sum" else torch.prod

    return torch.clamp(
        operator(torch.stack(semantics), dim=0), -1000000000000.0, 1000000000000.0
    )
