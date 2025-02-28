from algorithms.GSGP.representations.tree_utils import apply_tree, nested_depth_calculator, nested_nodes_calculator
from algorithms.GP.representations.tree_utils import flatten, tree_depth
import torch
from scipy.optimize import fmin
import numpy as np

class Tree:
    FUNCTIONS = None
    TERMINALS = None
    CONSTANTS = None

    def __init__(self, structure, train_semantics, test_semantics, reconstruct):
        self.FUNCTIONS = Tree.FUNCTIONS
        self.TERMINALS = Tree.TERMINALS
        self.CONSTANTS = Tree.CONSTANTS
        
        if structure is not None and reconstruct:
            self.structure = structure # either repr_ from gp(tuple) or list of pointers

        self.train_semantics = train_semantics
        self.test_semantics = test_semantics

        if isinstance(structure, tuple):
            self.depth = tree_depth(Tree.FUNCTIONS)(structure)
            self.nodes = len(list(flatten(structure)))
        elif reconstruct:
            self.depth = nested_depth_calculator(self.structure[0],
                                                  [tree.depth for tree in self.structure[1:] if isinstance(tree, Tree)])
            # operator_nodes = [5, self.structure[-1].nodes] if self.structure[0].__name__ == 'geometric_crossover' else [4]
            self.nodes = nested_nodes_calculator(self.structure[0],
                                                 [tree.nodes for tree in self.structure[1:] if isinstance(tree, Tree)])



        self.fitness = None
        self.test_fitness = None

    def calculate_semantics(self, inputs, testing = False, logistic=False, adjusted_sigmoid = False):

        if adjusted_sigmoid:
            structure = flatten_structure(self.structure)
            scaling_factor = torch.max(torch.abs(inputs[:, [int(el[1:]) for el in structure if 'x' in el]]))
        else:
            scaling_factor = None

        if testing and self.test_semantics is None:
            # checking if the individual is part of the initial population (table) or is a random tree (table)
            if isinstance(self.structure, tuple):
                self.test_semantics = modified_sigmoid(apply_tree(self, inputs), scaling_factor=scaling_factor) if logistic else apply_tree(self, inputs)
            else:
                # self.structure[0] is the operator (mutation or xo) while the remaining of the structure dare pointers to the trees
                self.test_semantics = self.structure[0](*self.structure[1:], testing=True)
        elif self.train_semantics is None:
            if isinstance(self.structure, tuple):
                self.train_semantics =modified_sigmoid(apply_tree(self, inputs), scaling_factor=scaling_factor) if logistic else apply_tree(self, inputs)
            else:
                self.train_semantics = self.structure[0](*self.structure[1:], testing=False)


    def evaluate(self, ffunction, y, testing=False):

        """
        evaluates the tree given a certain fitness function (ffunction) x) and target data (y).

        This evaluation is performed by applying ffunction to the semantics of self and the expected output y. The
        optional parameter testing is used to control whether the training y or test y is being used as well as which
        semantics should be used. The result is stored as an attribute of self.

        Parameters
        ----------
        ffunction: function
            fitness function to evaluate the individual
        y: torch tensor
            the expected output (target) values
        testing: bool
            Flag symbolizing if the y is from testing

        Returns
        -------
        None
            attributes a fitness value to the tree
        """

        # attributing the tree fitness
        if testing:
            self.test_fitness = ffunction(y, self.test_semantics)
        else:
            self.fitness = ffunction(y, self.train_semantics)

def modified_sigmoid(tensor, scaling_factor = None):
    if scaling_factor:
        return torch.div(1, torch.add(1, torch.exp(torch.mul(-1, torch.div(tensor, scaling_factor)))))
    else:
        return torch.sigmoid(tensor)

def flatten_structure(struct):
    result = []
    if isinstance(struct, tuple):
        result.append(struct[0])  # Add the operator
        for item in struct[1:]:
            result.extend(flatten_structure(item))
    else:
        result.append(struct)  # Base case: add the variable directly
    return result

def evaluate_structure(struct, values):
    if isinstance(struct, tuple):
        operator = struct[0]
        operands = [evaluate_structure(item, values) for item in struct[1:]]
        if operator == 'add':
            return sum(operands)
        elif operator == 'subtract':
            return operands[0] - operands[1]
    else:
        return values.get(struct, 0)  # Default to 0 if variable not in dictionary

def generate_variables(struct):
    flattened = flatten_structure(struct)
    return sorted(set(item for item in flattened if isinstance(item, str) and not item in ['add', 'subtract']))

def objective_function(x, struct, variables):
    values = dict(zip(variables, x))
    return evaluate_structure(struct, values)

def find_minimum(struct):
    variables = generate_variables(struct)
    initial_guess = np.zeros(len(variables))
    result = fmin(objective_function, initial_guess, args=(struct, variables), disp=False)
    min_value = objective_function(result, struct, variables)
    return min_value