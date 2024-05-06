import random
import torch
from utils.utils import get_random_tree
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.representations.individual import Individual

def std_xo_delta(operator='sum'):

    def stdxo_delta(p1, p2, tr, testing):

        if testing:
            return torch.add(torch.mul(p1.test_semantics, tr.test_semantics),
                             torch.mul(torch.sub(1, tr.test_semantics), p2.test_semantics)) if operator == 'sum' else \
                    torch.mul(torch.pow(p1.test_semantics, tr.test_semantics),
                              torch.pow(p2.test_semantics, torch.sub(1,  tr.test_semantics)))

        else:
            return torch.add(torch.mul(p1.train_semantics, tr.train_semantics),
                             torch.mul(torch.sub(1, tr.train_semantics), p2.train_semantics)) if operator == 'sum' else \
                torch.mul(torch.pow(p1.train_semantics, tr.train_semantics),
                          torch.pow(p2.train_semantics, torch.sub(1, tr.train_semantics)))


    return stdxo_delta

def std_xo_ot_delta(which, operator='sum'):

    def stdxo_ot_delta(p, tr, testing):

        if which == 'first':

            if testing:
                return torch.mul(p.test_semantics, tr.test_semantics) if operator == 'sum' else \
                    torch.pow(p.test_semantics, tr.test_semantics)

            else:
                return torch.mul(p.train_semantics, tr.train_semantics) if operator == 'sum' else \
                    torch.pow(p.train_semantics, tr.train_semantics)
        else:

            if testing:
                return torch.mul(p.test_semantics, torch.sub(1, tr.test_semantics)) if operator == 'sum' else \
                    torch.pow(p.test_semantics, torch.sub(1, tr.test_semantics))

            else:
                return torch.mul(p.train_semantics, torch.sub(1, tr.train_semantics)) if operator == 'sum' else \
                    torch.pow(p.train_semantics, torch.sub(1, tr.train_semantics))


    return stdxo_ot_delta


def slim_geometric_crossover(FUNCTIONS, TERMINALS, CONSTANTS, operator, max_depth = 8, grow_probability = 1, p_c = 0):

    def inner_xo(p1, p2, X, X_test = None):

        random_tree = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c,
                                       grow_probability=grow_probability, logistic=True)

        random_tree.calculate_semantics(X, testing=False, logistic=True)
        if X_test != None:
            random_tree.calculate_semantics(X_test, testing=True, logistic=True )

        offs = [Tree([std_xo_delta(operator=operator),
                      p1.collection[i], p2.collection[i],
                      random_tree]) for i in range(min(p1.size, p2.size))]

        if p1.size > p2.size:

            which = 'first'

            offs += [Tree([std_xo_ot_delta(which, operator=operator),
                          p1.collection[i], random_tree]) for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]

        else:

            which = 'second'

            offs += [Tree([std_xo_ot_delta(which, operator=operator),
                           p2.collection[i], random_tree]) for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]


        offs = Individual(offs)
        offs.calculate_semantics(X)
        if X_test is not None:
            offs.calculate_semantics(X_test, testing=True)

        return offs

    return inner_xo



