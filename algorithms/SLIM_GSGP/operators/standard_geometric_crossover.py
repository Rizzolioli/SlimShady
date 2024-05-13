import random
import torch
from utils.utils import get_random_tree
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.representations.individual import Individual

def std_xo_delta(operator='sum'):

    def stdxo_delta(p1, p2, tr, testing = False):

        if isinstance(tr, Tree):        
            if testing:
                p1 = p1.test_semantics
                tr = tr.test_semantics
                p2 = p2.test_semantics
            else:
                p1 = p1.train_semantics
                tr = tr.train_semantics
                p2 = p2.train_semantics


        return torch.add(torch.mul(p1, tr),
                         torch.mul(torch.sub(1, tr), p2)) if operator == 'sum' else \
                torch.mul(torch.pow(p1, tr),
                          torch.pow(p2, torch.sub(1,  tr)))

    return stdxo_delta

def std_xo_ot_delta(which, operator='sum'):

    def stdxo_ot_delta(p, tr, testing =False):
        
        if isinstance(tr, Tree):        
            if testing:
                p = p.test_semantics
                tr = tr.test_semantics

            else:
                p = p.train_semantics
                tr = tr.train_semantics

        if which == 'first':

            return torch.mul(p, tr) if operator == 'sum' else \
                torch.pow(p, tr)
        else:


            return torch.mul(p, torch.sub(1, tr)) if operator == 'sum' else \
                    torch.pow(p, torch.sub(1, tr))

    stdxo_ot_delta.__name__ += '_' + which


    return stdxo_ot_delta


def slim_geometric_crossover(FUNCTIONS, TERMINALS, CONSTANTS, operator, max_depth = 8, grow_probability = 1, p_c = 0):

    def inner_xo(p1, p2, X, X_test = None, reconstruct = True):

        random_tree = get_random_tree(max_depth, FUNCTIONS, TERMINALS, CONSTANTS, inputs=X, p_c=p_c,
                                       grow_probability=grow_probability, logistic=True)

        random_tree.calculate_semantics(X, testing=False, logistic=True)
        if X_test != None:
            random_tree.calculate_semantics(X_test, testing=True, logistic=True )

        random_tree.depth += 1

        if reconstruct:
            offs_collection = [Tree([std_xo_delta(operator=operator),
                          p1.collection[i], p2.collection[i],
                          random_tree]) for i in range(min(p1.size, p2.size))]


        offs_train_semantics = torch.stack([std_xo_delta(operator=operator)(p1.train_semantics[i], p2.train_semantics[i],
                                random_tree.train_semantics) for i in range(min(p1.size, p2.size))])
        if X_test is not None:
            offs_test_semantics = torch.stack([std_xo_delta(operator=operator)(p1.test_semantics[i], p2.test_semantics[i],
                                    random_tree.test_semantics) for i in range(min(p1.size, p2.size))])

        offs_nodes_collection = [p1.nodes_collection[i] + p2.nodes_collection[i] + 2 * random_tree.nodes + 5
                                 for i in range(min(p1.size, p2.size))]
        offs_depth_collection = [max([p1.depth_collection[i]+2, p2.depth_collection[i]+2, random_tree.depth + 3])
                                 for i in range(min(p1.size, p2.size))]

        if p1.size > p2.size:

            random_tree.depth -= 1

            which = 'first'

            if reconstruct:
                offs_collection += [Tree([std_xo_ot_delta(which, operator=operator),
                          p1.collection[i], random_tree]) for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]

            offs_train_semantics = torch.stack([*offs_train_semantics,
                                    *[std_xo_ot_delta(which, operator=operator)(p1.train_semantics[i], random_tree.train_semantics)
                                    for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]])

            if X_test is not None:
                offs_test_semantics = torch.stack([*offs_test_semantics,
                                      *[std_xo_ot_delta(which, operator=operator)(p1.test_semantics[i], random_tree.test_semantics)
                                       for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]])

            offs_nodes_collection += [p1.nodes_collection[i] + random_tree.nodes + 1
                                     for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]
            offs_depth_collection += [ max([p1.depth_collection[i] + 1,  random_tree.depth + 1])
                                      for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]

        else:

            which = 'second'

            if reconstruct:
                offs_collection += [Tree([std_xo_ot_delta(which, operator=operator),
                           p2.collection[i], random_tree]) for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]

            offs_train_semantics = torch.stack([*offs_train_semantics,
                                    *[std_xo_ot_delta(which, operator=operator)(p2.train_semantics[i], random_tree.train_semantics)
                                    for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]])

            if X_test is not None:
                offs_test_semantics = torch.stack([*offs_test_semantics,
                                      *[std_xo_ot_delta(which, operator=operator)(p2.test_semantics[i], random_tree.test_semantics)
                                       for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]])


            offs_nodes_collection += [p2.nodes_collection[i] + random_tree.nodes + 3
                                     for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]
            offs_depth_collection += [ max([p2.depth_collection[i] + 1,  random_tree.depth + 2])
                                      for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]


        offs = Individual(collection=offs_collection if reconstruct else None,
                          train_semantics=torch.Tensor(offs_train_semantics),
                          test_semantics=torch.Tensor(offs_test_semantics) if X_test is not None else None,
                          reconstruct=reconstruct)

        offs.size = max(p1.size, p2.size)

        offs.nodes_collection = offs_nodes_collection
        offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

        offs.depth_collection = offs_depth_collection
        offs.depth = max([depth - (i - 1) if i != 0 else depth
                          for i, depth in enumerate(offs.depth_collection)]) + (offs.size - 1)


        # offs.calculate_semantics(X)
        # if X_test is not None:
        #     offs.calculate_semantics(X_test, testing=True)

        return offs

    return inner_xo



