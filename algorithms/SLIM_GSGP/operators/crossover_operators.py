import torch
from algorithms.GP.operators.crossover_operators import crossover_trees
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.representations.individual import Individual


def slim_head_crossover(FUNCTIONS):
    _xo = crossover_trees(FUNCTIONS)

    def _test_row(tree_semantics, n_test):
        if tree_semantics.shape == torch.Size([]):
            return tree_semantics.repeat(n_test)
        return tree_semantics

    def head_xo(p1, p2, X_train=None, X_test=None, reconstruct=True):
        h1, h2 = p1.collection[0], p2.collection[0]

        if not (isinstance(h1.structure, tuple) and isinstance(h2.structure, tuple)):
            return p1, p2

        s1, s2 = _xo(h1.structure, h2.structure, h1.nodes, h2.nodes)

        new_h1 = Tree(structure=s1, train_semantics=None, test_semantics=None, reconstruct=reconstruct)
        new_h2 = Tree(structure=s2, train_semantics=None, test_semantics=None, reconstruct=reconstruct)

        # Compute train semantics eagerly (like inflate_mutation does) so offs_pop.calculate_semantics
        # is a no-op for XO offspring, matching the O(1) cost of mutation generations.
        if X_train is not None:
            new_h1.calculate_semantics(X_train)
            new_h2.calculate_semantics(X_train)
            n_train = len(X_train)
            tr1 = torch.cat([_test_row(new_h1.train_semantics, n_train).unsqueeze(0), p1.train_semantics[1:]], dim=0)
            tr2 = torch.cat([_test_row(new_h2.train_semantics, n_train).unsqueeze(0), p2.train_semantics[1:]], dim=0)
        else:
            tr1 = tr2 = None

        if X_test is not None and p1.test_semantics is not None and p2.test_semantics is not None:
            new_h1.calculate_semantics(X_test, testing=True)
            new_h2.calculate_semantics(X_test, testing=True)
            n = len(X_test)
            ts1 = torch.cat([_test_row(new_h1.test_semantics, n).unsqueeze(0), p1.test_semantics[1:]], dim=0)
            ts2 = torch.cat([_test_row(new_h2.test_semantics, n).unsqueeze(0), p2.test_semantics[1:]], dim=0)
        else:
            ts1 = ts2 = None

        off1 = Individual(
            collection=[new_h1] + p1.collection[1:],
            train_semantics=tr1,
            test_semantics=ts1,
            reconstruct=reconstruct,
        )
        off2 = Individual(
            collection=[new_h2] + p2.collection[1:],
            train_semantics=tr2,
            test_semantics=ts2,
            reconstruct=reconstruct,
        )
        return off1, off2

    return head_xo
