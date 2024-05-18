


class Tree:
    LABELS = None
    PERCENTILES = None

    def __init__(self, structure, train_semantics, test_semantics, reconstruct=True):
        if structure is not None and reconstruct:
            self.structure = structure  # either repr_ from gp(tuple) or list of pointers

        self.train_semantics = train_semantics
        self.test_semantics = test_semantics

        self.fitness = None
        self.test_fitness = None

        # TODO: Depth calculation in utils and call it here

