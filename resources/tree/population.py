from tree.utils.utils import flatten

class Population():


    def __init__(self, pop):


        self.pop = pop
        self.size = len(pop)
        self.nodes_count = sum([len(list(flatten(ind.repr_))) for ind in pop])

    def evaluate(self, solver, std_params, test = False):

        self.fit = [individual.evaluate_tree(solver, std_params, test)
                    for individual in self.pop]

        self.nodes = sum([individual.nodes for individual in self.pop])