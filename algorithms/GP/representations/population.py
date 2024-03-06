
class Population:

    def __init__(self, pop):
        self.pop = pop
        self.size = len(pop)
        self.nodes_count = sum([ind.node_count for ind in pop])

    def evaluate(self, ffunction, X, y):

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

        # evaluating all the individuals in the population on training
        [individual.evaluate(ffunction, X, y) for individual in self.pop]

        self.fit = [individual.fitness for individual in self.pop]
