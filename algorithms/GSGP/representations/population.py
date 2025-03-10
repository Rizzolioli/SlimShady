class Population():

    def __init__(self, pop):

        self.population = pop
        self.size = len(pop)
        self.nodes_count = sum([ind.nodes for ind in pop])

    def calculate_semantics(self, inputs, testing = False):

        [individual.calculate_semantics(inputs, testing) for individual in self.population]

        if testing:
            self.test_semantics = [individual.test_semantics for individual in self.population]

        else:
            self.train_semantics = [individual.train_semantics for individual in self.population]


    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.population[item]


    def evaluate(self, ffunction, y):
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
        [individual.evaluate(ffunction, y) for individual in self.population]

        self.fit = [individual.fitness for individual in self.population]