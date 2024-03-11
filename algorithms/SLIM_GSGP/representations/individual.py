import torch

class Individual():

    def __init__(self, collection):

        self.collection = collection
        self.structure = [tree.structure for tree in collection]
        self.size = len(collection) #size == number of blocks
        self.nodes_count = sum([tree.nodes for tree in collection])

        self.train_semantics = None
        self.test_semantics = None

    def calculate_semantics(self, inputs, testing = False):

        [tree.calculate_semantics(inputs, testing) for tree in self.collection]

        if testing:
            self.test_semantics = [tree.test_semantics for tree in self.collection]

        else:
            self.train_semantics = [tree.train_semantics for tree in self.collection]


    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.collection[item]

    def remove_block(self, index):

        new_collection = [*self.collection[:index], *self.collection[index+1:]]

        new_individual = Individual(new_collection)

        return new_individual


    def add_block(self, tree):

        new_collection = [*self.collection, tree]
        new_individual = Individual(new_collection)

        return new_individual



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
            operator = sum
        else:
            # TODO could improve, change self.train_semantics to tensor??, the way it's coded now it's worng
            operator = lambda x:[ x[i]*x[i+1] for i in range(len(x)-1)][0]
            
            
        if testing:
            self.test_fitness = ffunction(operator(self.test_semantics), y)

        else:
            self.fitness = ffunction(operator(self.train_semantics), y)
