import torch

class Individual():

    def __init__(self, collection):

        self.collection = collection
        self.structure = [tree.structure for tree in collection]
        self.size = len(collection) #TODO not really size(it's not the depth) it's the number of blocks
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

        new_collection = self.collection[:index] + self.collection[index+1:]

        new_individual = Individual(new_collection)

        return new_individual
        #
        # self.structure.pop(index)
        # self.size -= 1
        # self.nodes_count = sum([ind.nodes for ind in self.collection])
        #
        # if self.train_semantics != None:
        #     self.train_semantics.pop(index)
        # if self.test_semantics != None:
        #     self.test_semantics.pop(index)


    def add_block(self, tree): #TODO should this be a method of Individual? I think it's more appropriate to put it somewhere else
        # Old
        # self.collection.append(tree)
        # New
        new_collection = self.collection + [tree]
        new_individual = Individual(new_collection)

        return new_individual
        #
        #
        # self.structure.append(tree)
        # self.size += 1
        # self.nodes_count += tree.nodes
        #
        # if self.train_semantics != None:
        #     self.train_semantics.append(tree.train_semantics)
        # if self.test_semantics != None:
        #     self.test_semantics.append(tree.test_semantics)



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
            operator = lambda x:[ x[i]*x[i+1] for i in range(len(x)-1)][0] #TODO could improve, change self.train_semantics to tensor??
            
            
        if testing:
            self.test_fitness = ffunction(operator(self.test_semantics), y)

        else:
            self.fitness = ffunction(operator(self.train_semantics), y)
