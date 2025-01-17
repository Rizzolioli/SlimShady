from datasets.data_loader import *
from algorithms.GSGP.representations.population import Population
from algorithms.GSGP.representations.tree import Tree
from algorithms.GP.representations.tree import Tree as GP_Tree
from algorithms.GP.operators.initializers import rhh
import matplotlib.pyplot as plt
from algorithms.GSGP.operators.mutators import standard_geometric_mutation
from utils import generate_random_uniform, protected_div, specular_log, modified_sigmoid, pearson_corr
import torch

img = False

FUNCTIONS = {
    'add': {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide': {'function': lambda x, y: protected_div(x, y), 'arity': 2}
}

"""

'mean': {'function': lambda x, y: mean_(x, y), 'arity': 2},
    'tan': {'function': lambda x: torch.tan(x), 'arity': 1},
    'sin': {'function': lambda x: torch.sin(x), 'arity': 1},
    'cos': {'function': lambda x: torch.cos(x), 'arity': 1},

"""

CONSTANTS = {
    'constant_2': lambda x: torch.tensor(2).float(),
    'constant_3': lambda x: torch.tensor(3).float(),
    'constant_4': lambda x: torch.tensor(4).float(),
    'constant_5': lambda x: torch.tensor(5).float(),
    'constant__1': lambda x: torch.tensor(-1).float()
}

########################################################################################################################

# DATASETS & ALGORITHMS

########################################################################################################################

algos = ["GSGP"]

# data_loaders = [ "airfoil", "concrete_slump", "concrete_strength", "ppb", "ld50", "bioavalability", "yatch"]
data_loaders = [load_yatch, load_airfoil, load_concrete_slump, load_concrete_strength, load_ppb,
                load_bioav, load_ld50]

init_dict = {'init_pop_size': 100,
                'init_depth': 8,
                'FUNCTIONS': FUNCTIONS,
                'CONSTANTS': CONSTANTS,
                "p_c": 0}

Tree.FUNCTIONS = FUNCTIONS
Tree.CONSTANTS = CONSTANTS
GP_Tree.FUNCTIONS = FUNCTIONS
GP_Tree.CONSTANTS = CONSTANTS


for loader in data_loaders:
    # Loads the data via the dataset loader
    X, y = loader(X_y=True)

    # getting the name of the dataset
    dataset = loader.__name__.split("load_")[-1]

    TERMINALS = {f"x{i}": i for i in range(X.shape[1])}

    init_dict['TERMINALS'] = TERMINALS

    Tree.TERMINALS = TERMINALS
    GP_Tree.TERMINALS = TERMINALS

    # initializing the population
    population = Population([Tree(structure=tree,
                                  train_semantics=None,
                                  test_semantics=None,
                                  reconstruct=True) for tree in rhh(**init_dict)])  # reconstruct set as true to calculate the initial pop semantics

    # getting the individuals' semantics
    population.calculate_semantics(X)

    semantics = torch.flatten(torch.cat(population.train_semantics))

    # initializing the population
    population_2 = Population([Tree(structure=tree,
                                  train_semantics=None,
                                  test_semantics=None,
                                  reconstruct=True) for tree in rhh(**init_dict)])  # reconstruct set as true to calculate the initial pop semantics

    # getting the individuals' semantics
    population_2.calculate_semantics(X)

    semantics_2 = torch.flatten(torch.cat(population_2.train_semantics))

    if img:

        plt.hist(semantics)
        plt.title(dataset)
        plt.savefig(os.getcwd() + f'\\tree_output_distr\\{dataset}.png')
        # plt.show()
        # plt.close()

        plt.clf()

    # plt.boxplot(semantics)
    # plt.title(dataset)
    # plt.savefig(os.getcwd() + f'\\tree_output_distr\\{dataset}_bp.png')
    # plt.show()
    # plt.close()

    # plt.clf()

    print(dataset)
    # print('Q1 input', torch.quantile(X, 0.25).item())
    # print('Q1 tree', torch.quantile(semantics, 0.25).item())
    #
    # print('Q2 input', torch.quantile(X, 0.5).item())
    # print('Q2 tree', torch.quantile(semantics, 0.5).item())
    #
    # print('Q3 input', torch.quantile(X, 0.75).item())
    # print('Q3 tree', torch.quantile(semantics, 0.75).item())



    log_sem = specular_log(semantics)
    print('CORR LOG: ', pearson_corr(semantics, log_sem).item())
    # plt.hist(log_tens[torch.isfinite(log_tens)])
    if img:
        plt.hist(log_sem)
        plt.title(dataset + '_log')
        plt.savefig(os.getcwd() + f'\\tree_output_distr\\{dataset}_log.png')
        # plt.show()
        # plt.close()

        plt.clf()

    abs_sem = torch.abs(semantics)
    print('CORR ABS: ', pearson_corr(semantics, abs_sem).item())
    # plt.hist(log_tens[torch.isfinite(log_tens)])
    if img:
        plt.hist(abs_sem)
        plt.title(dataset + '_abs')
        plt.savefig(os.getcwd() + f'\\tree_output_distr\\{dataset}_abs.png')
        # plt.show()
        # plt.close()

        plt.clf()

    ms_abs = torch.sub(1, torch.div(2, torch.add(1, abs_sem)))
    plt.hist(ms_abs)
    plt.title(dataset + '_ms_abs')
    # plt.savefig(os.getcwd() + f'\\tree_output_distr\\{dataset}_abs.png')
    plt.show()
    # plt.close()

    plt.clf()

    adj_abs_sem = torch.add(semantics, torch.abs(torch.min(semantics)))
    print('CORR ADJ ABS: ', pearson_corr(semantics, adj_abs_sem).item())
    # plt.hist(log_tens[torch.isfinite(log_tens)])
    if img:
        plt.hist(adj_abs_sem)
        plt.title(dataset + '_adj_abs')
        plt.savefig(os.getcwd() + f'\\tree_output_distr\\{dataset}_adj_abs.png')
        # plt.show()
        # plt.close()

        plt.clf()


    sig_sem = torch.sigmoid(semantics)
    print('CORR SIG: ', pearson_corr(semantics, sig_sem).item())

    if img:
        plt.hist(sig_sem)
        plt.title(dataset + '_sigmoid')
        plt.savefig(os.getcwd() + f'\\tree_output_distr\\{dataset}_sigmoid.png')
        # plt.show()
        # plt.close()

        plt.clf()

    ms_sig2 = torch.sub(sig_sem, torch.sigmoid(semantics_2))
    plt.hist(ms_sig2)
    plt.title(dataset + '_ms_sig2')
    # plt.savefig(os.getcwd() + f'\\tree_output_distr\\{dataset}_abs.png')
    plt.show()
    # plt.close()

    plt.clf()

    ms_sig1 = torch.sub(torch.mul(2, sig_sem),1)
    plt.hist(ms_sig1)
    plt.title(dataset + '_ms_sig1')
    # plt.savefig(os.getcwd() + f'\\tree_output_distr\\{dataset}_abs.png')
    plt.show()
    # plt.close()

    plt.clf()

    # scaling_factor = max(abs(torch.quantile(X, 0.25).item()), abs(torch.quantile(X, 0.75).item()))
    # print('SCALING FACTOR: ', scaling_factor)
    # msig_sem = modified_sigmoid(semantics, scaling_factor)
    # print('CORR MSIG: ', pearson_corr(semantics, msig_sem).item())

    # scaling_factor = 100000000
    # print('SCALING FACTOR: ', scaling_factor)
    # msig_sem = modified_sigmoid(semantics, scaling_factor)
    # print('CORR MSIG: ', pearson_corr(semantics, msig_sem).item())

    # scaling_factor = max(abs(torch.min(X).item()), abs(torch.max(X).item()))
    # print('SCALING FACTOR: ', scaling_factor)
    # msig_sem = modified_sigmoid(semantics, scaling_factor)
    # print('CORR MSIG: ', pearson_corr(semantics, msig_sem).item())
    #
    # scaling_factor = max(abs(torch.quantile(X, 0.01).item()), abs(torch.quantile(X, 0.99).item()))
    # print('SCALING FACTOR: ', scaling_factor)
    # msig_sem = modified_sigmoid(semantics, scaling_factor)
    # print('CORR MSIG: ', pearson_corr(semantics, msig_sem).item())

    # if img:
    #     plt.hist(msig_sem)
    #     plt.title(dataset + '_modified_sigmoid')
    #     plt.savefig(os.getcwd() + f'\\tree_output_distr\\{dataset}_modified_sigmoid.png')
    #     # plt.show()
    #     # plt.close()
    #
    #     plt.clf()

    print('\n')