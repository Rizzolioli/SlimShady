from datasets.data_loader import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as npù
import torch

data_loaders = [load_yatch, load_airfoil, load_concrete_slump, load_concrete_strength, load_ppb,
                load_bioav, load_ld50]

for loader in data_loaders:
    print(loader.__name__.split("load_")[-1])
    X,y = loader(X_y = True)
    plt.hist(y)
    # print(torch.median(y).item())
    # print(torch.std(y).item())
    plt.title(loader.__name__.split("load_")[-1])
    # plt.savefig(f'input_distr/{loader.__name__.split("load_")[-1]}.png')
    plt.show()
    plt.close()

    y = StandardScaler().fit_transform(y.reshape(-1, 1))
    plt.hist(y)
    plt.title(loader.__name__.split("load_")[-1] + '_scaled')
    # plt.savefig(f'input_distr/{loader.__name__.split("load_")[-1]}.png')
    plt.show()
    plt.close()

