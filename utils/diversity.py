from scipy.stats import entropy
import torch


def niche_entropy(repr_, n_niches=10):
    # https://www.semanticscholar.org/paper/Entropy-Driven-Adaptive-RoscaComputer/ab5c8a8f415f79c5ec6ff6281ed7113736615682
    # https://strathprints.strath.ac.uk/76488/1/Marchetti_etal_Springer_2021_Inclusive_genetic_programming.pdf

    num_nodes = [len(ind) - 1 for ind in repr_]

    min_ = min(num_nodes)
    pop_size = len(repr_)

    stride = (max(num_nodes) - min_) / n_niches

    distributions = []
    for i in range(1, n_niches + 1):

        distribution = (
            sum(
                list(
                    map(
                        lambda x: (
                            True
                            if (i - 1) * stride + min_ <= x < i * stride + min_
                            else False
                        ),
                        num_nodes,
                    )
                )
            )
            / pop_size
        )
        if distribution > 0:
            distributions.append(distribution)

    return entropy(distributions)


def gsgp_pop_div_from_vectors(
    sem_vectors,
):  # https://ieeexplore.ieee.org/document/9283096

    return torch.sum(torch.cdist(sem_vectors, sem_vectors)) / (
        sem_vectors.shape[0] ** 2
    )
    # sem_vectors[~torch.isinf(sem_vectors)]
