import numpy as np
import random

def tournament_selection_min_slim(pool_size):

    """
    Performs tournament selection to select an individual with the lowest fitness from a population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    function
        Function to execute tournament selection for minimum fitness.
    """

    def ts(pop, deflate=False):

        # if deflate:
        #
        #     valid_pop = [ind for ind in pop.population if ind.size > 1]
        #
        #     if len(valid_pop) < pool_size:
        #
        #          return None
        #
        #     else:
        #
        #         pool = random.sample(valid_pop, k=pool_size)

        # else:

        pool = random.choices(pop.population, k=pool_size)

        return pool[np.argmin([ind.fitness for ind in pool])]

    return ts


def tournament_selection_max_slim(pool_size):
    """
    Performs tournament selection to select an individual with the lowest fitness from a population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    function
        Function to execute tournament selection for minimum fitness.
    """

    def ts(pop, deflate=False):

        if deflate:

            valid_pop = [ind for ind in pop.population if ind.size > 1]

            pool = random.sample(valid_pop, k=pool_size)

        else:

            pool = random.sample(pop.population, k=pool_size)


        return pool[np.argmax([ind.fitness for ind in pool])]

    return ts