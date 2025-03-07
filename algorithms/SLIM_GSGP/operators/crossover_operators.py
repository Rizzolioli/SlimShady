# MIT License
#
# Copyright (c) 2024 sofiaccap
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import torch
import numpy as np
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.representations.individual import Individual
# from utils.utils import get_random_tree, _evaluate_slim_individual


def swap_base_crossover(individual1, individual2, reconstruct = True ):
    """
    Performs crossover between two individuals by swapping the first block of them.

    Parameters
    ----------
    individual1 : Tree
        The first parent individual.
    individual2 : Tree
        The second parent individual.
    reconstruct : bool, optional
        Whether to reconstruct the Individual's collection after crossover (default: True).

    Returns
    -------
    Individual
            The base swaped crossover tree Individuals.
    """

    # changing the first block of two individuals will lead to two new individuals
    offs1 = Individual(collection=(
            [
                *individual2.collection[:1],
                *individual1.collection[1:],
            ]
            if reconstruct
            else None
        ),
         train_semantics=torch.stack(
            [
                *individual2.train_semantics[:1],
                *individual1.train_semantics[1:],
            ]
        ),
        test_semantics=(
            torch.stack(
                [
                    *individual2.test_semantics[:1],
                    *individual1.test_semantics[1:],
                ]
            )
            if individual1.test_semantics is not None and individual2.test_semantics is not None
            else None
        ),
        reconstruct=reconstruct,
        
        )
    
    offs2 = Individual(collection = ( 
            [
                *individual1.collection[:1],
                *individual2.collection[1:],
            ] 
            if reconstruct
            else None
            ),
             train_semantics=torch.stack(
            [
                *individual1.train_semantics[:1],
                *individual2.train_semantics[1 :],
            ]
        ),
        test_semantics=(
            torch.stack(
                [
                    *individual1.test_semantics[:1],
                    *individual2.test_semantics[1 :],
                ]
            )
            if individual1.test_semantics is not None and individual2.test_semantics is not None
            else None
        ),
        reconstruct=reconstruct,
            
            )
    
    # computing offspring attributes
    offs1.size = individual1.size 
    offs2.size = individual2.size 

    offs1.nodes_collection = [
        *individual2.nodes_collection[:1],
        *individual1.nodes_collection[1:],
    ]

    offs2.nodes_collection = [
        *individual1.nodes_collection[:1],
        *individual2.nodes_collection[1:],
    ]

   
    offs1.nodes_count = sum(offs1.nodes_collection) + (offs1.size - 1)
    offs2.nodes_count = sum(offs2.nodes_collection) + (offs2.size - 1)

    offs1.depth_collection = [
        *individual2.depth_collection[:1],
        *individual1.depth_collection[1:],
    ]

    offs2.depth_collection = [
        *individual1.depth_collection[:1],
        *individual2.depth_collection[1:],
    ]

    offs1.depth = max(
        [
            depth - (i - 1) if i != 0 else depth
            for i, depth in enumerate(offs1.depth_collection)
        ]
    ) + (offs1.size - 1)

    offs2.depth = max(
        [
            depth - (i - 1) if i != 0 else depth
            for i, depth in enumerate(offs2.depth_collection)
        ]
    ) + (offs2.size - 1)

    return offs1, offs2
def donor_xo(individual1, individual2, reconstruct = True):
    """
    Performs crossover between 2 individuals by donating a block of the donor to the reciever (selected at random between 2 individuals)

    Parameters
    ----------
    individual1 : Tree
        The first individual.
    individual2: Tree
        The second individual.
    reconstruct : bool, optional
        Whether to reconstruct the Individual's collection after crossover (default: True).

    Returns
    -------
    Individuals
            The donor_gxo crossover tree Individuals.
    """
      
    #choose at random the index of the donor (0 = individual1, 1= individual2)
    donor_idx = random.randint(0,1)
    donor = individual1 if donor_idx == 0 else individual2
    recipient = individual1 if donor_idx == 1 else individual2

    #choose at random a block from the donor
    if donor.size > 1:
        donation_idx = random.randint(1, donor.size - 1 )

        recipient_offs = Individual(collection=[*recipient.collection, donor[donation_idx]]
                         if reconstruct else None,
                          train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
                          test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
                          if donor.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)

        donor_offs = Individual(collection=donor.collection[:donation_idx] + donor.collection[donation_idx+1:]
                         if reconstruct else None,
                          train_semantics=torch.stack(
                                [*donor.train_semantics[:donation_idx], *donor.train_semantics[donation_idx + 1:]]),
                          test_semantics=torch.stack(
                [*donor.test_semantics[:donation_idx], *donor.test_semantics[donation_idx + 1:]])
                          if donor.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)

        recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
        donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[donation_idx + 1:]

        recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
        donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[donation_idx + 1:]

        recipient_offs.size = recipient.size + 1
        donor_offs.size = donor.size - 1

        recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
        donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)

        recipient_offs.depth = max([ depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(recipient_offs.depth_collection)]
                                    ) + (recipient_offs.size - 1)
        
        donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                for i, depth in enumerate(donor_offs.depth_collection)]
                            ) + (donor_offs.size - 1)


        return donor_offs, recipient_offs

    else:
        return individual1, individual2
def best_donor_xo(measure = 'min_fitness'):
    """
    Generate a best_donor_xo function.

    Parameters
    ----------
    measure: string
        Measure to be considered when selecting the 'best' individual to be donor.
    Returns
    -------
    Callable
        An best_donor_xo crossover function (`best_d_xo`).

        Performs crossover between 2 individuals by donating a block of the donor to the reciever.
        The donor is selected based on which individual is the 'best' in terms of the selected measure.

        Parameters
        ----------
        individual1 : Tree
            The first individual.
        individual2: list
            The second individual.
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after crossover (default: True).

        Returns
        -------
        Individuals
            The best_donor_xo crossover tree Individuals.
    """
    def best_d_xo(individual1, individual2, reconstruct = True):

        #choose the best individual
        if measure == 'biggest':
            donor_idx = np.argmax([ind.size for ind in [individual1, individual2]])
        elif measure == 'min_fitness':
            donor_idx = np.argmin([ind.fitness for ind in [individual1, individual2]])
        elif measure == 'max_fitness':
            donor_idx = np.argmax([ind.fitness for ind in [individual1, individual2]])

        #asign donor and reciever
        donor = individual1 if donor_idx == 0 else individual2
        recipient = individual1 if donor_idx == 1 else individual2

        if donor.size > 1:
            # choose at random a block from the donor
            donation_idx = random.randint(1, donor.size - 1 )

            #create new recipient offspring
            recipient_offs = Individual(collection=[*recipient.collection, donor[donation_idx]]
                          if reconstruct else None,
                          train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
                          test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
                          if donor.test_semantics is not None else None,
                          reconstruct=reconstruct)

            #create new donor offspring
            donor_offs = Individual(collection=donor.collection[:donation_idx] + donor.collection[donation_idx+1:]
                         if reconstruct else None,
                         train_semantics=torch.stack([*donor.train_semantics[:donation_idx], *donor.train_semantics[donation_idx + 1:]]),
                          test_semantics=torch.stack([*donor.test_semantics[:donation_idx], *donor.test_semantics[donation_idx + 1:]])
                          if donor.test_semantics is not None else None,
                          reconstruct=reconstruct)

            #add missing information about each offspring
            recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
            donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[donation_idx + 1:]

            recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
            donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[donation_idx + 1:]

            recipient_offs.size = recipient.size + 1
            donor_offs.size = donor.size - 1

            recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
            donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)

            recipient_offs.depth = recipient_offs.depth = max([ depth - (i - 1) if i != 0 else depth
                                                                 for i, depth in enumerate(recipient_offs.depth_collection)]) + (recipient_offs.size - 1)
            donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                for i, depth in enumerate(donor_offs.depth_collection)]) + (donor_offs.size - 1)


            return donor_offs, recipient_offs

        else:
            return individual1, individual2

    return best_d_xo
def donor_n_xo(n=5):
    """
    Generate a donor_n_xo function.

    Parameters
    ----------
    n: int
        Number of individuals to be used as recievers in crossover (1 donor, n receivers).

    Returns
    -------
    Callable
        A donor_n_xo crossover function (`d_n_xo`).
        Performs crossover between n individuals by donating a random block of the donor to the n individuals in the receiver_list.

    Parameters
    ----------
    individual_list: list
        A list of Trees with n+1 individuals to be used on donor_n_xo.
    reconstruct : bool, optional
        Whether to reconstruct the Individual's collection after crossover (default: True).

    Returns
    -------
    Individuals
        The donor_n_gxo crossover tree n+1 Individuals.
    """

    def crossover_for_recipient(donor, recipient, donation_idx, reconstruct):
        """Helper function to apply crossover on a single recipient. Created to make the crossover more efficient"""
        recipient_offs = Individual(
            collection=[*recipient.collection, donor[donation_idx]] if reconstruct else None,
            train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
            test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
            if donor.test_semantics is not None else None,
            reconstruct=reconstruct
        )

        recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
        recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
        recipient_offs.size = recipient.size + 1
        recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
        recipient_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(recipient_offs.depth_collection)]) + (
                                           recipient_offs.size - 1)

        return recipient_offs

    def d_n_xo(individual_list, reconstruct=True):

        # Choose at random the index of the donor
        #donor_idx = random.randint(0, n - 1)
        donor_idx = random.randint(0, n) #n recievers means n+1 individuals
        donor = individual_list[donor_idx]
        recipients = individual_list[:donor_idx] + individual_list[donor_idx + 1:]

        if donor.size > 1:
            # Choose at random a block from the donor
            donation_idx = random.randint(1, donor.size - 1)
            #perform crossover on each of the recipients
            xo_recipients = list(crossover_for_recipient(donor, recipient, donation_idx, reconstruct) for recipient in recipients)

            # Apply crossover on the donor, removing the block selected above
            donor_offs = Individual(
                collection=donor.collection[:donation_idx] + donor.collection[donation_idx + 1:] if reconstruct else None,
                train_semantics=torch.stack(
                    [*donor.train_semantics[:donation_idx], *donor.train_semantics[donation_idx + 1:]]),
                test_semantics=torch.stack(
                    [*donor.test_semantics[:donation_idx], *donor.test_semantics[donation_idx + 1:]])
                if donor.test_semantics is not None else None,
                reconstruct=reconstruct
            )

            donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[
                                                                                  donation_idx + 1:]
            donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[
                                                                                  donation_idx + 1:]
            donor_offs.size = donor.size - 1
            donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)
            donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(donor_offs.depth_collection)]) + (donor_offs.size - 1)

            # Return the donor and the crossover recipients
            xo_recipients = xo_recipients[:donation_idx] + [donor_offs] + xo_recipients[donor_idx:]
            return *xo_recipients,
        else:
            return *individual_list,
    return d_n_xo
def best_donor_n_xo(n=5, measure='min_fitness'):
    """
    Generate a best_donor_n_xo function.

    Parameters
    ----------
    n: int
        Number of individuals to be used as recievers in crossover (1 donor, n recievers).
    measure: string
        Measure to be considered when selecting the 'best' individual to be donor.

    Returns
    -------
    Callable
        An best_donor_n_xo crossover function (`best_d_n_xo`).

        Performs crossover between n+1 individuals by donating a block of the donor to the n individuals in the reciever_list.
        Donor is selected as the best individual considering a specific "measure"

        Parameters
        ----------
        individual_list: list
            A list of Trees with n+1 individuals to be used on best_donor_n_xo.
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after crossover (default: True).

        Returns
        -------
        Individuals
            The best_donor_n_xo crossover tree n+1 Individuals.
    """
    def crossover_for_recipient(donor, recipient, donation_idx, reconstruct):
        """Helper function to apply crossover on a single recipient. Used to make the crossover more efficient"""
        recipient_offs = Individual(
            collection=[*recipient.collection, donor[donation_idx]] if reconstruct else None,
            train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
            test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
            if donor.test_semantics is not None else None,
            reconstruct=reconstruct
        )

        recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
        recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
        recipient_offs.size = recipient.size + 1
        recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
        recipient_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(recipient_offs.depth_collection)]) + (
                                           recipient_offs.size - 1)

        return recipient_offs
    def best_d_n_xo(individual_list, reconstruct=True):

        _ = n  # dummy variable so n is identifiable (specific to the original way i called the crossover)

        # choose best individual to be donor:
        if measure == 'biggest':
            donor_idx = np.argmax([ind.size for ind in individual_list])
        elif measure == 'min_fitness':
            donor_idx = np.argmin([ind.fitness for ind in individual_list])
        elif measure == 'max_fitness':
            donor_idx = np.argmax([ind.fitness for ind in individual_list])

        donor = individual_list[donor_idx]
        recipients = individual_list[:donor_idx] + individual_list[donor_idx + 1:]

        # choose at random a block from the donor
        if donor.size > 1:
            # choose at random a block from the donor
            donation_idx = random.randint(1, donor.size - 1)
            #Apply the crossover to each of the recipients
            xo_recipients = list(
                crossover_for_recipient(donor, recipient, donation_idx, reconstruct) for recipient in recipients)

            donor_offs = Individual(collection=donor.collection[:donation_idx] + donor.collection[donation_idx + 1:]
                                    if reconstruct else None,
                                    train_semantics=torch.stack([*donor.train_semantics[:donation_idx],
                                                                *donor.train_semantics[donation_idx + 1:]]),
                                    test_semantics=torch.stack([*donor.test_semantics[:donation_idx],
                                                                *donor.test_semantics[donation_idx + 1:]])
                                    if donor.test_semantics is not None
                                    else None,
                                    reconstruct=reconstruct)
            #Add missing information in the donor individual
            donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[
                                                                                  donation_idx + 1:]
            donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[
                                                                                  donation_idx + 1:]
            donor_offs.size = donor.size - 1
            donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)
            donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(donor_offs.depth_collection)]) + (donor_offs.size - 1)

            #xo_recipients += (donor_offs,)
            xo_recipients = xo_recipients[:donation_idx] + [donor_offs] + xo_recipients[donor_idx:]
            return *xo_recipients,

        else:
            return *individual_list,

    return best_d_n_xo
def dif_donor_n_xo(n=5):
    """
    Generate a  dif_donor_n_xo function.

    Parameters
    ----------
    n: int
        Number of individuals to be used as recievers in crossover (1 donor, n receivers).

    Returns
    -------
    Callable
        A dif_donor_n_xo crossover function (`dif_d_n_xo`).
        Performs crossover between n+1 individuals by donating a diferent random block of the donor to the n individuals in the receiver_list.

    Parameters
    ----------
    individual_list: list
        A list of Trees with n+1 individuals to be used on donor_n_gxo.
    reconstruct : bool, optional
        Whether to reconstruct the Individual's collection after crossover (default: True).

    Returns
    -------
    Individuals
        The donor_n_gxo crossover tree n+1 Individuals.
    """

    def crossover_for_recipient(donor, recipient, donation_idx, reconstruct):
        """Helper function to apply crossover on a single recipient."""

        recipient_offs = Individual(
            collection=[*recipient.collection, donor[donation_idx]] if reconstruct else None,
            train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
            test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
            if donor.test_semantics is not None else None,
            reconstruct=reconstruct
        )

        recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
        recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
        recipient_offs.size = recipient.size + 1
        recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
        recipient_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(recipient_offs.depth_collection)]) + (
                                           recipient_offs.size - 1)

        return recipient_offs

    def dif_d_n_xo(individual_list, reconstruct=True):
        # Choose at random the index of the donor (0 = p1, 1= p2)
        donor_idx = random.randint(0, n)
        donor = individual_list[donor_idx]
        recipients = individual_list[:donor_idx] + individual_list[donor_idx + 1:]

        # Choose at random a block from the donor
        if donor.size > 1:
            #still select a random block to remove from the donor (this is what i am not sure)
            donation_idx = random.randint(1, donor.size - 1)

            xo_recipients = list(crossover_for_recipient(donor, recipient, random.randint(1, donor.size - 1), reconstruct) for recipient in recipients)

            #donor
            donor_offs = Individual(collection=donor.collection[:donation_idx] + donor.collection[donation_idx + 1:]
            if reconstruct else None,
                                    train_semantics=torch.stack(
                                        [*donor.train_semantics[:donation_idx],
                                         *donor.train_semantics[donation_idx + 1:]]),
                                    test_semantics=torch.stack(
                                        [*donor.test_semantics[:donation_idx],
                                         *donor.test_semantics[donation_idx + 1:]])
                                    if donor.test_semantics is not None
                                    else None,
                                    reconstruct=reconstruct)

            donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[
                                                                                  donation_idx + 1:]
            donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[
                                                                                  donation_idx + 1:]
            donor_offs.size = donor.size - 1
            donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)
            donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(donor_offs.depth_collection)]
                                   ) + (donor_offs.size - 1)
            # Return the donor and the crossover recipients
            xo_recipients = [donor_offs] + xo_recipients
            return *xo_recipients,

        else:
            return *individual_list,

    return dif_d_n_xo
def dif_best_donor_n_xo(n=5, measure='min_fitness'):
    """
    Generate a dif_best_donor_n_xo function.

    Parameters
    ----------
    n: int
        Number of individuals to be used as recievers in crossover (1 donor, n recievers).
    measure: string
        Measure to be considered when selecting the 'best' individual to be donor.

    Returns
    -------
    Callable
        A dif_best_donor_n_xo crossover function (`dif_best_d_n_xo`).

        Performs crossover between n+1 individuals by donating random a block of the donor to the n individuals in the reciever_list.
        The donor is selected based on which individual is the 'best' in terms of the selected measure.


        Parameters
        ----------
        individual_list: list
            A list of Trees with n individuals to be used on donor_n_gxo.
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after crossover (default: True).

        Returns
        -------
        Individuals
            The best_donor_n_xo crossover tree n Individuals.
    """
    def crossover_for_recipient(donor, recipient, donation_idx, reconstruct):
        """Helper function to apply crossover on a single recipient."""
        recipient_offs = Individual(
            collection=[*recipient.collection, donor[donation_idx]] if reconstruct else None,
            train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
            test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
            if donor.test_semantics is not None else None,
            reconstruct=reconstruct
        )

        recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
        recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
        recipient_offs.size = recipient.size + 1
        recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
        recipient_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(recipient_offs.depth_collection)]) + (
                                           recipient_offs.size - 1)

        return recipient_offs
    def dif_best_d_n_xo(individual_list, reconstruct=True):

        _ = n  # dummy variable so n is identifiable

        # choose best individual to be donor:
        if measure == 'biggest':
            donor_idx = np.argmax([ind.size for ind in individual_list])
        elif measure == 'min_fitness':
            donor_idx = np.argmin([ind.fitness for ind in individual_list])
        elif measure == 'max_fitness':
            donor_idx = np.argmax([ind.fitness for ind in individual_list])

        donor = individual_list[donor_idx]
        recipients = individual_list[:donor_idx] + individual_list[donor_idx + 1:]

        # choose at random a block from the donor
        if donor.size > 1:
            donation_idx = random.randint(1, donor.size - 1)

            xo_recipients = list(
                crossover_for_recipient(donor, recipient, random.randint(1, donor.size - 1), reconstruct) for recipient in recipients)

            donor_offs = Individual(collection=donor.collection[:donation_idx] + donor.collection[donation_idx + 1:]
            if reconstruct else None,
                                    train_semantics=torch.stack(
                                        [*donor.train_semantics[:donation_idx],
                                         *donor.train_semantics[donation_idx + 1:]]),
                                    test_semantics=torch.stack(
                                        [*donor.test_semantics[:donation_idx],
                                         *donor.test_semantics[donation_idx + 1:]])
                                    if donor.test_semantics is not None
                                    else None,
                                    reconstruct=reconstruct)

            donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[
                                                                                  donation_idx + 1:]
            donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[
                                                                                  donation_idx + 1:]
            donor_offs.size = donor.size - 1
            donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)
            donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(donor_offs.depth_collection)]
                                   ) + (donor_offs.size - 1)

            #xo_recipients += (donor_offs,)
            #xo_recipients = [donor_offs] + xo_recipients[donor_idx:] what was before
            xo_recipients = [donor_offs] + xo_recipients
            return *xo_recipients,

        else:
            return *individual_list,

    return dif_best_d_n_xo
def improved_donor_xo(type = "fitness"):
    """
      Generate a improved_donor_xo function.

      Parameters
      ----------
      type: string
          Measure to be considered when selecting the 'best' and 'worst' index. Default is 'fitness'.

      Returns
      -------
      Callable
          A improved_donor_xo crossover function (`imp_donor_xo`).

          Performs crossover between 2 individuals by donating a block of the donor to the reciever individual and removing a block from the donor.
          The donor is selected randomly and the 'best' and 'worst' blocks from it are selected based on the fitness of the donor with the block removed (original-without_block).
          The 'best' block is the one that leads to the worst without_block fitness (meaning it is the most impactful one).
          The 'worst' block is the one that leads to the best without_block fitness (meaning it is the least impactul/ the one with the worst impact).
          The 'best' block is donated to the reciever and the 'worst' block is removed from the donor.

          Parameters
          ----------
          individual1 : Tree
            The first individual.
          individual2: Tree
            The second individual.
          ffunction_: function
            The fitness function to be considered when selecting the blocks (usually the fitness used so assess slim)
          y_train: array-like
            Training output data for fitness to be calculated with.
          operator_: {'sum', 'prod'}
            Operator to apply to the semantics, either "sum" or "prod".
          reconstruct : bool, optional
              Whether to reconstruct the Individual's collection after crossover (default: True).

          Returns
          -------
          Individuals
              The improved_donor_n_xo crossover tree Individuals.
      """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def block_selector_fit(donor, ffunction_, y_train, operator_): #done based on _evalute_slim_individual from utils
        """Function that selects the best and worst block indexes"""
        operator = torch.sum if operator_ == "sum" else torch.prod
        num_blocks = donor.size

        # Move tensors to GPU
        y_train = y_train.to(device)
        baseline_ = donor.fitness

        # Compute all block modifications in parallel
        modified_trains = [operator(torch.cat((donor.train_semantics[:i], donor.train_semantics[i + 1:])), dim=0)
                           for i in range(1, num_blocks)]

        modified_trains = torch.stack(modified_trains)
        modified_trains = torch.clamp(modified_trains, -1e12, 1e12)

        # Compute all fitness values at once
        train_fitnesses = ffunction_(y_train, modified_trains)
        # Compute coefficients
        coefs = (baseline_ - train_fitnesses)

        # Get best and worst indices
        #worst
        worst_idx = torch.argmax(coefs) + 1 #+1 because index 0 is not used so we start conting from 1
        #best
        best_idx = torch.argmin(coefs) + 1

        return worst_idx.item(), best_idx.item()

    def imp_donor_xo(individual1, individual2, ffunction_, y_train, operator_, reconstruct=True):
        donor_idx = random.randint(0, 1)
        donor, recipient = (individual1, individual2) if donor_idx == 0 else (individual2, individual1)

        if donor.size > 1:
            if type == "fitness":
                remove_idx, donation_idx = block_selector_fit(donor, ffunction_, y_train, operator_)

            recipient_offs = Individual(
                collection=[*recipient.collection, donor.collection[donation_idx]] if reconstruct else None,
                train_semantics=torch.cat((recipient.train_semantics, donor.train_semantics[donation_idx].unsqueeze(0))),
                test_semantics=torch.cat((recipient.test_semantics, donor.test_semantics[donation_idx].unsqueeze(0)))
                if donor.test_semantics is not None else None,
                reconstruct=reconstruct
            )

            donor_offs = Individual(
                collection=donor.collection[:remove_idx] + donor.collection[remove_idx + 1:] if reconstruct else None,
                train_semantics=torch.cat((donor.train_semantics[:remove_idx], donor.train_semantics[remove_idx + 1:])),
                test_semantics=torch.cat((donor.test_semantics[:remove_idx], donor.test_semantics[remove_idx + 1:]))
                if donor.test_semantics is not None else None,
                reconstruct=reconstruct
            )

            # Efficient attribute updates
            recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
            donor_offs.nodes_collection = donor.nodes_collection[:remove_idx] + donor.nodes_collection[remove_idx + 1:]

            recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
            donor_offs.depth_collection = donor.depth_collection[:remove_idx] + donor.depth_collection[remove_idx + 1:]

            recipient_offs.size, donor_offs.size = recipient.size + 1, donor.size - 1
            recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
            donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)

            recipient_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                        for i, depth in enumerate(recipient_offs.depth_collection)]
                                       ) + (recipient_offs.size - 1)

            donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(donor_offs.depth_collection)]
                                   ) + (donor_offs.size - 1)

            return (donor_offs, recipient_offs) if donor_idx == 0 else (recipient_offs, donor_offs)

        return individual1, individual2
    return imp_donor_xo
def best_improved_donor_xo(type = "fitness",measure = 'min_fitness' ): #parallelization imporved nothing, just amde it slower
    """
          Generate a best_improved_donor_xo function.

          Parameters
          ----------
          type: string
              Measure to be considered when selecting the 'best' and 'worst' index. Default is 'fitness'.
          measure: string
              Measure to be considered when selecting the 'best' individual to be donor.

          Returns
          -------
          Callable
              A best_improved_donor_xo crossover function (`best_imp_donor_xo`).

              Performs crossover between 2 individuals by donating a block of the donor to the reciever individual and removing a block from the donor.
              The donor is selected based on 'measure' and the 'best' and 'worst' blocks from it are selected based on the fitness of the donor with the block removed (original-without_block).
              The 'best' block is the one that leads to the worst without_block fitness (meaning it is the most impactful one).
              The 'worst' block is the one that leads to the best without_block fitness (meaning it is the least impactul/ the one with the worst impact).
              The 'best' block is donated to the reciever and the 'worst' block is removed from the donor.

              Parameters
              ----------
              individual1 : Tree
                The first individual.
              individual2: Tree
                The second individual.
              ffunction_: function
                The fitness function to be considered when selecting the blocks (usually the fitness used so assess slim)
              y_train: array-like
                Training output data for fitness to be calculated with.
              operator_: {'sum', 'prod'}
                Operator to apply to the semantics, either "sum" or "prod".
              reconstruct : bool, optional
                  Whether to reconstruct the Individual's collection after crossover (default: True).

              Returns
              -------
              Individuals
                  The improved_donor_n_xo crossover tree Individuals.
          """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def block_selector_fit(donor, ffunction_, y_train, operator_): #done using _evalute_slim_individual from utils
        """Function that selects the best and worst block indexes"""
        operator = torch.sum if operator_ == "sum" else torch.prod
        num_blocks = donor.size

        # Move tensors to GPU
        y_train = y_train.to(device)
        baseline_ = donor.fitness

        # Compute all block modifications in parallel
        modified_trains = [operator(torch.cat((donor.train_semantics[:i], donor.train_semantics[i + 1:])), dim=0)
                           for i in range(1, num_blocks)]

        modified_trains = torch.stack(modified_trains)
        modified_trains = torch.clamp(modified_trains, -1e12, 1e12)

        # Compute all fitness values at once
        train_fitnesses = ffunction_(y_train, modified_trains)
        # Compute coefficients
        coefs = (baseline_ - train_fitnesses)

        # Get best and worst indices

        #worst
        worst_idx = torch.argmax(coefs) + 1 #+1 because index 0 is not used in function so we start conting from 1
        #best
        best_idx = torch.argmin(coefs) + 1

        return worst_idx.item(), best_idx.item()

    def best_imp_donor_xo(individual1, individual2, ffunction_, y_train, operator_, reconstruct=True):
        #donor_idx = random.randint(0, 1)
        #donor, recipient = (individual1, individual2) if donor_idx == 0 else (individual2, individual1)

        # choose the best individual
        if measure == 'biggest':
            donor_idx = np.argmax([ind.size for ind in [individual1, individual2]])
        elif measure == 'min_fitness':
            donor_idx = np.argmin([ind.fitness for ind in [individual1, individual2]])
        elif measure == 'max_fitness':
            donor_idx = np.argmax([ind.fitness for ind in [individual1, individual2]])

        donor, recipient = (individual1, individual2) if donor_idx == 0 else (individual2, individual1)

        if donor.size > 1:
            if type == "fitness":
                remove_idx, donation_idx = block_selector_fit(donor, ffunction_, y_train, operator_)

            recipient_offs = Individual(
                collection=[*recipient.collection, donor.collection[donation_idx]] if reconstruct else None,
                train_semantics=torch.cat((recipient.train_semantics, donor.train_semantics[donation_idx].unsqueeze(0))),
                test_semantics=torch.cat((recipient.test_semantics, donor.test_semantics[donation_idx].unsqueeze(0)))
                if donor.test_semantics is not None else None,
                reconstruct=reconstruct
            )

            donor_offs = Individual(
                collection=donor.collection[:remove_idx] + donor.collection[remove_idx + 1:] if reconstruct else None,
                train_semantics=torch.cat((donor.train_semantics[:remove_idx], donor.train_semantics[remove_idx + 1:])),
                test_semantics=torch.cat((donor.test_semantics[:remove_idx], donor.test_semantics[remove_idx + 1:]))
                if donor.test_semantics is not None else None,
                reconstruct=reconstruct
            )

            # Efficient attribute updates
            recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
            donor_offs.nodes_collection = donor.nodes_collection[:remove_idx] + donor.nodes_collection[remove_idx + 1:]

            recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
            donor_offs.depth_collection = donor.depth_collection[:remove_idx] + donor.depth_collection[remove_idx + 1:]

            recipient_offs.size, donor_offs.size = recipient.size + 1, donor.size - 1
            recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
            donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)

            recipient_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                        for i, depth in enumerate(recipient_offs.depth_collection)]
                                       ) + (recipient_offs.size - 1)

            donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(donor_offs.depth_collection)]
                                   ) + (donor_offs.size - 1)

            return (donor_offs, recipient_offs) if donor_idx == 0 else (recipient_offs, donor_offs)

        return individual1, individual2
    return best_imp_donor_xo
def best_block_n_xo(n=5):
    """
          Generate a best_block_n_xo function.

          Parameters
          ----------
          n: int
            Number of individuals to be used as recievers in crossover (1 donor, n receivers).

          Returns
          -------
          Callable
              A best_block_n_xo crossover function (`best_block_n`).

              Performs crossover between n+1 individuals by donating a block of the donor to the reciever individuals and removing a block from all individuals.
              The donor is selected as the individual with the 'best' block, with 'best' and 'worst' blocks selected based on the fitness of the individual with the block removed (original-without_block).
              The 'best' block is the one that leads to the worst without_block fitness (meaning it is the most impactful one).
              The 'worst' block is the one that leads to the best without_block fitness (meaning it is the least impactul/ the one with the worst impact).
              The 'best' overall block is donated to the n recievers.
              All individuals have their 'worst' block removed, only if this block leads to an imporvement in fitness (original-without_block is positive).

              Parameters
              ----------
              individual_list: list
                 A list of Trees with n+1 individuals to be used on donor_n_xo.
              ffunction_: function
                The fitness function to be considered when selecting the blocks (usually the fitness used so assess slim)
              y_train: array-like
                Training output data for fitness to be calculated with.
              operator_: {'sum', 'prod'}
                Operator to apply to the semantics, either "sum" or "prod".
              reconstruct : bool, optional
                  Whether to reconstruct the Individual's collection after crossover (default: True).

              Returns
              -------
              Individuals
                  The best_improved_donor_n_xo crossover tree Individuals.
          """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def block_selector_fit(individual,ffunction_, y_train, operator_):
        """Function that selects the 'best' and 'worst' blocks to be used"""

        operator = torch.sum if operator_ == "sum" else torch.prod
        num_blocks = individual.size
        baseline_ = individual.fitness

        train_semantics = individual.train_semantics.to(device)
        modified_trains = torch.stack([
            operator(torch.cat((train_semantics[:i], train_semantics[i + 1:])), dim=0)
            for i in range(1, num_blocks)
        ]) if num_blocks > 1 else torch.tensor([])
        modified_trains = torch.clamp(modified_trains, -1e12, 1e12) if num_blocks > 1 else modified_trains

        #calculate difference for individual blocks
        train_fitnesses = ffunction_(y_train, modified_trains) if num_blocks > 1 else torch.tensor([baseline_])
        coefs = baseline_ - train_fitnesses

        #select the best and worst ones
        worst_idx = (torch.argmax(coefs)+1).item() if num_blocks > 1 else 0
        best_idx = (torch.argmin(coefs)+1).item() if num_blocks > 1 else 0


        #you give back the coef difference, the biggest positive one is the one that is chosen because it represents the block that is most imporatant
        return worst_idx, best_idx, coefs.max().item() if num_blocks > 1 else 0

    def update_individual(individual, remove_idx, donor=None, best_block = None, reconstruct = True):
        if donor is not None:
            new_ind = Individual(
                collection=[*individual.collection, donor.collection[best_block]] if reconstruct else None,
                train_semantics=torch.cat(
                    (individual.train_semantics, donor.train_semantics[best_block].unsqueeze(0))),
                test_semantics=torch.cat((individual.test_semantics, donor.test_semantics[best_block].unsqueeze(0)))
                if donor.test_semantics is not None else None,
                reconstruct=reconstruct
            )

            new_ind.nodes_collection = individual.nodes_collection + [donor.nodes_collection[best_block]]
            new_ind.depth_collection = individual.depth_collection + [donor.depth_collection[best_block]]
            new_ind.size = individual.size + 1
            new_ind.nodes_count = sum(new_ind.nodes_collection) + (new_ind.size - 1)
            new_ind.depth = max([depth - (i - 1) if i != 0 else depth
                                        for i, depth in enumerate(individual.depth_collection)]) + (
                                           individual.size - 1)

        if individual.size > 1:

            new_ind = Individual(
                collection=individual.collection[:remove_idx] + individual.collection[remove_idx + 1:] if reconstruct else None,
                train_semantics=torch.cat((individual.train_semantics[:remove_idx], individual.train_semantics[remove_idx + 1:])),
                test_semantics=torch.cat((individual.test_semantics[:remove_idx], individual.test_semantics[remove_idx + 1:]))
                if individual.test_semantics is not None else None,
                reconstruct=reconstruct
            )

            new_ind.nodes_collection = individual.nodes_collection[:remove_idx] + individual.nodes_collection[remove_idx + 1:]
            new_ind.depth_collection = individual.depth_collection[:remove_idx] + individual.depth_collection[remove_idx + 1:]
            new_ind.size = individual.size - 1
            new_ind.nodes_count = sum(new_ind.nodes_collection) + (new_ind.size - 1)
            new_ind.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(new_ind.depth_collection)]
                                   ) + (new_ind.size - 1)

            return new_ind
        else:
            return individual
    def best_block_n(individual_list, ffunction_, y_train, operator_, reconstruct=True):
        _ = n
        y_train = y_train.to(device)

        #necessary variables
        worst_blocks = {}
        best_blocks = {}
        coefs = []
        best_fitness = float("-inf")
        best_donor_idx = None
        best_block_idx = None

        #iterate through all individuals to select donor (best overall block) and the 'worst' block from each individual.
        for idx, ind in enumerate(individual_list):
            worst_idx, best_idx, ind_best_fitness = block_selector_fit(ind, ffunction_, y_train, operator_)

            worst_blocks[idx] = worst_idx
            best_blocks[idx] = best_idx
            coefs.append(ind_best_fitness)

            #if the current individual, has a positive coeficient that is bigger than the previous,it means the block is better
            #We select only from positive coeficients so no 'bad' blocks are passed
            if 0 <= ind_best_fitness > best_fitness:
                best_fitness = ind_best_fitness
                best_donor_idx = idx
                best_block_idx = best_idx

        # If no improvement is found for any individual (all values negative), select the least negative change
        #This occurs mostly when few individuals are considered for the crossover (n=5)
        if best_fitness == float("-inf"): #if best fitness was not changed
            #select the donor as teh individual with 'best' out of the 'worst' blocks
            best_donor_idx = torch.argmax(torch.tensor(coefs)).item()
            #select the best block from donor
            best_block_idx = best_blocks[best_donor_idx]
            #not all crossovers lead to a positive change, but we decrease probability of negative ones


        #select donor and recievers
        donor = individual_list[best_donor_idx]
        recipients = [ind for i, ind in enumerate(individual_list) if i != best_donor_idx]

        #add 'best' block to recievers
        recipients = [update_individual(recipient, -1, donor, best_block_idx) for i, recipient in
                      enumerate(recipients) ]

        #add everything back together
        individual_list = (recipients[:best_donor_idx]+ [donor]+ recipients[best_donor_idx:])
        #remove the 'worst' block when it leads to a positive change
        individual_list = [
        update_individual(ind, worst_blocks[i]) if coefs[i] > 0 else ind
        for i, ind in enumerate(individual_list)
                            ]

        return *individual_list,

    return best_block_n
