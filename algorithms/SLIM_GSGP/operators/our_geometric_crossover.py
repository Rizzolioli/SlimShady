import random
import torch
from algorithms.GSGP.representations.tree import Tree
from algorithms.SLIM_GSGP.representations.individual import Individual


def generate_mask(n, k):
    my_mask = [0] * n
    positions = set()
    while len(positions) < k-1:
        positions.add(random.randint(1, n-1))
    for pos in positions: 
        my_mask[pos] = 1
    #the first block needs to always be saved(root block)
    my_mask[0] = 1
    return my_mask

def std_xo_alpha_delta(operator='sum'):

    def stdxo_a_delta(p1, p2, alpha, testing):

        if testing:
            return torch.add(torch.mul(p1.test_semantics, alpha),
                             torch.mul(torch.sub(1, alpha), p2.test_semantics)) if operator == 'sum' else \
                    torch.mul(torch.pow(p1.test_semantics, alpha),
                              torch.pow(p2.test_semantics, torch.sub(1,  alpha)))

        else:
            return torch.add(torch.mul(p1.train_semantics, alpha),
                             torch.mul(torch.sub(1, alpha), p2.train_semantics)) if operator == 'sum' else \
                torch.mul(torch.pow(p1.train_semantics, alpha),
                          torch.pow(p2.train_semantics, torch.sub(1, alpha)))


    return stdxo_a_delta

def std_xo_alpha_ot_delta(which, operator='sum'):

    def stdxo_ot_a_delta(p, alpha, testing):

        if which == 'first':

            if testing:
                return torch.mul(p.test_semantics, alpha) if operator == 'sum' else \
                    torch.pow(p.test_semantics, alpha)

            else:
                return torch.mul(p.train_semantics, alpha) if operator == 'sum' else \
                    torch.pow(p.train_semantics, alpha)
        else:

            if testing:
                return torch.mul(p.test_semantics, torch.sub(1, alpha)) if operator == 'sum' else \
                    torch.pow(p.test_semantics, torch.sub(1, alpha))

            else:
                return torch.mul(p.train_semantics, torch.sub(1, alpha)) if operator == 'sum' else \
                    torch.pow(p.train_semantics, torch.sub(1, alpha))

    stdxo_ot_a_delta.__name__ += '_' + which


    return stdxo_ot_a_delta


def slim_alpha_geometric_crossover(operator):

    def inner_xo(p1, p2, X, X_test = None):


        offs = [Tree([std_xo_alpha_delta(operator=operator),
                      p1.collection[i], p2.collection[i],
                      random.random()]) for i in range(min(p1.size, p2.size))]

        if p1.size > p2.size:

            which = 'first'

            offs += [Tree([std_xo_alpha_ot_delta(which, operator=operator),
                          p1.collection[i], random.random()]) for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]

        else:

            which = 'second'

            offs += [Tree([std_xo_alpha_ot_delta(which, operator=operator),
                           p2.collection[i], random.random()]) for i in range(min(p1.size, p2.size), max(p1.size, p2.size))]


        offs = Individual(offs)
        offs.calculate_semantics(X)
        if X_test is not None:
            offs.calculate_semantics(X_test, testing=True)

        return offs

    return inner_xo



def slim_swap_geometric_crossover(p1, p2, X = None, X_test = None):

    mask = [random.randint(0,1) for _ in range(max(p1.size, p2.size))]
    inv_mask = [abs(v-1) for v in mask]

    off1 = [
        p1.collection[idx] if mask[idx] == 0 and idx < p1.size
        else p2.collection[idx]
        for idx in range(len(mask))
        if (mask[idx] == 0 and idx < p1.size) or (mask[idx] == 1 and idx < p2.size)
    ]

    off1 = Individual(off1)
    off1.train_semantics = [
        p1.train_semantics[idx] if mask[idx] == 0 and idx < p1.size
        else p2.train_semantics[idx]
        for idx in range(len(mask))
        if (mask[idx] == 0 and idx < p1.size) or (mask[idx] == 1 and idx < p2.size)
    ]
    if p1.test_semantics is not None:
        off1.test_semantics = [
            p1.test_semantics[idx] if mask[idx] == 0 and idx < p1.size
            else p2.test_semantics[idx]
            for idx in range(len(mask))
            if (mask[idx] == 0 and idx < p1.size) or (mask[idx] == 1 and idx < p2.size)
        ]

    off2 = [
        p1.collection[idx] if inv_mask[idx] == 0 and idx < p1.size
        else p2.collection[idx]
        for idx in range(len(inv_mask))
        if (inv_mask[idx] == 0 and idx < p1.size) or (inv_mask[idx] == 1 and idx < p2.size)
    ]

    off2 = Individual(off2)
    off2.train_semantics = [
        p1.train_semantics[idx] if inv_mask[idx] == 0 and idx < p1.size
        else p2.train_semantics[idx]
        for idx in range(len(inv_mask))
        if (inv_mask[idx] == 0 and idx < p1.size) or (inv_mask[idx] == 1 and idx < p2.size)
    ]
    if p1.test_semantics is not None:
        off2.test_semantics = [
            p1.test_semantics[idx] if inv_mask[idx] == 0 and idx < p1.size
            else p2.test_semantics[idx]
            for idx in range(len(inv_mask))
            if (inv_mask[idx] == 0 and idx < p1.size) or (inv_mask[idx] == 1 and idx < p2.size)
        ]

    return off1, off2

def slim_alpha_deflate_geometric_crossover(operator, perc_off_blocks):

    def inner_xo(p1, p2, X, X_test = None):

        mask = generate_mask(max(p1.size, p2.size), int(perc_off_blocks * max(p1.size, p2.size)))


        offs = [Tree([std_xo_alpha_delta(operator=operator),
                      p1.collection[i], p2.collection[i],
                      random.random()]) for i in range(min(p1.size, p2.size)) if mask[i] == 1]

        if p1.size > p2.size:

            which = 'first'

            offs += [Tree([std_xo_alpha_ot_delta(which, operator=operator),
                          p1.collection[i], random.random()])
                     for i in range(min(p1.size, p2.size), max(p1.size, p2.size)) if mask[i] == 1]

        else:

            which = 'second'

            offs += [Tree([std_xo_alpha_ot_delta(which, operator=operator),
                           p2.collection[i], random.random()])
                     for i in range(min(p1.size, p2.size), max(p1.size, p2.size)) if mask[i] == 1]


        offs = Individual(offs)
        offs.calculate_semantics(X)
        if X_test is not None:
            offs.calculate_semantics(X_test, testing=True)

        return offs

    return inner_xo


def slim_swap_deflate_geometric_crossover(perc_off_blocks):
    
    def inner_ssd_gxo(p1, p2, X = None, X_test = None):

        mask_selection = generate_mask(max(p1.size, p2.size), int(perc_off_blocks * max(p1.size, p2.size)))
        
        mask_parents = [random.randint(0,1) for _ in range(max(p1.size, p2.size))]
        inv_mask_parents = [abs(v-1) for v in mask_parents]

        print(mask_selection)
        print(mask_parents)
    
        off1 = [
            p1.collection[idx] if mask_parents[idx] == 0 and idx < p1.size
            else p2.collection[idx]
            for idx in range(len(mask_parents))
            if ((mask_parents[idx] == 0 and idx < p1.size) or (mask_parents[idx] == 1 and idx < p2.size)) and mask_selection[idx] == 1
        ]
    
        off1 = Individual(off1)
        off1.train_semantics = [
            p1.train_semantics[idx] if mask_parents[idx] == 0 and idx < p1.size
            else p2.train_semantics[idx]
            for idx in range(len(mask_parents))
            if ((mask_parents[idx] == 0 and idx < p1.size) or (mask_parents[idx] == 1 and idx < p2.size)) and mask_selection[idx] == 1
        ]
        if p1.test_semantics is not None:
            off1.test_semantics = [
                p1.test_semantics[idx] if mask_parents[idx] == 0 and idx < p1.size
                else p2.test_semantics[idx]
                for idx in range(len(mask_parents))
                if ((mask_parents[idx] == 0 and idx < p1.size) or (mask_parents[idx] == 1 and idx < p2.size)) and mask_selection[idx] == 1
            ]
    
        off2 = [
            p1.collection[idx] if inv_mask_parents[idx] == 0 and idx < p1.size
            else p2.collection[idx]
            for idx in range(len(inv_mask_parents))
            if ((inv_mask_parents[idx] == 0 and idx < p1.size) or (inv_mask_parents[idx] == 1 and idx < p2.size)) and mask_selection[idx] == 1
        ]
    
        off2 = Individual(off2)
        off2.train_semantics = [
            p1.train_semantics[idx] if inv_mask_parents[idx] == 0 and idx < p1.size
            else p2.train_semantics[idx]
            for idx in range(len(inv_mask_parents))
            if ((inv_mask_parents[idx] == 0 and idx < p1.size) or (inv_mask_parents[idx] == 1 and idx < p2.size)) and mask_selection[idx] == 1
        ]
        if p1.test_semantics is not None:
            off2.test_semantics = [
                p1.test_semantics[idx] if inv_mask_parents[idx] == 0 and idx < p1.size
                else p2.test_semantics[idx]
                for idx in range(len(inv_mask_parents))
                if ((inv_mask_parents[idx] == 0 and idx < p1.size) or (inv_mask_parents[idx] == 1 and idx < p2.size)) and mask_selection[idx] == 1
            ]
    
        return off1, off2
    
    return inner_ssd_gxo


