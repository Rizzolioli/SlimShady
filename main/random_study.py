from algorithms.GP.representations.tree_utils import create_grow_random_tree
from parametrization import FUNCTIONS, generate_random_uniform
from random_utils import *
from algorithms.GP.representations.tree_utils import flatten

t1 = create_grow_random_tree(depth = 6,
                             FUNCTIONS = FUNCTIONS,
                             TERMINALS = {f"x{i}": i for i in range(20)},
                             CONSTANTS = None,
                             p_c = 0)
t2 = create_grow_random_tree(depth = 6,
                             FUNCTIONS = FUNCTIONS,
                             TERMINALS = {f"x{i}": i for i in range(20)},
                             CONSTANTS = None,
                             p_c = 0)

print(t1)
print(t2)

ms = generate_random_uniform(0,1)()

m1 = mutate_tree(t1, mutation_type='sigmoid', ms=ms)
m2 = mutate_tree(t1, mutation_type='abs', ms=ms)
m3 = mutate_tree(t1, mutation_type='sigmoid2', ms=ms, t2=t2)

print("Mutation 1 (sigmoid):", m1)
print("Mutation 2 (abs):", m2)
print("Mutation 3 (sigmoid2):", m3)

# Create sympy symbols
x_symbols = {f'x{i}': sp.Symbol(f'x{i}') for i in range(100)}

s_m1 = tree_to_sympy(m1, x_symbols)
ss_m1 = sp.simplify(s_m1)
print(ss_m1)
ss_m1 = ss_m1.rewrite(sp.exp)
print(ss_m1)
print(len(list(flatten(sympy_to_tree(ss_m1)))))
print(sympy_to_tree(ss_m1))
print(count_nodes(ss_m1))

variables = list(x_symbols.values())
data = np.random.uniform(-2, 2, size=(100, 100))

print("Kolmogorov Complexity:", kolmogorov_complexity(ss_m1))
print("SFC:", slope_complexity(ss_m1, variables, data))
