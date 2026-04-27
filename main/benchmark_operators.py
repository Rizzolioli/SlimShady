"""
Benchmark script to compare runtime performance of genetic operators:
- Inflate mutation
- Deflate mutation
- Donor crossover
- Best donor crossover

The operators are applied to randomly generated individuals outside the evolution process.
Statistical comparison is performed using the Wilcoxon signed-rank test.

Usage:
    Run from the main directory:
    python benchmark_operators.py
"""

import time
import numpy as np
import torch
from scipy.stats import wilcoxon
import pandas as pd
from algorithms.GP.operators.initializers import rhh
from algorithms.GSGP.representations.tree import Tree
from algorithms.GP.representations.tree import Tree as GP_Tree
from algorithms.SLIM_GSGP.representations.individual import Individual
from algorithms.SLIM_GSGP.operators.mutators import inflate_mutation, deflate_mutation
from algorithms.SLIM_GSGP.operators.crossover_operators import donor_xo, best_donor_xo
from utils.utils import protected_div


# Define functions and constants
FUNCTIONS = {
    'add': {'function': lambda x, y: torch.add(x, y), 'arity': 2},
    'subtract': {'function': lambda x, y: torch.sub(x, y), 'arity': 2},
    'multiply': {'function': lambda x, y: torch.mul(x, y), 'arity': 2},
    'divide': {'function': lambda x, y: protected_div(x, y), 'arity': 2}
}

CONSTANTS = {
    'constant_2': lambda x: torch.tensor(2).float(),
    'constant_3': lambda x: torch.tensor(3).float(),
    'constant_4': lambda x: torch.tensor(4).float(),
    'constant_5': lambda x: torch.tensor(5).float(),
    'constant__1': lambda x: torch.tensor(-1).float()
}

# Number of features for synthetic data
N_FEATURES = 10
TERMINALS = {f'x_{i}': i for i in range(N_FEATURES)}


def create_random_individual(X, X_test=None, init_depth=6, p_c=0.1, reconstruct=True):
    """
    Create a random SLIM individual with a single tree block.

    Parameters
    ----------
    X : torch.Tensor
        Training data
    X_test : torch.Tensor, optional
        Testing data
    init_depth : int
        Initial depth for tree creation
    p_c : float
        Probability of choosing a constant
    reconstruct : bool
        Whether to reconstruct the individual

    Returns
    -------
    Individual
        A randomly generated SLIM individual
    """
    # Create a single random tree using rhh
    trees = rhh(init_pop_size=1, init_depth=init_depth,
                FUNCTIONS=FUNCTIONS, TERMINALS=TERMINALS,
                CONSTANTS=CONSTANTS, p_c=p_c)

    tree = trees[0]

    # Calculate semantics
    tree.calculate_semantics(X, testing=False)
    if X_test is not None:
        tree.calculate_semantics(X_test, testing=True)

    # Create Individual
    train_semantics = tree.train_semantics if tree.train_semantics.shape != torch.Size([]) \
                      else tree.train_semantics.repeat(len(X))

    test_semantics = None
    if X_test is not None:
        test_semantics = tree.test_semantics if tree.test_semantics.shape != torch.Size([]) \
                        else tree.test_semantics.repeat(len(X_test))

    individual = Individual(
        collection=[tree] if reconstruct else None,
        train_semantics=torch.stack([train_semantics]),
        test_semantics=torch.stack([test_semantics]) if test_semantics is not None else None,
        reconstruct=reconstruct
    )

    # Set fitness to enable best_donor_xo
    individual.fitness = np.random.random()

    return individual


def create_multi_block_individual(X, X_test=None, n_blocks=5, init_depth=6, p_c=0.1, reconstruct=True):
    """
    Create a random SLIM individual with multiple tree blocks.

    Parameters
    ----------
    X : torch.Tensor
        Training data
    X_test : torch.Tensor, optional
        Testing data
    n_blocks : int
        Number of blocks in the individual
    init_depth : int
        Initial depth for tree creation
    p_c : float
        Probability of choosing a constant
    reconstruct : bool
        Whether to reconstruct the individual

    Returns
    -------
    Individual
        A randomly generated SLIM individual with multiple blocks
    """
    # Set Tree class attributes for both GSGP Tree and GP Tree
    Tree.FUNCTIONS = FUNCTIONS
    Tree.TERMINALS = TERMINALS
    Tree.CONSTANTS = CONSTANTS

    GP_Tree.FUNCTIONS = FUNCTIONS
    GP_Tree.TERMINALS = TERMINALS
    GP_Tree.CONSTANTS = CONSTANTS

    # Create multiple random tree tuples from rhh
    tree_tuples = rhh(init_pop_size=n_blocks, init_depth=init_depth,
                      FUNCTIONS=FUNCTIONS, TERMINALS=TERMINALS,
                      CONSTANTS=CONSTANTS, p_c=p_c)

    # Wrap each tree tuple in a Tree object
    trees = [Tree(tree_tuple, train_semantics=None, test_semantics=None, reconstruct=True)
             for tree_tuple in tree_tuples]

    # Calculate semantics for all trees
    train_semantics_list = []
    test_semantics_list = []

    for tree in trees:
        tree.calculate_semantics(X, testing=False)
        train_sem = tree.train_semantics if tree.train_semantics.shape != torch.Size([]) \
                   else tree.train_semantics.repeat(len(X))
        train_semantics_list.append(train_sem)

        if X_test is not None:
            tree.calculate_semantics(X_test, testing=True)
            test_sem = tree.test_semantics if tree.test_semantics.shape != torch.Size([]) \
                      else tree.test_semantics.repeat(len(X_test))
            test_semantics_list.append(test_sem)

    # Create Individual
    individual = Individual(
        collection=trees if reconstruct else None,
        train_semantics=torch.stack(train_semantics_list),
        test_semantics=torch.stack(test_semantics_list) if X_test is not None else None,
        reconstruct=reconstruct
    )

    # Set fitness to enable best_donor_xo
    individual.fitness = np.random.random()

    return individual


def benchmark_inflate_mutation(n_blocks, X, X_test, n_runs=100):
    """
    Benchmark inflate mutation operator.

    Parameters
    ----------
    n_blocks : int
        Number of blocks in the individuals to create
    X : torch.Tensor
        Training data
    X_test : torch.Tensor
        Testing data
    n_runs : int
        Number of repetitions

    Returns
    -------
    list
        List of execution times
    """
    times = []

    # Create the inflate mutator
    inflate_mutator = inflate_mutation(
        FUNCTIONS=FUNCTIONS,
        TERMINALS=TERMINALS,
        CONSTANTS=CONSTANTS,
        two_trees=True,
        operator='sum',
        sig=False
    )

    for _ in range(n_runs):
        # Create a fresh random individual for each run
        ind = create_multi_block_individual(X, X_test, n_blocks=n_blocks)

        # Measure time
        start_time = time.perf_counter()
        ms = np.random.uniform(0, 1)
        _ = inflate_mutator(ind, ms, X, max_depth=8, p_c=0.1,
                           X_test=X_test, grow_probability=1, reconstruct=True)
        end_time = time.perf_counter()

        times.append(end_time - start_time)

    return times


def benchmark_deflate_mutation(n_blocks, X, X_test, n_runs=100):
    """
    Benchmark deflate mutation operator.

    Parameters
    ----------
    n_blocks : int
        Number of blocks in the individuals to create
    X : torch.Tensor
        Training data
    X_test : torch.Tensor
        Testing data
    n_runs : int
        Number of repetitions

    Returns
    -------
    list
        List of execution times
    """
    times = []

    for _ in range(n_runs):
        # Create a fresh random individual for each run (at least 3 blocks for deflate)
        ind = create_multi_block_individual(X, X_test, n_blocks=max(n_blocks, 3))

        # Measure time
        start_time = time.perf_counter()
        _ = deflate_mutation(ind, reconstruct=True)
        end_time = time.perf_counter()

        times.append(end_time - start_time)

    return times


def benchmark_donor_crossover(n_blocks, X, X_test, n_runs=100):
    """
    Benchmark donor crossover operator.

    Parameters
    ----------
    n_blocks : int
        Number of blocks in the individuals to create
    X : torch.Tensor
        Training data
    X_test : torch.Tensor
        Testing data
    n_runs : int
        Number of repetitions

    Returns
    -------
    list
        List of execution times
    """
    times = []

    for _ in range(n_runs):
        # Create fresh random individuals for each run
        ind1 = create_multi_block_individual(X, X_test, n_blocks=n_blocks)
        ind2 = create_multi_block_individual(X, X_test, n_blocks=n_blocks)

        # Measure time
        start_time = time.perf_counter()
        _ = donor_xo(ind1, ind2, reconstruct=True)
        end_time = time.perf_counter()

        times.append(end_time - start_time)

    return times


def benchmark_best_donor_crossover(n_blocks, X, X_test, n_runs=100):
    """
    Benchmark best donor crossover operator.

    Parameters
    ----------
    n_blocks : int
        Number of blocks in the individuals to create
    X : torch.Tensor
        Training data
    X_test : torch.Tensor
        Testing data
    n_runs : int
        Number of repetitions

    Returns
    -------
    list
        List of execution times
    """
    times = []

    # Create the best_donor_xo operator
    best_d_xo = best_donor_xo(measure='min_fitness')

    for _ in range(n_runs):
        # Create fresh random individuals for each run
        ind1 = create_multi_block_individual(X, X_test, n_blocks=n_blocks)
        ind2 = create_multi_block_individual(X, X_test, n_blocks=n_blocks)

        # Measure time
        start_time = time.perf_counter()
        _ = best_d_xo(ind1, ind2, reconstruct=True)
        end_time = time.perf_counter()

        times.append(end_time - start_time)

    return times


def wilcoxon_pairwise_comparison(results_dict):
    """
    Perform pairwise Wilcoxon signed-rank tests between all operators.

    Parameters
    ----------
    results_dict : dict
        Dictionary with operator names as keys and lists of times as values

    Returns
    -------
    pd.DataFrame
        DataFrame containing p-values for all pairwise comparisons
    """
    operators = list(results_dict.keys())
    n_operators = len(operators)

    # Create a matrix to store p-values
    p_value_matrix = np.zeros((n_operators, n_operators))

    for i, op1 in enumerate(operators):
        for j, op2 in enumerate(operators):
            if i == j:
                p_value_matrix[i, j] = 1.0  # Same operator
            elif i < j:
                # Perform Wilcoxon signed-rank test
                statistic, p_value = wilcoxon(results_dict[op1], results_dict[op2])
                p_value_matrix[i, j] = p_value
                p_value_matrix[j, i] = p_value

    # Create DataFrame
    df = pd.DataFrame(p_value_matrix, index=operators, columns=operators)

    return df


def print_statistics(results_dict):
    """
    Print summary statistics for each operator.

    Parameters
    ----------
    results_dict : dict
        Dictionary with operator names as keys and lists of times as values
    """
    print("\n" + "="*80)
    print("RUNTIME STATISTICS (in seconds)")
    print("="*80)

    stats_data = []
    for operator, times in results_dict.items():
        stats_data.append({
            'Operator': operator,
            'Mean': np.mean(times),
            'Median': np.median(times),
            'Std Dev': np.std(times),
            'Min': np.min(times),
            'Max': np.max(times)
        })

    df_stats = pd.DataFrame(stats_data)
    print(df_stats.to_string(index=False))
    print("="*80)


def run_benchmark(n_samples=100, n_features=10, n_runs=100, n_blocks=5):
    """
    Run the complete benchmark comparing all genetic operators.

    Parameters
    ----------
    n_samples : int
        Number of samples in synthetic data
    n_features : int
        Number of features in synthetic data
    n_runs : int
        Number of repetitions for each operator
    n_blocks : int
        Number of blocks in multi-block individuals
    """
    print("="*80)
    print("GENETIC OPERATORS RUNTIME BENCHMARK")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Number of samples: {n_samples}")
    print(f"  - Number of features: {n_features}")
    print(f"  - Number of runs per operator: {n_runs}")
    print(f"  - Number of blocks in individuals: {n_blocks}")
    print("="*80)

    # Create synthetic data
    print("\nGenerating synthetic data...")
    X_train = torch.randn(n_samples, n_features).float()
    X_test = torch.randn(n_samples // 5, n_features).float()

    # Dictionary to store results
    results = {}

    # Benchmark Inflate Mutation
    print("\n[1/4] Benchmarking Inflate Mutation...")
    print(f"  (Creating {n_runs} random individuals with {n_blocks} blocks each)")
    results['Inflate Mutation'] = benchmark_inflate_mutation(
        n_blocks, X_train, X_test, n_runs=n_runs
    )
    print(f"  Mean time: {np.mean(results['Inflate Mutation']):.6f} seconds")

    # Benchmark Deflate Mutation
    print("\n[2/4] Benchmarking Deflate Mutation...")
    print(f"  (Creating {n_runs} random individuals with {n_blocks} blocks each)")
    results['Deflate Mutation'] = benchmark_deflate_mutation(
        n_blocks, X_train, X_test, n_runs=n_runs
    )
    print(f"  Mean time: {np.mean(results['Deflate Mutation']):.6f} seconds")

    # Benchmark Donor Crossover
    print("\n[3/4] Benchmarking Donor Crossover...")
    print(f"  (Creating {n_runs * 2} random individuals with {n_blocks} blocks each)")
    results['Donor Crossover'] = benchmark_donor_crossover(
        n_blocks, X_train, X_test, n_runs=n_runs
    )
    print(f"  Mean time: {np.mean(results['Donor Crossover']):.6f} seconds")

    # Benchmark Best Donor Crossover
    print("\n[4/4] Benchmarking Best Donor Crossover...")
    print(f"  (Creating {n_runs * 2} random individuals with {n_blocks} blocks each)")
    results['Best Donor Crossover'] = benchmark_best_donor_crossover(
        n_blocks, X_train, X_test, n_runs=n_runs
    )
    print(f"  Mean time: {np.mean(results['Best Donor Crossover']):.6f} seconds")

    # Print statistics
    print_statistics(results)

    # Perform Wilcoxon tests
    print("\n" + "="*80)
    print("WILCOXON SIGNED-RANK TEST P-VALUES")
    print("="*80)
    print("(Testing if runtime distributions are significantly different)")
    print()

    p_value_df = wilcoxon_pairwise_comparison(results)
    print(p_value_df.to_string())

    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("  - p < 0.001: *** (highly significant difference)")
    print("  - p < 0.01:  **  (very significant difference)")
    print("  - p < 0.05:  *   (significant difference)")
    print("  - p >= 0.05:     (no significant difference)")
    print("="*80)

    # Annotate p-values
    print("\nAnnotated p-values:")
    for i, op1 in enumerate(p_value_df.index):
        for j, op2 in enumerate(p_value_df.columns):
            if i < j:  # Only print upper triangle
                p_val = p_value_df.iloc[i, j]
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = ""

                print(f"  {op1} vs {op2}: p={p_val:.6f} {sig}")

    return results, p_value_df


if __name__ == "__main__":
    # Run the benchmark with default parameters
    results, p_values = run_benchmark(
        n_samples=100,
        n_features=10,
        n_runs=100,
        n_blocks=5
    )

    print("\n" + "="*80)
    print("BENCHMARK COMPLETED")
    print("="*80)