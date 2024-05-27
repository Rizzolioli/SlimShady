# SLIM (Semantic Learning algorithm based on Inflate and deflate Mutation)

SLIM is a Python library that implements the SLIM algorithm, which is a variant of the Geometric Semantic Genetic Programming (GSGP). This library includes functions for running standard Genetic Programming (GP), GSGP, and all developed versions of the SLIM algorithm. Users can specify the version of SLIM they wish to use and obtain results accordingly.

## Installation

To install the library, use the following command:
```sh
pip install slim
```
## Usage
### Running GP 
To use the GP algorithm, you can use the following example:

```python
from slim.main_gp import gp

datasets = ["toxicity"]
n_runs = 1
pop_size = 10
n_iter = 100
p_xo = 0.8

gp(datasets=datasets, n_runs=n_runs, pop_size=pop_size, n_iter=n_iter, p_xo=p_xo)
```
The arguments for the **gp** function are: 
* `datasets`: A list of strings specifying the datasets to be used. The user can specify one or more datasets. The dataset which can be tested are: Toxicity, Istanbul and Energy 
* `n_runs`: An integer specifying the number of runs for the algorithm.
* `pop_size`: An integer specifying the population size.
* `n_iter`: An integer specifying the number of iterations.
* `p_xo`: A float specifying the crossover probability.