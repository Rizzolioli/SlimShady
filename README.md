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

gp(datasets=datasets, n_runs=n_runs, pop_size=pop_size, n_iter=n_iter)
```
The arguments for the **gp** function are: 
* `datasets`: A list of strings specifying the datasets to be used. The user can specify one or more datasets
  * The dataset which can be tested are: _resid_build_sale_price, concrete, toxicity, istanbul, energy, ppb_
* `n_runs`: An integer specifying the number of runs for the algorithm. 
  * default: 30
* `pop_size`: An integer specifying the population size.
  * default: 100
* `n_iter`: An integer specifying the number of iterations.
  * default: 1000
* `p_xo`: A float specifying the crossover probability.
  * default: 0.8
* `elitism` : A boolean specifying the presence of elitism. 
  * default: True
* `n_elites` : An integer specifying the number of elites. 
  * default: 1
* `max_depth` : An integer specifying the maximum depth of the GP tree.
  * default: 17
* `init_depth` : An integer specifying the initial depth of the GP tree.
  * default: 6
* `log_path` : A string specifying where the results are going to be saved.
  * default: 
  ``` os.path.join(os.getcwd(), "log", "gp.csv")```

### Running GSGP 
To use the GP algorithm, you can use the following example: