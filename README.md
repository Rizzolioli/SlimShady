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

data = 'instanbul'

X_train, y_train = load_preloaded(data, seed= 1, training=True, X_y=True)
X_test, y_test = load_preloaded(data, seed= 1, training=False, X_y=True)
n_runs = 30
pop_size = 100
n_iter = 100

gp(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
         dataset_name=data, n_runs=n_runs, pop_size=pop_size, n_iter=n_iter)
```
The arguments for the **gp** function are: 
* `X_train`: A torch Tensor with the training input data.
* `y_train`: A torch Tensor with the training output data.
* `X_test`: A torch Tensor with the testing input data.
  * default: None
* `y_test`: A torch Tensor with the testing output data.
   * default: None
dataset_name : str, optional
    Dataset name, for logging purposes
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
To use the GSGP algorithm, you can use the following example:

```python
from slim.main_gsgp import gsgp

data = 'instanbul'

X_train, y_train = load_preloaded(data, seed= 1, training=True, X_y=True)
X_test, y_test = load_preloaded(data, seed= 1, training=False, X_y=True)
n_runs = 30
pop_size = 100
n_iter = 100

gsgp(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
         dataset_name=data, n_runs=n_runs, pop_size=pop_size, n_iter=n_iter)
```

Where the arguments for the **gsgp** function are the same as the ones for the **gp** function, except for the absence of the parameter
`max_depth` and for the default value of the parameter `log_path` which is 
``` os.path.join(os.getcwd(), "log", "gsgp.csv")```
