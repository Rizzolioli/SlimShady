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
from slim.main_gp import gp  # import the slim library
from datasets.data_loader import load_ppb  # import the loader for the dataset PPB
from slim.evaluators.fitness_functions import rmse  # import the rmse fitness metric
from slim.utils.utils import train_test_split  # import the train-test split function

X, y = load_ppb(X_y=True)  # load the PPB dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)  # split into train and test
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)  # split into the test and validation

# apply the gp algorithm
final_tree = gp(X_train=X_train, y_train=y_train,
                X_test=X_val, y_test=y_val,
                dataset_name='ppb', pop_size=100, n_iter=100)

final_tree.print_tree_representation()  # show the best individual structure at last generation
predictions = final_tree.predict(X_test)  # get the prediction of the best individual in the test set
print(float(rmse(y_true=y_test, y_pred=predictions)))  # compute metrics on the test set
```
The arguments for the **gp** function are: 
* `X_train`: A torch Tensor with the training input data (*default: None*).
* `y_train`: A torch Tensor with the training output data (*default: None*).
* `X_test`: A torch Tensor with the testing input data (*default: None*).
* `y_test`: A torch Tensor with the testing output data (*default: None*). 
* `dataset_name` : A string containing how the results will be logged (*default: None*).
* `pop_size`: An integer specifying the population size (*default: 100*).
* `n_iter`: An integer specifying the number of iterations (*default: 1000*).
* `p_xo`: A float specifying the crossover probability (*default: 0.8*).
* `elitism` : A boolean specifying the presence of elitism (*default: True*).
* `n_elites` : An integer specifying the number of elites (*default: 1*).
* `max_depth` : An integer specifying the maximum depth of the GP tree (*default: 17*).
* `init_depth` : An integer specifying the initial depth of the GP tree (*default: 6*)
* `log_path` : A string specifying where the results are going to be saved (*default: 
  ``` os.path.join(os.getcwd(), "log", "gp.csv")```*).

### Running GSGP 
To use the GSGP algorithm, you can use the following example:

```python
from slim.main_gsgp import gsgp

data = 'instanbul'

```

Where the arguments for the **gsgp** function are the same as the ones for the **gp** function, except for the absence of the parameter
`max_depth` and for the default value of the parameter `log_path` which is 
``` os.path.join(os.getcwd(), "log", "gsgp.csv")```
