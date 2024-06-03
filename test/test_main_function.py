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
