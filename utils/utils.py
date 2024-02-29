import torch

"""
Taken from GPOL
"""

def train_test_split(X, y, p_test=0.3, shuffle=True, indices_only=False, seed=0):
    """ Splits X and y tensors into train and test subsets

    This method replicates the behaviour of Sklearn's 'train_test_split'.

    Parameters
    ----------
    X : torch.Tensor
        Input data instances,
    y : torch.Tensor
        Target vector.
    p_test : float (default=0.3)
        The proportion of the dataset to include in the test split.
    shuffle : bool (default=True)
        Whether or not to shuffle the data before splitting.
    indices_only : bool (default=False)
        Whether or not to return only the indices representing training and test partition.
    seed : int (default=0)
        The seed for random numbers generators.

    Returns
    -------
    X_train : torch.Tensor
        Training data instances.
    y_train : torch.Tensor
        Training target vector.
    X_test : torch.Tensor
        Test data instances.
    y_test : torch.Tensor
        Test target vector.
    train_indices : torch.Tensor
        Indices representing the training partition.
    test_indices : torch.Tensor
    Indices representing the test partition.
    """
    # Sets the seed before generating partition's indexes
    torch.manual_seed(seed)
    # Generates random indices
    if shuffle:
        indices = torch.randperm(X.shape[0])
    else:
        indices = torch.arange(0, X.shape[0], 1)
    # Splits indices
    split = int(math.floor(p_test * X.shape[0]))
    train_indices, test_indices = indices[split:], indices[:split]

    if indices_only:
        return train_indices, test_indices
    else:
        # Generates train/test partitions
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test
