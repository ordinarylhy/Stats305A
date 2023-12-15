## Helper file for generating data with outliers and also solving the
## robust ridge regression problem with logistic-type loss.
## This loss is
##
## l(t) = log(1 + e^t) + log(1 + e^{-t}).
##
## Note that for numerical stability, when evaluating this loss, it may
## be important to use the identities
##
## log(1 + e^t) = log(1 + e^{-t}) + t
## log(1 + e^{-t}) = log(1 + e^t) - t,
##
## depending on whether t is highly positive or highly negative.

import cvxpy as cp
import numpy as np

## generate_data(n, d, num_outliers)
##
## Generates training data for the robust regression problem. The
## number of outliers num.outliers defaults to 10.
def generate_data(nn = 100, dd = 25, num_outliers = 10):
    X_train = np.random.normal(size=(nn, dd))
    X_test = np.random.normal(size=(nn, dd))

    beta_star = np.random.normal(size=dd)
    beta_star /= np.linalg.norm(beta_star);  # Makes X * beta ~ N(0, 1)

    train_noise = np.random.normal(size=nn)
    train_outliers = np.random.choice(nn, num_outliers, replace=False)
    test_noise = np.random.normal(size=nn)
    test_outliers = np.random.choice(nn, num_outliers, replace=False)

    ## Generate outlier measurements

    y_train = X_train.dot(beta_star) + train_noise
    signs = np.sign(np.random.normal(size=num_outliers)) # Symmetric random outliers
    y_train[train_outliers] = 5 * signs * np.random.normal(size=num_outliers)**4
    y_test = X_test.dot(beta_star) + test_noise
    signs = np.sign(np.random.normal(size=num_outliers)) # Symmetric noise
    y_test[test_outliers] = 5 * signs * np.random.normal(size=num_outliers)**4
    return X_train, y_train, X_test, y_test

## Function to fit the best model to this data using the log(1 + exp)
## loss. To use this function, simply call
##
## minimize_robust_ridge(X.train, y.train, lambda),
##
## which will return a vector minimizing
##
##  (1/n) sum_{i = 1}^n l(y - x_i' * b) + (lambda/2) * ||b||^2
##
## where ||.|| denotes the l2 norm.
def minimize_robust_ridge(X, y, lmda):
    nn, dd = X.shape
    beta = cp.Variable(dd)
    obj = ((1./nn) * cp.sum(cp.logistic(cp.matmul(X, beta) - y))
        + (1./nn) * cp.sum(cp.logistic(y - cp.matmul(X, beta)))
        + (lmda/2.) * cp.sum_squares(beta))
    problem = cp.Problem(cp.Minimize(obj))
    result = problem.solve()
    return beta.value
