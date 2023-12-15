## kernel_quantile_fitting.py
##
## Methods that allow fitting a kernel-based (local regression) method
## for prediction of quantiles.

import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap # this is so cringe...
import numpy as np
import matplotlib.pyplot as plt

## predictKRR(X, Z, beta, tau = 1, offset = 0)
##
## Calculates predictions from a kernel regression fit originally on
## data in X on a new dataset with covariates Z.  Returns the
## predicted target values. In particular, returns a vector y of size
## nrow(Z) whose entries are of the form
##
##  y[i] = sum_{j = 1}^n k(x_j, z_i) beta_i + offset
##
## where x_j denotes the jth row of X and z_i the ith row of Z, and k is the
## Gaussian (RBF) kernel function
##
##   k(x, z) = exp(-||x - z||^2 / (2 * tau^2))
##
## Does this with (reasonable) efficiency
##
## Expects X and Z to be 1-dimensional.
def predictKRR(X, Z, beta, tau = .1, offset = 0):
    nn = len(X)
    mm = len(Z)
    ip_matrix = np.outer(X, Z)                          # n-by-m matrix with entries x_i' * z_j
    squared_Xs = np.transpose(np.tile(X*X, (mm,1)))     # n-by-m matrix whose ith
                                                        # row is all x_i' * x_i
    squared_Zs = np.tile(Z*Z, (nn,1))   # n-by-m matrix whose ith column
                                        # is all z_i' * z_i
    dist_squared_matrix = squared_Xs - 2 * ip_matrix + squared_Zs
    kernel_matrix = np.transpose(np.exp(-dist_squared_matrix / (2 * tau**2)))
    predictions = offset + kernel_matrix.dot(beta)
    return predictions

## fitQuantileKernel(X, y, quantile_level, lambda, tau)
##
## Fits a kernel predictor for prediction of the *quantile* on the
## data given in X and y. Uses the Gaussian kernel
##
##  k(x, z) = exp(-||x - z||^2 / (2*tau^2))
##
## and ridge regularization with the given lambda regularizer.
## In particular, sets beta to minimize
##
##  L(b) = (1/n) sum_{i=1}^n l_q(y_i - G_i' * b) + (lambda/2) * b' * G * b
##
## where l_q(t) = q * max(t, 0) + (1 - q) * max(-t, 0) is the usual
## quantile loss with q= quantile_level, and G is Gram matrix whose
## entries are
##
##   G[i, j] = k(x_i, x_j)
##
## Expects X to be 1-dimensional.
def fitQuantileKernel(X, y, quantile_level = .5,
                      lmda = .01, tau = .1):
    nn = len(X)

    ## Construct the Gram matrix
    ip_matrix = np.outer(X, X)                  # n-by-n matrix with entries x_i' * x_j
    squared_Xs = np.outer(X * X, np.ones(nn))   # turn it into an n-by-n matrix
    dist_squared = squared_Xs + np.transpose(squared_Xs) - 2 * ip_matrix
    G = np.exp(-dist_squared / (2. * tau**2))

    ## We formulate the problem as solving
    ##
    ##  minimize    sum_i l_alpha(z_i) + (lambda/2) * beta' * G * beta
    ##  subject to  z = y - G * beta
    ##
    ## but we don't actually introduce the new variable z...
    beta = cp.Variable(nn)
    G = psd_wrap(G) # wrap the matrix as PSD to bypass PSD check in cvxpy.
    obj = ((1./nn) * cp.sum(quantile_level * cp.pos(y - G @ beta) +
                        (1 - quantile_level) * cp.neg(y - G @ beta)) +
           (lmda/2) * cp.quad_form(beta, G))
    problem = cp.Problem(cp.Minimize(obj))
    result = problem.solve()
    return beta.value

## fitKernelRidge(X, y, lambda, tau)
##
## Fits a Kernel ridge regression with regularization lambda, Gaussian
## kernel
##
##  k(x, z) = exp(-||x - z||^2 / (2 * tau^2))
##
## on the data with responses y.
##
## Expects X and y to be 1-dimensional.
def fitKernelRidge(X, y, lmda = .01, tau = .1):
    nn = len(X)

    ## Construct the Gram matrix
    ip_matrix = np.outer(X, X)                  # n-by-n matrix with entries x_i' * x_j
    squared_Xs = np.outer(X*X, np.ones(nn))     # turn it into an n-by-n matrix
    dist_squared = squared_Xs + np.transpose(squared_Xs) - 2 * ip_matrix
    G = np.exp(-dist_squared / (2. * tau**2))
    np.fill_diagonal(G, 1 + lmda)
    beta = np.linalg.solve(G, y)
    return beta

## plotUpperAndLower(X, y, y_low, y_high, filename)
##
## Plots and saves given data, scatterplotted as y versus X, and
## plots a semi-transparent band between the low and high confidence bands.
def plotUpperAndLower(X, y, y_low, y_high, filename=None):
    plt.scatter(X, y)
    plt.fill_between(X, y_low, y_high, color='blue', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    #plt.savefig(filename, format='pdf')
    #plt.close()
