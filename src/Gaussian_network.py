# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
import networkx as nx
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import os
from time import gmtime, strftime
import scipy as sp
import cvxopt
from sklearn.linear_model import lars_path
from decimal import Decimal
from sklearn.covariance.empirical_covariance_ import log_likelihood

def _dual_gap(emp_cov, precision, alpha):
    """Expression of the dual gap convergence criterion
    The specific definition is given in Duchi "Projected Subgradient Methods
    for Learning Sparse Gaussians".
    """
    gap = np.sum(emp_cov * precision)
    gap -= precision.shape[0]
    gap += alpha * (np.abs(precision).sum()
                    - np.abs(np.diag(precision)).sum())
    return gap 

def test_convergence( previous_W, new_W, S, t):
    d = S.shape[0]
    x = np.abs( previous_W - new_W ).mean()
    print(x - t*( np.abs(S).sum() + np.abs( S.diagonal() ).sum() )/(d*d-d))
    if np.abs( previous_W - new_W ).mean() < t*( np.abs(S).sum() + np.abs( S.diagonal() ).sum() )/(d*d-d):
        return True
    else:
        return False

def _objective(mle, precision_, alpha):
    """Evaluation of the graph-lasso objective function
    the objective function is made of a shifted scaled version of the
    normalized log-likelihood (i.e. its empirical mean over the samples) and a
    penalisation term to promote sparsity
    """
    p = precision_.shape[0]
    cost = - 2. * log_likelihood(mle, precision_) + p * np.log(2 * np.pi)
    cost += alpha * (np.abs(precision_).sum()
                     - np.abs(np.diag(precision_)).sum())
    return cost


def sparse_inv_cov_glasso(X, n, m, alpha=0.1, max_iter=100, convg_threshold=1e-3, return_costs=False):
    '''
         inverse covariance estimation by maximum log-likelihood estimate given X

         -log p(X | J) + alpha ||J||_{1} := -log(det(J)) + tr(S*J) + alpha*||J||_{1}

         S:= np.dot(X,X.T)/m

         using gLasso 

    '''
    S = np.cov(X)
    if alpha == 0:
        if return_costs:
            precision = linalg.inv(S)
            cost = - 2. * log_likelihood(S, precision)
            cost += n_features * np.log(2 * np.pi)
            d_gap = np.sum(S * precision) - n
            return S, precision, (cost, d_gap)
        else:
            return S, linalg.inv(S)

    costs = list()
    covariance = S.copy()
    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.

    covariance *= 0.95
    diagonal = S.flat[::n+1]
    covariance.flat[::n+1] = diagonal

 #+ alpha*np.eye(n, dtype=X.dtype)
    mle_estimate = S.copy()
    precision = np.linalg.pinv(covariance)
    indices = np.arange(n)
    try:
        d_gap = np.inf
        for t in range(max_iter):
            for i in range(n):
                #index = [x for x in indices if x !=i]
                covariance_11 = np.ascontiguousarray(
                                     covariance[indices != i].T[indices != i])
                covariance_12 = mle_estimate[indices != i , i]
                #solve lasso for each column
                alpha_min = alpha/(n-1)
                #alpha_min = float(round(alpha_min, 5))
                _, _, coeffs = lars_path(covariance_11, covariance_12, Xy=covariance_12, 
                                         Gram=covariance_11, alpha_min = alpha_min, 
                                         copy_Gram = True, method='lars', return_path=False  ) 
                #coeffs = coeffs[:,-1]   
                #update the precision matrix
                precision[i,i]   = 1./( covariance[i,i] - np.dot(covariance[indices != i, i], coeffs))
                precision[indices != i, i] = -precision[i,i] * coeffs
                precision[i, indices != i] = -precision[i,i] * coeffs
                temp_coeffs = np.dot(covariance_11, coeffs)
                covariance[indices != i ,i] = temp_coeffs
                covariance[i, indices != i] = temp_coeffs
    
            d_gap = _dual_gap(mle_estimate, precision, alpha) 
            cost = _objective(mle_estimate, precision, alpha)
            if return_costs:
                costs.append((cost, d_gap))
            if np.abs(d_gap) < convg_threshold:
                break
            if not np.isfinite(cost) and t > 0:
                    raise FloatingPointError('Non SPD result: the system is '
                                             'too ill-conditioned for this solver')
        else:
            #this triggers if not break command occurs
            print("The algorithm did not coverge. Try increasing the max number of iterations.")
    except FloatingPointError as e:
        e.args = (e.args[0]
                  + '. The system is too ill-conditioned for this solver',)
        raise e

    if return_costs:
        return covariance, precision, costs
    else:
        return (covariance ,  precision )






def sparse_inv_cov_cvx(X, n, m, alpha):
    '''
         inverse covariance estimation by maximum log-likelihood estimate given X

         -log p(X | J) + alpha ||J||_{1} := -log(det(J)) + tr(S*J) + alpha*||J||_{1}

         S:= np.dot(X,X.T)/m
 
         using CVX packages

    '''
    return np.cov(X)
   



def inv_cov(X, n, m):
    '''
         inverse covariance estimation by maximum log-likelihood estimate given X

       1/m* sum log p(X | J) := 0.5*log(det(J)) - 0.5*tr(S*J)

         S:= np.dot(X,X.T)/m

    '''
    return np.cov(X)



class Gaussian_Random_Field(object): 
    '''
         A Gaussian random field Covarianceith measurement X on undirected graph G 
    '''
    def __init__ (self, G, n, option):
        self.G = G, self.n = n
       


    def fit(self, X):
        self.X = X
        self.m = X.shape[1]
        





