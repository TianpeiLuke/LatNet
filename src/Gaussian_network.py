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
from sklearn.linear_model import lars_path, ridge_regression
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


def sparse_inv_cov_glasso(X, alpha=0.1, S_init=None, max_iter=100, verbose=False, convg_threshold=1e-3, return_costs=False):
    '''
         inverse covariance estimation by maximum log-likelihood estimate given X

         -log p(X | J) + alpha ||J||_{1} := -log(det(J)) + tr(S*J) + alpha*||J||_{1}

         S:= np.dot(X,X.T)/m

         using gLasso 

    '''
    n, m = X.shape
    S = np.cov(X)
    if alpha == 0:
        if return_costs:
            precision = np.linalg.pinv(S)
            cost = - 2. * log_likelihood(S, precision)
            cost += n_features * np.log(2 * np.pi)
            d_gap = np.sum(S * precision) - n
            return S, precision, (cost, d_gap)
        else:
            return S, np.linalg.pinv(S)

    costs = list()
    if S_init is None:
        covariance = S.copy()
    else:
        covariance = S_init.copy()
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
            if verbose:
                print(
                    '[graph_lasso] Iteration % 3i, cost % 3.2e, dual gap %.3e'
                    % (t, cost, d_gap))
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



def sparse_inv_cov_glasso_v2(emp_cov, block_index,  alpha=0.1, S_init=None, max_iter=100, verbose=False, convg_threshold=1e-3, return_costs=False):
    '''
         inverse covariance estimation by maximum log-likelihood estimate given X

         -log p(X | J) + alpha ||J||_{1} := -log(det(J)) + tr(S*J) + alpha*||J||_{1}

         S:= np.dot(X,X.T)/m

         using gLasso 

    '''
    n = emp_cov.shape[0]
    #S = np.cov(X)
    if alpha == 0:
        if return_costs:
            precision = np.linalg.pinv(emp_cov)
            cost = - 2. * log_likelihood(emp_cov, precision)
            cost += n_features * np.log(2 * np.pi)
            d_gap = np.sum(emp_cov * precision) - n
            return emp_cov, precision, (cost, d_gap)
        else:
            return emp_cov, np.linalg.pinv(emp_cov)

    costs = list()
    if S_init is None:
        covariance = emp_cov.copy()
    else:
        covariance = S_init.copy()
    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.

    covariance *= 0.95
    diagonal = emp_cov.flat[::n+1]
    covariance.flat[::n+1] = diagonal

 #+ alpha*np.eye(n, dtype=X.dtype)
    mle_estimate = emp_cov.copy()
    precision = np.linalg.pinv(covariance)
    indices = np.arange(n)
    try:
        d_gap = np.inf
        for t in range(max_iter):
            for i in block_index: #range(n):
                #index = [x for x in indices if x !=i]
                covariance_11 = np.ascontiguousarray(
                                     covariance[indices != i].T[indices != i])
                covariance_12 = mle_estimate[indices != i , i]
                #solve lasso for each column
                alpha_min = alpha/(n-1)
                #alpha_min = float(round(alpha_min, 5))
                #if i in block_index:
                _, _, coeffs = lars_path(covariance_11, covariance_12, Xy=covariance_12, 
                                         Gram=covariance_11, alpha_min = alpha_min, 
                                         copy_Gram = True, method='lars', return_path=False  ) 
                #else:
                #    coeffs  = ridge_regression(covariance_11, covariance_12, alpha=alpha)                
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
            if verbose:
                print(
                    '[graph_lasso] Iteration % 3i, cost % 3.2e, dual gap %.3e'
                    % (t, cost, d_gap))
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
   



def latent_variable_inv_cov(X, h_dim=None,  alpha=0.1, S_init=None, max_iter_out=100, max_iter_in=100, verbose=False, convg_threshold=1e-3, return_hists=False):
    '''
       A EM algorithm implementation of the Latent Variable Gaussian Graphical Model 
      
       see review of  "Venkat Chandrasekaran, Pablo A Parrilo, and Alan S Willsky. Latent variable graphical model selection via convex optimization. The Annals of Statistics, 40(4):1935–1967, 2012."


       Loop for t= 1,2,...,
 
       1. M-step:
          solve a sparse inverse covariance estimation using gLasso 
             with expectation of empirical covariance over (observed, latent) data

       2. E-step:
          given the estimated sparse inverse covariance \Sigma_{(o,h)}, find the expectation of covariance over (o,h) given the observed covariance data S

        = [
            [S, -S*Sigma_{oh} ]
            [-S*Sigma_{ho}, eye(h) + Sigma_{ho}*S*Sigma_{oh}]
          ]
 
    '''
    n, m = X.shape
    S = np.cov(X)

    if h_dim is None or h_dim > n:
        h_dim = int(np.ceil(float(n)/2.0))   #size of hidden variables
        print('Invalid hidden dimension. Choose '+ str(h_dim))

    n_all = n + h_dim
    #print(n_all)

    if alpha == 0:
        if return_costs:
            precision = np.linalg.pinv(S)
            cost = - 2. * log_likelihood(S, precision)
            cost += n_features * np.log(2 * np.pi)
            d_gap = np.sum(S * precision) - n
            return S, precision, (cost, d_gap)
        else:
            return S, np.linalg.pinv(S)

    costs = list()
    if S_init is None:
        covariance_o = S.copy()
    else:
        covariance_o = S_init.copy()
    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.

    covariance_o *= 0.95
    diagonal = S.flat[::n+1]
    covariance_o.flat[::n+1] = diagonal

    mle_estimate_o = S.copy()
    # initialize a random block for precision_(oh)
    #precision_oh = 0.1*np.random.randn(n, h_dim)
    covariance_all = np.zeros((n_all, n_all))
    #print(covariance_all.shape)
    factor = 0.3
    covariance_oh = (factor/np.sqrt(n*h_dim))*np.random.randn(n, h_dim)#-np.dot(covariance_o, precision_oh)
    # by Schur complement
    # covariance_hh - covariance_oh.T*inv(covariance_o)*covariance_oh \succeq 0 
    covariance_hh = np.eye(h_dim) #- np.dot(precision_oh.T, covariance_oh)  

    subblock1_index = np.arange(n)
    subblock2_index = n+ np.arange(h_dim)

    covariance_all[np.ix_(subblock1_index, subblock1_index)] = covariance_o
    covariance_all[np.ix_(subblock1_index, subblock2_index)] = covariance_oh
    covariance_all[np.ix_(subblock2_index, subblock1_index)] = covariance_oh.T 
    covariance_all[np.ix_(subblock2_index, subblock2_index)] = covariance_hh

    precision_all = np.linalg.pinv(covariance_all)

    cov_all_list = list()
    cov_all_list.append(covariance_all)
    prec_all_list = list()
    prec_all_list.append(precision_all)
   
    # EM-loop
    for t in range(max_iter_out):
        # M-step: find the inverse covariance matrix for entire graph
        _, precision_t = sparse_inv_cov_glasso_v2(covariance_all, block_index=subblock1_index, alpha=alpha, max_iter=max_iter_in, convg_threshold=convg_threshold, verbose=verbose)
         
        precision_all = precision_t
        prec_all_list.append(precision_all)
  
        precision_oh = precision_all[np.ix_(subblock1_index, subblock2_index)]
        # E-step: find the expectation of covariance over (o, h)
        covariance_oh = -np.dot(covariance_o, precision_oh)
        covariance_hh = np.eye(h_dim) - np.dot(precision_oh.T, covariance_oh)  
 
        covariance_all[np.ix_(subblock1_index, subblock1_index)] = covariance_o
        covariance_all[np.ix_(subblock1_index, subblock2_index)] = covariance_oh
        covariance_all[np.ix_(subblock2_index, subblock1_index)] = covariance_oh.T 
        covariance_all[np.ix_(subblock2_index, subblock2_index)] = covariance_hh

        cov_all_list.append(covariance_all)

    if return_hists:
        return (covariance_all[np.ix_(subblock1_index, subblock1_index)], precision_all[np.ix_(subblock1_index, subblock1_index)],  cov_all_list, prec_all_list)
    else:
        return (covariance_all[np.ix_(subblock1_index, subblock1_index)], precision_all[np.ix_(subblock1_index, subblock1_index)]) 


class Gaussian_Random_Field(object): 
    '''
         A Gaussian random field Covarianceith measurement X on undirected graph G 
    '''
    def __init__ (self, G, n, option):
        self.G = G, self.n = n
        try:
            self.alpha = option['alpha']
        except KeyError:
            self.alpha = 0.5

        try:
            self.threshold = option['threshold']
        except KeyError:
            self.threshold = 1e-3

        try:
            self.max_iter = option['max_iter']
        except KeyError:
            self.max_iter = 100

        try:
            self.h_dim = option['h_dim']
        except KeyError:
            self.h_dim = 10

    def fit(self, X, option):
        self.X = X
        self.m = X.shape[1]
        if option['method'] == 'Sparse_GGM':
            self.covariance, self.precision =  sparse_inv_cov_glasso(X, alpha=self.alpha, max_iter=self.max_iter,  convg_threshold=self.threshold)

        elif option['method'] == 'Latent_variable_GGM':
            self.covariance, self.precision =  latent_variable_inv_cov(X, self.h_dim, alpha=self.alpha, max_iter_out=self.max_iter, max_iter_in=100,  convg_threshold=self.threshold)




