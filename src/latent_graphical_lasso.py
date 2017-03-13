
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.linalg import eigsh
import os
import scipy as sp
import cvxpy as cvx
from sklearn.linear_model import lars_path, lasso_path,  ridge_regression
from decimal import Decimal
from sklearn.covariance.empirical_covariance_ import log_likelihood
import sys
from inverse_covariance import quic
from graphical_lasso import *


def latent_variable_gmm_cvx(X_o, alpha=1, lambda_s= 1, S_init=None, verbose=False):
    '''
          A cvx implementation of  the Latent Variable Gaussian Graphical Model 
      
       see review of  "Venkat Chandrasekaran, Pablo A Parrilo, and Alan S Willsky. Latent variable graphical model selection via convex optimization. The Annals of Statistics, 40(4):1935–1967, 2012."
    
        
           min_{S, L} -log det (S-L) + trace(emp_Cov*(S-L)) + alpha*lambda_s*\|S\|_{1} + alpha*\|L\|_{*}
                  s.t.  S-L \succeq 0
                        L \succeq 0

         return S, L

    '''
    n, m = X_o.shape
    emp_cov = np.cov(X_o)
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
        covariance_o = emp_cov.copy()
    else:
        covariance_o = S_init.copy()
    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.

    covariance_o *= 0.95
    diagonal_o = emp_cov.flat[::n+1]
    covariance_o.flat[::n+1] = diagonal_o
    
    # define the low-rank term L and sparse term S
    L = cvx.Semidef(n)
    S = cvx.Symmetric(n)
    # define the SDP problem 
    objective = cvx.Minimize(- cvx.log_det(S-L) + cvx.trace(covariance_o*(S-L)) + alpha*lambda_s*cvx.norm(S,1) + alpha*cvx.norm(L, "nuc")  )
    constraints = [S-L >> 0]
  
    # solve the problem
    problem = cvx.Problem(objective, constraints)
    problem.solve(verbose = verbose)

    return (S.value, L.value)




def latent_variable_glasso_data(X_o, X_h=None,  alpha=0.1, S_init=None, max_iter_out=100, verbose=False, threshold=1e-1, return_hists=False):
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
    n, m = X_o.shape
    X_o -= np.mean(X_o, axis=0)
    X_o /= X_o.std(axis=0)

    S = np.cov(X_o)

    if X_h is None:
        sigma_hidden = 1
        h_dim = int(np.ceil(float(n)/2.0))   #size of hidden variables
        X_h = sigma_hidden*np.random.randn(h_dim, m)
    else:
        h_dim = X_h.shape[0]

    n_all = n + h_dim

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
    mle_estimate_o = S.copy()

    # stack rows 
    X_all = np.concatenate((X_o, X_h), axis=0)
    # compute the covariance of the new (o,h) data
    covariance_all = np.cov(X_all)
    covariance_all[np.ix_(np.arange(n), np.arange(n))] = covariance_o

    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.
    covariance_all *= 0.95
    diagonal_all = covariance_all.flat[::n_all+1]
    covariance_all.flat[::n_all+1] = diagonal_all

    subblock1_index = np.arange(n)
    subblock2_index = n+ np.arange(h_dim)

    precision_all = np.linalg.pinv(covariance_all)

    cov_all_list = list()
    cov_all_list.append(covariance_all)
    prec_all_list = list()
    prec_all_list.append(precision_all)

    dsol_list = list()
    # compute a mask that are all ones in subblock1
    mask = np.zeros((n_all, n_all))
    mask[np.ix_(subblock1_index, subblock1_index)] = np.ones((n, n))
   
    # EM-loop
    from tqdm import tqdm
    for t in tqdm(range(max_iter_out)):
        # M-step: find the inverse covariance matrix for entire graph
        # use a package in skggm to solve glaphical lasso with matrix regularizer
        precision_t, _, _, _, _, _ = quic(covariance_all, lam=alpha*mask)
        

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
        if t == 0 :
            precision_pre = precision_t
            if verbose: print("| d-sol | ")
        else:
            diff = np.linalg.norm(precision_pre-precision_t)/np.sqrt(n)
            dsol_list.append(diff)
            if verbose: print("| %.3f  |" % (diff) )
            if diff < threshold:
                break
            else:
                precision_pre = precision_t
             

    if return_hists:
        return (covariance_all[np.ix_(subblock1_index, subblock1_index)], precision_all[np.ix_(subblock1_index, subblock1_index)],  cov_all_list, prec_all_list, dsol_list)
    else:
        return (covariance_all[np.ix_(subblock1_index, subblock1_index)], precision_all[np.ix_(subblock1_index, subblock1_index)]) 



def latent_variable_glasso_cvx(X_o, X_h=None,  alpha=0.1, S_init=None, max_iter_out=100, max_iter_in=100, verbose=False, convg_threshold=1e-3, return_hists=False):
    '''
       A EM algorithm implementation of the Latent Variable Gaussian Graphical Model 
      
       see review of  "Venkat Chandrasekaran, Pablo A Parrilo, and Alan S Willsky. Latent variable graphical model selection via convex optimization. The Annals of Statistics, 40(4):1935–1967, 2012."


       Loop for t= 1,2,...,
 
       1. M-step:
          solve a sparse inverse covariance estimation using gLasso 
             with expectation of empirical covariance over (observed, latent) data
          implemented via cvx solver

       2. E-step:
          given the estimated sparse inverse covariance \Sigma_{(o,h)}, find the expectation of covariance over (o,h) given the observed covariance data S

        = [
            [S, -S*Sigma_{oh} ]
            [-S*Sigma_{ho}, eye(h) + Sigma_{ho}*S*Sigma_{oh}]
          ]
 
    '''
    n, m = X_o.shape
    X_o -= np.mean(X_o, axis=0)
    X_o /= X_o.std(axis=0)

    S = np.cov(X_o)

    if X_h is None:
        sigma_hidden = 1
        h_dim = int(np.ceil(float(n)/2.0))   #size of hidden variables
        X_h = sigma_hidden*np.random.randn(h_dim, m)
    else:
        h_dim = X_h.shape[0]

    n_all = n + h_dim

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
    mle_estimate_o = S.copy()

    # stack rows 
    X_all = np.concatenate((X_o, X_h), axis=0)
    # compute the covariance of the new (o,h) data
    covariance_all = np.cov(X_all)
    covariance_all[np.ix_(np.arange(n), np.arange(n))] = covariance_o

    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.
    covariance_all *= 0.95
    diagonal_all = covariance_all.flat[::n_all+1]
    covariance_all.flat[::n_all+1] = diagonal_all

    subblock1_index = np.arange(n)
    subblock2_index = n+ np.arange(h_dim)

    precision_all = np.linalg.pinv(covariance_all)

    cov_all_list = list()
    cov_all_list.append(covariance_all)
    prec_all_list = list()
    prec_all_list.append(precision_all)
   
    mask = np.zeros((n_all, n_all))
    mask[np.ix_(subblock1_index, subblock1_index)] = np.ones((n, n))
    # EM-loop
    for t in range(max_iter_out):
        # M-step: find the inverse covariance matrix for entire graph
        # solve the inverse covariance estimation 
        Theta = cvx.Semidef(n_all)
        # define the SDP problem 
        objective = cvx.Minimize(- cvx.log_det(Theta) + cvx.trace(covariance_all*Theta) + alpha*cvx.norm(cvx.mul_elemwise(mask, Theta),1))
  
        # solve the problem
        problem = cvx.Problem(objective)
        problem.solve(verbose = verbose)


        precision_all = Theta.value
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
