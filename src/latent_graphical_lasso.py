
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


#=================================================================================


def hard_threshold(seq, tau):
    seq_copy = seq.copy()
    sortind = np.argsort(np.abs(seq_copy), axis=0)[::-1]
    seq_copy[sortind[tau:]] = 0
    return seq_copy

def prox_R_lvgmm_admm(emp_cov, Z, xi=1):
    '''
      solve for the proximal minimization
         min_R  0.5*||R-Z||**2/xi + trace(R, empcov) - logdet(R)

    '''
    cov_diff = xi*emp_cov - Z
    eigvals, eigvecs = np.linalg.eigh(cov_diff)
    def eig_transform(eig, xi):
        return 0.5*(-eig + np.sqrt(eig**2 + 4*xi))
    #vec_eig_transform = np.vectorize(eig_transform)
    transformed_eigvals = eig_transform(eigvals, xi)
    return np.dot(transformed_eigvals*eigvecs, eigvecs.T)


def prox_S_lvgmm_admm(Z, alpha=1, xi=1):
    '''
      solve for the proximal minimization
         min_R  0.5*||S-Z||**2/xi + alpha||S||_1

    '''
    import pywt
    return pywt.threshold(Z, alpha*xi, 'soft')

def prox_L_lvgmm_admm(Z, gamma=1, xi=1):
    '''
      solve for the proximal minimization
         min_R  0.5*||L-Z||**2/xi + gamma||S||_1

    '''
    eigvals, eigvecs = np.linalg.eigh(Z)
    def eig_transform(eig, gamma, xi):
        return np.maximum(eig-xi*gamma, 0)
    #vec_eig_transform = np.vectorize(eig_transform)
    transformed_eigvals = eig_transform(eigvals, gamma, xi)
    return np.dot(transformed_eigvals*eigvecs, eigvecs.T)


def latent_variable_gmm_admm(X_o, alpha=1, gamma=1,  mu=10, S_init=None, max_iter=1000, threshold=1e-3, verbose=False, return_hists=False):
    '''
          An implementation of  the Latent Variable Gaussian Graphical Model via alteranting direction of multipliers method (ADMM)
      
       see review of  "Ma, Shiqian, Lingzhou Xue, and Hui Zou. "Alternating direction methods for latent variable Gaussian graphical model selection." Neural computation 25, no. 8 (2013): 2172-2198"
    
           min_{R, S, L} f(R) + g(S) + h(L)+ I(R*-S*+L* = 0)

         return S, L

        where f(R) = trace(emp_cov*R) - logdet R
              g(S) = alpha*||S||_{1}
              h(L) = beta*trace(L) + I(L>>0)
       
     The ADMM algorithm is described as below:

     Z = (R, S, L)
     For t in range(max_iters):
         W = Z + mu* Lambda

         R = prox_R(W_R, mu)
         S = prox_S(W_S, alpha, mu)
         L = prox_L(W_L, gamma, mu)
         Z = (R, S, L)

         T = Z - mu* Lambda

         R' = T_R - (T_R-T_S+T_L)/3
         S' = T_S + (T_R-T_S+T_L)/3
         L' = T_L - (T_R-T_S+T_L)/3
         Z' = (R', S', L')
         
         Lambda = Lambda - (Z - Z')/mu

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
    
    # initialization 
    precision_o_init = np.linalg.pinv(covariance_o)
    #S = hard_threshold(precision_o_init.reshape((precision_o_init.size,)), tau_s).reshape((n,n))
    
    #diff_pd = precision_o_init - S
    #eigval, eigvec = np.linalg.eigh(diff_pd)
    #eigvec = np.asarray(eigvec)
    #index_sig = np.argsort(eigval)[::-1]
    #eigval = eigval[index_sig]
    #eigvec = eigvec[:,index_sig]
    #k_neg = np.where(eigval < 0)
    #print("rank of diff: %d" % (k_neg))
    #k  = min(k_l, k_neg)
    #Z = np.sqrt(eigval[:k])*eigvec[:,:k] #right multiplication
    R = np.zeros((n,n))
    S = np.zeros((n,n))
    L = np.zeros((n,n))
    W_R = np.zeros((n,n))
    W_S = np.zeros((n,n))
    W_L = np.zeros((n,n))
    T_R = np.zeros((n,n))
    T_S = np.zeros((n,n))
    T_L = np.zeros((n,n))
    Lambda_R = np.zeros((n,n))
    Lambda_S = np.zeros((n,n))
    Lambda_L = np.zeros((n,n))

    R = precision_o_init
    S = np.eye(n)
    L = S - R
    eigval, eigvec = np.linalg.eigh(L)
    eigvec = np.asarray(eigvec)
    index_sig = np.argsort(eigval)[::-1]
    eigval = eigval[index_sig]
    eigvec = eigvec[:,index_sig]
    k_neg = np.where(eigval < 0)[0][0]
    if verbose: print("rank of initial L: %d" % (k_neg))
    L = np.dot(eigval[:k_neg]*eigvec[:,:k_neg], eigvec[:,:k_neg].T )#right multiplication
    #k  = min(k_l, k_neg)
    mu0 = mu    

    cost_hists = list([])
    diff_pre_hists = list([])
    constraint_loss_hists = list([])

    for t in range(max_iter):
        R_pre = R
        S_pre = S
        L_pre = L

        W_R = R + mu*Lambda_R
        W_S = S + mu*Lambda_S
        W_L = L + mu*Lambda_L
        # proximal projection 
        R = prox_R_lvgmm_admm(covariance_o, W_R, mu)
        S = prox_S_lvgmm_admm(W_S, alpha,  mu)
        L = prox_L_lvgmm_admm(W_L, gamma,  mu)
        
        # concensus modification
        T_R = R - mu*Lambda_R
        T_S = S - mu*Lambda_S
        T_L = L - mu*Lambda_L

        R_tmp = T_R*2/3 + T_S/3 - T_L/3
        S_tmp = T_S*2/3 + T_R/3 + T_L/3
        L_tmp = T_L*2/3 - T_R/3 + T_S/3
        #  dual direction 
        Lambda_R = Lambda_R - (R - R_tmp)/mu
        Lambda_S = Lambda_S - (S - S_tmp)/mu
        Lambda_L = Lambda_L - (L - L_tmp)/mu

        diff_pre = np.linalg.norm(R-R_pre) + np.linalg.norm(S-S_pre) + np.linalg.norm(L-L_pre)  

        constraint_loss = np.linalg.norm(R- S + L)
        cost = -2. * log_likelihood(covariance_o, R) + m * np.log(2* np.pi) + alpha*(np.abs(S).sum() - np.abs(np.diag(S)).sum()) + gamma * np.trace(L)
        if return_hists:
            cost_hists.append(cost)
            constraint_loss_hists.append(constraint_loss)
            diff_pre_hists.append(diff_pre)

        if verbose:
            if t == 0:
                print("iterations | cost  | succ-diff | constraint-loss |")
            
            print("    %d     |  %.4f |  %.4f     |      %.4f       |" \
                       % (t, cost, diff_pre, constraint_loss))
        if diff_pre/3 < threshold and constraint_loss < threshold:
            break

    if return_hists:
        return (R, S, L, cost_hists, constraint_loss_hists, diff_pre_hists)
    else:
        return (R, S, L)




#============================================================================

def latent_variable_glasso_data(X_o, X_h=None,  alpha=0.1, mask=None, S_init=None, max_iter_out=100, verbose=False, threshold=1e-1, return_hists=False):
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
        if mask is not None:
            raise ValueError("Please decide the initial latent variables. ")
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
    if mask is None:
        mask = np.zeros((n_all, n_all))
        mask[np.ix_(subblock1_index, subblock1_index)] = np.ones((n, n))
    else:
        if mask.shape[0] != n_all:
            raise ValueError("mask must be of size (%d, %d)" % (n_all, n_all))
        if mask.shape[0] != mask.shape[1]:
            raise ValueError("mask must be square. shape now (%d, %d)." % (mask.shape) )
        if np.linalg.norm(mask-mask.T) > 1e-3:
            raise ValueError("mask must be symmetric.")
        

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



#def latent_variable_glasso_cvx(X_o, X_h=None,  alpha=0.1, S_init=None, max_iter_out=100, max_iter_in=100, verbose=False, convg_threshold=1e-3, return_hists=False):
#    '''
#       A EM algorithm implementation of the Latent Variable Gaussian Graphical Model 
#      
#       see review of  "Venkat Chandrasekaran, Pablo A Parrilo, and Alan S Willsky. Latent variable graphical model selection via convex optimization. The Annals of Statistics, 40(4):1935–1967, 2012."
#
#
#       Loop for t= 1,2,...,
# 
#       1. M-step:
#          solve a sparse inverse covariance estimation using gLasso 
#             with expectation of empirical covariance over (observed, latent) data
#          implemented via cvx solver
#
#       2. E-step:
#          given the estimated sparse inverse covariance \Sigma_{(o,h)}, find the expectation of covariance over (o,h) given the observed covariance data S
#
#        = [
#            [S, -S*Sigma_{oh} ]
#            [-S*Sigma_{ho}, eye(h) + Sigma_{ho}*S*Sigma_{oh}]
#          ]
# 
#    '''
#    n, m = X_o.shape
#    X_o -= np.mean(X_o, axis=0)
#    X_o /= X_o.std(axis=0)
#
#    S = np.cov(X_o)
#
#    if X_h is None:
#        sigma_hidden = 1
#        h_dim = int(np.ceil(float(n)/2.0))   #size of hidden variables
#        X_h = sigma_hidden*np.random.randn(h_dim, m)
#    else:
#        h_dim = X_h.shape[0]
#
#    n_all = n + h_dim
#
#    if alpha == 0:
#        if return_costs:
#            precision = np.linalg.pinv(S)
#            cost = - 2. * log_likelihood(S, precision)
#            cost += n_features * np.log(2 * np.pi)
#            d_gap = np.sum(S * precision) - n
#            return S, precision, (cost, d_gap)
#        else:
#            return S, np.linalg.pinv(S)
#
#    costs = list()
#    if S_init is None:
#        covariance_o = S.copy()
#    else:
#        covariance_o = S_init.copy()
#    mle_estimate_o = S.copy()
#
#    # stack rows 
#    X_all = np.concatenate((X_o, X_h), axis=0)
#    # compute the covariance of the new (o,h) data
#    covariance_all = np.cov(X_all)
#    covariance_all[np.ix_(np.arange(n), np.arange(n))] = covariance_o
#
#    # As a trivial regularization (Tikhonov like), we scale down the
#    # off-diagonal coefficients of our starting point: This is needed, as
#    # in the cross-validation the cov_init can easily be
#    # ill-conditioned, and the CV loop blows. Beside, this takes
#    # conservative stand-point on the initial conditions, and it tends to
#    # make the convergence go faster.
#    covariance_all *= 0.95
#    diagonal_all = covariance_all.flat[::n_all+1]
#    covariance_all.flat[::n_all+1] = diagonal_all
#
#    subblock1_index = np.arange(n)
#    subblock2_index = n+ np.arange(h_dim)
#
#    precision_all = np.linalg.pinv(covariance_all)
#
#    cov_all_list = list()
#    cov_all_list.append(covariance_all)
#    prec_all_list = list()
#    prec_all_list.append(precision_all)
#   
#    mask = np.zeros((n_all, n_all))
#    mask[np.ix_(subblock1_index, subblock1_index)] = np.ones((n, n))
#    # EM-loop
#    for t in range(max_iter_out):
#        # M-step: find the inverse covariance matrix for entire graph
#        # solve the inverse covariance estimation 
#        Theta = cvx.Semidef(n_all)
#        # define the SDP problem 
#        objective = cvx.Minimize(- cvx.log_det(Theta) + cvx.trace(covariance_all*Theta) + alpha*cvx.norm(cvx.mul_elemwise(mask, Theta),1))
#  
#        # solve the problem
#        problem = cvx.Problem(objective)
#        problem.solve(verbose = verbose)
#
#
#        precision_all = Theta.value
#        prec_all_list.append(precision_all)
#  
#        precision_oh = precision_all[np.ix_(subblock1_index, subblock2_index)]
#        # E-step: find the expectation of covariance over (o, h)
#        covariance_oh = -np.dot(covariance_o, precision_oh)
#        covariance_hh = np.eye(h_dim) - np.dot(precision_oh.T, covariance_oh)  
# 
#        covariance_all[np.ix_(subblock1_index, subblock1_index)] = covariance_o
#        covariance_all[np.ix_(subblock1_index, subblock2_index)] = covariance_oh
#        covariance_all[np.ix_(subblock2_index, subblock1_index)] = covariance_oh.T 
#        covariance_all[np.ix_(subblock2_index, subblock2_index)] = covariance_hh
#
#        cov_all_list.append(covariance_all)
#
#    if return_hists:
#        return (covariance_all[np.ix_(subblock1_index, subblock1_index)], precision_all[np.ix_(subblock1_index, subblock1_index)],  cov_all_list, prec_all_list)
#    else:
#        return (covariance_all[np.ix_(subblock1_index, subblock1_index)], precision_all[np.ix_(subblock1_index, subblock1_index)]) 
