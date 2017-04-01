

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
from pywt import threshold


def dilat_lvggm_ccg_cvx(X_o, alpha=1, beta=1, gamma=1, mu=10, precision_h=np.eye(5), S_init=None, max_iter=1000, threshold=1e-3, return_hist=False,  verbose=False):
    '''
         A cvx implementation of the Decayed-influence Latent variable Gaussian Graphial Model

        Using Convex-Concave Procedure

        For t=1, ...
        
            min_{R} -log det(R) + trace(R*S_t) + alpha*||[1,0]*R*[1;0]||_1 + gamma*||[-Cov(X), 0]*R*[0;1]||_{1,2} 

                s.t.   [0,1]*R*[0;1] = Theta^{-1}

                       R >= 0 

            S_t = [Cov(X), -Cov*B*Theta; -Theta*B'*Cov, beta I]

    '''
    n1, m = X_o.shape
    emp_cov = np.cov(X_o)

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
    diagonal_o = emp_cov.flat[::n1+1]
    covariance_o.flat[::n1+1] = diagonal_o
    precision_o = np.linalg.pinv(covariance_o)
    # the conditional precision on hidden variables
    n2 = precision_h.shape[0]
    n  = n1 + n2

    eff_rank_h = np.trace(precision_h)/np.linalg.norm(precision_h, 2)
    #while eff_rank_h < 0.3*float(n2):
    #    step = 1e-2
    #    precision_h = precision_h + step*np.eye(n2)
    #    eff_rank_h = np.trace(precision_h)/np.linalg.norm(precision_h, 2)


    eigvals_h, eigvecs_h = np.linalg.eigh(precision_h)
    index_sig0 = np.argsort(eigvals_h)
    eigvals_h = eigvals_h[index_sig0[::-1]]
    eigvecs_h = eigvecs_h[:,index_sig0[::-1]]
    eigvecs_h = np.asarray(eigvecs_h)
    eps = 1e-4
    nonzeros_indices = np.argwhere(eigvals_h> 1e-4)
    transformed_eigvals_h = eigvals_h.copy()
    transformed_eigvals_h[nonzeros_indices] = 1/(eps + transformed_eigvals_h[nonzeros_indices])
    covariance_h = np.dot(transformed_eigvals_h*eigvecs_h, eigvecs_h.T)


    # ==========  initialization =============
    # precision_o_vec = precision_o.reshape((precision_o.size,))
    # idx_sort = np.argsort(precision_o_vec)[::-1]
    # tmp=precision_o_vec[idx_sort]
    # hard_threshold = tmp[18]
    # S = threshold(precision_o. hard_threshold, 'hard')
    S = threshold(precision_o, mu*alpha, 'soft')    
    L = S - precision_o
    eigvals_Ls, eigvecs_Ls = np.linalg.eigh(L)
    index_sig1 = np.argsort(eigvals_Ls) #arange in decreasing order
    eigvals_Ls = eigvals_Ls[index_sig1[::-1]]
    eigvecs_Ls = eigvecs_Ls[:,index_sig1[::-1]]
    eigvecs_Ls = np.asarray(eigvecs_Ls)
    eigvals_Ls = threshold(eigvals_Ls, 1e-3, 'hard') # L is enforced to be psd
    transformed_eigvals_D = eigvals_h.copy()
    transformed_eigvals_D = np.sqrt(np.multiply(transformed_eigvals_D, eigvals_Ls[np.arange(n2)]))
    # transformed_eigvals_D = np.sqrt(transformed_eigvals_D)
    # D = B*precision_h = U*(sqrt(eigvals_Ls*eigvals_h))*V.T
    # B = U*sqrt(eigvals_Ls/eigvals_h)*V.T
    # where  S-precision_o = U*eigvals_Ls*U.T
    #        precision_h = V*eigvals_h*V.T
    D1 = np.dot(transformed_eigvals_D*eigvecs_Ls[:,np.arange(n2)], eigvecs_h.T)
    S_12 = np.dot(-covariance_o, D1)
    #for t in range(S_12.shape[0]):
    #    row_t = S_12[t,:]
    #    if np.linalg.norm(row_t,2) < 
    # =========
    cov_all = np.zeros((n,n))
    cov_all[np.ix_(np.arange(n1), np.arange(n1))] = covariance_o
    cov_all[np.ix_(np.arange(n1), np.arange(n1, n))] = S_12
    cov_all[np.ix_(np.arange(n1,n), np.arange(n1))] = S_12.T
    L_tmp = np.dot(S_12.T, np.dot(precision_o, S_12))
    eig_max = np.linalg.norm(L_tmp,2)
    gamma = eig_max+0.1
    cov_all[np.ix_(np.arange(n1,n), np.arange(n1,n))] = gamma*np.eye(n2)
    #=================
    hist_R = list()
    hist_diff_R = list()
    if alpha == 0 and gamma == 0:
        # B = U*sqrt(eigvals_Ls/eigvals_h)*V.T
        # where  S-precision_o = U*eigvals_Ls*U.T
        #        precision_h = V*eigvals_h*V.T
        transformed_eigvals_B = eigvals_h.copy()
        transformed_eigvals_B[nonzeros_indices] = 1/(eps + transformed_eigvals_B[nonzeros_indices])
        transformed_eigvals_B = np.sqrt(np.multiply(transformed_eigvals_B, eigvals_Ls[np.arange(n2)])) 
        B = np.dot(transformed_eigvals_B*eigvecs_Ls[:,np.arange(n2)], eigvecs_h.T)
        if return_hist:
            precision = np.linalg.pinv(emp_cov)
            return (precision, S, B, (hist_R, hist_diff_R))
        else:
            return (np.linalg.pinv(emp_cov), S, B)


    for t in range(max_iter):
        # ============   call sub-routine =================
        # solve a convex sub-problem 
        R = dilat_lvggm_ccg_cvx_sub(cov_all, alpha, beta, covariance_o, covariance_h, max_iter_in=1000, threshold_in=1e-3, verbose=verbose)
        pre_R = R.copy()
        hist_R.append(R)
        D2 = np.dot(-covariance_o, R[np.ix_(np.arange(n1), np.arange(n1,n))])
        D2 = threshold(D2, 1e-3, 'hard')
        S_12 = np.dot(D2, precision_h)
        cov_all[np.ix_(np.arange(n1), np.arange(n1, n))] = S_12
        cov_all[np.ix_(np.arange(n1,n), np.arange(n1))] = S_12.T
        L_tmp = np.dot(S_12.T, np.dot(precision_o, S_12))
        eig_max = np.linalg.norm(L_tmp,2)
        gamma = eig_max+0.1
        cov_all[np.ix_(np.arange(n1,n), np.arange(n1,n))] = gamma*np.eye(n2)
        if t > 0:
            diff_R = np.linalg.norm(R-pre_R, 'fro')
            hist_diff_R.append(diff_R)
            if diff_R < threshold:
                break
    S = R[np.ix_(np.arange(n1), np.arange(n1))]
    B = R[np.ix_(np.arange(n1), np.arange(n1,N))]  
    L = np.dot(B, np.dot(precision_h,B.T))
    precision_marginal = S - L
    if return_hist:
        return (precision_marginal, S, B, (hist_R, hist_diff_R))
    else:
        return (precision_marginal, S, B)
    
def dilat_lvggm_ccg_cvx_sub(S, alpha, beta, covariance, covariance_h, max_iter_in=1000, threshold_in=1e-3, verbose=False):
    '''
         A cvx implementation of the Decayed-influence Latent variable Gaussian Graphial Model

        The subproblem in Convex-Concave Procedure

        
            min_{R} -log det(R) + trace(R*S_t) + alpha*||[1,0]*R*[1;0]||_1 + gamma*||[-Cov(X), 0]*R*[0;1]||_{1,2} 

                s.t.   [0,1]*R*[0;1] = Theta^{-1}

                       R >= 0 

            S_t = [Cov(X), -Cov*B*Theta; -Theta*B'*Cov, beta I]

    '''
    if np.linalg.norm(S-S.T) > 1e-3:
        raise ValueError("Covariance matrix should be symmetric.")

    n = S.shape[0]
    n1 = covariance.shape[0]
    n2 = covariance_h.shape[0]
    if n != (n1+n2):
       raise ValueError("dimension mismatch n=%d, n1=%d, n2=%d" % (n,n1,n2))
    
    mask = np.zeros((n,n))
    mask[np.ix_(np.arange(n1), np.arange(n1))]
    J1 = np.zeros((n, n1))
    J1[np.arange(n1),:] = np.eye(n1)
    J2 = np.zeros((n, n2))
    J2[np.arange(n1,n), :] = np.eye(n2)
    Q  = np.zeros((n, n1))
    Q[np.arange(n1),:] = -covariance

    J1 = np.asmatrix(J1)
    J2 = np.asmatrix(J2)
    Q = np.asmatrix(Q)
    S - np.asmatrix(S)
 
    R = cvx.Semidef(n)
    # define the SDP problem 
    objective = cvx.Minimize(-cvx.log_det(R) + cvx.trace(S*R) + alpha*cvx.norm((J1.T*R*J1), 1) + beta*cvx.mixed_norm( (Q.T*R*J2), 2, 1) )
    constraints = [J2.T*R*J2 ==  covariance_h ]
    # solve the problem
    problem = cvx.Problem(objective, constraints)
    problem.solve(verbose = verbose)

    return np.asarray(R.value)