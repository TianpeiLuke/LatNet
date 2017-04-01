

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


def dilat_lvggm_ccg_cvx(X_o, alpha=1, gamma=1, beta=1, precision_h=np.eye(5), S_init=None, max_iter=1000, threshold=1e-3, verbose=False):
    '''
         A cvx implementation of the Decayed-influence Latent variable Gaussian Graphial Model

        Using Convex-Concave Procedure

        For t=1, ...
        
            min_{R} -log det(R) + trace(R*S_t) + alpha*||[1,0]*R*[1;0]||_1 + gamma*||[-Cov(X), 0]*R*[0;1]||_{1,2} 

                s.t.   [0,1]*R*[0;1] = Theta^{-1}

                       R >= 0 

            S_t = [Cov(X), -Cov*B*Theta; -Theta*B'*Cov, beta I]

    '''
    n, m = X_o.shape
    emp_cov = np.cov(X_o)
    if alpha == 0 and gamma == 0:
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

    n2 = precision_h.shape[0]
    eff_rank_h = np.trace(precision_h)/np.linalg.norm(precision_h, 2)
    #while eff_rank_h < 0.3*float(n2):
    #    step = 1e-2
    #    precision_h = precision_h + step*np.eye(n2)
    #    eff_rank_h = np.trace(precision_h)/np.linalg.norm(precision_h, 2)
    eigvals_h, eigvecs_h = np.linalg.eigh(precision_h)
    eigvecs_h = np.asarray(eigvecs_h)
    eps = 1e-4

    nonzeros_indices = np.argwhere(eigvals_h> 1e-4)
    transformed_eigvals_h = eigvals_h.copy()
    transformed_eigvals_h[nonzeros_indices] = 1/(eps + transformed_eigvals_h[nonzeros_indices])
    covariance_h = np.dot(transformed_eigvals_h*eigvecs_h, eigvecs_h.T)


    for t in range(max_iter):
        R = dilat_lvggm_ccg_cvx_sub(S, alpha, gamma, covariance_o, covariance_h, max_iter_in=1000, threshold_in=1e-3, verbose=verbose)





    
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