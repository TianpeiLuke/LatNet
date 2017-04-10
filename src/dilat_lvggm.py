

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

import matplotlib.pyplot as plt


def dilat_lvggm_ccg_cvx(X_o, alpha, beta, option, S_init=None, max_iter=10, threshold_convg=5e-3, show_plot=False, return_hist=False,  verbose=False):
    '''
         A cvx implementation of the Decayed-influence Latent variable Gaussian Graphial Model

        Using Convex-Concave Procedure

        For t=1, ...
        
            min_{R} -log det(R) + trace(R*S_t) + alpha*||[1,0]*R*[1;0]||_1 + gamma*||[-Cov(X), 0]*R*[0;1]||_{1,2} 

                s.t.   [0,1]*R*[0;1] = Theta^{-1}

                       R >= 0 

            S_t = [Cov(X), -Cov*B*Theta; -Theta*B'*Cov, beta I]

    '''
    #=======================  extract solver parameter ================
    try:
        gamma = option['gamma']
    except KeyError:
        gamma = 1

    try:
        mu = option['mu']
    except KeyError:
        mu = 10

    try:
        max_iter_sub = option['max_iter_sub']
    except KeyError:
        max_iter_sub = 1000

    try:
        precision_h = option['precision_h']
    except KeyError:
        precision_h = np.eye(5)

    try:
        sparse_init_method= option['sparse_init_method']
    except KeyError:
        sparse_init_method='identity'
    if sparse_init_method == 'hard':
        try:
            sparsify_hard_nonzeros = option['sparse_init_nonzeros']
        except KeyError:
            sparsify_hard_nonzeros = 10

    try:
        noise = option['noise_level']
    except KeyError:
        noise = 0
    #
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

    eigvals_o, eigvecs_o = np.linalg.eigh(covariance_o)
    index_sig1 = np.argsort(eigvals_o)[::-1] #arange in decreasing order
    eigvals_o = eigvals_o[index_sig1]
    eigvecs_o = eigvecs_o[:,index_sig1]
    eigvecs_o = np.asarray(eigvecs_o)
    # enforce eig >= 0
    thresh = eigvals_o[5]
    eigvals_o = threshold(eigvals_o, thresh, 'greater') # L is enforced to be psd
    eigvals_o = np.zeros((n1,))
    eigvals_o[np.arange(5)] = np.ones((5,))
    covariance_principal = np.dot(eigvals_o*eigvecs_o, eigvecs_o.T)
    # the conditional precision on hidden variables
    n2 = precision_h.shape[0]
    n  = n1 + n2

    eff_rank_h = np.trace(precision_h)/np.linalg.norm(precision_h, 2)
    if verbose: 
        print("Input latent precision matrix  ")
        print("size %d x %d, effective rank %.3f" % (precision_h.shape[0], precision_h.shape[1], eff_rank_h))
    #while eff_rank_h < 0.3*float(n2):
    #    step = 1e-2
    #    precision_h = precision_h + step*np.eye(n2)
    #    eff_rank_h = np.trace(precision_h)/np.linalg.norm(precision_h, 2)

    # ========= take inversion of precision_h  =======================
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
    # precision_o_vec = precision_o[np.triu_indices(n1)]
    # idx_sort = np.argsort(precision_o_vec)[::-1]
    # tmp=precision_o_vec[idx_sort]
    # hard_threshold = tmp[9]
    # S = threshold(precision_o. hard_threshold, 'hard')
    if sparse_init_method == 'identity':
        eig_max_0 = np.linalg.norm(precision_o,2)
        S = (eig_max_0+1)*np.eye(n1)
    elif sparse_init_method == 'soft':
        S = threshold(precision_o, alpha*mu, 'soft')
    elif sparse_init_method == 'hard':
        iu = np.triu_indices(n1)
        precision_o_vec = precision_o[iu]
        id_sort = np.argsort(precision_o_vec)[::-1]
        tmp = precision_o_vec[id_sort]
        lower_limit = min(sparsify_hard_nonzeros, len(tmp))
        threshold_hard = tmp[lower_limit]
        S = threshold(precision_o, threshold_hard, 'hard')


    L = S - precision_o
    eigvals_Ls, eigvecs_Ls = np.linalg.eigh(L)
    index_sig1 = np.argsort(eigvals_Ls)[::-1] #arange in decreasing order
    eigvals_Ls = eigvals_Ls[index_sig1]
    eigvecs_Ls = eigvecs_Ls[:,index_sig1]
    eigvecs_Ls = np.asarray(eigvecs_Ls)
    # enforce eig >= 0
    eigvals_Ls = threshold(eigvals_Ls, 1e-3, 'greater') # L is enforced to be psd
    transformed_eigvals_D = eigvals_h.copy()
    transformed_eigvals_D = np.sqrt(np.multiply(transformed_eigvals_D, eigvals_Ls[np.arange(n2)]))
    # transformed_eigvals_D = np.sqrt(transformed_eigvals_D)
    # D = B*precision_h = U*(sqrt(eigvals_Ls*eigvals_h))*V.T
    # B = U*sqrt(eigvals_Ls/eigvals_h)*V.T
    # where  S-precision_o = U*eigvals_Ls*U.T
    #        precision_h = V*eigvals_h*V.T
    try:
        D1 = option['cross_init']
    except KeyError:
        D1 = np.dot(transformed_eigvals_D*eigvecs_Ls[:,np.arange(n2)], eigvecs_h.T)
    if verbose: print("construct off diagonal terms. size %d x %d" % D1.shape)
    S_12 = np.dot(-covariance_o, D1)
    # ========= construct parameters ==================================
    cov_all = np.zeros((n,n))
    cov_all[np.ix_(np.arange(n1), np.arange(n1))] = covariance_o
    cov_all[np.ix_(np.arange(n1), np.arange(n1, n))] = S_12
    cov_all[np.ix_(np.arange(n1,n), np.arange(n1))] = S_12.T
    L_tmp = np.dot(S_12.T, np.dot(precision_o, S_12))
    eig_max = np.linalg.norm(L_tmp,2)
    gamma = eig_max+0.1
    cov_all[np.ix_(np.arange(n1,n), np.arange(n1,n))] = gamma*np.eye(n2)
    #=================  if no sparsity needed ==============================
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

    pre_R = np.zeros((n,n))
    S_list = list()
    for t in range(max_iter):
        # ============   call sub-routine =================
        # solve a convex sub-problem 
        R = dilat_lvggm_ccg_cvx_sub(cov_all, alpha, beta, covariance_h, precision_h, max_iter_in=1000, threshold_in=1e-3, verbose=verbose)
        
        if show_plot:        
            fig1 = plt.figure(1)
            ax = fig1.add_subplot(111)
            cax = ax.matshow(R)
            fig1.colorbar(cax)
            plt.show()
            filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_R_"+ str(t) + "_alpha_%.2f_beta_%.2f"%(alpha,beta) +  ".eps"
            if t%4 == 0: fig1.savefig(filename)
            fig2 = plt.figure(2)
            ax = fig2.add_subplot(111)
            cax = ax.matshow(np.sign(abs(threshold(R, 1e-4, 'hard'))))
            fig2.colorbar(cax)
            plt.show()
            filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_R2_"+ str(t) + "_alpha_%.2f_beta_%.2f"%(alpha,beta) + ".eps"
            if t%4 == 0: fig2.savefig(filename)

        S = R[np.ix_(np.arange(n1), np.arange(n1))]
        #if t > 0: print("difference in S = %.3f" % (np.linalg.norm(S-S_list[-1])))
        S_list.append(S)
        D2 = np.dot(R[np.ix_(np.arange(n1), np.arange(n1,n))], precision_h)
        if show_plot: 
            fig2 = plt.figure(2)
            ax = fig2.add_subplot(111)
            cax = ax.matshow(D2)
            fig2.colorbar(cax)
            plt.show()
            filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_D1_"+ str(t) + "_alpha_%.2f_beta_%.2f"%(alpha,beta) + ".eps"
            if t%4 == 0: fig2.savefig(filename)
            fig3 = plt.figure(3)
            ax = fig3.add_subplot(111)
            cax = ax.matshow(np.sign(abs(threshold(D2, 1e-4, 'hard'))), vmin=-0.2, vmax=0.2)
            fig3.colorbar(cax)
            plt.show()
            filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_D2_"+ str(t) +  "_alpha_%.2f_beta_%.2f"%(alpha,beta) +".eps"
            if t%4 == 0: fig3.savefig(filename)
        #D2 = threshold(D2, 1e-3, 'hard')
        S_12 = np.dot(-covariance_o, D2) #+ noise/(np.sqrt(t+1))*np.random.randn(n1,n2)
        cov_all[np.ix_(np.arange(n1), np.arange(n1))] = covariance_o
        cov_all[np.ix_(np.arange(n1), np.arange(n1, n))] = S_12
        cov_all[np.ix_(np.arange(n1,n), np.arange(n1))] = S_12.T
        L_tmp = np.dot(S_12.T, np.dot(precision_o, S_12))
        eig_max = np.linalg.norm(L_tmp,2)
        gamma = eig_max+0.1
        cov_all[np.ix_(np.arange(n1,n), np.arange(n1,n))] = gamma*np.eye(n2)
        
        if t > 0:
            diff_R = np.linalg.norm(R-pre_R, 'fro')
            print("|        %.4f           |" % (diff_R))

            hist_diff_R.append(diff_R)
            pre_R = R.copy()
            hist_R.append(R)
            if t > 1 and diff_R < threshold_convg:
                break
        else:
            print("|  difference between R |")
            pre_R = R.copy()
            hist_R.append(R)
    S = R[np.ix_(np.arange(n1), np.arange(n1))]
    B = R[np.ix_(np.arange(n1), np.arange(n1,n))]  
    L = np.dot(B, np.dot(precision_h,B.T))
    precision_marginal = S - L
    if return_hist:
        return (precision_marginal, S, B, (hist_R, hist_diff_R))
    else:
        return (precision_marginal, S, B)
    
def dilat_lvggm_ccg_cvx_sub(S, alpha, beta, covariance_h, precision_h, max_iter_in=1000, threshold_in=1e-3, verbose=False):
    '''
         A cvx implementation of the Decayed-influence Latent variable Gaussian Graphial Model

        The subproblem in Convex-Concave Procedure

        
            min_{R} -log det(R) + trace(R*S_t) + alpha*||[1,0]*R*[1;0]||_1 + gamma*||[0, Theta]*R*[1;0]||_{1,2} 

                s.t.   [0,1]*R*[0;1] = Theta^{-1}

                       R >= 0 

            S_t = [Cov(X), -Cov*B*Theta; -Theta*B'*Cov, beta I]

    '''
    if np.linalg.norm(S-S.T) > 1e-3:
        raise ValueError("Covariance matrix should be symmetric.")

    n = S.shape[0]
    #n1 = covariance.shape[0]
    n2 = covariance_h.shape[0]
    n1 = n - n2
    #if n != (n1+n2):
    if n1 < 0:
        raise ValueError("dimension mismatch n=%d, n1=%d, n2=%d" % (n,n1,n2))
    
    mask = np.zeros((n,n))
    mask[np.ix_(np.arange(n1), np.arange(n1))]
    J1 = np.zeros((n, n1))
    J1[np.arange(n1),:] = np.eye(n1)
    J2 = np.zeros((n, n2))
    J2[np.arange(n1,n), :] = np.eye(n2)
    Q = np.zeros((n,n2))
    Q[np.arange(n1,n),:] = precision_h
    #Q  = np.zeros((n, n1))
    #Q[np.arange(n1),:] = -covariance

    J1 = np.asmatrix(J1)
    J2 = np.asmatrix(J2)
    Q = np.asmatrix(Q)
    S - np.asmatrix(S)
 
    R = cvx.Semidef(n)
    # define the SDP problem 
    objective = cvx.Minimize(-cvx.log_det(R) + cvx.trace(S*R) + alpha*(cvx.norm((J1.T*R*J1), 1) + beta*cvx.mixed_norm((J1.T*R*Q).T, 2, 1)  ))#beta*cvx.norm( (J1.T*R*Q), 1)) )
    constraints = [J2.T*R*J2 ==  covariance_h]
    # solve the problem
    problem = cvx.Problem(objective, constraints)
    problem.solve(verbose = verbose)

    return np.asarray(R.value)

#==================================================================================

def dilat_lvggm_ccg_admm(X_o, alpha, beta, option, S_init=None, max_iter=10, threshold_convg=5e-3, show_plot=False, return_hist=False,  verbose=False):
    '''
         A cvx implementation of the Decayed-influence Latent variable Gaussian Graphial Model

        Using Convex-Concave Procedure

        For t=1, ...
        
            min_{R} -log det(R) + trace(R*S_t) + alpha*||[1,0]*R*[1;0]||_1 + gamma*||[-Cov(X), 0]*R*[0;1]||_{1,2} 

                s.t.   [0,1]*R*[0;1] = Theta^{-1}

                       R >= 0 

            S_t = [Cov(X), -Cov*B*Theta; -Theta*B'*Cov, beta I]

    '''
    #=======================  extract solver parameter ================
    try:
        gamma = option['gamma']
    except KeyError:
        gamma = 1

    try:
        mu = option['mu']
    except KeyError:
        mu = 10

    try:
        max_iter_sub = option['max_iter_sub']
    except KeyError:
        max_iter_sub = 1000

    try:
        precision_h = option['precision_h']
    except KeyError:
        precision_h = np.eye(5)

    try:
        sparse_init_method= option['sparse_init_method']
    except KeyError:
        sparse_init_method='identity'
    if sparse_init_method == 'hard':
        try:
            sparsify_hard_nonzeros = option['sparse_init_nonzeros']
        except KeyError:
            sparsify_hard_nonzeros = 10

    #try:
    #    noise = option['noise_level']
    #except KeyError:
    #    noise = 0

    try:
        threshold_in = option['threshold_in']
    except KeyError:
        threshold_in = 1e-4

    try:
        precision_h_type = option['precision_h_type']
    except KeyError:
        precision_h_type = 'diag'

    if verbose: print("input precision type %s" % (precision_h_type))
    if precision_h_type == 'diag':
        precision_h = np.diag(np.diag(precision_h))#precision_h[np.diag_indices(precision_h.shape[0])]
    elif precision_h_type == 'eig':
        eigvals_h, _ = np.linalg.eigh(precision_h)
        precision_h = np.diag(eigvals_h) 

    if show_plot:
        fig1 = plt.figure(1)
        ax = fig1.add_subplot(111)
        cax = ax.matshow(precision_h)
        fig1.colorbar(cax)
        plt.show()
    #
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

    eigvals_o, eigvecs_o = np.linalg.eigh(covariance_o)
    index_sig1 = np.argsort(eigvals_o)[::-1] #arange in decreasing order
    eigvals_o = eigvals_o[index_sig1]
    eigvecs_o = eigvecs_o[:,index_sig1]
    eigvecs_o = np.asarray(eigvecs_o)
    # enforce eig >= 0
    thresh = eigvals_o[5]
    eigvals_o = threshold(eigvals_o, thresh, 'greater') # L is enforced to be psd
    eigvals_o = np.zeros((n1,))
    eigvals_o[np.arange(5)] = np.ones((5,))
    covariance_principal = np.dot(eigvals_o*eigvecs_o, eigvecs_o.T)
    # the conditional precision on hidden variables
    n2 = precision_h.shape[0]
    n  = n1 + n2

    eff_rank_h = np.trace(precision_h)/np.linalg.norm(precision_h, 2)
    if verbose: 
        print("Input latent precision matrix  ")
        print("size %d x %d, effective rank %.3f" % (precision_h.shape[0], precision_h.shape[1], eff_rank_h))
    #while eff_rank_h < 0.3*float(n2):
    #    step = 1e-2
    #    precision_h = precision_h + step*np.eye(n2)
    #    eff_rank_h = np.trace(precision_h)/np.linalg.norm(precision_h, 2)

    # ========= take inversion of precision_h  =======================
    #if precision_h_type == 'diag':
    #    covariance_h = np.linalg.inv(precision_h)
    #else:
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
    # precision_o_vec = precision_o[np.triu_indices(n1)]
    # idx_sort = np.argsort(precision_o_vec)[::-1]
    # tmp=precision_o_vec[idx_sort]
    # hard_threshold = tmp[9]
    # S = threshold(precision_o. hard_threshold, 'hard')
    if sparse_init_method == 'identity':
        eig_max_0 = np.linalg.norm(precision_o,2)
        S = (eig_max_0+1)*np.eye(n1)
    elif sparse_init_method == 'soft':
        S = threshold(precision_o, alpha*mu, 'soft')
    elif sparse_init_method == 'hard':
        iu = np.triu_indices(n1)
        precision_o_vec = precision_o[iu]
        id_sort = np.argsort(precision_o_vec)[::-1]
        tmp = precision_o_vec[id_sort]
        lower_limit = min(sparsify_hard_nonzeros, len(tmp))
        threshold_hard = tmp[lower_limit]
        S = threshold(precision_o, threshold_hard, 'hard')


    L = S - precision_o
    eigvals_Ls, eigvecs_Ls = np.linalg.eigh(L)
    index_sig1 = np.argsort(eigvals_Ls)[::-1] #arange in decreasing order
    eigvals_Ls = eigvals_Ls[index_sig1]
    #print(len(eigvals_Ls))
    eigvecs_Ls = eigvecs_Ls[:,index_sig1]
    eigvecs_Ls = np.asarray(eigvecs_Ls)
    # enforce eig >= 0
    eigvals_Ls = threshold(eigvals_Ls, 1e-3, 'greater') # L is enforced to be psd
    transformed_eigvals_D = eigvals_h.copy()
    if n1 > n2 :
        transformed_eigvals_D = np.sqrt(np.multiply(transformed_eigvals_D, eigvals_Ls[np.arange(n2)]))
    else:
        transformed_eigvals_D = np.sqrt(np.multiply(transformed_eigvals_D[np.arange(n1)], eigvals_Ls))
    # transformed_eigvals_D = np.sqrt(transformed_eigvals_D)
    # D = B*precision_h = U*(sqrt(eigvals_Ls*eigvals_h))*V.T
    # B = U*sqrt(eigvals_Ls/eigvals_h)*V.T
    # where  S-precision_o = U*eigvals_Ls*U.T
    #        precision_h = V*eigvals_h*V.T
    try:
        D1 = option['cross_init']
    except KeyError:
        if n1 > n2:
            D1 = np.dot(transformed_eigvals_D*eigvecs_Ls[:,np.arange(n2)], eigvecs_h.T)
        else:
            D1 = np.dot(transformed_eigvals_D*eigvecs_Ls, eigvecs_h[:,np.arange(n1)].T)
    if verbose: print("construct off diagonal terms. size %d x %d" % D1.shape)
    S_12 = np.dot(-covariance_o, D1)
    # ========= construct parameters ==================================
    cov_all = np.zeros((n,n))
    cov_all[np.ix_(np.arange(n1), np.arange(n1))] = covariance_o
    cov_all[np.ix_(np.arange(n1), np.arange(n1, n))] = S_12
    cov_all[np.ix_(np.arange(n1,n), np.arange(n1))] = S_12.T
    L_tmp = np.dot(S_12.T, np.dot(precision_o, S_12))
    eig_max = np.linalg.norm(L_tmp,2)
    gamma = eig_max+0.1
    cov_all[np.ix_(np.arange(n1,n), np.arange(n1,n))] = gamma*np.eye(n2)
    #=================  if no sparsity needed ==============================
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

    pre_R = np.zeros((n,n))
    S_list = list()
    for t in range(max_iter):
        # ============   call sub-routine =================
        # solve a convex sub-problem
        if precision_h_type == 'diag' or precision_h_type == 'eig':
            R,_ = dilat_lvggm_ccg_sub_admm_diag(cov_all, alpha, beta*alpha, covariance_h, precision_h, mu=1,  max_iter_in=1000, threshold_in=threshold_in, return_costs=False, verbose=verbose)
        else:
            R,_,_ = dilat_lvggm_ccg_sub_admm_full(cov_all, alpha, beta*alpha, covariance_h, precision_h, mu=1, mu_w=1, max_iter_in=1000, threshold_in=threshold_in, return_costs=False, verbose=verbose)


        if show_plot:        
            fig1 = plt.figure(1)
            ax = fig1.add_subplot(111)
            cax = ax.matshow(R)
            fig1.colorbar(cax)
            plt.show()
            filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_R_"+ str(t) + "_alpha_%.2f_beta_%.2f"%(alpha,beta) +  ".eps"
            if t%4 == 0: fig1.savefig(filename)
            fig2 = plt.figure(2)
            ax = fig2.add_subplot(111)
            cax = ax.matshow(np.sign(abs(threshold(R, 1e-4, 'hard'))))
            fig2.colorbar(cax)
            plt.show()
            filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_R2_"+ str(t) + "_alpha_%.2f_beta_%.2f"%(alpha,beta) + ".eps"
            if t%4 == 0: fig2.savefig(filename)

        S = R[np.ix_(np.arange(n1), np.arange(n1))]
        #if t > 0: print("difference in S = %.3f" % (np.linalg.norm(S-S_list[-1])))
        S_list.append(S)
        D2 = np.dot(R[np.ix_(np.arange(n1), np.arange(n1,n))], precision_h)
        if show_plot: 
            fig2 = plt.figure(2)
            ax = fig2.add_subplot(111)
            cax = ax.matshow(D2)
            fig2.colorbar(cax)
            plt.show()
            filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_D1_"+ str(t) + "_alpha_%.2f_beta_%.2f"%(alpha,beta) + ".eps"
            if t%4 == 0: fig2.savefig(filename)
            fig3 = plt.figure(3)
            ax = fig3.add_subplot(111)
            cax = ax.matshow(np.sign(abs(threshold(D2, 1e-4, 'hard'))), vmin=-0.2, vmax=0.2)
            fig3.colorbar(cax)
            plt.show()
            filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_D2_"+ str(t) +  "_alpha_%.2f_beta_%.2f"%(alpha,beta) +".eps"
            if t%4 == 0: fig3.savefig(filename)
        #D2 = threshold(D2, 1e-3, 'hard')
        S_12 = np.dot(-covariance_o, D2) #+ noise/(np.sqrt(t+1))*np.random.randn(n1,n2)
        cov_all[np.ix_(np.arange(n1), np.arange(n1))] = covariance_o
        cov_all[np.ix_(np.arange(n1), np.arange(n1, n))] = S_12
        cov_all[np.ix_(np.arange(n1,n), np.arange(n1))] = S_12.T
        L_tmp = np.dot(S_12.T, np.dot(precision_o, S_12))
        eig_max = np.linalg.norm(L_tmp,2)
        gamma = eig_max+0.1
        cov_all[np.ix_(np.arange(n1,n), np.arange(n1,n))] = gamma*np.eye(n2)
        
        if t > 0:
            diff_R = np.linalg.norm(R-pre_R, 'fro')
            if verbose: print("|        %.4f           |" % (diff_R))

            hist_diff_R.append(diff_R)
            pre_R = R.copy()
            hist_R.append(R)
            if t > 1 and diff_R < threshold_convg:
                break
        else:
            if verbose: print("|  difference between R |")
            pre_R = R.copy()
            hist_R.append(R)
    S = R[np.ix_(np.arange(n1), np.arange(n1))]
    B = R[np.ix_(np.arange(n1), np.arange(n1,n))]  
    L = np.dot(B, np.dot(precision_h,B.T))
    precision_marginal = S - L
    if return_hist:
        return (precision_marginal, threshold(S, 1e-5, 'hard'), B, (hist_R, hist_diff_R))
    else:
        return (precision_marginal, threshold(S, 1e-5, 'hard'), B)

#-----------------------------------------------------------------
def _prox_R_lvgmm_admm(emp_cov, Z, xi=1):
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


def _prox_S_lvgmm_admm(Z, alpha=1, xi=1):
    '''
      solve for the proximal minimization
         min_R  0.5*||S-Z||**2/xi + alpha||S||_1

    '''
    import pywt
    return pywt.threshold(Z, alpha*xi, 'soft')

def _prox_W_lvggm_admm(Z, beta=1, xi=1):
    '''
        solves for proximal minimization
          min_W  0.5*||W -Z||**2/xi + beta||W||_{2,1}
    '''
    import pywt
    n,m = Z.shape
    result  = Z.copy()
    for i in range(n):
        tmp = Z[i,:]
        tmp_norm = np.linalg.norm(tmp)
        tmp_ratio = (1-xi*beta/tmp_norm)
        tmp_weight = np.max([tmp_ratio, 0])
        tmp_row = tmp_weight*tmp
        result[i,:] = tmp_row
    return result

def _prox_P_lvggm_admm(Z, Z_w, Theta, xi=1, xi_w=1):
    '''
        solves for proximal minimization

            min_P 0.5||P- Z||**2/xi + 0.5*||P-Theta*Z_w||**2/xi_w

    '''
    s = float(xi/xi_w)
    eigval, eigvec = np.linalg.eigh(Theta)
    eigvec = np.asarray(eigvec)
    index_sig = np.argsort(eigval)[::-1]
    eigval = eigval[index_sig]
    eigvec = eigvec[:,index_sig]
    def eig_l2_Z(eigval, s):
        return (1/(1+s*eigval**2))
    def eig_l2_Z_w(eigval, s):
        return (s*eigval/(1+s*eigval**2))

    eig_Z_vec = np.vectorize(eig_l2_Z)
    eig_Z_w_vec = np.vectorize(eig_l2_Z_w)
    
    eigval_trans_Z = eig_Z_vec(eigval, s)
    eigval_trans_Z_w = eig_Z_w_vec(eigval, s)


    Z_trans = np.dot(eigval_trans_Z*eigvec, np.dot(eigvec.T, Z))
    Z_w_trans = np.dot(eigval_trans_Z_w*eigvec, np.dot(eigvec.T, Z_w))

    return (Z_trans + Z_w_trans)

#---------------------------------------------------------------

def dilat_lvggm_ccg_sub_admm_diag(S, alpha, beta, covariance_h, precision_h, mu=1, max_iter_in=1000, threshold_in=1e-6, init_R=None, return_costs=False, verbose=False):

    '''
         implementation of sub-routine of the DiLat-GGM using ADMM where precision_h is diagonal
    '''
    if np.linalg.norm(S-S.T) > 1e-3:
        raise ValueError("Covariance matrix should be symmetric.")

    n = S.shape[0]
    #n1 = covariance.shape[0]
    n2 = covariance_h.shape[0]
    n1 = n - n2
    #if n != (n1+n2):
    if n1 < 0:
        raise ValueError("dimension mismatch n=%d, n1=%d, n2=%d" % (n,n1,n2))

    covariance_h = np.diag(np.diag(covariance_h))
    precision_h = np.diag(np.diag(precision_h))
    
    # initialization
    R = np.zeros((n,n))
    if init_R is None:
        R[np.ix_(np.arange(n1,n), np.arange(n1,n))] = covariance_h.copy()
        tmp = np.random.randn(n1,n2)
        max_eig = np.linalg.norm(( np.dot(tmp, np.dot(precision_h, tmp.T)) ),2)
        R[np.ix_(np.arange(n1), np.arange(n1, n))] = tmp
        R[np.ix_(np.arange(n1,n), np.arange(n1))] = tmp.T
        R[np.ix_(np.arange(n1), np.arange(n1))] = (max_eig+0.1)*np.eye(n1)
    else:
        R = init_R.copy()
 
    P = np.zeros((n,n))
    P = R.copy()
    Lambda = np.zeros((n, n))
    cost_hists = list()
    diff_pre_hists = list()
    constraint_loss_hists = list()
    
    for t in range(max_iter_in):
        R_pre = R
        P_pre = P

        T = R + mu*Lambda
        # proxmial on Sparsity 
        P1 = _prox_S_lvgmm_admm(T[np.ix_(np.arange(n1), np.arange(n1))], alpha, mu)
        # proximal on row sparisty
        P21 = _prox_W_lvggm_admm(np.dot(precision_h, T[np.ix_(np.arange(n1,n), np.arange(n1))]), beta, mu)
        # update
        P[np.ix_(np.arange(n1), np.arange(n1))] = P1.copy()
        P[np.ix_(np.arange(n1,n), np.arange(n1))] = P21.copy()
        P[np.ix_(np.arange(n1), np.arange(n1,n))] = P21.copy().T
        P[np.ix_(np.arange(n1,n), np.arange(n1,n))] = covariance_h.copy()
        
        Q = P - mu*Lambda
        # proximal on R
        R = _prox_R_lvgmm_admm(S, Q, mu)
        # update dual variables
        Lambda = Lambda + 1/mu*(R - P)

        diff_pre = np.linalg.norm(R-R_pre) + np.linalg.norm(P-P_pre)
        constraint_loss = np.linalg.norm(R-P)

        tmp = np.dot(precision_h, R[np.ix_(np.arange(n1,n), np.arange(n1))])
        row_l2_norm = np.dot(tmp, tmp.T)
        l12_norm = np.sqrt(np.diag(row_l2_norm))
        cost = -2. * log_likelihood(S, R) + alpha*(np.abs(R[np.ix_(np.arange(n1), np.arange(n1))]).sum() - np.abs(np.diag(R[np.ix_(np.arange(n1), np.arange(n1))])).sum()) + beta*l12_norm.sum()
        if return_costs:
            cost_hists.append(cost)
            constraint_loss_hists.append(constraint_loss)
            diff_pre_hists.append(diff_pre)

        if verbose:
            if t == 0:
                print("iterations | cost  | succ-diff | constraint-loss |")
            
            print("    %d     |  %.4e |  %.4e     |      %.4e       |" \
                       % (t, cost, diff_pre, constraint_loss))
        if diff_pre/3 < threshold_in and constraint_loss < threshold_in:
            if verbose: print("optimal solution found.")
            break

    if return_costs:
        return (R, P, cost_hists, constraint_loss_hists, diff_pre_hists)
    else:
        return (R, P)
        
       
def dilat_lvggm_ccg_sub_admm_full(S, alpha, beta, covariance_h, precision_h, mu=1, mu_w=1, max_iter_in=1000, threshold_in=1e-6, init_R=None, return_costs=False, verbose=False):

    '''
         implementation of sub-routine of the DiLat-GGM using ADMM where precision_h is full positive definite matrix
    '''
    if np.linalg.norm(S-S.T) > 1e-3:
        raise ValueError("Covariance matrix should be symmetric.")

    n = S.shape[0]
    #n1 = covariance.shape[0]
    n2 = covariance_h.shape[0]
    n1 = n - n2
    #if n != (n1+n2):
    if n1 < 0:
        raise ValueError("dimension mismatch n=%d, n1=%d, n2=%d" % (n,n1,n2))

    eigvals_h, eigvecs_h = np.linalg.eigh(precision_h)
    index_sig0 = np.argsort(eigvals_h)
    eigvals_h = eigvals_h[index_sig0[::-1]]
    eigvecs_h = eigvecs_h[:,index_sig0[::-1]]
    eigvecs_h = np.asarray(eigvecs_h)
    # initialization
    R = np.zeros((n,n))
    if init_R is None:
        R[np.ix_(np.arange(n1,n), np.arange(n1,n))] = covariance_h.copy()
        tmp = np.random.randn(n1,n2)
        max_eig = np.linalg.norm(( np.dot(tmp, np.dot(precision_h, tmp.T)) ),2)
        R[np.ix_(np.arange(n1), np.arange(n1, n))] = tmp
        R[np.ix_(np.arange(n1,n), np.arange(n1))] = tmp.T
        R[np.ix_(np.arange(n1), np.arange(n1))] = (max_eig+0.1)*np.eye(n1)
    else:
        R = init_R.copy()
 
    P = np.zeros((n,n))
    P = R.copy()
    Lambda = np.zeros((n, n))

    P21 = P[np.ix_(np.arange(n1,n), np.arange(n1))].copy()
    W = np.zeros((n2, n1))
    W = np.dot(precision_h, P21)
    Lambda_w = np.zeros((n2, n1))

    cost_hists = list()
    diff_pre_hists = list()
    constraint_loss_hists = list()
   
 
    for t in range(max_iter_in):
        R_pre = R
        P_pre = P
        W_pre = W

        T = R + mu*Lambda
        # proxmial on Sparsity 
        P1 = _prox_S_lvgmm_admm(T[np.ix_(np.arange(n1), np.arange(n1))], alpha, mu)

        #================= for row sparsity ============================
        # proximal on row sparisty

        V = np.dot(precision_h, P21) - mu_w*Lambda_w
        W = _prox_W_lvggm_admm(V, beta, mu_w)

        U = W + mu_w*Lambda_w

        P21 = _prox_P_lvggm_admm(T[np.ix_(np.arange(n1,n), np.arange(n1))], U, precision_h, mu, mu_w)
        
        # update Lambda_w
        Lambda_w = Lambda_w + 1/(mu_w)*(W - np.dot(precision_h, P21))

        #===============================================================
        # update
        P[np.ix_(np.arange(n1), np.arange(n1))] = P1.copy()
        P[np.ix_(np.arange(n1,n), np.arange(n1))] = P21.copy()
        P[np.ix_(np.arange(n1), np.arange(n1,n))] = P21.copy().T
        P[np.ix_(np.arange(n1,n), np.arange(n1,n))] = covariance_h.copy()
        
        Q = P - mu*Lambda
        # proximal on R
        R = _prox_R_lvgmm_admm(S, Q, mu)
        # update dual variables
        Lambda = Lambda + 1/mu*(R - P)

        diff_pre = np.linalg.norm(R-R_pre) + np.linalg.norm(P-P_pre) + np.linalg.norm(W-W_pre)
        constraint_loss = np.linalg.norm(R-P) + np.linalg.norm(W - np.dot(precision_h, P[np.ix_(np.arange(n1,n), np.arange(n1))]))

        tmp = np.dot(precision_h, R[np.ix_(np.arange(n1,n), np.arange(n1))])
        row_l2_norm = np.dot(tmp, tmp.T)
        l12_norm = np.sqrt(np.diag(row_l2_norm))
        cost = -2. * log_likelihood(S, R) + alpha*(np.abs(R[np.ix_(np.arange(n1), np.arange(n1))]).sum() - np.abs(np.diag(R[np.ix_(np.arange(n1), np.arange(n1))])).sum()) + beta*l12_norm.sum()
        if return_costs:
            cost_hists.append(cost)
            constraint_loss_hists.append(constraint_loss)
            diff_pre_hists.append(diff_pre)

        if verbose:
            if t == 0:
                print("iterations | cost  | succ-diff | constraint-loss |")
            
            print("    %d     |  %.4e |  %.4e     |      %.4e       |" \
                       % (t, cost, diff_pre, constraint_loss))
        if diff_pre/3 < threshold_in and constraint_loss < threshold_in:
            if verbose: print("optimal solution found.")
            break

    if return_costs:
        return (R, P, W,  cost_hists, constraint_loss_hists, diff_pre_hists)
    else:
        return (R, P, W)
 #def latent_variable_gmm_cvx_v2(X_o, alpha=1, gamma= 1, beta=1, S_init=None, verbose=False):
 #    '''
 #          A cvx implementation of  the Latent Variable Gaussian Graphical Model 
 #      
 #       see review of  "Venkat Chandrasekaran, Pablo A Parrilo, and Alan S Willsky. Latent variable graphical model selection via convex optimization. The Annals of Statistics, 40(4):1935–1967, 2012."
 #    
 #        
 #           min_{S, L} -log det (S-L) + trace(emp_Cov*(S-L)) + alpha*lambda_s*\|S\|_{1} + alpha*\|L\|_{*}
 #                  s.t.  S-L \succeq 0
 #                        L \succeq 0
 #
 #         return S, L
 #
 #    '''
 #    n, m = X_o.shape
 #    emp_cov = np.cov(X_o)
 #    if alpha == 0:
 #        if return_costs:
 #            precision = np.linalg.pinv(emp_cov)
 #            cost = - 2. * log_likelihood(emp_cov, precision)
 #            cost += n_features * np.log(2 * np.pi)
 #            d_gap = np.sum(emp_cov * precision) - n
 #            return emp_cov, precision, (cost, d_gap)
 #        else:
 #            return emp_cov, np.linalg.pinv(emp_cov)
 #
 #    costs = list()
 #    if S_init is None:
 #        covariance_o = emp_cov.copy()
 #    else:
 #        covariance_o = S_init.copy()
 #    # As a trivial regularization (Tikhonov like), we scale down the
 #    # off-diagonal coefficients of our starting point: This is needed, as
 #    # in the cross-validation the cov_init can easily be
 #    # ill-conditioned, and the CV loop blows. Beside, this takes
 #    # conservative stand-point on the initial conditions, and it tends to
 #    # make the convergence go faster.
 #
 #    covariance_o *= 0.95
 #    diagonal_o = emp_cov.flat[::n+1]
 #    covariance_o.flat[::n+1] = diagonal_o
 #   
 #    
 #    eigvals_o, eigvecs_o = np.linalg.eigh(covariance_o)
 #    index_sig1 = np.argsort(eigvals_o)[::-1] #arange in decreasing order
 #    eigvals_o = eigvals_o[index_sig1]
 #    eigvecs_o = eigvecs_o[:,index_sig1]
 #    eigvecs_o = np.asarray(eigvecs_o)
 #    # enforce eig >= 0
 #    thresh = eigvals_o[5]
 #    eigvals_o = threshold(eigvals_o, thresh, 'greater') # L is enforced to be psd
 #    covariance_principal = np.dot(eigvals_o*eigvecs_o, eigvecs_o.T)
 #    #transformed_eigvals_o = eigvals_o.copy()
 #    #transformed_eigvals_o = np.sqrt(np.multiply(transformed_eigvals_, eigvals_Ls[np.arange(n2)]))
 # 
 #    # define the low-rank term L and sparse term S
 #    L = cvx.Semidef(n)
 #    S = cvx.Symmetric(n)
 #    # define the SDP problem 
 #    objective = cvx.Minimize(- cvx.log_det(S-L) + cvx.trace(covariance_o*(S-L)) + alpha*cvx.norm(S,1) + gamma*cvx.norm(L, "nuc") + beta*cvx.mixed_norm( -covariance_principal*L, 2, 1))
 #    constraints = [S-L >> 0]
 #  
 #    # solve the problem
 #    problem = cvx.Problem(objective, constraints)
 #    problem.solve(verbose = verbose)
 #
 #    return (S.value, L.value)