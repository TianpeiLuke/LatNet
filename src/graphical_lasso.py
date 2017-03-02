
# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import solve, cond, lstsq
import cvxpy as cvx
from sklearn.linear_model import lars_path, lasso_path, ridge_regression
from sklearn.covariance.empirical_covariance_ import log_likelihood

import matplotlib.pyplot as plt
from time import gmtime, strftime

#from adaptive_lasso import lasso_path_adaptive 
#=============================================================================================================================
#========================   Auxilary function ==========================================================


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

def _objective_mask(mle, precision_, alpha, mask):
    """Evaluation of the graph-lasso objective function
    the objective function is made of a shifted scaled version of the
    normalized log-likelihood (i.e. its empirical mean over the samples) and a
    penalisation term to promote sparsity
    """
    p = precision_.shape[0]
    cost = - 2. * log_likelihood(mle, precision_) + p * np.log(2 * np.pi)
    cost += alpha * (np.abs(mask*precision_).sum()
                     - np.abs(np.diag(mask*precision_)).sum())
    return cost
#===============================================================================================================================

def sparse_inv_cov_glasso(X, alpha=1, S_init=None, max_iter=100, verbose=False, convg_threshold=1e-3, return_costs=False):
    '''
         inverse covariance estimation by maximum log-likelihood estimate given X

         -log p(X | J) + alpha ||J||_{1} := -log(det(J)) + tr(S*J) + alpha*||J||_{1}

         S:= np.dot(X,X.T)/m

         using graphical Lasso 

         Friedman, Jerome, Trevor Hastie, and Robert Tibshirani. 
         "Sparse inverse covariance estimation with the graphical lasso." 
         Biostatistics 9, no. 3 (2008): 432-441.

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
                #update the precision matrix
                #  coeffs = cov(not_i)^-1*cov_i,noti
                #  precision_ii = 1/(cov_ii - cov_i,noti.T*cov(not_i)^{-1}*cov_i,noti )
                #  precision_noti,i = -cov(not_i)^-1*cov_i,noti*precision_ii
                precision[i,i]   = 1./( covariance[i,i] - np.dot(covariance[indices != i, i], coeffs))
                precision[indices != i, i] = -precision[i,i] * coeffs
                precision[i, indices != i] = -precision[i,i] * coeffs
                temp_coeffs = np.dot(covariance_11, coeffs)
                #  cov_i,noti = cov(not_i)*coeffs
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

#==========================================================================#
#                                                                          #
#             implementation of adaptive lasso                             #
#==========================================================================#

#def sparse_inv_cov_glasso_mask(X, mask=None,  alpha=1, S_init=None, max_iter=100, verbose=False, convg_threshold=1e-3, return_costs=False):
#    '''
#        inverse covariance estimation by maximum log-likelihood estimate given X
#
#        -log p(X | J) + alpha ||J||_{1} := -log(det(J)) + tr(S*J) + alpha*||mask*J||_{1}
#
#        S:= np.dot(X,X.T)/m
#
#        using gLasso with mask
#
#        Friedman, Jerome, Trevor Hastie, and Robert Tibshirani. 
#        "Sparse inverse covariance estimation with the graphical lasso." 
#        Biostatistics 9, no. 3 (2008): 432-441.
#
#    '''
#    if mask is None:
#        return sparse_inv_cov_glasso(X, alpha, S_init, max_iter, verbose, convg_threshold, return_costs)
#   
#    # mask must be symmetric 
#    if not np.array_equal(mask.T, mask):
#        raise ValueError("Mask must be symmetric!")
#    
#
#    n, m = X.shape
#    emp_cov = np.cov(X)
#    if emp_cov.shape != mask.shape:
#        raise ValueError("mask must have the same size as the covariance of X")
#
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
#        covariance = emp_cov.copy()
#    else:
#        covariance = S_init.copy()
#    # As a trivial regularization (Tikhonov like), we scale down the
#    # off-diagonal coefficients of our starting point: This is needed, as
#    # in the cross-validation the cov_init can easily be
#    # ill-conditioned, and the CV loop blows. Beside, this takes
#    # conservative stand-point on the initial conditions, and it tends to
#    # make the convergence go faster.
#
#    covariance *= 0.95
#    diagonal = emp_cov.flat[::n+1]
#    covariance.flat[::n+1] = diagonal
#
# #+ alpha*np.eye(n, dtype=X.dtype)
#    mle_estimate = emp_cov.copy()
#    precision = np.linalg.pinv(covariance)
#    plot_mat(precision, [-2,2])
#    indices = np.arange(n)
#    try:
#        d_gap = np.inf
#        for t in range(max_iter):
#            for i in range(n):
#                #index = [x for x in indices if x !=i]
#                covariance_11 = np.ascontiguousarray(
#                                     covariance[indices != i].T[indices != i])
#                covariance_12 = mle_estimate[indices != i , i]
#                #solve adaptive lasso for each column 
#                alpha_min = alpha/(n-1)
#                mask_i = mask[indices!=i, i]
#                _, coeffs, _ = lasso_path_adaptive(X=covariance_11, y=covariance_12, 
#                                             mask=mask_i, eps=1e-3, 
#                                             n_alphas=None, alphas=[alpha_min],
#                                             precompute=covariance_11, Xy=covariance_12,
#                                             copy_X=True, coef_init= np.zeros(n-1),
#                                  verbose=verbose, return_n_iter=False, positive=False)
#                
#                #update the precision matrix
#                coeffs = np.squeeze(coeffs)
#                precision[i,i]   = 1./( covariance[i,i] - np.dot(covariance[indices != i, i], coeffs))
#                precision[indices != i, i] = -precision[i,i] * coeffs
#                precision[i, indices != i] = -precision[i,i] * coeffs
#                temp_coeffs = np.dot(covariance_11, coeffs)
#                covariance[indices != i ,i] = temp_coeffs
#                covariance[i, indices != i] = temp_coeffs
#    
#            plot_mat(precision, [-2,2])
#            d_gap = _dual_gap(mle_estimate, precision, alpha)
#            cost = _objective_mask(mle_estimate, precision, alpha, mask)
#            #print("iter "+ str(t) +"cost " + str(cost))
#            
#            if verbose:
#                print(
#                    '[graph_lasso] Iteration % 3i, cost % 3.2e, dual gap %.3e'
#                    % (t, cost, d_gap))
#            if return_costs:
#                costs.append((cost, d_gap))
#            if np.abs(d_gap) < convg_threshold:
#                break
#            if not np.isfinite(cost) and t > 0:
#                    raise FloatingPointError('Non SPD result: the system is '
#                                             'too ill-conditioned for this solver')
#        else:
#            #this triggers if not break command occurs
#            print("The algorithm did not coverge. Try increasing the max number of iterations.")
#    except FloatingPointError as e:
#        e.args = (e.args[0]
#                  + '. The system is too ill-conditioned for this solver',)
#        raise e
#
#    if return_costs:
#        return covariance, precision, costs
#    else:
#        return (covariance ,  precision )




def sparse_inv_cov_glasso_partial(X, observed_indices=None,  alpha=1, S_init=None, max_iter=100, verbose=False, convg_threshold=1e-3, return_costs=False):
    '''
        inverse covariance estimation by maximum log-likelihood estimate given X

        -log p(X | J) + alpha ||J||_{1} := -log(det(J)) + tr(S*J) + alpha*||mask*J||_{1}

        S:= np.dot(X,X.T)/m

        using gLasso with mask

        Friedman, Jerome, Trevor Hastie, and Robert Tibshirani. 
        "Sparse inverse covariance estimation with the graphical lasso." 
        Biostatistics 9, no. 3 (2008): 432-441.

    '''
    if observed_indices is None:
        return sparse_inv_cov_glasso(X, alpha, S_init, max_iter, verbose, convg_threshold, return_costs)
   
    n, m = X.shape
    
    if not set(observed_indices).issubset(set(list(range(n)))):
        raise ValueError("observed indices should be in range(n)")
    
    emp_cov = np.cov(X)

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
    plot_mat(precision, [-2,2])
    indices = np.arange(n)
    try:
        d_gap = np.inf
        for t in range(max_iter):
            for i in np.arange(n):
                if i in observed_indices:
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
                    
                    #update the precision matrix
                    precision[i,i]   = 1./( covariance[i,i] - np.dot(covariance[indices != i, i], coeffs))
                    precision[indices != i, i] = -precision[i,i] * coeffs
                    precision[i, indices != i] = -precision[i,i] * coeffs
                    temp_coeffs = np.dot(covariance_11, coeffs)
                    covariance[indices != i ,i] = temp_coeffs
                    covariance[i, indices != i] = temp_coeffs
                else:
                    covariance_11 = np.ascontiguousarray(
                                         covariance[indices != i].T[indices != i])
                    covariance_12 = mle_estimate[indices != i , i]
                    print(cond(covariance_11)) 
                    if cond(covariance_11) < 100:
                        coeffs = solve(covariance_11, covariance_12)
                    else:
                        X_o = X[indices != i,:]
                        X_h = X[i,:]
                        coeffs, _, _, _ = lstsq(X_o.T, X_h.T) #solve for least square
                    #print(coeffs)
                    #coeffs = np.asarray(coeffs)
                    precision[i,i]   = 1./( covariance[i,i] - np.dot(covariance[indices != i, i], coeffs))
                    precision[indices != i, i] = -precision[i,i] * coeffs
                    precision[i, indices != i] = -precision[i,i] * coeffs
                    temp_coeffs = np.dot(covariance_11, coeffs)
                    covariance[indices != i ,i] = temp_coeffs
                    covariance[i, indices != i] = temp_coeffs
                   
            plot_mat(precision, [-2,2])
            d_gap = _dual_gap(mle_estimate, precision, alpha)
            cost = _objective(mle_estimate, precision, alpha)
            #print("iter "+ str(t) +"cost " + str(cost))
            
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











def plot_mat(mat, vrange, savefigure=False):
    fig2= plt.figure(2)
    ax = fig2.add_subplot(111)
    cax = ax.matshow(mat, vmin=vrange[0], vmax=vrange[1])
    fig2.colorbar(cax)
    
    plt.show()
    filename = "../figures/"+strftime("%d%m%Y_%H%M%S", gmtime()) + "_precision_laplacian.eps"
    if savefigure : fig2.savefig(filename)
