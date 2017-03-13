
# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import solve, cond, lstsq
import cvxpy as cvx
from sklearn.linear_model import lars_path, lasso_path, ridge_regression
from sklearn.covariance.empirical_covariance_ import log_likelihood

import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.optimize import minimize
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

def _dual_gap_Laplacian(emp_cov, precision):
    """Expression of the dual gap convergence criterion
    The specific definition is given in Duchi "Projected Subgradient Methods
    for Learning Sparse Gaussians".
    """
    gap = np.sum(emp_cov * precision)
    gap -= precision.shape[0]
    #gap += (np.abs(np.diag(precision)).sum())
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


#==============================================================================
def dual_optimizer_gle(Q, q, stepsize=1e-2, max_iters=2000, threshold=1e-5, verbose=False):
    '''    
         coeff = argmin_{z>=0} (z+covariance_12).T*precision_{11}*(z+covariance_12)
    '''
    n = len(q)
    coeffs = np.zeros((n,))
    def objective(Q, q, coeffs):
        return np.dot(np.dot((q+coeffs).T, Q), (q+coeffs))

    gap_seq = list([])
    for t in range(max_iters):
        pre_objective_value = objective(Q, q, coeffs)
        gradient = np.dot(Q, (q+coeffs))
        coeffs = coeffs - stepsize*gradient
        coeffs[coeffs < 0] = 0
        objective_value = objective(Q, q, coeffs)
        gap = np.linalg.norm(pre_objective_value - objective_value)
        gap_seq.append(gap)
        if t >=1 :
            if gap_seq[-1] > gap_seq[-2]:
                break
        if verbose: print("gap %.4f" % (gap))
        if gap <= threshold:
            break
    return coeffs

def dual_optimizer_gle_cvx(Q, q, verbose=False):
    '''    
         coeff = argmin_{z>=0} (z+covariance_12).T*precision_{11}*(z+covariance_12)
    '''
    n = len(q)
    x = cvx.Variable(n)
    objective = cvx.Minimize(cvx.quad_form(x+q, Q))
    constraints = [x >= 0]
    problem = cvx.Problem(objective, constraints)
    problem.solve(verbose = verbose)
    coeffs = np.squeeze(np.asarray(x.value))
    return coeffs




def generalized_Laplacian_estimate(X, S_init=None, max_iter=100, verbose=False, convg_threshold=1e-3, return_costs=False):
    '''
        Generalized Laplacian estimate
          

        find the generalized Laplacian matrix by minimize log-likelihood

       -log(deg((J))) + tr(S*J)

        s.t.   J \ge 0
               J_{i,j} \le 0,  



        implemenation based on    

        Pavez, Eduardo, and Antonio Ortega. 
        "Generalized Laplacian precision matrix estimation for graph signal processing." In Acoustics, Speech and Signal Processing (ICASSP), 2016 IEEE International Conference on, pp. 6350-6354. IEEE, 2016.

    '''
    n, m = X.shape
    S = np.cov(X)

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

    #covariance *= 0.9
    diagonal = S.flat[::n+1]
    #covariance.flat[::n+1] = diagonal

 #+ alpha*np.eye(n, dtype=X.dtype)
    mle_estimate = S.copy()
    precision = np.diag(1.0/diagonal)  #np.linalg.pinv(covariance)
    indices = np.arange(n)
    try:
        d_gap = np.inf
        for t in range(max_iter):
            for i in range(n):
                #index = [x for x in indices if x !=i]
                precision_11 = np.ascontiguousarray(
                                     precision[indices != i].T[indices != i])
                covariance_12 = mle_estimate[indices != i , i]
                # solve dual optimization problem
                #  coeffs = argmin_{z>=0} (z+covariance_12).T*precision_{11}*(z+covariance_12)
                coeffs = dual_optimizer_gle_cvx(precision_11, covariance_12, verbose)
                coeffs[coeffs<1e-4] = 0
                coeffs_plus_covariance = coeffs + covariance_12
                #dual_optimizer_gle(precision_11, covariance_12.copy(), stepsize=1e-3, max_iters=2000, threshold=1e-4)
                #update the precision matrix
                #  coeffs = cov(not_i)^-1*cov_i,noti
                #  precision_ii =  1/covariance[i,i] - precision[-i,i].T*precision[-i,-i]^-1*prcision[-i,i]
                #  precision_noti,i = -cov(not_i)^-1*cov_i,noti*precision_ii
                q12 = -np.dot(precision_11, coeffs_plus_covariance)/mle_estimate[i,i]
                q12[coeffs>1e-4] = 0
                precision[indices != i, i] = q12
                precision[i, indices != i] = q12.T
                precision[i,i]   = 1.0/mle_estimate[i,i] - 1.0/mle_estimate[i,i]* np.dot(q12.T, coeffs_plus_covariance)
    
            d_gap = _dual_gap_Laplacian(mle_estimate, precision) 
            cost = _objective(mle_estimate, precision, 0)
            if verbose:
                print(
                    '[dp-glasso] Iteration % 3i, cost % 3.2e, dual gap %.3e'
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
        return precision, costs
    else:
        return precision
