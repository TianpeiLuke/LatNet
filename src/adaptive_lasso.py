
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from sklearn.linear_model import lars_path, lasso_path, ridge_regression
from sklearn.linear_model.base import  _pre_fit
from sklearn.utils import check_array
from sklearn.covariance.empirical_covariance_ import log_likelihood
from copy import deepcopy

#import pyximport; pyximport.install()
import sys
sys.path.append('/home/tianpei/Dropbox/Codes/Python/LatNet/src/')
import LatNet.src.cd_fast_adaptive




#============================================================================
#-----------   lasso method via coordinate descent

def soft_thresholding(x, tau):
    '''
         
    '''
    if x > 0 and tau < abs(x):
        return x - tau
    elif x < 0 and tau < abs(x):
        return x + tau
    else:
        return 0 



###############################################################################
# Paths functions

def _alpha_grid(X, y, Xy=None, l1_ratio=1.0, fit_intercept=True,
                eps=1e-3, n_alphas=100, normalize=False, copy_X=True):
    """ Compute the grid of alpha values for elastic net parameter search
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication
    y : ndarray, shape (n_samples,)
        Target values
    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed.
    l1_ratio : float
        The elastic net mixing parameter, with ``0 < l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. (currently not
        supported) ``For l1_ratio = 1`` it is an L1 penalty. For
        ``0 < l1_ratio <1``, the penalty is a combination of L1 and L2.
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``
    n_alphas : int, optional
        Number of alphas along the regularization path
    fit_intercept : boolean, default True
        Whether to fit an intercept or not
    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.
        This parameter is ignored when ``fit_intercept`` is set to ``False``.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        :class:`preprocessing.StandardScaler` before calling ``fit`` on an estimator
        with ``normalize=False``.
    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    """
    if l1_ratio == 0:
        raise ValueError("Automatic alpha grid generation is not supported for"
                         " l1_ratio=0. Please supply a grid by providing "
                         "your estimator with the appropriate `alphas=` "
                         "argument.")
    n_samples = len(y)

    sparse_center = False
    if Xy is None:
        X_sparse = sparse.isspmatrix(X)
        sparse_center = X_sparse and (fit_intercept or normalize)
        X = check_array(X, 'csc',
                        copy=(copy_X and fit_intercept and not X_sparse))
        if not X_sparse:
            # X can be touched inplace thanks to the above line
            X, y, _, _, _ = _preprocess_data(X, y, fit_intercept,
                                             normalize, copy=False)
        Xy = safe_sparse_dot(X.T, y, dense_output=True)

        if sparse_center:
            # Workaround to find alpha_max for sparse matrices.
            # since we should not destroy the sparsity of such matrices.
            _, _, X_offset, _, X_scale = _preprocess_data(X, y, fit_intercept,
                                                      normalize,
                                                      return_mean=True)
            mean_dot = X_offset * np.sum(y)

    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    if sparse_center:
        if fit_intercept:
            Xy -= mean_dot[:, np.newaxis]
        if normalize:
            Xy /= X_scale[:, np.newaxis]

    alpha_max = (np.sqrt(np.sum(Xy ** 2, axis=1)).max() /
                 (n_samples * l1_ratio))

    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    return np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max),
                       num=n_alphas)[::-1]


def lasso_path_adaptive(X, y, mask,  eps=1e-3, n_alphas=100, alphas=None,
               precompute='auto', Xy=None, copy_X=True, coef_init=None,
               verbose=False, return_n_iter=False, positive=False, **params):
    """Compute Lasso path with coordinate descent
    The Lasso optimization function varies for mono and multi-outputs.
    For mono-output tasks it is::
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    For multi-output tasks it is::
        (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21
    Where::
        ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}
    i.e. the sum of norm of each row.
    Read more in the :ref:`User Guide <lasso>`.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication. If ``y`` is mono-output then ``X``
        can be sparse.
    y : ndarray, shape (n_samples,), or (n_samples, n_outputs)
        Target values
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``
    n_alphas : int, optional
        Number of alphas along the regularization path
    alphas : ndarray, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically
    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.
    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.
    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    coef_init : array, shape (n_features, ) | None
        The initial values of the coefficients.
    verbose : bool or integer
        Amount of verbosity.
    params : kwargs
        keyword arguments passed to the coordinate descent solver.
    positive : bool, default False
        If set to True, forces coefficients to be positive.
    return_n_iter : bool
        whether to return the number of iterations or not.
    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.
    coefs : array, shape (n_features, n_alphas) or \
            (n_outputs, n_features, n_alphas)
        Coefficients along the path.
    dual_gaps : array, shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.
    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
    Notes
    -----
    See examples/linear_model/plot_lasso_coordinate_descent_path.py
    for an example.
    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.
    Note that in certain cases, the Lars solver may be significantly
    faster to implement this functionality. In particular, linear
    interpolation can be used to retrieve model coefficients between the
    values output by lars_path
    Examples
    ---------
    Comparing lasso_path and lars_path with interpolation:
    #>>> X = np.array([[1, 2, 3.1], [2.3, 5.4, 4.3]]).T
    #>>> y = np.array([1, 2, 3.1])
    #>>> # Use lasso_path to compute a coefficient path
    #>>> _, coef_path, _ = lasso_path(X, y, alphas=[5., 1., .5])
    #>>> print(coef_path)
    [[ 0.          0.          0.46874778]
     [ 0.2159048   0.4425765   0.23689075]]
    #>>> # Now use lars_path and 1D linear interpolation to compute the
    #>>> # same path
    #>>> from sklearn.linear_model import lars_path
    #>>> alphas, active, coef_path_lars = lars_path(X, y, method='lasso')
    #>>> from scipy import interpolate
    #>>> coef_path_continuous = interpolate.interp1d(alphas[::-1],
    ...                                             coef_path_lars[:, ::-1])
    #>>> print(coef_path_continuous([5., 1., .5]))
    [[ 0.          0.          0.46915237]
     [ 0.2159048   0.4425765   0.23668876]]
    See also
    --------
    lars_path
    Lasso
    LassoLars
    LassoCV
    LassoLarsCV
    sklearn.decomposition.sparse_encode
    """
    return enet_path_adaptive(X, y, mask, l1_ratio=1., eps=eps, n_alphas=n_alphas,
                     alphas=alphas, precompute=precompute, Xy=Xy,
                     copy_X=copy_X, coef_init=coef_init, verbose=verbose,
                     positive=positive, return_n_iter=return_n_iter, **params)




def enet_path_adaptive(X, y, mask, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
              precompute='auto', Xy=None, copy_X=True, coef_init=None,
              verbose=False, return_n_iter=False, positive=False,
              check_input=True, **params):
    """Compute elastic net path with coordinate descent
    The elastic net optimization function varies for mono and multi-outputs.
    For mono-output tasks it is::
        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
    For multi-output tasks it is::
        (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
    Where::
        ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}
    i.e. the sum of norm of each row.
    Read more in the :ref:`User Guide <elastic_net>`.
    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication. If ``y`` is mono-output then ``X``
        can be sparse.
    y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Target values
    l1_ratio : float, optional
        float between 0 and 1 passed to elastic net (scaling between
        l1 and l2 penalties). ``l1_ratio=1`` corresponds to the Lasso
    eps : float
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``
    n_alphas : int, optional
        Number of alphas along the regularization path
    alphas : ndarray, optional
        List of alphas where to compute the models.
        If None alphas are set automatically
    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.
    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.
    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    coef_init : array, shape (n_features, ) | None
        The initial values of the coefficients.
    verbose : bool or integer
        Amount of verbosity.
    params : kwargs
        keyword arguments passed to the coordinate descent solver.
    return_n_iter : bool
        whether to return the number of iterations or not.
    positive : bool, default False
        If set to True, forces coefficients to be positive.
    check_input : bool, default True
        Skip input validation checks, including the Gram matrix when provided
        assuming there are handled by the caller when check_input=False.
    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.
    coefs : array, shape (n_features, n_alphas) or \
            (n_outputs, n_features, n_alphas)
        Coefficients along the path.
    dual_gaps : array, shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.
    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
        (Is returned when ``return_n_iter`` is set to True).
    Notes
    -----
    See examples/linear_model/plot_lasso_coordinate_descent_path.py for an example.
    See also
    --------
    MultiTaskElasticNet
    MultiTaskElasticNetCV
    ElasticNet
    ElasticNetCV
    """
    # We expect X and y to be already Fortran ordered when bypassing
    # checks
    if check_input:
        X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                        order='F', copy=copy_X)
        y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                        ensure_2d=False)
        if Xy is not None:
            # Xy should be a 1d contiguous array or a 2D C ordered array
            Xy = check_array(Xy, dtype=X.dtype.type, order='C', copy=False,
                             ensure_2d=False)

    n_samples, n_features = X.shape

    multi_output = False
    if y.ndim != 1:
        multi_output = True
        _, n_outputs = y.shape

    # MultiTaskElasticNet does not support sparse matrices
    if not multi_output and sparse.isspmatrix(X):
        if 'X_offset' in params:
            # As sparse matrices are not actually centered we need this
            # to be passed to the CD solver.
            X_sparse_scaling = params['X_offset'] / params['X_scale']
            X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
        else:
            X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

    # X should be normalized and fit already if function is called
    # from ElasticNet.fit
    if check_input:
        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, Xy, precompute, normalize=False,
                     fit_intercept=False, copy=False)

    if len(mask) != n_samples:
        tmp_alpha = np.zeros(n_samples)
        tmp_alpha[np.arange(len(mask))] = mask
        mask = tmp_alpha


    if alphas is None:
        # No need to normalize of fit_intercept: it has been done
        # above
        alphas = _alpha_grid(X, y, Xy=Xy, l1_ratio=l1_ratio,
                         fit_intercept=False, eps=eps, n_alphas=n_alphas,
                         normalize=False, copy_X=False)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    dual_gaps = np.empty(n_alphas)
    n_iters = []

    rng = check_random_state(params.get('random_state', None))
    selection = params.get('selection', 'cyclic')
    if selection not in ['random', 'cyclic']:
        raise ValueError("selection should be either random or cyclic.")
    random = (selection == 'random')

    if not multi_output:
        coefs = np.empty((n_features, n_alphas), dtype=X.dtype)
    else:
        coefs = np.empty((n_outputs, n_features, n_alphas),
                         dtype=X.dtype)

    if coef_init is None:
        coef_ = np.asfortranarray(np.zeros(coefs.shape[:-1], dtype=X.dtype))
    else:
        coef_ = np.asfortranarray(coef_init, dtype=X.dtype)

    for i, alpha in enumerate(alphas):
        # a vector now
        l1_reg = alpha * l1_ratio * n_samples * mask 
        #
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples
        if not multi_output and sparse.isspmatrix(X):
            model = cd_fast_adaptive.sparse_enet_coordinate_descent_adaptive(
                coef_, l1_reg, l2_reg, X.data, X.indices,
                X.indptr, y, X_sparse_scaling,
                max_iter, tol, rng, random, positive)
        elif multi_output:
            l1_reg_new = alpha* l1_ratio * n_samples
            model = cd_fast_adaptive.enet_coordinate_descent_multi_task(
                coef_, l1_reg_new, l2_reg, X, y, max_iter, tol, rng, random)
        elif isinstance(precompute, np.ndarray):
            # We expect precompute to be already Fortran ordered when bypassing
            # checks
            if check_input:
                precompute = check_array(precompute, dtype=np.float64,
                                         order='C')
            model = cd_fast_adaptive.enet_coordinate_descent_gram_adaptive(
                coef_, l1_reg, l2_reg, precompute, Xy, y, max_iter,
                tol, rng, random, positive)
        elif precompute is False:
            model = cd_fast_adaptive.enet_coordinate_descent_adaptive(
                coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random,
                positive)
        else:
            raise ValueError("Precompute should be one of True, False, "
                             "'auto' or array-like. Got %r" % precompute)
        coef_, dual_gap_, eps_, n_iter_ = model
        coefs[..., i] = coef_
        dual_gaps[i] = dual_gap_
        n_iters.append(n_iter_)
        if dual_gap_ > eps_:
            warnings.warn('Objective did not converge.' +
                          ' You might want' +
                          ' to increase the number of iterations.' +
                          ' Fitting data with very small alpha' +
                          ' may cause precision problems.',
                          ConvergenceWarning)

        if verbose:
            if verbose > 2:
                print(model)
            elif verbose > 1:
                print('Path: %03i out of %03i' % (i, n_alphas))
            else:
                sys.stderr.write('.')

    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    return alphas, coefs, dual_gaps
