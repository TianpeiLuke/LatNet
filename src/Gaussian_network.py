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
import cvxpy as cvx
from sklearn.linear_model import lars_path, ridge_regression
from decimal import Decimal
from sklearn.covariance.empirical_covariance_ import log_likelihood
from graphical_lasso import *
from latent_graphical_lasso import *

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




